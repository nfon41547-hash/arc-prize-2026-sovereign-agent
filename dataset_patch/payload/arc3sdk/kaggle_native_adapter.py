"""Kaggle-native ARC-AGI-3 adapter (v3 integration + sovereign explorer tier).

Merges the proven ``ARC3_RTX6000_Kaggle_Native_v3`` contract with this repo's
cross-game exploration transfer, in ONE self-contained module (stdlib + numpy
at import time; ``arcengine`` and the explorer resolve lazily):

  * Official interface: ``is_done(frames, latest_frame)`` /
    ``choose_action(frames, latest_frame) -> GameAction``. Use
    :func:`make_official_agent` worker-side for a real ``MyAgent(Agent)``.
  * Single-action guarantee (v3 double-execution fix): tiers NEVER touch the
    environment; ``choose_action`` builds exactly ONE ``GameAction`` and the
    official framework executes it.
  * ``FrameData.frame`` (list of 2-D frames) -> latest grid; ``available_actions``
    (ids / names / members) -> ``GameAction``; ACTION6 coords clamped 0..63
    (pydantic rejects negatives, so this is a crash guard, not policy);
    GAME_OVER / NOT_PLAYED -> RESET; WIN -> done.
  * 9-hour scheduler (32400s hard, 600s reserve, process-wide, injectable clock).
  * LLM watchdog (90s timeout enforced with threads — works on any OS, unlike
    SIGALRM — 45s slow-call detection, degraded planning mode, always falls
    back to a legal action).
  * VRAM governor (normal <88%, pressure >=88%, critical >=94%; trims cache,
    scales generation budget; never raises).
  * Explorer tier: this repo's cross-game transfer (dead-action avoidance,
    positive transfer, pattern + sequence memory) votes alongside heuristics.
    LLM tiers are optional (``llm=None`` runs fully offline).

No sockets, no subprocesses at import, never raises out of ``choose_action``.
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from collections.abc import Callable

import numpy as np

from arc3sdk.competition_contract import Decision, Observation
from arc3sdk.egcd import EGCDPolicy

HARD_DEADLINE_S = 32400.0
RESERVE_S = 600.0
LLM_TIMEOUT_S = 90.0
LLM_SLOW_S = 45.0
VRAM_PRESSURE = 0.88
VRAM_CRITICAL = 0.94
COORD_MAX = 63
MAX_ACTIONS_PER_GAME = 120


# The official harness may construct the agent on one thread and execute the
# first decision on another.  Keep the import/registry prime process-wide and
# idempotent so that a lazy import cannot become a first-turn latency spike.
_WARMUP_LOCK = threading.Lock()
_WARMUP_STATUS: dict[str, object] = {"done": False, "ok": False, "errors": []}


def warmup() -> dict[str, object]:
    """Prime adapter dependencies before an episode starts.

    This performs no environment calls and remains fail-open: a missing
    optional registry must not prevent the legal fallback path.  The returned
    status is suitable for structured worker logs.
    """
    global _WARMUP_STATUS
    if bool(_WARMUP_STATUS.get("done")):
        return dict(_WARMUP_STATUS)
    with _WARMUP_LOCK:
        if bool(_WARMUP_STATUS.get("done")):
            return dict(_WARMUP_STATUS)
        started = time.perf_counter()
        errors: list[str] = []
        try:
            # Resolve the lazy module and exercise its read-only proposal path
            # once.  Reset immediately so the prime cannot affect a game.
            from arc3sdk import exploration_registry as er

            er.propose("__adapter_warmup__", np.zeros((2, 2), dtype=np.uint8), [1, 2])
            with contextlib.suppress(Exception):
                er.reset_game("__adapter_warmup__")
        except Exception as exc:
            errors.append(f"explorer:{type(exc).__name__}")
        try:
            # The official GameAction model pulls in arcengine/pydantic on its
            # first construction.  Resolve that cost during warmup so the
            # first live action is not charged with a 100-200ms import spike.
            _game_action_cls()
        except Exception as exc:
            errors.append(f"arcengine:{type(exc).__name__}")
        _WARMUP_STATUS = {
            "done": True,
            "ok": not errors,
            "errors": errors,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        return dict(_WARMUP_STATUS)


# ── frame / action mapping ──────────────────────────────────────────────

def frame_to_grid(latest_frame: Any) -> np.ndarray:
    """Latest 2-D frame -> uint8 grid. Never raises."""
    try:
        raw = getattr(latest_frame, "frame", None)
        if raw is None:
            return np.zeros((8, 8), dtype=np.uint8)
        # FrameData normally carries a list of frames, but direct ndarray
        # frames are valid in local/API adapters.  Avoid boolean evaluation of
        # numpy arrays (``array or []`` raises on multi-element arrays).
        if isinstance(raw, np.ndarray):
            arr = np.asarray(raw, dtype=np.uint8)
            if arr.ndim == 2:
                cur = arr
            elif arr.ndim >= 3:
                cur = arr[-1]
            else:
                return np.zeros((8, 8), dtype=np.uint8)
        else:
            # A FrameData list may contain differently shaped historical
            # frames, so converting the whole list can raise a ragged-array
            # error. Only the latest frame is part of the decision contract.
            try:
                if len(raw) == 0:
                    return np.zeros((8, 8), dtype=np.uint8)
                cur = raw[-1]
            except Exception:
                return np.zeros((8, 8), dtype=np.uint8)
        g = np.asarray(cur, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return np.zeros((8, 8), dtype=np.uint8)
        return g
    except Exception:
        return np.zeros((8, 8), dtype=np.uint8)


def available_to_ids(avail: Any) -> list[int]:
    """available_actions (ids/names/members) -> sorted legal ids 1..7."""
    try:
        out: list[int] = []
        for v in list(avail or []):
            try:
                if hasattr(v, "value") and isinstance(v.value, int):
                    out.append(int(v.value))
                    continue
                if isinstance(v, str):
                    s = v.strip().upper()
                    if s == "RESET":
                        continue
                    if s.startswith("ACTION") and s[6:].isdigit():
                        out.append(int(s[6:]))
                    continue
                out.append(int(v))
            except Exception:
                continue
        out = sorted({a for a in out if 1 <= a <= 7})
        return out or [1, 2, 3, 4, 5, 6, 7]
    except Exception:
        return [1, 2, 3, 4, 5, 6, 7]


def clamp_coord(v: Any) -> int:
    try:
        return max(0, min(COORD_MAX, int(v)))
    except Exception:
        return 0


def _split_action(act: Any) -> tuple[Any, Any, Any]:
    """Consensus/explorer output -> (action_id, x, y). Never raises.

    Accepts ints, {'action'|'name'|'id'} dicts (x/y/col/row/data coords),
    (action, x, y) tuples, and objects with .id. RESET/bools/garbage -> None.
    """
    try:
        if act is None or isinstance(act, bool):
            return None, None, None
        if isinstance(act, int):
            return act, None, None
        if isinstance(act, str):
            s = act.strip().upper()
            if s == "RESET":
                return None, None, None
            if "ACTION" in s:
                try:
                    return int(s.replace("ACTION", "")), None, None
                except Exception:
                    return None, None, None
            return None, None, None
        if isinstance(act, dict):
            n = act.get("action", act.get("name", act.get("id", None)))
            if isinstance(n, str):
                s = n.strip().upper()
                if s == "RESET":
                    return None, None, None
                n = int(s.replace("ACTION", "")) if "ACTION" in s else None
            data = act.get("data")
            if not isinstance(data, dict):
                data = {}
            x = act.get("x", act.get("col", data.get("x")))
            y = act.get("y", act.get("row", data.get("y")))
            return n, x, y
        if isinstance(act, (tuple, list)):
            n = act[0] if len(act) > 0 else None
            x = act[1] if len(act) > 1 else None
            y = act[2] if len(act) > 2 else None
            return n, x, y
        _id = getattr(act, "id", None)
        if isinstance(_id, int) and not isinstance(_id, bool):
            data = getattr(act, "data", None)
            if not isinstance(data, dict):
                data = {}
            x = getattr(act, "x", data.get("x"))
            y = getattr(act, "y", data.get("y"))
            return _id, x, y
    except Exception:
        pass
    return None, None, None


def _valid_coord(v: Any) -> int | None:
    try:
        if isinstance(v, bool):
            return None
        f = float(v)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return max(0, min(COORD_MAX, int(f)))
    except Exception:
        return None


def normalize_action_output(act: Any, legal: Any, game_id: str = "",
                            grid: Any = None) -> tuple[Any, bool]:
    """Consensus/explorer output -> TAAF action payload. Never raises.

    Returns (payload, True): GameAction with coords for ACTION6 when the
    runtime wheel supports it, else the bare legal int (status quo).
    Returns (None, False): caller must fall through to Duck. In particular
    a coardless consensus dict never becomes a blind click (wasted turn);
    a coardless explorer vote keeps the proven bare-int path.
    When *grid* is provided, ACTION6 coords additionally pass ClickGuard
    (massive-background / pixel-noise / out-of-bounds rejected to Duck).
    """
    try:
        n, x, y = _split_action(act)
        if not isinstance(n, int) or isinstance(n, bool):
            return None, False
        if n not in list(legal or []):
            return None, False
        if n == 6:
            cx, cy = _valid_coord(x), _valid_coord(y)
            if cx is not None and cy is not None:
                if grid is not None:
                    try:
                        from arc3sdk import zero_waste as _zw
                        _ok, _why = _zw.click_guard(grid, cx, cy)
                        if not _ok:
                            return None, False  # guarded: let Duck decide
                    except Exception:
                        pass
                try:
                    return build_game_action(6, game_id, cx, cy), True
                except Exception:
                    pass
                return 6, True  # local wheel: bare id (status quo)
            if isinstance(act, dict):
                return None, False  # coardless consensus: let Duck decide
            return 6, True  # coardless vote: proven bare-int path
        if isinstance(act, int):
            return n, True
        if hasattr(act, "id"):
            return act, True
        return n, True
    except Exception:
        return None, False


def _game_action_cls() -> Any:
    from arcengine import GameAction  # local wheel / Kaggle runtime wheel
    return GameAction


def build_game_action(act_id: int, game_id: str = "", x: Any = None, y: Any = None) -> Any:
    """Construct exactly ONE GameAction. ACTION6 coords clamped 0..63."""
    GameAction = _game_action_cls()
    action = GameAction.from_id(int(act_id))
    data: dict[str, Any] = {"game_id": str(game_id or "")}
    try:
        if bool(action.is_complex()):
            data["x"] = clamp_coord(x)
            data["y"] = clamp_coord(y)
    except Exception:
        pass
    action.set_data(data)
    with contextlib.suppress(Exception):
        action.reasoning = {"agent": "arc3-kaggle-native", "source": "sovereign+xfer"}
    return action


def validate_click(grid: Any, x: Any, y: Any) -> tuple[bool, str]:
    """ClickGuard without importing zero_waste at module top. Never raises."""
    try:
        from arc3sdk import zero_waste as _zw
        return _zw.click_guard(grid, x, y)
    except Exception:
        return False, "guard-unavailable"


def reset_allowed_for(failed_count: int, legal: Any, stagnation: int = 0,
                      zero_diff_streak: int = 0) -> bool:
    """Reset economy: reset ONLY when the pool is exhausted or deeply stalled."""
    try:
        try:
            n_legal = len(list(legal or []))
        except Exception:
            n_legal = 7
        from arc3sdk import zero_waste as _zw
        return _zw.reset_allowed(int(failed_count), n_legal, int(stagnation),
                                 int(zero_diff_streak))
    except Exception:
        return False


def rhae_level_score(baseline: float, taken: float) -> float:
    """Exact per-level RHAE score (mirrors eval.py). Never raises."""
    try:
        from arc3sdk import zero_waste as _zw
        return _zw.rhae_level_score(baseline, taken)
    except Exception:
        return 0.0


def _state_name(state: Any) -> str:
    try:
        name = getattr(state, "name", None)
        if isinstance(name, str):
            return name.upper()
        value = getattr(state, "value", None)
        if isinstance(value, str):
            return value.upper()
        return str(state).upper()
    except Exception:
        return ""


# ── scheduler / watchdog / governor ─────────────────────────────────────

class RunScheduler:
    """Process-wide monotonic budget. Injectable clock for tests."""

    def __init__(self, hard_seconds: float = HARD_DEADLINE_S,
                 reserve_seconds: float = RESERVE_S,
                 clock: Callable[[], float] | None = None) -> None:
        try:
            self.hard = float(hard_seconds)
        except Exception:
            self.hard = HARD_DEADLINE_S
        try:
            self.reserve = float(reserve_seconds)
        except Exception:
            self.reserve = RESERVE_S
        self._clock = clock or time.monotonic
        try:
            self.t0 = float(self._clock())
        except Exception:
            self.t0 = 0.0

    def elapsed(self) -> float:
        try:
            return max(0.0, float(self._clock()) - self.t0)
        except Exception:
            return 0.0

    def remaining(self) -> float:
        return max(0.0, self.hard - self.elapsed())

    def soft_remaining(self) -> float:
        return max(0.0, self.hard - self.reserve - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def critical(self) -> bool:
        return self.remaining() <= self.reserve


class LLMWatchdog:
    """Enforced generation timeout (threads, portable) + slow-call tracking."""

    def __init__(self, timeout_s: float = LLM_TIMEOUT_S, slow_s: float = LLM_SLOW_S,
                 executor_factory: Callable[[], Any] | None = None) -> None:
        self.timeout = float(timeout_s)
        self.slow = float(slow_s)
        self._factory = executor_factory or (lambda: ThreadPoolExecutor(max_workers=1))
        self.calls = 0
        self.slow_calls = 0
        self.timeouts = 0
        self._lock = threading.Lock()

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[bool, Any, float]:
        """Returns (ok, value, elapsed). Timeout -> (False, None, elapsed)."""
        start = time.monotonic()
        ex = self._factory()
        try:
            fut = ex.submit(fn, *args, **kwargs)
            try:
                value = fut.result(timeout=self.timeout)
                ok = True
            except Exception:
                with contextlib.suppress(Exception):
                    fut.cancel()
                with self._lock:
                    self.timeouts += 1
                ok, value = False, None
        finally:
            with contextlib.suppress(Exception):
                ex.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - start
        with self._lock:
            self.calls += 1
            if elapsed >= self.slow:
                self.slow_calls += 1
        return ok, value, elapsed

    @property
    def degraded(self) -> bool:
        try:
            with self._lock:
                if self.timeouts >= 1:
                    return True
                return self.calls >= 3 and self.slow_calls / max(1, self.calls) > 0.5
        except Exception:
            return False


def _read_vram_mib() -> tuple[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip().splitlines()[0]
        used, total = [int(float(x.strip())) for x in out.split(",")[:2]]
        return used, total
    except Exception:
        return 0, 0


class VRAMGovernor:
    """VRAM pressure governor. Injectable reader for tests. Never raises."""

    def __init__(self, reader: Callable[[], tuple[int, int]] | None = None) -> None:
        self._reader = reader or _read_vram_mib
        self._cache: tuple[float, tuple[int, int]] = (0.0, (0, 0))
        try:
            self.pressure = float(os.environ.get("ARC3_VRAM_PRESSURE", str(VRAM_PRESSURE)))
            self.critical = float(os.environ.get("ARC3_VRAM_CRITICAL", str(VRAM_CRITICAL)))
        except Exception:
            self.pressure, self.critical = VRAM_PRESSURE, VRAM_CRITICAL

    def state(self) -> dict:
        try:
            now = time.monotonic()
            if now - self._cache[0] > 5.0:
                with contextlib.suppress(Exception):
                    self._cache = (now, self._reader())
            used, total = self._cache[1]
            frac = (used / total) if total else 0.0
            if frac >= self.critical:
                level = "critical"
            elif frac >= self.pressure:
                level = "pressure"
            else:
                level = "normal"
            return {"used_mib": used, "total_mib": total, "fraction": frac, "level": level}
        except Exception:
            return {"used_mib": 0, "total_mib": 0, "fraction": 0.0, "level": "normal"}

    def budget_scale(self) -> float:
        try:
            level = self.state()["level"]
            return {"normal": 1.0, "pressure": 0.5, "critical": 0.25}.get(level, 1.0)
        except Exception:
            return 1.0

    def trim(self) -> None:
        with contextlib.suppress(Exception):
            gc.collect()
        try:
            import torch  # lazy: absent on CPU-only workers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ── explorer tier (cross-game transfer vote) ────────────────────────────

_TAG_CONF = {"explorer_xfer_prior": 0.55, "pattern_": 0.65, "chain_": 0.60}


def _default_propose(grid: np.ndarray, legal: list[int], game_id: str, stagnation: int) -> Any:
    from arc3sdk import exploration_registry as er  # vendored / runtime copy
    return er.propose(game_id, grid, legal, stagnation=stagnation)


def _default_observe(game_id: str, action_name: str, grid: np.ndarray, changed: bool) -> None:
    try:
        from arc3sdk import exploration_registry as er  # vendored / runtime copy
        name = str(action_name or "ACTION1").upper()
        act = 6 if name == "ACTION6" else int(name.replace("ACTION", "")) if name.startswith("ACTION") else 1
        if act not in (1, 2, 3, 4, 5, 6, 7):
            return
        er.observe(game_id, grid, act, None, None, bool(changed))
        er.record_transition(game_id, act, bool(changed), grid)
    except Exception:
        pass


def _grid_sig(grid: np.ndarray) -> str:
    try:
        import hashlib as _hl
        return _hl.blake2b(np.ascontiguousarray(grid).tobytes(), digest_size=8).hexdigest()
    except Exception:
        return ""


class ExplorerTier:
    """Cross-game transfer vote++. Injectable propose_fn for hermetic tests."""

    def __init__(self, propose_fn: Callable[..., Any] | None = None) -> None:
        self._propose = propose_fn or _default_propose

    def vote(self, grid: np.ndarray, legal: list[int], game_id: str = "",
             stagnation: int = 0) -> dict | None:
        try:
            got = self._propose(grid, list(legal), game_id, stagnation)
            if not got:
                return None
            act, x, y, tag, conf = got
            act = int(act)
            if act not in legal:
                return None
            try:
                conf = float(conf)
            except Exception:
                conf = 0.48
            for prefix, boost in _TAG_CONF.items():
                if str(tag).startswith(prefix):
                    conf = max(conf, boost)
                    break
            out: dict[str, Any] = {"action": f"ACTION{act}", "confidence": conf, "tag": str(tag)}
            if act == 6:
                out["data"] = {"x": x, "y": y}
            return out
        except Exception:
            return None


# ── the agent ───────────────────────────────────────────────────────────

class KaggleNativeAgent:
    """Framework-agnostic core. Never touches the environment (no env handle exists)."""

    def __init__(self, game_id: str = "", max_actions: int = MAX_ACTIONS_PER_GAME,
                 scheduler: RunScheduler | None = None,
                 governor: VRAMGovernor | None = None,
                 watchdog: LLMWatchdog | None = None,
                 propose_fn: Callable[..., Any] | None = None,
                 observe_fn: Callable[..., Any] | None = None) -> None:
        self.game_id = str(game_id or "")
        try:
            self.max_actions = int(max_actions)
        except Exception:
            self.max_actions = MAX_ACTIONS_PER_GAME
        self.scheduler = scheduler or RunScheduler()
        self.governor = governor or VRAMGovernor()
        self.watchdog = watchdog or LLMWatchdog()
        self.explorer = ExplorerTier(propose_fn)
        self._observe = observe_fn or _default_observe
        self.steps = 0
        self._track: dict[str, dict] = {}
        self.egcd = EGCDPolicy() if os.getenv("ARC3_ENABLE_EGCD", "0").lower() not in {"0", "false", "off", "no"} else None
        self._egcd_previous: dict[str, np.ndarray] = {}
        # Construction is outside the action loop in the official harness;
        # this call is idempotent and only primes imports/registries.
        warmup()

    def _track_step(self, gid: str, grid: np.ndarray, action_name: str | None = None) -> int:
        """Update per-game stagnation + close the learning loop. Returns stagnation.

        With ``action_name=None`` the stored last action is preserved (observe the
        previous transition, then the caller records the newly chosen action).
        """
        try:
            if len(self._track) > 32:
                for k in list(self._track)[:8]:
                    self._track.pop(k, None)
            tr = self._track.setdefault(gid, {"sig": None, "stag": 0, "last_action": ""})
            sig = _grid_sig(grid)
            prev_sig = tr.get("sig")
            if prev_sig is None:
                tr["sig"] = sig
                tr["stag"] = 0
            else:
                changed = bool(sig and prev_sig and sig != prev_sig)
                if changed:
                    tr["stag"] = 0
                else:
                    tr["stag"] = min(99, int(tr.get("stag", 0)) + 1)
                last_action = str(tr.get("last_action") or "")
                if last_action:
                    with contextlib.suppress(Exception):
                        self._observe(gid, last_action, grid, changed)
                tr["sig"] = sig
            if action_name:
                tr["last_action"] = str(action_name)
            return int(tr.get("stag", 0))
        except Exception:
            return 0

    def is_done(self, frames: Any, latest_frame: Any) -> bool:
        try:
            if _state_name(getattr(latest_frame, "state", "")) == "WIN":
                return True
            try:
                if self.scheduler.expired():
                    return True
            except Exception:
                pass
            try:
                n = len(frames) if frames is not None else self.steps
            except Exception:
                n = self.steps
            return int(n) >= self.max_actions
        except Exception:
            return True  # fail-closed on done-ness: never loop forever

    def _fallback_name(self, legal: list[int]) -> str:
        try:
            for a in legal:
                if a not in (6,):
                    return f"ACTION{a}"
            if legal:
                return f"ACTION{legal[0]}"
        except Exception:
            pass
        return "ACTION1"

    def choose_action(self, frames: Any, latest_frame: Any) -> Any:
        """Exactly ONE GameAction out; zero environment calls. Never raises."""
        try:
            self.steps += 1
            state = _state_name(getattr(latest_frame, "state", ""))
            if state in ("GAME_OVER", "NOT_PLAYED"):
                GameAction = _game_action_cls()
                return GameAction.RESET
            grid = frame_to_grid(latest_frame)
            legal = available_to_ids(getattr(latest_frame, "available_actions", None))
            gid = str(getattr(latest_frame, "game_id", None) or self.game_id or "native")
            # Validate the exact boundary before any policy output reaches the
            # official wheel.  Invalid observations fail closed to a legal move.
            obs = Observation(gid, int(getattr(latest_frame, "levels_completed", 0) or 0), grid, tuple(legal), state)
            if self.egcd is not None:
                previous = self._egcd_previous.get(gid)
                if previous is not None:
                    self.egcd.observe(gid, previous, int(self._track.get(gid, {}).get("last_id", legal[0] if legal else 1)), grid,
                                      game_over=state == "GAME_OVER")
                self._egcd_previous[gid] = grid.copy()
            stag = self._track_step(gid, grid)  # observe prev transition; keep last action
            vote = self.explorer.vote(grid, legal, gid, stagnation=stag)
            if self.egcd is not None:
                egcd_action, egcd_conf, egcd_reason = self.egcd.choose(gid, grid, legal, stagnation=stag)
                vote = {"action": f"ACTION{egcd_action}", "data": {}, "confidence": egcd_conf, "reason": egcd_reason}
            if vote is not None:
                name = str(vote.get("action", "ACTION1")).upper()
                data = vote.get("data") if isinstance(vote.get("data"), dict) else {}
            else:
                name, data = self._fallback_name(legal), {}
            num = int(name.replace("ACTION", "")) if name.startswith("ACTION") else 1
            x = (data or {}).get("x") if isinstance(data, dict) else None
            y = (data or {}).get("y") if isinstance(data, dict) else None
            if num == 6 and (x is None or y is None):
                # Preserve the adapter's historical legal ACTION6 fallback;
                # coordinates are made explicit for the typed contract and
                # clipped by the wheel builder below.
                x, y = 0, 0
            if num == 6:
                x = max(0, min(int(x), grid.shape[1] - 1))
                y = max(0, min(int(y), grid.shape[0] - 1))
            Decision(num, "fusion" if vote is not None else "fallback", 0.0, x, y).validate(obs)
            with contextlib.suppress(Exception):
                self._track[gid]["last_action"] = name
                self._track[gid]["last_id"] = num
            return self._build_named(name, legal, gid, data)
        except Exception:
            try:
                GameAction = _game_action_cls()
                return GameAction.from_id(1)
            except Exception:
                return None

    def _build_named(self, name: str, legal: list[int], gid: str, data: dict) -> Any:
        try:
            GameAction = _game_action_cls()
            if name == "RESET":
                return GameAction.RESET
            num = int(name.replace("ACTION", "")) if name.startswith("ACTION") else 1
            if num not in legal:
                num = legal[0] if legal else 1
            x = clamp_coord((data or {}).get("x", 0))
            y = clamp_coord((data or {}).get("y", 0))
            return build_game_action(num, gid, x, y)
        except Exception:
            return build_game_action(1, gid, 0, 0)


def make_official_agent() -> Any:
    """Worker-side ``MyAgent(Agent)`` delegating to :class:`KaggleNativeAgent`.

    Importing ``agents.agent``/``arcengine`` happens HERE (not at module import)
    so this module stays importable without the official wheel.
    """
    from agents.agent import Agent  # official starter layout, worker-side

    class MyAgent(Agent):
        MAX_ACTIONS = MAX_ACTIONS_PER_GAME  # v3: 120, not the starter default 80

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._core = KaggleNativeAgent(game_id=getattr(self, "game_id", ""))

        @property
        def name(self) -> str:  # type: ignore[override]
            try:
                return f"{self.game_id}.arc3_kaggle_native"
            except Exception:
                return "arc3_kaggle_native"

        def is_done(self, frames: Any, latest_frame: Any) -> bool:
            return self._core.is_done(frames, latest_frame)

        def choose_action(self, frames: Any, latest_frame: Any) -> Any:
            return self._core.choose_action(frames, latest_frame)

    return MyAgent
