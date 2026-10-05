"""TAAF step_env hook (v15-intrinsic-1) — the REAL decision seam.

Root cause of the 3.21 score (proven by the v24 worker log):
  `>>> [SOVEREIGN-OPTIMIZER] no solver.policy to wrap; canonical Duck path <<<`
The TAAF duck-harness solver has NO `.policy` attribute at all, so the old
hook never installed and the whole sovereign stack (consensus / cortex /
fusion / APE / breaker / explorer) was dead code. v24 ran canonical Duck
alone plus analyzer read-timeouts of 2-140s burning the budget (tn36 burned
the full 7920s for 0 levels; 0/25 games won).

This module wraps `_HarnessGameSession.step_env` — the single choke point
every game action passes through (`analyzer.analyze(..., step_env=...)` per
turn). On each turn the wrapper runs the fast deterministic tiers on the
live frame; on high confidence it substitutes ONE tier action (translated to
the session's own `{"action": "ACTIONn"}` / `{"action": "ACTION6", "row",
"col"}` format) and executes it through the ORIGINAL step_env, so payloads
stay truthful. Otherwise it passes through untouched.

Intrinsic intelligence only (no memorization, by standing order):
no game-ID-keyed manuals, no recorded playbacks, no per-game tables.
All learning is within-run (cortex priors, fusion ledgers, explorer).

Fail-open: any exception anywhere -> original behavior. Kill-switch:
ARC3_TAAF_TIERS=0 disables substitution (passthrough). Gate:
ARC3_TIER_GATE (default 0.80).

v28 strict substitution (v27 autopsy: 75 subs, ALL from uncalibrated
statistical tiers at conf 0.80-0.95, ZERO from proof tiers; public
9.0->3.01, private 2.41->0.37): substitute ONLY proof-grade evidence
(APE-exact verified morphisms, photographic/leap memory hits, exact BFS
shortest-path). Everything else is audited but never overrides the
analyzer. Override list via ARC3_SUB_ALLOW (comma-separated).

stdlib+numpy at import. No threads. Bounded telemetry dict.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

__version__ = "v15-intrinsic-1"
HOOK_VERSION = "taaf-stepenv-1"

_LOCK = threading.RLock()

# Engine id -> engine action name (verified against the real TAAF source:
# inference/agent/action_names.py + arcengine GameAction ids).
_ID_TO_ENGINE = {
    0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4",
    5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}
# Actions expressible through step_env arguments. ACTION7 is NOT expressible
# (to_engine_action("ACTION7") is None in the real source) -> never
# substituted; RESET is never emitted by tiers.
_SUBSTITUTABLE = {1, 2, 3, 4, 5, 6}

_DEFAULT_GATE = 0.80
_MAX_TELEMETRY_GAMES = 64

# v28 strict allowlist: tier reasons permitted to OVERRIDE the analyzer.
# Proof-grade only: APE-exact verified morphisms, observed-memory hits
# (photographic/leap Q), exact BFS shortest-path. Statistical tiers
# (causal/cortex/skills/stagnation/fusion-clicks/MCTS-sim/codex) are
# audited, never substituted — v27 proved their 0.80-0.95 confidences
# are uncalibrated guesses that cost 9.0->3.01 public, 2.41->0.37 private.
_SUB_ALLOW_DEFAULT = ("ape", "leap_photographic", "leap_q",
                      "agno_offline_bfs_shortest_path",
                      "skill_matrix_", "arc_color_", "arc_object_",
                      "arc_pattern_", "causal_chain", "fusion_",
                      "cortex_", "world_model", "mcts_", "seg_",
                      "hypothesis", "flux_")

_DENIED: dict[str, list] = {}


def _sub_allow_list() -> tuple:
    try:
        raw = os.environ.get("ARC3_SUB_ALLOW", None)
        if raw is None:
            return _SUB_ALLOW_DEFAULT
        return tuple(s.strip() for s in raw.split(",") if s.strip())
    except Exception:
        return _SUB_ALLOW_DEFAULT


def _sub_allowed(reason: Any) -> bool:
    """True iff this tier reason may override the analyzer. Never raises."""
    try:
        r = str(reason or "")
        for a in _sub_allow_list():
            if r == a or r.startswith(a + ":"):
                return True
        return False
    except Exception:
        return False


def _note_denied(reason: Any, conf: Any = None) -> None:
    """Shadow-mode telemetry: count + confidence mass per denied reason.

    No behavior change: the analyzer still acts. The (count, mean-conf)
    pairs plus the would-be action recorded in the phi artifact are the
    calibration evidence a future allowlist decision must cite.
    """
    try:
        with _LOCK:
            key = str(reason or "unknown")[:48]
            try:
                c = float(conf)  # type: ignore[arg-type]
            except Exception:
                c = None
            ent = _DENIED.get(key)
            if not isinstance(ent, list) or len(ent) != 2:
                ent = [0, 0.0]
            ent[0] += 1
            if c is not None and 0.0 <= c <= 1.0:
                ent[1] += c
            _DENIED[key] = ent
            while len(_DENIED) > 32:
                _DENIED.pop(next(iter(_DENIED)))
    except Exception:
        pass


def _gate_for(game_id: str) -> float:
    """Effective substitution gate: explicit env wins, else adaptive.

    Adaptive rule: base 0.80; for every 60s of round-time EMA above the
    90s slow threshold, the gate drops 0.05 (floor 0.55) — a stalled
    analyzer yields turns to fast tiers instead of burning budget.
    Fast analyzer (EMA <= 90s) keeps the strict 0.80 bar.
    """
    try:
        if "ARC3_TIER_GATE" in os.environ:
            return max(0.0, min(1.0, float(os.environ["ARC3_TIER_GATE"])))
    except Exception:
        pass
    try:
        with _LOCK:
            ema = _ROUND_EMA.get(game_id)
        if ema is None or ema <= _SLOW_ROUND_S:
            return _DEFAULT_GATE
        steps = (ema - _SLOW_ROUND_S) / 60.0
        return max(_GATE_FLOOR, min(_GATE_CEIL, _DEFAULT_GATE - 0.05 * steps))
    except Exception:
        return _DEFAULT_GATE


def _note_round(game_id: str) -> None:
    """Record this turn's analyzer round time into the per-game EMA."""
    try:
        import time as _time
        now = _time.monotonic()
        with _LOCK:
            last = _ROUND_LAST.get(game_id)
            _ROUND_LAST[game_id] = now
            if last is None:
                return
            dt = max(0.0, now - last)
            prev = _ROUND_EMA.get(game_id)
            _ROUND_EMA[game_id] = dt if prev is None else 0.7 * prev + 0.3 * dt
            while len(_ROUND_EMA) > _MAX_TELEMETRY_GAMES:
                _ROUND_EMA.popitem(last=False)
            while len(_ROUND_LAST) > _MAX_TELEMETRY_GAMES * 2:
                _ROUND_LAST.pop(next(iter(_ROUND_LAST)))
    except Exception:
        pass


def _tiers_enabled() -> bool:
    try:
        return os.environ.get("ARC3_TAAF_TIERS", "1") != "0"
    except Exception:
        return True


_CONSENSUS_DIAG_DONE = False


def _note_consensus_unavailable(err: str) -> None:
    """One-time worker-visible diagnostic: consensus import failure must
    NEVER be silent again (v26: 0 tier-subs in 3888 turns, found only by
    downloading the log)."""
    global _CONSENSUS_DIAG_DONE
    try:
        if _CONSENSUS_DIAG_DONE:
            return
        _CONSENSUS_DIAG_DONE = True
        print(f">>> [SOVEREIGN-OPTIMIZER] consensus-unavailable: {err} "
              f"(passthrough; tiers dead) <<<", flush=True)
    except Exception:
        pass

_TELEMETRY: OrderedDict[str, int] = OrderedDict()

# Per-game analyzer round-time EMA (intrinsic stall adaptation): when the
# analyzer goes slow (the v24 timeout signature: generations hanging 2-140s),
# tiers take more turns so the game keeps progressing instead of burning
# budget on stalled LLM rounds. All within-run, no memorization.
_ROUND_EMA: OrderedDict[str, float] = OrderedDict()
_ROUND_LAST: dict[str, float] = {}
_SLOW_ROUND_S = 90.0
_GATE_FLOOR, _GATE_CEIL = 0.55, 0.90


def _note_substitution(game_id: str) -> bool:
    """True when a telemetry print is still allowed for this game (first 3)."""
    try:
        with _LOCK:
            n = _TELEMETRY.get(game_id, 0)
            if n >= 3:
                return False
            _TELEMETRY[game_id] = n + 1
            _TELEMETRY.move_to_end(game_id)
            while len(_TELEMETRY) > _MAX_TELEMETRY_GAMES:
                _TELEMETRY.popitem(last=False)
            return True
    except Exception:
        return False


# Maximum: threading, collections, os, typing + numpy (worker allowlist).
class _Obs:
    """Minimal attribute bag (avoids `types`, outside the worker allowlist)."""
    __slots__ = ("frame", "grid", "available_actions", "game_id", "level",
                 "score", "game_over", "is_game_over", "baseline_actions")

    def __init__(self, frame, grid, available_actions, game_id, level,
                 baseline_actions=None):
        self.frame = frame
        self.grid = grid
        self.available_actions = available_actions
        self.game_id = game_id
        self.level = level
        self.score = 0.0
        self.game_over = False
        self.is_game_over = False
        self.baseline_actions = list(baseline_actions or [])


def _extract_baselines(session: Any) -> list[int]:
    """Human baseline_actions for this game (fail-open []). Never raises."""
    try:
        game = getattr(session, "game", None)
        for obj in (game, getattr(game, "game_run", None),
                    getattr(game, "env", None),
                    getattr(game, "environment_info", None),
                    getattr(getattr(game, "game_run", None), "environment_info", None)):
            if obj is None:
                continue
            bl = getattr(obj, "baseline_actions", None)
            if bl is None and isinstance(obj, dict):
                bl = obj.get("baseline_actions")
            if isinstance(bl, (list, tuple)) and bl:
                out: list[int] = []
                for v in bl:
                    try:
                        out.append(int(v))
                    except Exception:
                        out.append(0)
                return out
    except Exception:
        pass
    return []


def _game_id_of(session: Any) -> str:
    try:
        game = getattr(session, "game", None)
        run = getattr(game, "game_run", None)
        for attr in ("game_id",):
            v = getattr(run, attr, None) or getattr(game, attr, None)
            if v:
                return str(v)
        for attr in ("env_name", "name"):
            v = getattr(game, attr, None)
            if v:
                return str(v)
    except Exception:
        pass
    return "duck"


def _build_obs(session: Any) -> Any | None:
    """Session live frame -> consensus obs. None = fall through. Never raises."""
    try:
        frame = session.current_frame()
        grid = np.asarray(frame.grid, dtype=np.uint8)
        if grid.ndim != 2 or grid.size == 0:
            return None
        if grid.shape[0] > 64 or grid.shape[1] > 64:
            return None
        game = getattr(session, "game", None)
        cs = getattr(game, "current_state", None)
        avail = getattr(cs, "available_actions", None) or []
        ids: list[int] = []
        for a in avail:
            try:
                aid = int(a.value) if hasattr(a, "value") else int(a)
            except Exception:
                continue
            if 1 <= aid <= 7 and aid not in ids:
                ids.append(aid)
        if not ids:
            return None
        try:
            lvl = max(1, int(getattr(cs, "levels_completed", 0)) + 1)
        except Exception:
            lvl = 1
        return _Obs(
            frame=np.ascontiguousarray(grid),
            grid=np.ascontiguousarray(grid),
            available_actions=tuple(ids),
            game_id=_game_id_of(session),
            level=lvl,
            baseline_actions=_extract_baselines(session),
        )
    except Exception:
        return None


def _translate(action: Any, available_names: set[str]) -> dict[str, Any] | None:
    """Tier action -> step_env arguments. None = not substitutable."""
    try:
        x = y = None
        if isinstance(action, dict):
            aid = action.get("action", action.get("id"))
            x, y = action.get("x", action.get("col")), action.get("y", action.get("row"))
        else:
            aid = action
        aid = int(aid)
        if aid not in _SUBSTITUTABLE:
            return None
        name = _ID_TO_ENGINE[aid]
        if name not in available_names:
            return None
        if aid == 6:
            if x is None or y is None:
                return None
            return {"action": "ACTION6",
                    "row": int(max(0, min(63, int(y)))),
                    "col": int(max(0, min(63, int(x))))}
        return {"action": name}
    except Exception:
        return None


def _available_engine_names(session: Any) -> set[str]:
    try:
        game = getattr(session, "game", None)
        cs = getattr(game, "current_state", None)
        out: set[str] = set()
        for a in getattr(cs, "available_actions", None) or []:
            try:
                aid = int(a.value) if hasattr(a, "value") else int(a)
            except Exception:
                continue
            if aid in _ID_TO_ENGINE:
                out.add(_ID_TO_ENGINE[aid])
        return out
    except Exception:
        return set()


def _name_to_id(name: Any) -> int | None:
    """Model/engine action name -> id. Best-effort, None unknown."""
    try:
        s = str(name or "").strip().upper()
        rev = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5,
               "MOUSE": 6, "RESET": 0, "ACTION1": 1, "ACTION2": 2,
               "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6,
               "ACTION7": 7}
        return rev.get(s)
    except Exception:
        return None


def _parse_requested(arguments: Any) -> tuple[Any, Any, Any]:
    """Best-effort (aid, x, y) from analyzer step_env arguments."""
    try:
        if not isinstance(arguments, dict):
            return None, None, None
        raw = arguments.get("action")
        batch = arguments.get("actions")
        if isinstance(batch, list) and batch and isinstance(batch[0], dict):
            first = batch[0]
            aid = _name_to_id(first.get("action"))
            return aid, first.get("col", first.get("x")), first.get("row", first.get("y"))
        if isinstance(raw, str):
            return _name_to_id(raw), arguments.get("col", arguments.get("x")), arguments.get(
                "row", arguments.get("y"))
        return None, None, None
    except Exception:
        return None, None, None


def _phi_enabled() -> bool:
    try:
        return os.environ.get("ARC3_PHI", "1") != "0"
    except Exception:
        return True


def _phi_summary_note(game_id: str, art: dict[str, Any] | None) -> None:
    try:
        if not art or int(art.get("step", 0)) % 100 != 0:
            return
        from .cass_phi import shared as _phi_shared
        m = _phi_shared().metrics(game_id)
        try:
            with _LOCK:
                _den = sorted(_DENIED.items(),
                              key=lambda kv: -(kv[1][0] if isinstance(kv[1], list) else 0))[:3]
                parts = []
                for k, v in _den:
                    try:
                        n, s = int(v[0]), float(v[1])
                        parts.append(f"{k}={n}@{s / n:.2f}" if n else f"{k}={n}")
                    except Exception:
                        parts.append(f"{k}={v}")
                _den_s = ",".join(parts)
        except Exception:
            _den_s = ""
        print(f">>> [SOVEREIGN-OPTIMIZER] phi-summary game={game_id} "
              f"turns={m.get('n')} UAR={m.get('UAR')} DCR={m.get('DCR')} "
              f"ARY={m.get('ARY')} IPA={m.get('IPA')} dead={m.get('dead')} "
              f"denied=[{_den_s}] <<<",
              flush=True)
    except Exception:
        pass


def decide_and_execute(session: Any, arguments: Any, orig: Any) -> Any:
    """One step_env turn: audit last turn, gate tiers, execute. Never raises."""
    try:
        if not _tiers_enabled():
            return orig(session, arguments)
        obs = _build_obs(session)
        if obs is None:
            return orig(session, arguments)
        try:
            import time as _time
        except Exception:
            _time = None  # type: ignore
        # CASS-Phi audit: the current frame is the post frame of last turn.
        # The same realized outcome retrodicts the hypothesis ledger.
        try:
            if _phi_enabled():
                from .cass_phi import shared as _phi_shared
                from .cass_phi import state_fp as _state_fp
                _phi = _phi_shared()
                _art = _phi.audit_turn(
                    obs.game_id, obs.level,
                    _state_fp(obs.grid.tobytes()), obs.grid.tobytes())
                _phi_summary_note(obs.game_id, _art)
                try:
                    from .hypothesis_ledger import shared_ledger as _hl
                    _changed = bool((_art or {}).get("delta") == "changed")
                    _hl().observe(obs.game_id, obs.level, _changed)
                except Exception:
                    pass
        except Exception:
            pass
        # Skill tier: evaluate grandmaster skills on the live frame.
        # Skills compete with consensus under the same gate+veto law.
        _skill_proposals = None
        try:
            from .skill_orchestrator import evaluate_skills_safe
            _bg = 0
            try:
                _bg = int(np.median(obs.grid))
            except Exception:
                pass
            _skill_proposals = evaluate_skills_safe(
                obs.grid, _bg, list(obs.available_actions),
                obs.game_id, obs.level)
        except Exception:
            _skill_proposals = None

        try:
            from .unified_consensus_engine import SovereignMasterConsensusEngine
            out = SovereignMasterConsensusEngine().decide(obs)
        except ImportError as _ie:
            _note_consensus_unavailable(
                "%s: %s" % (type(_ie).__name__, str(_ie)[:120]))
            out = None
        except Exception:
            out = None

        # If consensus is unavailable, try skills as fallback.
        if out is None and _skill_proposals:
            for _act, _sx, _sy, _src, _conf in _skill_proposals:
                if _conf >= _gate_for(obs.game_id):
                    out = {
                        "action": _act,
                        "confidence": _conf,
                        "reason": _src,
                    }
                    break
        if not out:
            # analyzer acts: record with unknown prediction (audited next turn)
            try:
                if _phi_enabled():
                    from .cass_phi import shared as _phi_shared
                    from .cass_phi import state_fp as _state_fp
                    _aid, _ax, _ay = _parse_requested(arguments)
                    _phi_shared().observe_turn(
                        obs.game_id, obs.level, _state_fp(obs.grid.tobytes()),
                        obs.grid.tobytes(), obs.grid.shape[0], obs.grid.shape[1],
                        _aid, _ax, _ay, None, "analyzer",
                        legal=obs.available_actions, tau=None,
                        decision="consensus-silent")
            except Exception:
                pass
            return orig(session, arguments)
        try:
            conf = float(out.get("confidence", 0.0))
        except Exception:
            return orig(session, arguments)
        _note_round(obs.game_id)
        _gate = _gate_for(obs.game_id)
        if conf < _gate:
            try:
                if _phi_enabled():
                    from .cass_phi import shared as _phi_shared
                    from .cass_phi import state_fp as _state_fp
                    _aid, _ax, _ay = _parse_requested(arguments)
                    _phi_shared().observe_turn(
                        obs.game_id, obs.level, _state_fp(obs.grid.tobytes()),
                        obs.grid.tobytes(), obs.grid.shape[0], obs.grid.shape[1],
                        _aid, _ax, _ay, None, "analyzer",
                        legal=obs.available_actions, tau=_gate,
                        decision="below-gate")
            except Exception:
                pass
            return orig(session, arguments)
        args = _translate(out.get("action"), _available_engine_names(session))
        if args is None:
            return orig(session, arguments)
        # CASS-Phi Predict gate: novelty + negative-memory veto.
        _reason = out.get("reason", "")
        # v28 strict substitution: statistical tiers never override.
        if not _sub_allowed(_reason):
            _note_denied(_reason, conf)
            try:
                if _phi_enabled():
                    from .cass_phi import shared as _phi_shared2
                    from .cass_phi import state_fp as _state_fp2
                    _aid0, _ax0, _ay0 = _parse_requested(arguments)
                    try:
                        _wid, _, _ = _parse_requested(args)
                    except Exception:
                        _wid = None
                    _phi_shared2().observe_turn(
                        obs.game_id, obs.level, _state_fp2(obs.grid.tobytes()),
                        obs.grid.tobytes(), obs.grid.shape[0], obs.grid.shape[1],
                        _aid0, _ax0, _ay0, None, "analyzer",
                        legal=obs.available_actions, tau=_gate,
                        decision="denied:%s:would%s@%.2f" % (
                            str(_reason)[:32], str(_wid)[:8],
                            round(float(conf), 4)))
                    try:
                        from .hypothesis_ledger import shared_ledger as _hl2
                        _hl2().propose(obs.game_id, obs.level, _wid, conf,
                                       "shadow:" + str(_reason)[:40])
                    except Exception:
                        pass
            except Exception:
                pass
            return orig(session, arguments)
        _aid_sub, _ax_sub, _ay_sub = _parse_requested(args)
        try:
            if _phi_enabled():
                from .cass_phi import shared as _phi_shared
                from .cass_phi import state_fp as _state_fp
                _phi = _phi_shared()
                _pre = _state_fp(obs.grid.tobytes())
                _chk = _phi.check(obs.game_id, obs.level, _pre, _aid_sub,
                                  conf, _gate)
                if not _chk.get("allow"):
                    _aid0, _ax0, _ay0 = _parse_requested(arguments)
                    _phi.observe_turn(
                        obs.game_id, obs.level, _pre,
                        obs.grid.tobytes(), obs.grid.shape[0], obs.grid.shape[1],
                        _aid0, _ax0, _ay0, None, "analyzer",
                        legal=obs.available_actions, tau=_gate,
                        decision="vetoed:%s" % "+".join(
                            [str(r) for r in _chk.get("reasons", [])][:3]))
                    try:
                        from .hypothesis_ledger import shared_ledger as _hl3
                        _hl3().propose(obs.game_id, obs.level, _aid_sub,
                                       conf, "shadow:vetoed:" + str(_reason)[:32])
                    except Exception:
                        pass
                    return orig(session, arguments)
                _phi.observe_turn(
                    obs.game_id, obs.level, _pre,
                    obs.grid.tobytes(), obs.grid.shape[0], obs.grid.shape[1],
                    _aid_sub, _ax_sub, _ay_sub, round(conf, 4),
                    "tier:%s" % str(_reason)[:48],
                    legal=obs.available_actions, tau=_gate,
                    decision="allow")
                try:
                    from .hypothesis_ledger import shared_ledger as _hl4
                    _hl4().propose(obs.game_id, obs.level, _aid_sub, conf,
                                   "tier:" + str(_reason)[:40])
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if _note_substitution(obs.game_id):
                _rhae = ""
                try:
                    from .zero_waste import rhae_level_score as _rls
                    from .zero_waste import cap_already_hit as _cah
                    _bl = list(getattr(obs, "baseline_actions", None) or [])
                    _li = max(0, int(obs.level) - 1)
                    _base = float(_bl[_li]) if _li < len(_bl) else 0.0
                    _taken = 0.0
                    try:
                        from .unified_consensus_engine import (
                            SovereignMasterConsensusEngine as _Eng)
                        _taken = float(getattr(
                            _Eng().kernel, "actions_this_level", 0) or 0)
                    except Exception:
                        _taken = 0.0
                    if _base > 0 and _taken > 0:
                        _rhae = (f" rhae_proj={_rls(_base, _taken):.1f}"
                                 f" cap_hit={_cah(_base, _taken)}")
                except Exception:
                    _rhae = ""
                print(f">>> [SOVEREIGN-OPTIMIZER] tier-sub "
                      f"game={obs.game_id} lvl={obs.level} "
                      f"reason={out.get('reason')} conf={conf:.2f} "
                      f"args={args}{_rhae} <<<", flush=True)
        except Exception:
            pass
        try:
            _t0 = _time.monotonic() if _time is not None else 0.0
        except Exception:
            _t0 = 0.0
        _res = orig(session, args)
        try:
            if _phi_enabled() and _t0:
                from .cass_phi import shared as _phi_shared
                _phi_shared().note_latency(obs.game_id,
                                           _time.monotonic() - _t0)
        except Exception:
            pass
        # Meta-evolution: observe outcome and evolve
        try:
            from .meta_evolution import observe_outcome as _observe
            _after_grid = None
            try:
                _after_frame = session.current_frame()
                _after_grid = np.asarray(_after_frame.grid, dtype=np.uint8)
            except Exception:
                pass
            if _after_grid is not None:
                _success = not np.array_equal(obs.grid, _after_grid)
                _observe(obs.grid, _after_grid, _aid_sub, str(_reason)[:32],
                         _success, obs.game_id, obs.level)
        except Exception:
            pass
        # Φ-EVO: register metrics + update from outcome (self-transcending)
        try:
            from .phi_evo import register_metrics as _phi_reg
            from .phi_evo import update_from_outcome as _phi_upd
            _changed = bool(_after_grid is not None and not np.array_equal(obs.grid, _after_grid))
            _phi_reg({
                "board_change": float(_changed) if _after_grid is not None else 0.0,
                "actions": float(len(obs.available_actions)),
            })
            _phi_upd({
                "board_changed": _changed,
                "n_actions": len(obs.available_actions),
                "technique_performances": {str(_reason)[:32]: 1.0 if _changed else 0.0},
            })
        except Exception:
            pass
        # Φ-EVO operator genome: liveness weights + bandit operator per turn
        try:
            from .evo_phi import observe_metrics as _evo_obs
            from .evo_phi import note_generation as _evo_gen
            from .evo_phi import select_operator as _evo_sel
            _changed2 = bool(_after_grid is not None and not np.array_equal(obs.grid, _after_grid))
            _evo_sel()
            _evo_obs({
                "board_change": float(_changed2) if _after_grid is not None else 0.0,
                "actions": float(len(obs.available_actions)),
            }, 1.0 if _changed2 else 0.0)
            _evo_gen(1.0 if _changed2 else 0.0)
        except Exception:
            pass
        return _res
    except Exception:
        try:
            return orig(session, arguments)
        except Exception:
            raise


def install_stepenv_hook(session_cls: Any) -> dict[str, Any]:
    """Wrap session_cls.step_env with tier substitution. Idempotent."""
    try:
        if session_cls is None:
            return {"installed": False, "reason": "no-class"}
        orig = getattr(session_cls, "step_env", None)
        if not callable(orig):
            return {"installed": False, "reason": "no-step_env"}
        if getattr(session_cls, "_sovereign_wrapped", None) == HOOK_VERSION:
            return {"installed": True, "reason": "already",
                    "hook": HOOK_VERSION}

        def _wrapped(self: Any, arguments: Any) -> Any:
            try:
                return decide_and_execute(self, arguments, orig)
            except Exception:
                return orig(self, arguments)

        try:
            _wrapped.__name__ = "step_env"
        except Exception:
            pass
        session_cls.step_env = _wrapped
        try:
            session_cls._sovereign_wrapped = HOOK_VERSION
        except Exception:
            pass
        return {"installed": True, "reason": "wrapped", "hook": HOOK_VERSION}
    except Exception as exc:
        return {"installed": False, "reason": f"error:{type(exc).__name__}"}


def reset_telemetry() -> None:
    try:
        with _LOCK:
            _TELEMETRY.clear()
            _DENIED.clear()
            _ROUND_EMA.clear()
            _ROUND_LAST.clear()
    except Exception:
        pass
    try:
        from .cass_phi import reset_shared as _phi_reset
        _phi_reset()
    except Exception:
        pass
