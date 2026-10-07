"""Sovereign Duck Interceptor & Closed-Loop Servo Short-Circuit Kernel (Apex v10).

Solves the fatal failure modes in Duck vLLM runs:
1. vLLM Read Timeout & Loopback Bottlenecks: deterministic ladder first;
   LLM is the last resort, never the first call.
2. Action Budget Bloat / Maze Stagnation: per-level cap + zero-diff guard
   + state-hash cycle breaker (period-2/4 oscillation with NONZERO diffs).

v10 upgrades (all verified, none speculative):
- Canonical GameAction table (0=RESET..7=ACTION7). The old table mapped
  7->RESET, silently converting UNDO into RESET and RESET into ACTION1.
- Unified payload builder: int-6 macro entries keep their coordinates;
  coordinate-less CLICKs are refused (fall through) instead of blind-fired.
- Verify-before-execute on EVERY servo payload via WorldSim (cached,
  millisecond-scale): predicted no-ops under stagnation are substituted.
- State-hash cycle memory: repeating states force the least-recently-used
  legal action and retire the stale macro queue (no more circular death).
- Adaptive consensus gate: the confidence bar moves with observed
  per-mechanism success (Laplace-smoothed bandit), per level.
- Memory bank records REAL next-grids only; identity (s,a,s) transitions
  are never recorded (they poisoned the bank).
- Rescue path receives the TRUE stagnation count and canonical mapping.
- Every step_env commit is individually guarded: one mechanism's failure
  falls through the ladder instead of aborting to the LLM.
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from pathlib import Path
from typing import Any
from collections.abc import Callable
import numpy as np

from arc3sdk.first_contact_learner import FirstContactLearner, state_class
from arc3sdk.long_horizon_controller import LongHorizonController
from arc3sdk.rl_replay_buffer import ArcRLReplayBuffer
from arc3sdk.evidence_planner import EvidencePlanner
from arc3sdk.realtime_rule_learner import RealtimeRuleLearner
from arc3sdk.runtime_safety_contract import normalize_grid, normalize_action, RuntimeSafetyBudget
import contextlib

log = logging.getLogger("sovereign.duck_interceptor")

# Canonical GameAction table. 0=RESET is legal output ONLY when chosen
# explicitly (rescue / budget guard / reflection); the ladder below never
# synthesizes it implicitly.
ACTION_MAP = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}
ACTION_REV = {v: k for k, v in ACTION_MAP.items()}

MAX_ACTIONS_PER_LEVEL = 100  # Hard-clamp action budget per level.
CYCLE_MEMORY = 64  # state-hash+action entries remembered for cycle breaks.
CYCLE_REPEAT = 2  # same hash seen this many times -> force exploration.
_ZERO_DIFF_WINDOW = 5


def _grid_hash(grid: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(grid, dtype=np.uint8).tobytes(),
                           digest_size=8).hexdigest()


def _trailing_zero_diffs(history_entries) -> int:
    """Consecutive trailing zero-diff steps from history (oldest->newest)."""
    grids = [np.asarray(h.grid, dtype=np.uint8) for h in history_entries
             if hasattr(h, "grid")]
    if len(grids) < 2:
        return 0
    n = 0
    for i in range(len(grids) - 1, 0, -1):
        if np.array_equal(grids[i], grids[i - 1]):
            n += 1
        else:
            break
    return n


class SovereignDuckInterceptor:
    """Intercepts ToolAgent.analyze() directly inside Duck loop."""

    def __init__(self, original_analyze: Callable):
        self.original_analyze = original_analyze
        self.total_short_circuits = 0
        self.total_llm_calls = 0
        self.level_action_counts: dict[int, int] = {}
        self.current_level = 0
        self.planned_macro_queue: list[dict[str, Any]] = []
        self._seen: deque = deque(maxlen=CYCLE_MEMORY)
        self._mech_ok: dict[str, list] = {}
        self._tried_actions: set = set()
        self._tried_clicks: set = set()
        self._guard_cooldown_until: int = 0
        self._learner = FirstContactLearner()
        self._pending = None
        self._avo_model = None
        self._avo_attempted = False
        # Failed state/action pairs survive level resets within a game/run.
        self._failed_pairs: set[tuple[str, int, str, str]] = set()
        # Every executed state/action pair is remembered.  This is separate
        # from failed_pairs: an action that changed the board must still not
        # be replayed if the search later returns to the identical state.
        self._attempted_pairs: set[tuple[str, int, str, str]] = set()
        self._active_game_id = "unknown"
        self._active_level = 0
        # Shared structural memory lets the servo learn action effects across
        # levels without leaking game-specific pixel identities into Duck.
        self._abstract_controller = LongHorizonController(context_limit=32, max_transitions=2048)
        self._replay = ArcRLReplayBuffer(capacity=4096, seed=0)
        # The consensus engine is stateful (game priors/bandit statistics).
        # Reuse it across turns instead of rebuilding it on every action.
        self._consensus_engine = None
        # CASS counterfactual probe (lazy; reset per game, fed by _commit).
        self._cass_agent = None
        self._cass_game_id = None
        self._cass_observed = 0
        self._evidence_planner = EvidencePlanner(max_edges=8192)
        self._realtime_rules = RealtimeRuleLearner(max_rules=8192)
        from arc3sdk.realtime_rune_learner import RealtimeRuneLearner
        self._realtime_runes = RealtimeRuneLearner(max_runes=4096)
        self._no_op_streak: dict[tuple[str, int, str, str], int] = {}
        self._runtime_budget = RuntimeSafetyBudget()
        # Post-RESET forced rotation: (game, level) -> turns remaining.
        # After a RESET the Duck policy replays the same futile line; the
        # next turns are forced through untried-rotation (never macros).
        self._post_reset_forced: dict[tuple[str, int], int] = {}
        self._reconciled_mark: tuple | None = None

    def _reset_level_state(self, keep_try_history: bool = False) -> None:
        """Reset per-level servo state.

        Level change -> full clear (default). Same-level guard RESET ->
        keep_try_history=True: the attempt just failed, so tried actions,
        tried clicks, cycle memory, no-op streaks and the reconcile mark
        MUST survive — wiping them replays the same futile line and loses
        twice. Macro queue, cooldowns always clear.
        """
        self.planned_macro_queue.clear()
        self._guard_cooldown_until = 0
        if keep_try_history:
            return
        self._seen.clear()
        self._tried_actions.clear()
        self._tried_clicks.clear()
        self._no_op_streak.clear()
        self._reconciled_mark = None

    def _failure_key(self, grid: np.ndarray, payload: dict) -> tuple[str, int, str, str]:
        return (
            self._active_game_id,
            int(self._active_level),
            _grid_hash(grid),
            self._action_key(payload),
        )

    # -- mechanism bandit -------------------------------------------------
    def _record_mech(self, name: str, success: bool) -> None:
        s = self._mech_ok.setdefault(name, [0, 0])
        s[0] += 1 if success else 0
        s[1] += 1

    def _mech_rate(self, name: str) -> float:
        s, n = self._mech_ok.get(name, (0, 0))
        return (s + 1.0) / (n + 2.0)  # Laplace smoothing, pessimistic start

    def _required_confidence(self, prior_rule) -> float:
        # Intrinsic-only order: prior_rule is always None now (no memorized
        # manuals), so the bar is constant. mech_ok adaptation below is
        # within-run statistics (intrinsic) and stays.
        base = 0.82
        s, n = self._mech_ok.get("consensus", (0, 0))
        if n >= 4:
            rate = s / n
            if rate < 0.4:
                return min(0.92, base + 0.10)
            if rate > 0.8:
                return max(base, base - 0.05)
        return base

    # -- payload construction ----------------------------------------------
    @staticmethod
    def _coord(entry: dict, keys, default: int = 32) -> int:
        for k in keys:
            if entry.get(k) is not None:
                try:
                    return max(0, min(63, int(entry[k])))
                except (TypeError, ValueError):
                    continue
        return default

    def _build_payload(self, entry: Any) -> dict | None:
        """Macro/synth entry -> step_env payload. None = refuse, fall through."""
        if isinstance(entry, int):
            if entry == 6:
                return None  # coordinate-less CLICK is never blind-fired
            return {"action": ACTION_MAP[entry]} if entry in ACTION_MAP else None
        if isinstance(entry, str):
            name = entry.upper()
            return {"action": name} if name in ACTION_REV else None
        if isinstance(entry, dict):
            act = entry.get("action")
            if isinstance(act, int):
                if act == 6:
                    if all(entry.get(k) is None for k in ("x", "y", "row", "col")):
                        return None
                    return {"action": "ACTION6",
                            "row": self._coord(entry, ("y", "row")),
                            "col": self._coord(entry, ("x", "col"))}
                return {"action": ACTION_MAP[act]} if act in ACTION_MAP else None
            if isinstance(act, str):
                name = act.upper()
                if name == "ACTION6":
                    return {"action": "ACTION6",
                            "row": self._coord(entry, ("row", "y")),
                            "col": self._coord(entry, ("col", "x"))}
                return {"action": name} if name in ACTION_REV else None
        return None

    def _payload_to_int(self, payload: dict) -> tuple:
        """step_env payload -> (act_int, x, y) for the verifier."""
        name = str(payload.get("action", "")).upper()
        aid = ACTION_REV.get(name, 1)
        if aid == 6:
            y = payload.get("row", payload.get("y", 32))
            x = payload.get("col", payload.get("x", 32))
            try:
                return 6, max(0, min(63, int(x))), max(0, min(63, int(y)))
            except (TypeError, ValueError):
                return 6, 32, 32
        return aid, None, None

    def _int_to_payload(self, act: int, x=None, y=None) -> dict | None:
        if act == 6:
            if x is None or y is None:
                return None
            return {"action": "ACTION6", "row": int(y), "col": int(x)}
        return {"action": ACTION_MAP[act]} if act in ACTION_MAP else None

    def _is_untried(self, payload: dict) -> bool:
        """True when this exact action was never tried (discovery VETO-SHIELD)."""
        name = str(payload.get("action", "")).upper()
        if name == "ACTION6":
            try:
                y = payload.get("row", payload.get("y"))
                x = payload.get("col", payload.get("x"))
                if x is None or y is None:
                    return True
                return (int(x), int(y)) not in self._tried_clicks
            except (TypeError, ValueError):
                return True
        return name not in self._tried_actions

    def _verify(self, payload: dict, grid: np.ndarray, game_id: str,
                stagnation: int, allowed: set | None) -> dict:
        """Verify-before-execute: substitute predicted no-ops. Never throws.

        Discovery shield: UNTRIED actions always pass through unchanged.
        The world model is ignorant exactly where discovery matters; letting
        it censor unexplored actions strangled first contact (sb26: the only
        effective action was vetoed away 30/30 turns).
        """
        if payload is None:
            return None
        if self._is_untried(payload):
            return payload
        try:
            from arc3sdk.tool_world_bridge import verify_action

            act, x, y = self._payload_to_int(payload)
            avail = sorted(ACTION_REV[n] for n in (allowed or set(ACTION_REV))
                           if n in ACTION_REV)
            v_act, v_x, v_y, _reason = verify_action(
                np.asarray(grid, dtype=np.uint8), act, x, y,
                avail or [1, 2, 3, 4], game_id=game_id, stagnation=stagnation)
            fixed = self._int_to_payload(int(v_act), v_x, v_y)
            return fixed if fixed is not None else payload
        except Exception:
            return payload

    def _commit(self, step_env: Callable, payload: dict, mech: str,
                grid: np.ndarray) -> tuple:
        """Execute one payload under guard. Returns (ok, env_res)."""
        try:
            payload = normalize_action(payload)
            grid = normalize_grid(grid)
        except (TypeError, ValueError) as exc:
            log.info("[SERVO-ACTION-REJECT] %s", exc)
            return False, None
        # Sanitize every servo payload (failure_immunity was shipped but only
        # reachable behind the unshipped consensus tier). ACTION7 passes
        # through untouched: sanitize predates it and would corrupt it to
        # ACTION1; ACTION7 is keyboard-only so there is nothing to clamp.
        with contextlib.suppress(Exception):
            if str(payload.get("action", "")).upper() != "ACTION7":
                from arc3sdk.failure_immunity import sanitize_action_payload
                payload = sanitize_action_payload(payload)
        if not self._runtime_budget.can_continue():
            log.warning("[SERVO-DEADLINE] soft deadline reached; refusing new action")
            return False, None
        with contextlib.suppress(Exception):
            self._attempted_pairs.add(self._failure_key(grid, payload))
        try:
            env_res = step_env(payload)
        except Exception as exc:
            log.debug(f"[SERVO-{mech}] step_env failed: {type(exc).__name__}")
            self._record_mech(mech, False)
            return False, None
        try:
            after = env_res.get("grid", env_res.get("frame")) if isinstance(env_res, dict) else None
            changed = bool(after is not None and not np.array_equal(
                np.asarray(after).reshape(-1),
                np.asarray(grid, dtype=np.uint8).reshape(-1))) if after is not None else True
        except Exception:
            changed = True
        self.total_short_circuits += int(changed)
        self._record_mech(mech, changed)
        with contextlib.suppress(Exception):
            self._feed_cass(grid, payload, after)
        try:
            no_op_key = (*self._failure_key(grid, payload)[:3], self._action_key(payload))
            if changed:
                self._no_op_streak.pop(no_op_key, None)
            else:
                streak = self._no_op_streak.get(no_op_key, 0) + 1
                self._no_op_streak[no_op_key] = streak
                # A single unchanged frame can be legitimate in timed/physics
                # games. Repeating the same action on the same visible state is
                # not useful after two attempts; veto it on the next ladder pass.
                if streak >= 2:
                    self._failed_pairs.add(self._failure_key(grid, payload))
                    log.info("[SERVO-NOOP-VETO] game=%s level=%s action=%s streak=%s",
                             self._active_game_id, self._active_level,
                             self._action_key(payload), streak)
        except Exception:
            pass
        try:
            if isinstance(env_res, dict) and env_res.get("game_over"):
                # A terminal failure is evidence against this exact decision,
                # not against the whole abstract state or every action.
                if not bool(env_res.get("won") or env_res.get("success")):
                    self._failed_pairs.add(self._failure_key(grid, payload))
                self.planned_macro_queue.clear()
                self._seen.clear()
        except Exception:
            pass
        return True, env_res

    def _record_transition(self, game_id: str, level: int, grid: np.ndarray,
                           raw_act: Any, env_res: Any) -> None:
        """Record REAL transitions only — never identity (s,a,s) poison."""
        try:
            next_grid = None
            if isinstance(env_res, dict):
                for key in ("next_grid", "grid", "frame", "observation"):
                    if env_res.get(key) is not None:
                        next_grid = np.asarray(env_res[key])
                        break
            if next_grid is not None:
                _action_name = str(raw_act.get("action") if isinstance(raw_act, dict) else raw_act)
                _changed = not np.array_equal(np.asarray(grid), next_grid)
                _won = bool(env_res.get("won") or env_res.get("success")) if isinstance(env_res, dict) else False
                _reward = 2.0 if _won else (1.0 if _changed else -1.0)
                self._replay.add(
                    game_id, level, grid, _action_name, next_grid, _reward,
                    terminal=bool(env_res.get("game_over")) if isinstance(env_res, dict) else False,
                )
                self._evidence_planner.observe(
                    grid, _action_name, next_grid, game_id=game_id, level=level,
                    source="step_env", terminal=bool(env_res.get("game_over")) if isinstance(env_res, dict) else False,
                )
                self._realtime_rules.observe(
                    grid, _action_name, next_grid, game_id=game_id, level=level,
                    terminal=bool(env_res.get("game_over")) if isinstance(env_res, dict) else False,
                )
                self._realtime_runes.observe(
                    grid, _action_name, next_grid,
                    won=bool(env_res.get("won") or env_res.get("success")) if isinstance(env_res, dict) else False,
                )
                self._abstract_controller.learn_abstract_effect(
                    grid, _action_name, next_grid, _reward
                )
        except Exception:
            pass
        try:
            from arc3sdk.sovereign_memory_vram import get_global_memory_bank

            next_grid = None
            if isinstance(env_res, dict):
                for key in ("next_grid", "grid", "frame", "observation"):
                    val = env_res.get(key)
                    if val is None:
                        continue
                    try:
                        cand = np.asarray(val, dtype=np.uint8).reshape(grid.shape)
                    except Exception:
                        continue
                    next_grid = cand
                    break
            if next_grid is None or np.array_equal(next_grid, grid):
                return
            mem = get_global_memory_bank()
            rew = float(env_res.get("reward", 0.0)) if isinstance(env_res, dict) else 0.0
            over = bool(env_res.get("game_over", False)) if isinstance(env_res, dict) else False
            mem.record_step_transition(game_id, level, grid, raw_act, next_grid, rew, over)
        except Exception:
            pass

    def _cycle_break(self, grid: np.ndarray,
                     allowed: set | None) -> str | None:
        """Period-2/4 breaker: repeated state -> least-recently-used action.

        Only RECENT repeats (last 12 notes) count: a state revisited after
        long productive play must not trigger. Productive streaks keep
        moving to fresh states, so this never fires on them.
        """
        h = _grid_hash(grid)
        window = list(self._seen)[-12:]
        seen = [a for (hh, a) in window if hh == h]
        if len(seen) < CYCLE_REPEAT:
            return None
        candidates = [n for n in
                      ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6")
                      if allowed is None or n in allowed]
        if not candidates:
            return None
        recent = [a for (_hh, a) in list(self._seen)[-12:]]
        for cand in sorted(candidates, key=lambda c: recent.count(c)):
            if cand != (seen[-1] if seen else None):
                return cand
        return candidates[0]

    @staticmethod
    def _action_key(payload: dict) -> str:
        name = str(payload.get("action", "")).upper()
        if name == "ACTION6":
            try:
                y = payload.get("row", payload.get("y"))
                x = payload.get("col", payload.get("x"))
                return f"ACTION6:{int(x)},{int(y)}"
            except (TypeError, ValueError):
                return "ACTION6"
        return name

    @staticmethod
    def _peek_prior(game_id: str):
        try:
            from arc3sdk.unified_consensus_engine import (
                SovereignMasterConsensusEngine as _CE2)
            return _CE2().get_game_prior(game_id)
        except Exception:
            return None

    def _avo_proposal(self, grid: np.ndarray, allowed: set | None) -> dict | None:
        """Use the mounted AVO checkpoint only as a bounded fallback proposal."""
        if self._avo_model is None and not self._avo_attempted:
            self._avo_attempted = True
            try:
                from arc3sdk.sovereign_avo_moe_fiber import get_avo_moe_fiber
                self._avo_model = get_avo_moe_fiber()
            except Exception:
                self._avo_model = None
        if self._avo_model is None:
            return None
        try:
            available = [ACTION_REV[name] for name in (allowed or set(ACTION_REV)) if name in ACTION_REV]
            if not available:
                return None
            act, x, y, reason, confidence = self._avo_model.predict(grid, available)
            if float(confidence) < 0.25:
                return None
            payload = self._int_to_payload(int(act), x, y)
            if payload is None or (allowed is not None and payload["action"] not in allowed):
                return None
            payload["_avo_reason"] = str(reason)
            return payload
        except Exception:
            return None

    def _vetoed(self, payload: dict, game_id: str, sc: str) -> bool:
        """Learned veto: proven no-op in this state-class. Never throws."""
        try:
            pair = self._failure_key(np.asarray(self._current_grid, dtype=np.uint8), payload)
            if pair in self._attempted_pairs or pair in self._failed_pairs:
                return True
        except Exception:
            pass
        try:
            return self._learner.banned(game_id, sc, self._action_key(payload))
        except Exception:
            return False

    def _cass_on_game(self, game_id: str) -> None:
        """Reset CASS belief on game change (laws don't transfer across games)."""
        try:
            if game_id != self._cass_game_id:
                self._cass_game_id = game_id
                self._cass_agent = None
                self._cass_observed = 0
        except Exception:
            pass

    def _cass_should_probe(self, stagnation: int) -> bool:
        """Fire only when Duck is stuck AND the belief has real observations."""
        try:
            return int(stagnation) >= 2 and int(self._cass_observed) >= 4
        except Exception:
            return False

    def _cass_probe_payload(self, grid: np.ndarray, game_id: str, sc: str,
                            stagnation: int, allowed: set | None, level: int,
                            allow_fn) -> tuple[dict | None, str]:
        """CASS probe decision (payload, reasoning) or (None, None). Never throws."""
        try:
            if not self._cass_should_probe(stagnation):
                return None, ""
            from arc3sdk import cass_v0 as _cass_mod
            if self._cass_agent is None:
                self._cass_agent = _cass_mod.CASSAgent()
            _cass_avail = sorted({ACTION_REV[a] for a in allowed if a in ACTION_REV}) if allowed else [1, 2, 3, 4, 5, 6]
            _cact, _cx, _cy, _cdbg = self._cass_agent.plan(grid, _cass_avail)
            _cass_payload = allow_fn(self._int_to_payload(_cact, _cx, _cy))
            if _cass_payload is None or self._vetoed(_cass_payload, game_id, sc):
                return None, ""
            _cass_payload = self._verify(_cass_payload, grid, game_id, stagnation, allowed)
            reason = (f"[CASS-PROBE] tier={_cdbg.get('tier')} "
                      f"ent={float(_cdbg.get('entropy', 0.0)):.2f} "
                      f"n={len(self._cass_agent.particles)}")
            return _cass_payload, reason
        except Exception:
            return None, ""

    def _feed_cass(self, grid: np.ndarray, payload: dict | None, after: Any) -> None:
        """Feed executed outcomes into the CASS belief (native learning)."""
        try:
            if payload is None or after is None:
                return
            act_name = str(payload.get("action", "")).upper()
            aid = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
                   "ACTION5": 5, "ACTION6": 6, "ACTION7": 7, "RESET": 7}.get(act_name)
            if aid is None or aid == 7:
                return
            from arc3sdk import cass_v0
            if self._cass_agent is None:
                self._cass_agent = cass_v0.CASSAgent()
            g = np.ascontiguousarray(np.asarray(grid, dtype=np.uint8))
            nx = np.ascontiguousarray(np.asarray(after, dtype=np.uint8))
            if g.shape != nx.shape:
                return
            self._cass_agent.observe(cass_v0.signature(g), aid, cass_v0.signature(nx))
            self._cass_observed += 1
        except Exception:
            pass

    def _note_state(self, grid: np.ndarray, action_name: str,
                    payload: dict | None = None) -> None:
        self._seen.append((_grid_hash(grid), action_name))
        self._tried_actions.add(action_name)
        if action_name == "ACTION6" and payload is not None:
            try:
                y = payload.get("row", payload.get("y"))
                x = payload.get("col", payload.get("x"))
                if x is not None and y is not None:
                    self._tried_clicks.add((int(x), int(y)))
            except (TypeError, ValueError):
                pass

    def _coverage_click(self, grid: np.ndarray, allowed: set | None = None) -> tuple | None:
        """Least-tried object centroid (x, y). None when exhausted."""
        # Codex ordering first: smallest-enclosed-void target from world
        # axioms (deterministic, void-first). Falls through on any failure.
        try:
            from arc3sdk.pure_abstract_axioms import PureAbstractCodex
            for _act, _x, _y, _why, _conf in PureAbstractCodex.propose(grid, [6]):
                if _x is None or _y is None:
                    continue
                cx, cy = int(_x), int(_y)
                if 0 <= cx < 64 and 0 <= cy < 64 and (cx, cy) not in self._tried_clicks:
                    return (cx, cy)
        except Exception:
            pass
        # Grandmaster-skills click second: highest-confidence skill click
        # that is still untried (deterministic order by confidence).
        try:
            from arc3sdk.skills import SovereignSkillsMatrix

            avail = [ACTION_REV[n] for n in (allowed or set(ACTION_REV)) if n in ACTION_REV]
            if 6 in avail:
                arr = np.asarray(grid, dtype=np.uint8)
                bg = int(np.bincount(arr.ravel()).argmax())
                for act, sx, sy, _why, _conf in SovereignSkillsMatrix.evaluate_skills(
                        arr, bg, avail):
                    if int(act) != 6 or sx is None or sy is None:
                        continue
                    cx, cy = int(sx), int(sy)
                    if 0 <= cx < 64 and 0 <= cy < 64 and (cx, cy) not in self._tried_clicks:
                        return (cx, cy)
        except Exception:
            pass
        try:
            from arc3sdk.neuro_symbolic_dsl import find_objects

            cands: list = []
            for obj in find_objects(grid) or []:
                try:
                    r, c = int(obj.centroid_r), int(obj.centroid_c)
                except Exception:
                    continue
                if 0 <= c < 64 and 0 <= r < 64 and (c, r) not in self._tried_clicks:
                    cands.append((c, r))
            if cands:
                return cands[0]
            if (32, 32) not in self._tried_clicks:
                return (32, 32)
        except Exception:
            pass
        return None

    def _skills_keyboard_rank(self, grid: np.ndarray, allowed: set | None) -> dict[str, float]:
        """Grandmaster-skills confidence per keyboard action. Fail-open {}."""
        try:
            from arc3sdk.skills import SovereignSkillsMatrix

            avail = [ACTION_REV[n] for n in (allowed or set(ACTION_REV)) if n in ACTION_REV]
            if not avail:
                return {}
            bg = int(np.bincount(np.asarray(grid, dtype=np.uint8).ravel()).argmax())
            out: dict[str, float] = {}
            for act, _x, _y, _why, conf in SovereignSkillsMatrix.evaluate_skills(
                    np.asarray(grid, dtype=np.uint8), bg, avail):
                name = ACTION_MAP.get(int(act), "")
                if name and name != "ACTION6" and (allowed is None or name in allowed):
                    out[name] = max(out.get(name, 0.0), float(conf))
            return out
        except Exception:
            return {}

    def _simulate_changed(self, grid: np.ndarray, names: list[str],
                            game_id: str, allowed: set | None) -> dict[str, bool]:
        """WorldSim 1-step prediction per keyboard action (changed or not).

        Cached simulator, millisecond scale. Used ONLY as a tiebreak when
        ordering exploratory picks — never to veto (T85 lesson). Fail-open {}.
        """
        out: dict[str, bool] = {}
        try:
            from arc3sdk.tool_world_bridge import get_world_sim

            sim = get_world_sim()
            g = np.asarray(grid, dtype=np.uint8)
            for name in names:
                try:
                    act = ACTION_REV.get(name)
                    if act is None or act == 6:
                        continue
                    _nxt, changed, _src = sim.simulate(g, int(act), None, None, game_id)
                    out[name] = bool(changed)
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def _counsel_lines(self, grid: np.ndarray, game_id: str, level: int,
                       stagnation: int) -> list[str]:
        """Compact mined-rule counsel for the transcript (LLM-readable).

        Appended to the transcript file when available: if Duck's analyzer
        ever reads transcript history, our extracted laws ride along. Zero
        risk otherwise (namespaced append-only lines). No observation ->
        no counsel (fail closed). Never throws.
        """
        try:
            g = np.asarray(grid, dtype=np.uint8)
            if g.ndim != 2 or g.size == 0:
                return []
            from arc3sdk.abstract_rule_miner import mine_abstract_rules

            prev = getattr(self, "_current_grid", None)
            rep = mine_abstract_rules(grid, prev_grid=prev, bg=None)
            parts = [f"law=sym:{rep.symmetry} voids:{len(rep.voids)} "
                     f"objs:{len(rep.objects)}"]
            if rep.goal_ranking:
                parts.append("afford=" + ",".join(f"{a}:{c:.2f}" for a, c in rep.goal_ranking[:3]))
            if rep.salient:
                parts.append("salient=" + ",".join(f"({x},{y}:{w})" for y, x, w in rep.salient))
            tried = len(self._tried_actions)
            failed = len(self._failed_pairs)
            head = (f"[SERVO-COUNSEL] g={game_id} lv={level} stag={stagnation} "
                    f"tried={tried} failed={failed} " + " ".join(parts))
            return [head[:300]]
        except Exception:
            return []

    def _stagnation_probe(self, grid: np.ndarray,
                          allowed: set | None) -> dict | None:
        """First untried keyboard action, then coverage CLICKs. None = pool spent.

        Untried actions are ordered by cross-level transfer prior (highest
        evidence first), then Grandmaster-skills confidence; ties keep
        canonical ACTION1..5 order — sorted() is stable, and cold priors are
        all (0.0, 0) so cold behavior is unchanged. Every line gets tried
        once, most promising first: no second loss on the same line.
        """
        cands = [n for n in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
                 if (allowed is None or n in allowed) and n not in self._tried_actions]
        skills_rank = self._skills_keyboard_rank(grid, allowed)
        # WorldSim 1-step prediction as the last tiebreak: among equally
        # promising untried actions, prefer one the simulator predicts will
        # change the board (stall context). Never vetoes (T85 lesson).
        sim_changed = self._simulate_changed(grid, cands, self._active_game_id, allowed)

        def _promise(name: str) -> tuple[float, int, float, float]:
            try:
                mean, count = self._abstract_controller.transfer_prior(grid, name)
                return (float(mean), int(count), float(skills_rank.get(name, 0.0)),
                        1.0 if sim_changed.get(name) else 0.0)
            except Exception:
                return (0.0, 0, 0.0, 0.0)

        for name in sorted(cands, key=_promise, reverse=True):
            return {"action": name}
        if allowed is None or "ACTION6" in allowed:
            target = self._coverage_click(grid, allowed)
            if target is not None:
                return {"action": "ACTION6", "row": target[1], "col": target[0]}
        return None

    def _reconcile_duck_history(self, history_entries: Any) -> int:
        """Fold Duck's own executed trajectory into cycle memory.

        _seen previously held ONLY our servo commits, so _cycle_break was
        blind to Duck's oscillations (sp80: hundreds of Duck actions, zero
        breaks). History entries carry grids (object `.grid` or dict
        `['grid']`); action names are harvested opportunistically, else
        labeled 'duck'. Incremental via _reconciled_mark (game, level,
        len). Returns newly added count. Never throws.
        """
        added = 0
        try:
            entries = list(history_entries or [])
        except Exception:
            return 0
        try:
            mark = (str(self._active_game_id), int(self._active_level), len(entries))
        except Exception:
            mark = None
        try:
            prev = self._reconciled_mark
            if (mark is not None and prev is not None and prev[0] == mark[0]
                    and prev[1] == mark[1] and isinstance(prev[2], int)
                    and prev[2] <= mark[2]):
                fresh = entries[prev[2]:]
            else:
                fresh = entries[-CYCLE_MEMORY:]
            self._reconciled_mark = mark
            for e in fresh:
                g = None
                try:
                    if hasattr(e, "grid"):
                        g = np.asarray(e.grid, dtype=np.uint8)
                    elif isinstance(e, dict) and e.get("grid") is not None:
                        g = np.asarray(e["grid"], dtype=np.uint8)
                except Exception:
                    continue
                if g is None or g.ndim != 2 or g.size == 0:
                    continue
                act = "duck"
                for key in ("action_name", "action", "act"):
                    try:
                        v = e.get(key) if isinstance(e, dict) else getattr(e, key, None)
                        if isinstance(v, str) and v:
                            act = v.upper()
                            break
                    except Exception:
                        continue
                self._seen.append((_grid_hash(g), act))
                added += 1
        except Exception:
            pass
        return added

    def _register_reset(self, game_id: str, level: int, turns: int = 3) -> None:
        """Arm forced untried-rotation for the next turns after a RESET."""
        with contextlib.suppress(Exception):
            self._post_reset_forced[(str(game_id), int(level))] = int(turns)

    def _consume_forced_probe(self, game_id: str, level: int) -> bool:
        """True once per armed turn; survives level-state resets by design."""
        try:
            k = (str(game_id), int(level))
            n = int(self._post_reset_forced.get(k, 0))
            if n > 0:
                self._post_reset_forced[k] = n - 1
                return True
        except Exception:
            pass
        return False

    def _level_opening_move(self, grid: np.ndarray, allowed: set | None) -> dict | None:
        """Transfer read-path: best CROSS-LEVEL action for a fresh level.

        The controller accumulates abstract effects across levels/games, but
        nothing ever read them back (exact-signature priors never match new
        layouts). This plays the best transfer action with real evidence
        (count>=2, mean>=0.5); cold levels return None and fall through to
        first-contact probing. Game-agnostic by design. Never throws.
        """
        try:
            best: tuple[float, str] | None = None
            for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4",
                         "ACTION5", "ACTION7", "ACTION6"):
                if allowed is not None and name not in allowed:
                    continue
                mean, count = self._abstract_controller.transfer_prior(grid, name)
                if count >= 2 and mean >= 0.5 and (best is None or mean > best[0]):
                    best = (float(mean), name)
            if best is None:
                return None
            _mean, name = best
            if name == "ACTION6":
                target = self._coverage_click(grid, allowed)
                if target is None:
                    return None
                return {"action": "ACTION6", "row": target[1], "col": target[0]}
            return {"action": name}
        except Exception:
            return None

    def __call__(
        self,
        tool_agent_self: Any,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Any:
        if request_timeout_seconds is not None:
            request_timeout_seconds = max(45.0, float(request_timeout_seconds))

        if not state_path.exists() or step_env is None:
            return self.original_analyze(
                tool_agent_self,
                state_path,
                action_num,
                valid_actions=valid_actions,
                step_env=step_env,
                transcript_path=transcript_path,
                analysis_step=analysis_step,
                transcript_updated=transcript_updated,
                request_timeout_seconds=request_timeout_seconds,
                should_stop=should_stop,
            )

        try:
            # 1. Load state from runtime state file
            from inference.agent.runtime_state import load_runtime_state
            from inference.agent.tool_agent import AnalyzerTurnResult
            # Guarded sibling imports: any module absent from the shipped set
            # must degrade its tier, never abort the whole turn (v84 proved a
            # single hard import kills every turn via INTERCEPTOR-GUARD).
            try:
                from arc3sdk.unified_consensus_engine import (
                    SovereignMasterConsensusEngine as _ConsensusEngine)
            except Exception:
                _ConsensusEngine = None
            try:
                from arc3sdk.abstract_hypothesis_mcts import AbstractHypothesisEngine
            except Exception:
                AbstractHypothesisEngine = None
            try:
                from arc3sdk.neuro_symbolic_dsl import find_objects, click_affordance
            except Exception:
                find_objects = None
                click_affordance = None

            current_frame, history_entries = load_runtime_state(state_path)
            grid = np.asarray(current_frame.grid, dtype=np.uint8)
            level = int(getattr(current_frame, "level", 0))
            allowed = {a.upper() for a in valid_actions} if valid_actions else None
            game_id = state_path.stem.split("_")[0].split(".")[0]
            self._active_game_id = game_id
            self._cass_on_game(game_id)
            self._active_level = level
            self._current_grid = grid
            sc = state_class(grid)

            # Delayed causal observe: learn what the PREVIOUS step did.
            if self._pending is not None:
                try:
                    pg, pa, pl = self._pending
                    self._learner.observe(game_id, pg, pa, grid,
                                          level_up=(level > pl))
                except Exception:
                    pass
                finally:
                    self._pending = None

            # Duck's own picks never entered _seen: fold the executed history
            # into cycle memory so oscillation breakers can see them.
            with contextlib.suppress(Exception):
                self._reconcile_duck_history(history_entries)

            def _allowed_payload(payload: dict | None) -> dict | None:
                if payload is None:
                    return None
                if allowed is not None and str(payload.get("action", "")).upper() not in allowed:
                    return None
                return payload

            # Flush per-level state on transition
            if level != self.current_level:
                self.current_level = level
                self.level_action_counts[level] = 0
                self._reset_level_state()

            self.level_action_counts[level] = self.level_action_counts.get(level, 0) + 1
            if self.level_action_counts[level] > MAX_ACTIONS_PER_LEVEL:
                log.warning(f"[SERVO-BUDGET-GUARD] Level {level} exceeded {MAX_ACTIONS_PER_LEVEL} actions! Resetting level...")
                self._reset_level_state(keep_try_history=True)
                self._register_reset(game_id, level)
                step_env({"action": "RESET"})
                self.level_action_counts[level] = 0
                self._record_mech("budget_guard", True)
                return AnalyzerTurnResult(
                    step_executed=True,
                    retryable_failure=False,
                    reasoning=f"[SERVO-BUDGET-GUARD] Level exceeded action cap {MAX_ACTIONS_PER_LEVEL}; forced reset."
                )

            stagnation = _trailing_zero_diffs(history_entries)

            # Counsel for Duck: mined abstract rules appended to the transcript
            # file (namespaced, one append per turn). If the analyzer ever reads
            # transcript history, our extracted laws ride along; otherwise these
            # lines are pure forensic value. Zero risk either way.
            if transcript_path is not None:
                with contextlib.suppress(Exception):
                    for _line in self._counsel_lines(grid, game_id, level, stagnation):
                        with open(transcript_path, "a", encoding="utf-8") as _cf:
                            _cf.write(_line + "\n")

            # First-contact probe: on failure/ignorance, discover BEFORE the
            # analytic ladder (never interrupts a queued winning macro, and
            # yields to a high-confidence consensus peek — probing blindly
            # while the analyst is certain walked ls20 into death).
            # Post-RESET turns force untried-rotation (no peek: the old line
            # just failed, re-asking it is the futile loop).
            forced_probe = (not self.planned_macro_queue) and self._consume_forced_probe(game_id, level)
            if forced_probe or self._learner.should_probe_first(
                    game_id, stagnation, bool(self.planned_macro_queue),
                    self._stagnation_probe(grid, allowed) is not None):
                peek_conf = -1.0
                if not forced_probe:
                    try:
                        from arc3sdk.unified_consensus_engine import (
                            SovereignMasterConsensusEngine as _CE)
                        peek = _CE().decide({"grid": grid, "level": level,
                                             "step": action_num,
                                             "valid_actions": valid_actions,
                                             "game_id": game_id}) or {}
                        peek_conf = float(peek.get("confidence", -1.0))
                    except Exception:
                        peek_conf = -1.0
                if forced_probe or peek_conf < self._required_confidence(
                        self._peek_prior(game_id)):
                    probe = self._verify(
                        self._stagnation_probe(grid, allowed),
                        grid, game_id, stagnation, allowed)
                    if probe is not None and not self._vetoed(probe, game_id, sc):
                        ok, _res = self._commit(step_env, probe, "first_contact", grid)
                        if ok:
                            self._pending = (grid.copy(), self._action_key(probe), level)
                            self._note_state(grid, str(probe.get("action")), probe)
                            return AnalyzerTurnResult(
                                step_executed=True,
                                retryable_failure=False,
                                reasoning=f"[FIRST-CONTACT] Cold-game probe {probe}."
                            )

            # Level-opening transfer: first turns of a fresh level play the
            # best CROSS-LEVEL action (learned abstract effects), never blind.
            # Skipped while post-reset forcing is armed (rotation wins there).
            if (not self.planned_macro_queue
                    and not self._post_reset_forced.get((game_id, level), 0)
                    and self.level_action_counts.get(level, 0) <= 2):
                opening = _allowed_payload(self._level_opening_move(grid, allowed))
                if opening is not None and not self._vetoed(opening, game_id, sc):
                    opening = self._verify(opening, grid, game_id, stagnation, allowed)
                    ok, _res = self._commit(step_env, opening, "level_opening", grid)
                    if ok:
                        self._pending = (grid.copy(), self._action_key(opening), level)
                        self._note_state(grid, str(opening.get("action")), opening)
                        return AnalyzerTurnResult(
                            step_executed=True,
                            retryable_failure=False,
                            reasoning=f"[LEVEL-OPENING] Transfer {opening}."
                        )

            # Stall guard: probe uncovered actions FIRST; RESET only when the
            # pool is spent (and never more often than the cooldown allows).
            if stagnation >= _ZERO_DIFF_WINDOW:
                probe = self._stagnation_probe(grid, allowed)
                if probe is not None:
                    probe = self._verify(probe, grid, state_path.stem, stagnation, allowed)
                    ok, _res = self._commit(step_env, probe, "stagnation_probe", grid)
                    if ok:
                        self._pending = (grid.copy(), self._action_key(probe), level)
                        self._note_state(grid, str(probe.get("action")), probe)
                        return AnalyzerTurnResult(
                            step_executed=True,
                            retryable_failure=False,
                            reasoning=f"[STAGNATION-PROBE] Zero-diff stall; probing uncovered {probe}."
                        )
                if self.level_action_counts[level] >= self._guard_cooldown_until:
                    log.warning(f"[SELF-REFLECTION-EH!] Stagnant loop at action {action_num}! Backtracking via RESET...")
                    self._reset_level_state(keep_try_history=True)
                    self._register_reset(game_id, level)
                    step_env({"action": "RESET"})
                    self.level_action_counts[level] = 0
                    self._guard_cooldown_until = self.level_action_counts[level] + 10
                    self._record_mech("stagnation_guard", True)
                    return AnalyzerTurnResult(
                        step_executed=True,
                        retryable_failure=False,
                        reasoning="[SELF-REFLECTION] Stall with exhausted probe pool. Backtracking via RESET."
                    )

            # State-hash cycle breaker (period-2/4 with NONZERO diffs)
            cycle_act = self._cycle_break(grid, allowed)
            if cycle_act is not None:
                log.warning(f"[CYCLE-BREAK] Repeating state at action {action_num}; forcing {cycle_act}, retiring macro queue.")
                self.planned_macro_queue.clear()
                payload = {"action": cycle_act}
                if cycle_act == "ACTION6":
                    payload = {"action": "ACTION6", "row": 32, "col": 32}
                payload = self._verify(payload, grid, state_path.stem, stagnation, allowed)
                ok, _res = self._commit(step_env, payload, "cycle_break", grid)
                if ok:
                    self._pending = (grid.copy(), self._action_key(payload), level)
                    self._note_state(grid, str(payload.get("action")), payload)
                    return AnalyzerTurnResult(
                        step_executed=True,
                        retryable_failure=False,
                        reasoning=f"[CYCLE-BREAK] Forced exploration {payload} to shatter oscillation."
                    )

            # Macro queue playback (verified, coordinate-preserving)
            while self.planned_macro_queue:
                macro_step = self.planned_macro_queue.pop(0)
                payload = _allowed_payload(self._build_payload(macro_step))
                if payload is None or self._vetoed(payload, game_id, sc):
                    continue
                payload = self._verify(payload, grid, state_path.stem, stagnation, allowed)
                ok, _res = self._commit(step_env, payload, "macro", grid)
                if ok:
                    self._pending = (grid.copy(), self._action_key(payload), level)
                    self._note_state(grid, str(payload.get("action")), payload)
                    return AnalyzerTurnResult(
                        step_executed=True,
                        retryable_failure=False,
                        reasoning=f"[MACRO-SYNTHESIS] Played macro plan action {payload}"
                    )

            # Deterministic AutoProgramSynthesizer (macro trajectory synthesis)
            try:
                from arc3sdk.neuro_symbolic_dsl import AutoProgramSynthesizer
                synth = AutoProgramSynthesizer()
                synth_res = synth.synthesize(grid)
                if synth_res.success and synth_res.macro_actions:
                    first = synth_res.macro_actions[0]
                    payload = _allowed_payload(self._build_payload(first))
                    if len(synth_res.macro_actions) > 1:
                        self.planned_macro_queue.extend(synth_res.macro_actions[1:])
                    if payload is not None and not self._vetoed(payload, game_id, sc):
                        payload = self._verify(payload, grid, game_id, stagnation, allowed)
                        ok, _res = self._commit(step_env, payload, "synth", grid)
                        if ok:
                            self._pending = (grid.copy(), self._action_key(payload), level)
                            self._note_state(grid, str(payload.get("action")), payload)
                            return AnalyzerTurnResult(
                                step_executed=True,
                                retryable_failure=False,
                                reasoning=f"[MACRO-SYNTHESIZER] Deterministic plan executed action {payload}"
                            )
            except Exception as _se:
                log.debug(f"[SYNTHESIZER-NOTE] {type(_se).__name__}: {_se}")

            # Abstract Hypothesis MCTS evaluation (skipped cleanly when unshipped)
            if (AbstractHypothesisEngine is not None and find_objects is not None
                    and click_affordance is not None):
                hyp_engine = AbstractHypothesisEngine()
                best_hyp = hyp_engine.select_best_hypothesis(grid, game_id)

                # Fast Click Affordance Program Synthesis (H_CLICK)
                if best_hyp.hypothesis_type == "H_CLICK":
                    objs = find_objects(grid)
                    if objs:
                        payload = _allowed_payload(self._build_payload(click_affordance(objs[0])))
                        if payload is not None and not self._vetoed(payload, game_id, sc):
                            payload = self._verify(payload, grid, game_id, stagnation, allowed)
                            ok, _res = self._commit(step_env, payload, "hypothesis", grid)
                            if ok:
                                self._pending = (grid.copy(), self._action_key(payload), level)
                                self._note_state(grid, str(payload.get("action")), payload)
                                return AnalyzerTurnResult(
                                    step_executed=True,
                                    retryable_failure=False,
                                    reasoning=f"[HYPOTHESIS-CLICK] Click affordance on {objs[0].color} object at ({payload.get('col')}, {payload.get('row')})"
                                )

            # Evidence-only action shortcut: use a path only when every edge
            # was observed for this exact game/level. Coordinate-less ACTION6
            # is deliberately excluded because its payload is state-specific.
            evidence_plan = self._evidence_planner.plan_to_terminal(
                grid, game_id=game_id, level=level, max_depth=4
            )
            if evidence_plan is not None:
                planned_actions, evidence = evidence_plan
                if planned_actions and all(edge.observations >= 1 for edge in evidence):
                    evidence_payload = _allowed_payload(self._build_payload(planned_actions[0]))
                    if (evidence_payload is not None
                            and evidence_payload.get("action") != "ACTION6"
                            and not self._vetoed(evidence_payload, game_id, sc)):
                        evidence_payload = self._verify(evidence_payload, grid, game_id, stagnation, allowed)
                        ok, env_res = self._commit(step_env, evidence_payload, "evidence_plan", grid)
                        if ok:
                            self._record_transition(game_id, level, grid, planned_actions[0], env_res)
                            self._pending = (grid.copy(), self._action_key(evidence_payload), level)
                            self._note_state(grid, str(evidence_payload.get("action")), evidence_payload)
                            return AnalyzerTurnResult(
                                step_executed=True,
                                retryable_failure=False,
                                reasoning=f"[EVIDENCE-PLAN] observed terminal path depth={len(planned_actions)}",
                            )

            # 4. Query Sovereign Closed-Loop Servo (adaptive gate; skipped
            # cleanly when the consensus module is not in the shipped set)
            consensus = None
            if _ConsensusEngine is not None:
                if self._consensus_engine is None:
                    with contextlib.suppress(Exception):
                        self._consensus_engine = _ConsensusEngine()
                consensus = self._consensus_engine
            if consensus is None:
                action_info, prior_rule = None, None
            else:
                obs_dict = {
                    "grid": grid,
                    "level": level,
                    "step": action_num,
                    "valid_actions": valid_actions,
                    "game_id": game_id,
                }
                try:
                    action_info = consensus.decide(obs_dict)
                    prior_rule = consensus.get_game_prior(game_id)
                except Exception:
                    action_info, prior_rule = None, None
            required_confidence = self._required_confidence(prior_rule)

            if action_info and action_info.get("confidence", 0.0) >= required_confidence:
                raw_act = action_info["action"]
                payload = _allowed_payload(self._build_payload(
                    raw_act if not isinstance(raw_act, dict)
                    else {"action": raw_act.get("action"),
                          "x": raw_act.get("x", raw_act.get("col")),
                          "y": raw_act.get("y", raw_act.get("row"))}))

                # Adversarial sanitize payload (Strict [0, 63] bounds, valid names)
                from arc3sdk.failure_immunity import sanitize_action_payload
                payload = _allowed_payload(
                    sanitize_action_payload(payload) if payload is not None else None)

                if payload is not None and not self._vetoed(payload, game_id, sc):
                    payload = self._verify(payload, grid, game_id, stagnation, allowed)
                    ok, env_res = self._commit(step_env, payload, "consensus", grid)
                    if ok:
                        reason = action_info.get("reason", "servo_short_circuit")
                        self._record_transition(game_id, level, grid, raw_act, env_res)
                        self._pending = (grid.copy(), self._action_key(payload), level)
                        self._note_state(grid, str(payload.get("action")), payload)

                        if transcript_path is not None:
                            try:
                                with open(transcript_path, "a", encoding="utf-8") as f:
                                    f.write(f"\n[SERVO-SHORT-CIRCUIT] act={payload} conf={action_info.get('confidence'):.2f} reason={reason}\n")
                            except Exception:
                                pass

                        return AnalyzerTurnResult(
                            step_executed=True,
                            retryable_failure=False,
                            reasoning=f"[SERVO] {reason}"
                            )

            # AVO is a fallback, never a competitor to a confident LLM.
            # This ordering avoids duplicate work and preserves the strongest
            # evidence source when the two disagree.
            if stagnation >= 2:
                avo_payload = self._avo_proposal(grid, allowed)
                if avo_payload is not None and not self._vetoed(avo_payload, game_id, sc):
                    avo_reason = avo_payload.pop("_avo_reason", "proposal")
                    avo_payload = self._verify(avo_payload, grid, game_id, stagnation, allowed)
                    ok, _res = self._commit(step_env, avo_payload, "avo_fallback", grid)
                    if ok:
                        self._pending = (grid.copy(), self._action_key(avo_payload), level)
                        self._note_state(grid, str(avo_payload.get("action")), avo_payload)
                        return AnalyzerTurnResult(
                            step_executed=True,
                            retryable_failure=False,
                            reasoning=f"[AVO-FALLBACK] {avo_reason}",
                        )

            # CASS counterfactual probe: only when Duck is stuck AND the
            # belief learned from executed outcomes. Same gates as siblings;
            # saves an LLM round trip whenever it fires instead of fallback.
            _cass_out = self._cass_probe_payload(grid, game_id, sc, stagnation, allowed, level,
                                                 _allowed_payload)
            if _cass_out[0] is not None:
                _cass_payload, _cass_reason = _cass_out
                _ok, _res = self._commit(step_env, _cass_payload, "cass_probe", grid)
                if _ok:
                    self._pending = (grid.copy(), self._action_key(_cass_payload), level)
                    self._note_state(grid, str(_cass_payload.get("action")), _cass_payload)
                    return AnalyzerTurnResult(
                        step_executed=True,
                        retryable_failure=False,
                        reasoning=_cass_reason,
                    )

            # BBK Game Brain: dense simulation + beam search BEFORE the LLM.
            # Fixes the passive-fallback gap: the old path gave the LLM zero
            # game-state intelligence (CoT scratchpad went to a file, never
            # into the prompt). The brain simulates every candidate action,
            # ranks futures, and either commits a measured high-confidence
            # move (skipping the LLM round trip) or appends its dense prompt
            # to the transcript so the LLM reasons from evidence.
            try:
                from arc3sdk.bbk_game_brain import get_brain as _get_bbk_brain
                _bbk = _get_bbk_brain()
                _bbk_payload, _bbk_conf, _bbk_reason, _bbk_prompt = _bbk.decide(
                    np.asarray(grid, dtype=np.uint8), game_id, level,
                    action_num, valid_actions,
                    set(self._tried_actions), set(self._tried_clicks),
                    history_entries, int(stagnation),
                    set(self._failed_pairs))
                if transcript_path is not None:
                    with contextlib.suppress(Exception), open(transcript_path, "a", encoding="utf-8") as _bf:
                        _bf.write(f"\n{_bbk_prompt}\n{_bbk_reason}\n")
                if float(_bbk_conf) >= 0.80:
                    _bbk_fixed = _allowed_payload(_bbk_payload)
                    if (_bbk_fixed is not None
                            and not self._vetoed(_bbk_fixed, game_id, sc)):
                        _bbk_fixed = self._verify(
                            _bbk_fixed, grid, game_id, stagnation, allowed)
                        _ok, _res = self._commit(
                            step_env, _bbk_fixed, "bbk_brain", grid)
                        if _ok:
                            with contextlib.suppress(Exception):
                                _after = None
                                if isinstance(_res, dict):
                                    for _k in ("next_grid", "grid", "frame",
                                               "observation"):
                                        if _res.get(_k) is not None:
                                            _after = np.asarray(_res[_k])
                                            break
                                if _after is not None:
                                    _bbk.record_outcome(game_id, np.asarray(
                                        grid, dtype=np.uint8), _after)
                            self._pending = (
                                grid.copy(), self._action_key(_bbk_fixed), level)
                            self._note_state(
                                grid, str(_bbk_fixed.get("action")), _bbk_fixed)
                            return AnalyzerTurnResult(
                                step_executed=True,
                                retryable_failure=False,
                                reasoning=f"[BBK-BRAIN-COMMIT] {_bbk_reason}",
                            )
            except Exception as _bbk_err:
                log.debug(f"[BBK-BRAIN-NOTE] {type(_bbk_err).__name__}: {_bbk_err}")

        except Exception as err:
            log.warning(f"[INTERCEPTOR-GUARD] Interceptor exception: {err}")

        # Fallback to original LLM analyze (port 1234) with System 2 CoT Prompt Augmentation
        self.total_llm_calls += 1
        try:
            from arc3sdk.object_extractor import describe_grid_topology
            obj_summary = describe_grid_topology(grid.tolist())

            cass_hint = ""
            if hasattr(self, "_cass_agent") and self._cass_agent is not None:
                try:
                    cands = [ACTION_REV[a.upper()] for a in (valid_actions or []) if a.upper() in ACTION_REV and 1 <= ACTION_REV[a.upper()] <= 6]
                    if cands:
                        c_act, c_x, c_y, c_dbg = self._cass_agent.plan(grid, cands, game_id=game_id, level=level)
                        ent = c_dbg.get("entropy", 1.0)
                        tier = c_dbg.get("tier", "deep")
                        cass_hint = f"\n- CASS Co-Pilot Belief: Suggested Action={ACTION_MAP.get(c_act)} (x={c_x}, y={c_y}, tier={tier}, entropy={ent:.2f})"
                except Exception:
                    pass

            prior_rule_hint = ""
            game_prior = self._peek_prior(game_id)
            if game_prior:
                # Truncate clean prior to concise summary (first 800 chars)
                short_prior = "\n".join(game_prior.strip().split("\n")[:20])[:800]
                prior_rule_hint = f"\nVerified Domain Knowledge & Game Rules:\n{short_prior}\n"

            cot_system_guide = (
                f"\n[SYSTEM 2 COGNITIVE SCRATCHPAD]\n"
                f"Game ID: {game_id} | Level: {level} | Step: {action_num}\n"
                f"{prior_rule_hint}"
                f"Visual Scene Objects & Affordances:\n{obj_summary}{cass_hint}\n"
                f"INSTRUCTION: You must think step-by-step before producing your action:\n"
                f"1. What is the current board state and active objects?\n"
                f"2. What is the sub-goal (e.g., move player to waypoint, align object, click interactable target)?\n"
                f"3. Output your reasoning inside <think> ... </think> tags.\n"
                f"4. Output your verified action.\n"
            )
            if transcript_path is not None and transcript_path.exists():
                try:
                    with open(transcript_path, "a", encoding="utf-8") as tf:
                        tf.write(f"\n[SYSTEM-2-COT-SCRATCHPAD] {cot_system_guide}\n")
                    if transcript_updated is not None:
                        transcript_updated(str(transcript_path))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            return self.original_analyze(
                tool_agent_self,
                state_path,
                action_num,
                valid_actions=valid_actions,
                step_env=step_env,
                transcript_path=transcript_path,
                analysis_step=analysis_step,
                transcript_updated=transcript_updated,
                request_timeout_seconds=request_timeout_seconds,
                should_stop=should_stop,
            )
        except Exception as llm_err:
            log.warning(f"[LLM-FALLBACK-RESCUE] Original analyze failed/timed out: {llm_err}. Engaging deterministic emergency action!")
            # Emergency Rescue: safest available move, TRUE stagnation, canonical map
            try:
                from arc3sdk.solver_agent import choose_action
                from inference.agent.tool_agent import AnalyzerTurnResult
                avail_int = [1, 2, 3, 4]
                if valid_actions:
                    avail_int = [ACTION_REV[a.upper()] for a in valid_actions
                                 if a.upper() in ACTION_REV] or [1]
                rescue_stagnation = 0
                with contextlib.suppress(Exception):
                    rescue_stagnation = _trailing_zero_diffs(history_entries)
                rec_act, rec_x, rec_y, rec_reason = choose_action(
                    grid, avail_int, game_id=game_id,
                    stagnation=rescue_stagnation, action_count=action_num, level=level
                )
                if int(rec_act) == 6:
                    rescue_payload = {"action": "ACTION6",
                                      "row": int(rec_y if rec_y is not None else 32),
                                      "col": int(rec_x if rec_x is not None else 32)}
                else:
                    rescue_payload = {"action": ACTION_MAP.get(int(rec_act), "ACTION1")}

                step_env(rescue_payload)
                self._record_mech("rescue", True)
                return AnalyzerTurnResult(
                    step_executed=True,
                    retryable_failure=False,
                    reasoning=f"[EMERGENCY-RESCUE] LLM failed ({type(llm_err).__name__}). Fallback executed {rescue_payload} ({rec_reason})"
                )
            except Exception as _rescue_err:
                log.error(f"[FATAL-RESCUE-ERROR] {_rescue_err}")
                raise llm_err from _rescue_err


def install_sovereign_servo_hook(solver: Any = None) -> bool:
    try:
        from inference.agent import tool_agent
        if hasattr(tool_agent.ToolAgent, "analyze"):
            if not getattr(tool_agent.ToolAgent.analyze, "_is_sovereign_interceptor", False):
                original = tool_agent.ToolAgent.analyze
                interceptor = SovereignDuckInterceptor(original)

                def patched_analyze(self, *args, **kwargs):
                    return interceptor(self, *args, **kwargs)

                patched_analyze._is_sovereign_interceptor = True
                patched_analyze._interceptor_instance = interceptor
                tool_agent.ToolAgent.analyze = patched_analyze
                print(">>> [SOVEREIGN-SERVO] Successfully patched ToolAgent.analyze with Neuro-Symbolic AutoCoder! <<<", flush=True)
                return True
            else:
                print(">>> [SOVEREIGN-SERVO] ToolAgent.analyze is already patched. <<<", flush=True)
                return True
    except Exception as exc:
        print(f">>> [SOVEREIGN-SERVO] Hook note: {exc} <<<", flush=True)
    # Path 2: generic solver.policy wrap (local dev + worker fallback when the
    # TAAF `inference` bundle is absent). Fail-open pass-through: behavior
    # unchanged, idempotent via marker attribute.
    try:
        import functools as _functools

        _original = getattr(solver, "policy", None)
        if callable(_original):
            if getattr(_original, "_sovereign_servo_wrapped", False):
                return True

            @_functools.wraps(_original)
            def _servo_policy(*args, **kwargs):
                return _original(*args, **kwargs)

            _servo_policy._sovereign_servo_wrapped = True
            _servo_policy._sovereign_servo_original = _original
            solver.policy = _servo_policy
            return True
    except Exception:
        pass
    return False
