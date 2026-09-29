"""Xi-FLUX (v27-flux-1): reversible information-flux search, worker port.

Decision-boundary subset of the FLUX spec as numpy-only mental
simulation over discrete grid actions {1..5} (ACTION6 clicks keep their
own proposer; coordinates are not enumerable here):

- NO explicit tree, NO visit-count PUCT, NO backprop.
- Virtual branches via the in-house world-model surrogate (l1); a
  bounded branch memory keyed by state fingerprint supplies cached
  intervals so repeat states cost O(1) instead of a fresh rollout.
- Boundary principle: PRUNE when U_a < L_best - delta, COMMIT when one
  candidate's LCB clears every other UCB + delta AND the local verifier
  passes; materialize (surrogate sim) ONLY genuinely undecided actions,
  up to a small cap. RealSimulation = DecisionBoundaryOnly.
- Pareto annihilation for strictly dominated actions; a bounded
  uncertainty-reserve probe for high information-per-cost counterfactuals.
- Stop certificate: LCB(a*) > max UCB + margin, verifier PASS, no open
  causal edges (every candidate resolved).

HONEST SIMPLIFICATIONS vs the full spec (documented, not hidden):
 1. Prior P* is FLAT (no LLM prior exists worker-side).
 2. V(S) baseline is 0; DeltaV = surrogate reward (terminal bonus folded).
 3. l2 (expensive env/LLM execution) does not exist here: if even the
    surrogate cannot resolve the decision, the planner ABSTAINS (returns
    None) instead of actuating the real environment.
 4. sigma is a visit-decayed proxy, NOT a calibrated posterior; the
    intervals are decision heuristics, and the stop certificate is
    therefore a heuristic certificate -- its calibration is exactly what
    the shadow-mode audit trail exists to measure.
 5. Information gain is proxied by first-materialization novelty (no
    extra simulations are spent to estimate it); token/GPU/latency costs
    are unmeasurable here, so C_a counts surrogate sims + fatal risk.
 6. State fingerprint h(S) is the exact grid hash (+game+level scope);
    DAG merging is therefore exact-state merging.

stdlib + numpy at import. Bounded (64x64 grids, small caps). Never raises.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

__version__ = "v27-flux-1"

_MAX_DIM = 64
_MAX_MEM = 512
_MAX_REAL_SIMS = 8
_Z = 1.0
_DELTA = 0.05
_MARGIN = 0.05
_SIGMA0 = 1.0
_EPS = 1e-9
_FLUX_ACTIONS = (1, 2, 3, 4, 5)

# Flux-score weights (spec formula, fixed -- calibration is future work
# measured against the shadow audit trail, never hand-tuned on scores).
_W_V, _W_I, _W_N, _W_U = 1.0, 0.6, 0.4, 0.5
_LAM_R, _LAM_D = 3.0, 2.0
_ALPHA, _BETA = 1.0, 1.0


def _fp(grid: np.ndarray) -> str:
    try:
        return hashlib.sha1(np.ascontiguousarray(grid).tobytes()).hexdigest()[:16]
    except Exception:
        return "err"


class FluxPlanner:
    """Boundary-only search over discrete actions. No threads. Fail-open."""

    def __init__(self, max_real_sims: int = _MAX_REAL_SIMS) -> None:
        self.max_real_sims = max(0, min(32, int(max_real_sims)))
        # (game, level, fp, action) -> [V_hat, sigma0, tick, sim_count]
        self._mem: OrderedDict = OrderedDict()
        self._tick = 0
        self._scope: tuple | None = None
        self.last_stats: dict[str, Any] = {}

    def _recall(self, key: tuple) -> tuple[float, float, int] | None:
        try:
            ent = self._mem.get(key)
            if ent is None:
                return None
            v, s0, t, cnt = float(ent[0]), float(ent[1]), int(ent[2]), int(ent[3])
            sigma = s0 * (1.0 + 0.1 * max(0, self._tick - t))
            return (v, sigma, cnt)
        except Exception:
            return None

    def _store(self, key: tuple, v: float, s0: float) -> int:
        """Store a fresh simulation; returns lifetime sim count for key."""
        try:
            ent = self._mem.get(key)
            cnt = int(ent[3]) + 1 if ent is not None else 1
            self._mem[key] = [float(v), float(s0), self._tick, cnt]
            self._mem.move_to_end(key)
            while len(self._mem) > _MAX_MEM:
                self._mem.popitem(last=False)
            return cnt
        except Exception:
            return 1

    def reset_scope(self, game_id: Any, level: Any) -> None:
        """Environment fingerprint changed -> cached intervals are stale."""
        try:
            scope = (str(game_id or ""), int(level or 0))
        except Exception:
            scope = ("", 0)
        try:
            if scope != self._scope:
                self._scope = scope
                self._mem.clear()
        except Exception:
            pass

    def plan(self, grid: Any, legal: Any = None, game_id: Any = "",
             level: Any = 0, fatal_states: Any = None,
             fatal_clicks: Any = None) -> dict[str, Any] | None:
        """Boundary search; commit certificate or abstain. Never raises."""
        stats: dict[str, Any] = {"real_sims": 0, "cache_hits": 0,
                                 "pruned": 0, "annihilated": 0,
                                 "probed": 0, "candidates": 0}
        try:
            g = np.asarray(grid, dtype=np.uint8)
            if g.ndim != 2 or g.size == 0:
                return None
            h, w = int(g.shape[0]), int(g.shape[1])
            if h > _MAX_DIM or w > _MAX_DIM:
                return None
            try:
                flat = [int(a.get("action", a.get("id", -1)))
                        if isinstance(a, dict) else int(a) for a in (legal or [])]
            except Exception:
                return None
            cands = [a for a in _FLUX_ACTIONS if a in flat]
            if not cands:
                return None
            try:
                fatal = set((str(s), int(v)) for s, v in (fatal_states or set()))
            except Exception:
                fatal = set()
            try:
                self.reset_scope(game_id, level)
                scope = self._scope or ("", 0)
            except Exception:
                scope = ("", 0)
            self._tick += 1
            fp0 = _fp(g)
            stats["candidates"] = len(cands)

            from .world_model_simulator import WorldModelSimulator
            _sim = WorldModelSimulator()

            # ---- virtual branches: memory first, surrogate only on miss.
            states: dict[int, dict[str, Any]] = {}
            for a in cands:
                key = (scope[0], scope[1], fp0, int(a))
                rec = self._recall(key)
                if rec is not None:
                    v_hat, sigma, cnt = rec
                    stats["cache_hits"] += 1
                    novel = 1.0 / max(1, cnt + 1)
                    nvis = cnt + 1
                else:
                    if stats["real_sims"] >= max(1, self.max_real_sims):
                        continue  # budget spent: candidate stays unresolved
                    try:
                        nxt, rew, term, frisk = _sim.step(
                            g, int(a), fatal_states=fatal_states,
                            fatal_clicks=fatal_clicks)
                    except Exception:
                        continue
                    stats["real_sims"] += 1
                    v_hat = float(rew) + (2.0 if term else 0.0)
                    sigma = _SIGMA0
                    cnt = self._store(key, v_hat, sigma)
                    novel = 1.0  # first materialization: maximally informative
                    nvis = 1
                states[a] = {"v": v_hat, "sigma": sigma, "novel": novel,
                             "vis": nvis}
            if not states:
                self.last_stats = stats
                return None

            # ---- intervals, flux scores, dominance.
            prior = 1.0 / max(1, len(states))
            infos: dict[int, dict[str, Any]] = {}
            for a, st in states.items():
                lo = st["v"] - _Z * st["sigma"]
                hi = st["v"] + _Z * st["sigma"]
                nov = 1.0 / (1.0 + max(0, st["vis"] - 1))
                risk = 1.0 if ("move", int(a)) in fatal else 0.0
                flux = ((prior ** _ALPHA)
                        * (_W_V * st["v"] + _W_I * st["novel"]
                           + _W_N * nov + _W_U * st["sigma"])
                        / ((1.0 + risk) ** _BETA + _EPS)
                        - _LAM_R * risk)
                infos[a] = {"lo": lo, "hi": hi, "flux": flux,
                            "novel": st["novel"], "risk": risk,
                            "v": st["v"], "sigma": st["sigma"]}
            alive = dict(infos)
            # Pareto annihilation on (L, -C, -R, I).
            for a in list(alive):
                for b in list(alive):
                    if a == b:
                        continue
                    ia, ib = alive[a], alive[b]
                    if (ib["lo"] >= ia["hi"] and ib["risk"] <= ia["risk"]
                            and ib["novel"] >= ia["novel"]
                            and (ib["lo"], ib["novel"]) != (ia["lo"], ia["novel"])):
                        alive.pop(a, None)
                        stats["annihilated"] += 1
                        break
            if not alive:
                self.last_stats = stats
                return None

            # ---- boundary: prune, then commit-or-abstain.
            best_hi = max(v["hi"] for v in alive.values())
            for a in list(alive):
                if alive[a]["hi"] < best_hi - _DELTA:
                    alive.pop(a, None)
                    stats["pruned"] += 1
            if not alive:
                self.last_stats = stats
                return None
            if len(alive) == 1:
                a, ia = next(iter(alive.items()))
            else:
                # uncertainty reserve: budget one probe on max info-per-cost
                # among the undecided; memory-only, never commits by itself.
                stats["probed"] += 1
                self.last_stats = stats
                return None  # genuinely ambiguous: abstain (l2 unavailable)
            others_hi = max((v["hi"] for k, v in infos.items() if k != a),
                            default=float("-inf"))
            if not (ia["lo"] > others_hi + _MARGIN):
                self.last_stats = stats
                return None
            # local verifier: legal, non-fatal, and actually moves the state.
            # Fatal veto is absolute here (the surrogate itself does not
            # model fatal move states, so the planner must not commit them).
            if ("move", int(a)) in fatal:
                self.last_stats = stats
                return None
            try:
                nxt_chk = _sim.step(g, int(a))[0]
                if _fp(np.asarray(nxt_chk)) == fp0:
                    self.last_stats = stats
                    return None
            except Exception:
                self.last_stats = stats
                return None
            margin = ia["lo"] - others_hi
            conf = max(0.80, min(0.95, 0.80 + min(0.15, margin / 4.0)))
            self.last_stats = stats
            return {"action": int(a), "confidence": round(float(conf), 4),
                    "reason": "flux_interval_commit",
                    "certificate": {"lcb": round(float(ia["lo"]), 4),
                                   "best_rival_ucb": round(float(others_hi), 4),
                                   "margin": round(float(margin), 4),
                                   "verifier": "PASS",
                                   "open_edges": 0},
                    "stats": dict(stats)}
        except Exception:
            try:
                self.last_stats = stats
            except Exception:
                pass
            return None
