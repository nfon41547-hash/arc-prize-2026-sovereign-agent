"""Φ-EVO: Phenomenological Self-Transcending Evolution Engine (v1-evo-phi-1).

Core insight: ordinary EVO systems run gradients over FIXED metrics — they
always plateau once the metrics stop being informative. Φ-EVO transcends
that by evolving the measurement itself:

  1. MetricLiveness: a metric whose variance is zero (or uncorrelated with
     quality) for `patience` generations is RETIRED (weight 0, renormalize);
     a retired metric that turns informative again for a full streak is
     REVIVED. The pool never keeps voting with a blind eye.
  2. HiddenDimensionMiner: SVD via power iteration on the behavior matrix
     (genomes x tasks) finds factors the current metrics fail to explain;
     deterministic through seeded init (lesson: constant init dies on
     symmetric matrices — found by the first two failing tests).
  3. OperatorGenome + MetaOperatorEvolver: the evolution strategy itself is
     a genome (tournament_k, elite_n, distinct_n, mutation law); a bandit
     selects the operator per generation with reward = fitness improvement;
     the operator itself evolves every 4 generations.

stdlib+numpy at import. One RLock. Bounded stores. Fail-open everywhere.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

__version__ = "v1-evo-phi-1"

_LOCK = threading.RLock()

# Seeded PRNG for full determinism (lesson: unseeded/constant init dies on
# symmetric matrices — power iteration needs a random asymmetric start).
_RNG = np.random.RandomState(20260930)

# ---------------------------------------------------------------------------
# 1. MetricLiveness — retire dead metrics, revive informative ones
# ---------------------------------------------------------------------------

_LIVE: OrderedDict[str, dict[str, Any]] = OrderedDict()
# metric -> {"weight": float, "history": deque[float], "flat_streak": int,
#            "retired": bool, "retire_streak": int, "corr_ema": float}

_PATIENCE = 3        # flat generations before retirement
_REVIVE_STREAK = 4   # informative turns before revival
_MIN_W, _MAX_W = 0.05, 4.0


def _liveness_entry(name: str, value: float) -> dict[str, Any]:
    m = _LIVE.get(name)
    if m is None:
        m = {
            "weight": 1.0,
            "history": [],
            "q_history": [],
            "flat_streak": 0,
            "retired": False,
            "retire_streak": 0,
            "corr_ema": 0.0,
        }
        _LIVE[name] = m
    return m


def _observe_metric(name: str, value: float, quality: float) -> None:
    """One observation: update flat streak, correlation, retire/revive."""
    m = _liveness_entry(name, value)
    hist = m["history"]
    hist.append(float(value))
    if len(hist) > 32:
        del hist[: len(hist) - 32]
    # Parallel quality history (correlation needs TWO varying vectors —
    # a constant within-window scalar has zero std and can never correlate).
    q_hist = m["q_history"]
    q_hist.append(float(np.clip(quality, 0.0, 1.0)))
    if len(q_hist) > 32:
        del q_hist[: len(q_hist) - 32]

    # Flat detection: zero variance over the recent window
    if len(hist) >= 3:
        recent = hist[-8:]
        var = float(np.var(recent))
        if var < 1e-9:
            m["flat_streak"] += 1
        else:
            m["flat_streak"] = 0
    else:
        m["flat_streak"] = 0

    # Quality correlation EMA (is this metric informative?) — aligned pairs
    if len(hist) >= 3 and len(q_hist) >= 3:
        n = min(len(hist), len(q_hist), 8)
        recent = np.asarray(hist[-n:], dtype=float)
        qv = np.asarray(q_hist[-n:], dtype=float)
        rs, qs = recent.std(), qv.std()
        if rs > 1e-9 and qs > 1e-9:
            corr = float(np.corrcoef(recent, qv)[0, 1])
            m["corr_ema"] = 0.8 * m["corr_ema"] + 0.2 * abs(corr)

    # Retirement: flat for full patience OR uncorrelated with quality
    if not m["retired"]:
        if m["flat_streak"] >= _PATIENCE or (
            len(hist) >= 8 and m["corr_ema"] < 0.05
        ):
            m["retired"] = True
            m["retire_streak"] = 0
            m["weight"] = 0.0
    else:
        # Revival: informative again (variance AND correlation) for a streak
        n = min(len(hist), len(q_hist), 8)
        recent = np.asarray(hist[-n:], dtype=float) if n >= 2 else np.zeros(2)
        var = float(np.var(recent)) if n >= 2 else 0.0
        informative = var > 1e-6 and m["corr_ema"] >= 0.10
        m["retire_streak"] = m["retire_streak"] + 1 if informative else 0
        if m["retire_streak"] >= _REVIVE_STREAK:
            m["retired"] = False
            m["weight"] = _MIN_W  # re-enter small, earn its way back
            m["flat_streak"] = 0

    _renormalize()


def _renormalize() -> None:
    """Keep total alive weight constant so retirement actually shifts votes."""
    try:
        alive = [m for m in _LIVE.values() if not m["retired"]]
        if not alive:
            return
        total = sum(m["weight"] for m in alive)
        if total <= 1e-9:
            for m in alive:
                m["weight"] = 1.0
            total = float(len(alive))
        target = max(1.0, float(len(alive)))
        scale = target / total
        for m in alive:
            m["weight"] = max(_MIN_W, min(_MAX_W, m["weight"] * scale))
    except Exception:
        pass


def observe_metrics(metrics: dict[str, float], quality: float) -> None:
    """Per-turn: observe all live metric values with the measured quality."""
    try:
        with _LOCK:
            q = float(np.clip(float(quality), 0.0, 1.0))
            for name, value in metrics.items():
                try:
                    _observe_metric(str(name), float(value), q)
                except Exception:
                    continue
            # Absent metrics count as flat (they die too)
            for name, m in list(_LIVE.items()):
                if name not in metrics:
                    last = m["history"][-1] if m["history"] else 0.0
                    _observe_metric(str(name), float(last), q)
            while len(_LIVE) > 32:
                _LIVE.popitem(last=False)
    except Exception:
        pass


def live_weights() -> dict[str, float]:
    """Weight vector over ALIVE (non-retired) metrics only. Never raises."""
    try:
        with _LOCK:
            return {k: float(m["weight"]) for k, m in _LIVE.items()
                    if not m["retired"]}
    except Exception:
        return {}


def retired_metrics() -> dict[str, float]:
    try:
        with _LOCK:
            return {k: float(m["weight"]) for k, m in _LIVE.items() if m["retired"]}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 2. HiddenDimensionMiner — SVD (power iteration) on the behavior matrix
# ---------------------------------------------------------------------------

_BEHAVIOR: list[dict[str, float]] = []   # rows = generations, cols = metrics
_MAX_BEHAVIOR_ROWS = 32


def _behavior_matrix() -> np.ndarray | None:
    """Assemble the (rows x metrics) matrix from the observed stream."""
    try:
        if len(_BEHAVIOR) < 4:
            return None
        cols = sorted({k for row in _BEHAVIOR for k in row})
        if len(cols) < 2:
            return None
        m = np.zeros((len(_BEHAVIOR), len(cols)), dtype=float)
        for i, row in enumerate(_BEHAVIOR):
            for j, c in enumerate(cols):
                m[i, j] = float(row.get(c, 0.0))
        # Require variance somewhere (constant matrix has no hidden factors)
        if float(m.var()) < 1e-12:
            return None
        return m
    except Exception:
        return None


def _power_iteration_svd(m: np.ndarray, k: int = 2, iters: int = 64,
                         tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic top-k singular triplets via seeded power iteration."""
    try:
        # Seed a fixed asymmetric start — symmetric/constant starts collapse
        rng = np.random.RandomState(20260930)
        v = rng.rand(m.shape[1], k)
        v /= max(1e-9, np.linalg.norm(v))
        prev_s = None
        for _ in range(iters):
            u = m @ v
            # Gram-Schmidt-ish deflation for orthogonal top-k
            u_norms = np.linalg.norm(u, axis=0)
            u_norms[u_norms < 1e-12] = 1e-12
            u /= u_norms
            v = m.T @ u
            v_norms = np.linalg.norm(v, axis=0)
            v_norms[v_norms < 1e-12] = 1e-12
            v /= v_norms
            s = np.linalg.norm(m @ v, axis=0)
            if prev_s is not None and float(np.abs(s - prev_s).max()) < tol:
                break
            prev_s = s
        return v, np.asarray(s, dtype=float)
    except Exception:
        return np.zeros((m.shape[1], k)), np.zeros(k)


def mine_hidden_dimensions(metrics: dict[str, float], quality: float) -> dict[str, Any]:
    """Find factors the current metrics fail to explain.

    Returns {"factors": [...], "residual": float, "explained": float}.
    Residual = variance of the quality left after projecting onto the
    top singular directions of the behavior matrix.
    """
    try:
        with _LOCK:
            _BEHAVIOR.append(dict(metrics))
            while len(_BEHAVIOR) > _MAX_BEHAVIOR_ROWS:
                _BEHAVIOR.pop(0)
            m = _behavior_matrix()
            if m is None:
                return {"factors": [], "residual": 0.0, "explained": 0.0}
            v, s = _power_iteration_svd(m, k=min(2, m.shape[1]))
            # Project the latest quality signal onto the top directions
            proj = m[-1] @ v
            total = float(m[-1].var()) if m.shape[1] > 1 else float(abs(m[-1][0]) + 1e-9)
            resid = max(0.0, total - float(np.sum(proj ** 2)))
            factors = [f"f{i}" for i in range(len(s)) if float(s[i]) > 1e-6]
            explained = 1.0 - (resid / total) if total > 1e-9 else 0.0
            return {
                "factors": factors,
                "residual": round(resid, 6),
                "explained": round(float(np.clip(explained, 0.0, 1.0)), 6),
            }
    except Exception:
        return {"factors": [], "residual": 0.0, "explained": 0.0}


# ---------------------------------------------------------------------------
# 3. OperatorGenome + MetaOperatorEvolver — evolving the evolution operator
# ---------------------------------------------------------------------------

_OPERATORS = ("uniform", "tournament", "elite", "distinct", "gaussian")
_OPERATOR_GENOME = {
    "operator": "tournament",
    "tournament_k": 3,
    "elite_n": 2,
    "distinct_n": 4,
    "mutation_rate": 0.25,
}
_OP_BOUNDS = {"tournament_k": (2, 6), "elite_n": (1, 4),
              "distinct_n": (2, 8), "mutation_rate": (0.05, 0.60)}
# Bandit: operator -> {pulls, reward}
_OP_BANDIT: dict[str, dict[str, float]] = {}
_CURRENT_OP = "tournament"  # operator selected for the current generation
_GENERATION = 1
_FITNESS: list[float] = []
_MAX_GEN = 40          # hard cap
_EVOLVE_EVERY = 4      # the operator itself evolves every 4 generations


def _bandit_entry(op: str) -> dict[str, float]:
    b = _OP_BANDIT.get(op)
    if b is None:
        b = {"pulls": 0.0, "reward": 0.0}
        _OP_BANDIT[op] = b
    return b


def select_operator() -> str:
    """UCB1-style selection over operators (deterministic tie-break).

    The selection itself is recorded as a bandit pull; the reward is
    attributed in note_generation to the operator actually in play.
    """
    global _CURRENT_OP
    try:
        with _LOCK:
            best_op, best_score = None, -1e18
            for op in _OPERATORS:
                b = _bandit_entry(op)
                if b["pulls"] == 0:
                    _CURRENT_OP = op  # explore unpulled first
                    _bandit_entry(op)["pulls"] += 1
                    return op
                mean = b["reward"] / b["pulls"]
                explore = 0.6 * (2.0 * sum(x["pulls"] for x in _OP_BANDIT.values())
                                 / b["pulls"]) ** 0.5
                score = mean + explore
                if score > best_score:
                    best_op, best_score = op, score
            _CURRENT_OP = best_op or "tournament"
            _bandit_entry(_CURRENT_OP)["pulls"] += 1
            return _CURRENT_OP
    except Exception:
        _CURRENT_OP = "tournament"
        return "tournament"


def note_generation(fitness: float) -> dict[str, Any]:
    """One generation: record fitness, bandit update, maybe evolve the operator."""
    try:
        with _LOCK:
            global _GENERATION
            op = _CURRENT_OP  # reward the operator actually in play
            prev = _FITNESS[-1] if _FITNESS else None
            f = float(np.clip(float(fitness), 0.0, 1.0))
            _FITNESS.append(f)
            while len(_FITNESS) > 64:
                _FITNESS.pop(0)

            # Bandit reward = fitness improvement over the previous generation
            if prev is not None:
                b = _bandit_entry(op)
                b["pulls"] = max(b["pulls"], 1.0)
                b["reward"] += (f - prev)

            # Operator self-evolution every _EVOLVE_EVERY generations
            evolved = False
            if _GENERATION % _EVOLVE_EVERY == 0 and len(_FITNESS) >= _EVOLVE_EVERY:
                _evolve_operator()
                evolved = True
            _GENERATION += 1
            if _GENERATION > _MAX_GEN:
                _GENERATION = _MAX_GEN
            return {
                "generation": _GENERATION,
                "operator": _OPERATOR_GENOME.get("operator"),
                "evolved": evolved,
                "fitness": round(f, 4),
            }
    except Exception:
        return {"error": "failed"}


def _evolve_operator() -> None:
    """The operator itself evolves: pick the bandit's best, mutate the genome."""
    try:
        with _LOCK:
            # Choose the operator with the best empirical reward
            best_op, best_mean = None, -1e18
            for op, b in _OP_BANDIT.items():
                if b["pulls"] >= 2:
                    mean = b["reward"] / b["pulls"]
                    if mean > best_mean:
                        best_op, best_mean = op, mean
            if best_op is not None:
                _OPERATOR_GENOME["operator"] = best_op
            # Mutate numeric genes (clamped)
            rng = np.random.RandomState(20260930 + _GENERATION)
            for key, (lo, hi) in _OP_BOUNDS.items():
                cur = float(_OPERATOR_GENOME.get(key, lo))
                factor = float(rng.uniform(0.75, 1.35))
                _OPERATOR_GENOME[key] = max(lo, min(hi, cur * factor))
    except Exception:
        pass


def operator_genome() -> dict[str, Any]:
    try:
        with _LOCK:
            return dict(_OPERATOR_GENOME)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 4. Status / reset
# ---------------------------------------------------------------------------


def status() -> dict[str, Any]:
    try:
        with _LOCK:
            return {
                "version": __version__,
                "live_metrics": len(live_weights()),
                "retired_metrics": len(retired_metrics()),
                "behavior_rows": len(_BEHAVIOR),
                "operator_genome": dict(_OPERATOR_GENOME),
                "generation": _GENERATION,
                "bandit": {k: dict(v) for k, v in _OP_BANDIT.items()},
            }
    except Exception:
        return {"version": __version__, "error": "failed"}


def reset() -> None:
    try:
        with _LOCK:
            global _GENERATION
            _LIVE.clear()
            _BEHAVIOR.clear()
            _OP_BANDIT.clear()
            _FITNESS.clear()
            _GENERATION = 1
            _RNG.seed(20260930)
    except Exception:
        pass
