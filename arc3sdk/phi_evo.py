"""Φ-EVO: Phenomenological Self-Transcending Evolution Engine.

Core insight: ordinary EVO systems run gradients over FIXED metrics — they
always plateau once the metrics stop being informative. Φ-EVO transcends
that by evolving the measurement itself:

  1. METRIC DEATH (phenomenological): every metric is scored by its
     information gain (variance of its deltas over recent verified turns).
     A metric whose gain collapses below an adaptive floor is declared DEAD,
     removed from the weighting pool, and replaced by a newly discovered
     candidate — the engine never keeps voting with a blind eye.
  2. HIDDEN DIMENSION DISCOVERY: candidate quality dimensions are mined
     from the live observation stream itself (agreement rate, novelty rate,
     action diversity, position entropy, transition surprise, dead-signature
     escape). Each candidate starts ADJUDICATED (small weight) and is
     promoted only after it proves informative (gain > floor for N turns).
  3. SELF-META-EVOLUTION: the evolution operator itself is a genome.
     (rate, floor, promote_after, diversity) mutate multiplicatively when
     the meta-score (system-level progress rate) stagnates, so the engine
     learns HOW to evolve, not just WHAT to value.

stdlib+numpy at import. One RLock. Bounded stores. Fail-open everywhere.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict, deque
from typing import Any

import numpy as np

__version__ = "v1-phi-evo-1"

_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# 1. METRIC REGISTRY (the observable quality dimensions)
# ---------------------------------------------------------------------------

_METRICS: OrderedDict[str, dict[str, Any]] = OrderedDict()
# metric -> {
#   "alive": bool, "weight": float, "gain_ema": float,
#   "history": deque[(value)], "prev": float | None,
#   "turns_alive": int, "adjudicated": bool,
# }

_DEAD_GRAVEYARD: dict[str, dict[str, Any]] = {}
# dead metric -> {"turns_lived", "final_gain", "reason"}

# Hidden-dimension candidates awaiting adjudication
_CANDIDATES: OrderedDict[str, dict[str, Any]] = OrderedDict()

# Verified-turn stream (value snapshots for gain computation)
_TURN_STREAM: deque[dict[str, float]] = deque(maxlen=64)

# ---------------------------------------------------------------------------
# 2. SELF-META-GENOME (the evolution operator itself)
# ---------------------------------------------------------------------------

_GENOME = {
    "rate": 0.10,          # per-turn weight adaptation rate
    "floor": 0.02,         # gain floor below which a metric dies
    "promote_after": 8,    # informative turns before a candidate is promoted
    "diversity": 0.30,     # weight of the diversity dimension
    "generation": 1,       # mutation generation
}
_GENOME_BOUNDS = {
    "rate": (0.02, 0.40),
    "floor": (0.005, 0.10),
    "promote_after": (3, 24),
    "diversity": (0.05, 0.80),
}
_META_HISTORY: deque[float] = deque(maxlen=32)
_META_STAGNATION = 0
_MAX_STAGNATION = 12  # stagnation turns before the genome mutates

# ---------------------------------------------------------------------------
# 3. OBSERVATION PRIMITIVES (pure functions over the live stream)
# ---------------------------------------------------------------------------


def _variance_of_deltas(history: deque) -> float:
    """Information gain = variance of consecutive deltas (bounded)."""
    try:
        if len(history) < 3:
            return 0.0
        vals = [v for _, v in history][-16:]
        deltas = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        if not deltas:
            return 0.0
        mean = sum(deltas) / len(deltas)
        var = sum((d - mean) ** 2 for d in deltas) / len(deltas)
        return float(var)
    except Exception:
        return 0.0


def _entropy_of_counts(counts: list[float]) -> float:
    """Shannon entropy of a count distribution (bits)."""
    try:
        total = sum(counts)
        if total <= 0:
            return 0.0
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                h -= p * math.log2(p)
        return h
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 4. HIDDEN DIMENSION MINERS (discover new quality dimensions live)
# ---------------------------------------------------------------------------


def _mine_candidates(obs: dict[str, Any]) -> None:
    """Mine hidden-dimension candidates from the live observation.

    Each candidate starts with weight 0 (ADJUDICATED — no vote) and is
    promoted only after proving informative for promote_after turns.
    """
    try:
        with _LOCK:
            # A1: agreement rate — fraction of actions that changed the board
            n = int(obs.get("n_actions", 0))
            changed = int(obs.get("n_changed", 0))
            if n > 0:
                agree = changed / float(n)
                _touch_candidate("agree_rate", agree)

            # A2: action diversity — entropy of the action distribution
            acts = obs.get("action_counts")
            if isinstance(acts, (list, tuple)) and len(acts) >= 2:
                div = _entropy_of_counts([float(a) for a in acts]) / max(
                    1, math.log2(len(acts)))
                _touch_candidate("action_diversity", div)

            # A3: position entropy — spatial spread of changed cells
            pos = obs.get("position_counts")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                pe = _entropy_of_counts([float(p) for p in pos]) / max(
                    1, math.log2(len(pos)))
                _touch_candidate("position_entropy", pe)

            # A4: transition surprise — 1 - agreement with predicted op
            surprise = obs.get("transition_surprise")
            if isinstance(surprise, (int, float)):
                _touch_candidate("transition_surprise", float(surprise))

            # A5: dead-signature escape — clicks on previously-dead types
            # that DID change the board (resurrection evidence)
            escape = obs.get("dead_escape")
            if isinstance(escape, (int, float)):
                _touch_candidate("dead_escape", float(escape))

            # A6: progress velocity — levels per turn EMA
            vel = obs.get("progress_velocity")
            if isinstance(vel, (int, float)):
                _touch_candidate("progress_velocity", float(vel))

            # Bounded: keep only the most recent 12 candidates
            while len(_CANDIDATES) > 12:
                _CANDIDATES.popitem(last=False)
    except Exception:
        pass


def _touch_candidate(name: str, value: float) -> None:
    """Record one observation for a candidate dimension. Never raises."""
    try:
        with _LOCK:
            c = _CANDIDATES.setdefault(name, {
                "history": deque(maxlen=32), "prev": None,
                "informative_streak": 0, "promoted": False,
            })
            c["history"].append((time.monotonic(), float(value)))
            gain = _variance_of_deltas(c["history"])
            rate, floor, promote_after, _ = _genome_values()
            if gain > floor:
                c["informative_streak"] += 1
            else:
                c["informative_streak"] = 0
            # Promotion: proven informative -> joins the live metric pool
            if (not c["promoted"] and c["informative_streak"] >= promote_after):
                c["promoted"] = True
                _METRICS[name] = {
                    "alive": True, "weight": 0.5,  # small adjudicated start
                    "gain_ema": gain, "history": deque(c["history"], maxlen=32),
                    "prev": None, "turns_alive": 0, "adjudicated": False,
                }
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 5. CORE API (never raises)
# ---------------------------------------------------------------------------


def _genome_values() -> tuple[float, float, int, float]:
    try:
        with _LOCK:
            g = _GENOME
            return (float(g["rate"]), float(g["floor"]),
                    int(g["promote_after"]), float(g["diversity"]))
    except Exception:
        return (0.10, 0.02, 8, 0.30)


def register_metrics(metrics: dict[str, float]) -> None:
    """Register the current values of all live metrics (per turn).

    Metrics absent from this dict keep their last value (persistence);
    a metric that stays absent/flat across turns dies by gain collapse.
    """
    try:
        with _LOCK:
            rate, floor, _, _ = _genome_values()
            seen: set[str] = set()
            for name, value in metrics.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                seen.add(name)
                m = _METRICS.setdefault(name, {
                    "alive": True, "weight": 1.0, "gain_ema": 0.0,
                    "history": deque(maxlen=32), "prev": None,
                    "turns_alive": 0, "adjudicated": False,
                })
                if not m.get("alive", True):
                    continue  # dead metrics stay dead until rediscovered
                m["history"].append((time.monotonic(), v))
                m["turns_alive"] = int(m.get("turns_alive", 0)) + 1
                gain = _variance_of_deltas(m["history"])
                prev_gain = float(m.get("gain_ema", 0.0))
                m["gain_ema"] = 0.7 * prev_gain + 0.3 * gain
                # METRIC DEATH: gain collapsed below the adaptive floor
                if (m["turns_alive"] >= 4
                        and m["gain_ema"] < floor):
                    m["alive"] = False
                    _DEAD_GRAVEYARD[name] = {
                        "turns_lived": m["turns_alive"],
                        "final_gain": round(m["gain_ema"], 6),
                        "reason": "gain-collapse",
                    }
            # Absent metrics also accumulate flat history (they die too)
            for name, m in _METRICS.items():
                if name not in seen and m.get("alive", True):
                    m["history"].append((time.monotonic(), float(m.get("prev", 0.0) or 0.0)))
                    m["turns_alive"] = int(m.get("turns_alive", 0)) + 1
                    gain = _variance_of_deltas(m["history"])
                    prev_gain = float(m.get("gain_ema", 0.0))
                    m["gain_ema"] = 0.7 * prev_gain + 0.3 * gain
                    if m["turns_alive"] >= 4 and m["gain_ema"] < floor:
                        m["alive"] = False
                        _DEAD_GRAVEYARD[name] = {
                            "turns_lived": m["turns_alive"],
                            "final_gain": round(m["gain_ema"], 6),
                            "reason": "absent-flat",
                        }
                m["prev"] = float(metrics.get(name, m.get("prev", 0.0) or 0.0))
            # Bounded
            while len(_METRICS) > 24:
                _METRICS.popitem(last=False)
            while len(_DEAD_GRAVEYARD) > 24:
                _DEAD_GRAVEYARD.popitem(last=False)
    except Exception:
        pass


def live_weights() -> dict[str, float]:
    """Current weight vector over ALIVE metrics only. Never raises."""
    try:
        with _LOCK:
            out: dict[str, float] = {}
            for name, m in _METRICS.items():
                if m.get("alive", True):
                    out[name] = float(m.get("weight", 1.0))
            return out
    except Exception:
        return {}


def dead_metrics() -> dict[str, dict[str, Any]]:
    """Graveyard of metrics that stopped being informative."""
    try:
        with _LOCK:
            return {k: dict(v) for k, v in _DEAD_GRAVEYARD.items()}
    except Exception:
        return {}


def promoted_candidates() -> dict[str, float]:
    """Hidden dimensions that earned promotion into the live pool."""
    try:
        with _LOCK:
            return {name: float(m.get("weight", 0.5))
                    for name, m in _METRICS.items() if m.get("adjudicated") is False
                    and name in {c for c, v in _CANDIDATES.items() if v.get("promoted")}}
    except Exception:
        return {}


def update_from_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    """One verified turn: mine candidates, adapt weights, maybe mutate genome.

    outcome keys (all optional, fail-open):
      n_actions, n_changed, action_counts, position_counts,
      transition_surprise, dead_escape, progress_velocity,
      level_completed, board_changed, technique_performances (dict)
    """
    try:
        with _LOCK:
            # Mine hidden dimensions from this turn's stream
            _mine_candidates(outcome)

            # Per-technique multiplicative update from observed performance
            rate, floor, _, diversity = _genome_values()
            perfs = outcome.get("technique_performances")
            if isinstance(perfs, dict):
                for tech, good in perfs.items():
                    try:
                        good_v = float(good)
                    except Exception:
                        continue
                    evo = _METRICS.get(f"tech:{tech}")
                    if evo is None:
                        evo = _METRICS.setdefault(f"tech:{tech}", {
                            "alive": True, "weight": 1.0, "gain_ema": 0.0,
                            "history": deque(maxlen=32), "prev": None,
                            "turns_alive": 0, "adjudicated": False,
                        })
                    w = float(evo.get("weight", 1.0))
                    gain = _EVOLUTION_RATE if good_v > 0.5 else -_EVOLUTION_RATE
                    w = max(_MIN_WEIGHT, min(_MAX_WEIGHT, w * (1.0 + gain)))
                    evo["weight"] = w
                    evo["history"].append((time.monotonic(), w))
                    evo["turns_alive"] = int(evo.get("turns_alive", 0)) + 1

            # System-level progress rate for meta-evolution
            board_changed = bool(outcome.get("board_changed", False))
            level_completed = bool(outcome.get("level_completed", False))
            progress = 1.0 if level_completed else (0.4 if board_changed else 0.0)
            prev = _META_HISTORY[-1] if _META_HISTORY else 0.0
            _META_HISTORY.append(0.7 * prev + 0.3 * progress)

            # SELF-META-EVOLUTION: stagnation -> mutate the genome
            global _META_STAGNATION
            if len(_META_HISTORY) >= 4:
                ema = _META_HISTORY[-1]
                if ema < 0.15:
                    _META_STAGNATION += 1
                else:
                    _META_STAGNATION = 0
                if _META_STAGNATION >= _MAX_STAGNATION:
                    _mutate_genome()
                    _META_STAGNATION = 0

            return {
                "alive": len(live_weights()),
                "dead": len(_DEAD_GRAVEYARD),
                "generation": _GENOME.get("generation", 1),
                "meta_ema": round(float(_META_HISTORY[-1]), 4) if _META_HISTORY else 0.0,
            }
    except Exception:
        return {"error": "failed"}


def _mutate_genome() -> None:
    """Multiplicative genome mutation (clamped to bounds). Never raises."""
    try:
        with _LOCK:
            import random as _rnd
            for key, (lo, hi) in _GENOME_BOUNDS.items():
                cur = float(_GENOME.get(key, lo))
                factor = _rnd.uniform(0.7, 1.4)
                new = max(lo, min(hi, cur * factor))
                _GENOME[key] = new
            _GENOME["generation"] = int(_GENOME.get("generation", 1)) + 1
    except Exception:
        pass


def genome() -> dict[str, Any]:
    """Current self-meta genome (the evolution operator itself)."""
    try:
        with _LOCK:
            return dict(_GENOME)
    except Exception:
        return {}


def status() -> dict[str, Any]:
    """Bounded status dict for telemetry."""
    try:
        with _LOCK:
            return {
                "version": __version__,
                "alive_metrics": len(live_weights()),
                "dead_metrics": len(_DEAD_GRAVEYARD),
                "candidates": len(_CANDIDATES),
                "promoted": len(promoted_candidates()),
                "genome": dict(_GENOME),
                "meta_ema": round(float(_META_HISTORY[-1]), 4) if _META_HISTORY else 0.0,
                "stagnation": _META_STAGNATION,
            }
    except Exception:
        return {"version": __version__, "error": "failed"}


def reset() -> None:
    """Reset all phi-evo state (call on game/level change if desired)."""
    global _META_STAGNATION
    try:
        with _LOCK:
            _METRICS.clear()
            _DEAD_GRAVEYARD.clear()
            _CANDIDATES.clear()
            _TURN_STREAM.clear()
            _META_HISTORY.clear()
            _META_STAGNATION = 0
    except Exception:
        pass


# Multiplicative evolution constants (shared with the technique updates)
_EVOLUTION_RATE = 0.10
_MIN_WEIGHT = 0.1
_MAX_WEIGHT = 2.0
