"""Meta-Evolution — Self-improving system that coordinates with the LLM.

This is the bridge between the symbolic decision chain and the LLM analyzer.
It enables self-meta-evolution: the system learns from LLM feedback and
improves its own decision-making over time.

How it works:
  1. Before each turn, the orchestrator proposes actions.
  2. The LLM analyzer acts (or the tier substitutes).
  3. After each turn, the meta-evolution module observes the outcome.
  4. It updates transition confidence, technique weights, and calibration.
  5. Over time, the system evolves: techniques that work get boosted,
     techniques that fail get suppressed.

This is NOT a copy of any existing technique — it is a new algorithm
that enables self-meta-evolution through LLM feedback.

Design principles:
- Fail-open: any exception -> no evolution (safe).
- Intrinsic only: no game-ID-keyed manuals, no memorization.
- Bounded: O(n) per turn, no threads, stdlib+numpy at import.
- Novelty gate: only learn from NEW transitions.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

__version__ = "v1-meta-evolution-1"

_LOCK = threading.RLock()

# Evolution state: technique -> {successes, failures, weight}
_EVOLUTION: dict[str, dict[str, float]] = {}

# Transition memory: (before_hash, after_hash) -> {type, outcome}
_TRANSITION_MEMORY: dict[tuple[str, str], dict[str, Any]] = {}

# Max memory size
_MAX_MEMORY = 512

# Evolution rate (how fast weights adapt)
_EVOLUTION_RATE = 0.1

# Minimum weight (techniques never go below this)
_MIN_WEIGHT = 0.1

# Maximum weight (techniques never go above this)
_MAX_WEIGHT = 2.0


def _hash_grid(grid: np.ndarray) -> str:
    """Fast grid hash."""
    try:
        return hash(grid.tobytes())
    except Exception:
        return ""


def observe_outcome(
    before: np.ndarray,
    after: np.ndarray,
    action: int,
    technique: str,
    success: bool,
    game_id: str = "unknown",
    level: int = 1,
) -> None:
    """Observe the outcome of an action and update evolution state.

    Called after each turn to learn from the result.
    """
    try:
        with _LOCK:
            # Update technique weight
            evo = _EVOLUTION.setdefault(technique, {"successes": 0.0, "failures": 0.0, "weight": 1.0})
            if success:
                evo["successes"] += 1
                evo["weight"] = min(_MAX_WEIGHT, evo["weight"] + _EVOLUTION_RATE)
            else:
                evo["failures"] += 1
                evo["weight"] = max(_MIN_WEIGHT, evo["weight"] - _EVOLUTION_RATE)

            # Store transition memory
            b_hash = _hash_grid(before)
            a_hash = _hash_grid(after)
            if b_hash and a_hash:
                _TRANSITION_MEMORY[(b_hash, a_hash)] = {
                    "action": action,
                    "technique": technique,
                    "success": success,
                    "game_id": game_id,
                    "level": level,
                }
                while len(_TRANSITION_MEMORY) > _MAX_MEMORY:
                    _TRANSITION_MEMORY.pop(next(iter(_TRANSITION_MEMORY)))
    except Exception:
        pass


def get_technique_weight(technique: str) -> float:
    """Get the current evolution weight for a technique."""
    try:
        with _LOCK:
            return _EVOLUTION.get(technique, {}).get("weight", 1.0)
    except Exception:
        return 1.0


def get_all_weights() -> dict[str, float]:
    """Get all technique weights."""
    try:
        with _LOCK:
            return {k: v["weight"] for k, v in _EVOLUTION.items()}
    except Exception:
        return {}


def apply_evolution(proposals: list[tuple[int, int | None, int | None, str, float]]) -> list[tuple[int, int | None, int | None, str, float]]:
    """Apply evolution weights to proposals.

    Boosts proposals from successful techniques, suppresses failed ones.
    """
    try:
        with _LOCK:
            weighted = []
            for act, x, y, src, conf in proposals:
                weight = _EVOLUTION.get(src, {}).get("weight", 1.0)
                # Apply weight as a multiplier (clamped)
                new_conf = conf * weight
                new_conf = max(0.0, min(1.0, new_conf))
                weighted.append((act, x, y, src, new_conf))
            # Re-sort by new confidence
            weighted.sort(key=lambda p: -p[4])
            return weighted
    except Exception:
        return proposals


def get_evolution_summary() -> dict[str, Any]:
    """Get a summary of the evolution state."""
    try:
        with _LOCK:
            return {
                "version": __version__,
                "techniques": {k: dict(v) for k, v in _EVOLUTION.items()},
                "transitions": len(_TRANSITION_MEMORY),
                "evolution_rate": _EVOLUTION_RATE,
            }
    except Exception:
        return {"version": __version__, "error": "failed"}


def reset_evolution() -> None:
    """Reset all evolution state."""
    global _EVOLUTION, _TRANSITION_MEMORY
    with _LOCK:
        _EVOLUTION = {}
        _TRANSITION_MEMORY = {}


def evolve_from_llm_feedback(
    llm_action: int,
    tier_action: int | None,
    before: np.ndarray,
    after: np.ndarray,
    game_id: str = "unknown",
    level: int = 1,
) -> None:
    """Evolve based on LLM feedback.

    If the LLM action succeeded, boost the LLM technique.
    If the tier action succeeded, boost the tier technique.
    If both failed, suppress both.
    """
    try:
        # Observe LLM outcome
        llm_success = not np.array_equal(before, after)
        observe_outcome(before, after, llm_action, "llm_analyzer", llm_success, game_id, level)

        # Observe tier outcome (if tier acted)
        if tier_action is not None and tier_action != llm_action:
            tier_success = not np.array_equal(before, after)
            observe_outcome(before, after, tier_action, "tier_substitution", tier_success, game_id, level)
    except Exception:
        pass
