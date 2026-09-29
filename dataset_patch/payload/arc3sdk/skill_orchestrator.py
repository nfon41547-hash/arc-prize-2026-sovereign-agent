"""Skill Orchestrator — wires SovereignSkillsMatrix + AbstractionSkillRegistry
into the TAAF step_env hook decision chain.

This is the missing link: skills.py has 11 grandmaster skills but nothing calls
them. This module exposes evaluate_skills() which taaf_stepenv_hook invokes
every turn, returning prioritized proposals that compete with other tiers
(consensus, cortex, fusion, APE, breaker) under the same gate+veto law.

Design principles:
- Fail-open: any exception -> empty list (other tiers take over).
- Intrinsic only: no game-ID-keyed manuals, no memorization.
- Bounded: O(n) per turn, no threads, stdlib+numpy at import.
- Novelty gate: skills only fire on NEW states (sym_repeat>=2 suppressed).
- Veto: never overrides a legal+safe action with an illegal one.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__version__ = "v1-skill-orchestrator-1"

# Minimum confidence for a skill proposal to compete with other tiers.
_SKILL_GATE = 0.75

# Skills only fire on new states — repeated states are stagnation, not skill.
_MAX_STATE_REPEAT = 1

# Per-(game,level) repeat tracker: (game_id, level) -> {state_hash: count}
_STATE_SEEN: dict[tuple[str, int], dict[int, int]] = {}


def _state_hash(grid: np.ndarray) -> int:
    """Fast deterministic hash for novelty gating."""
    try:
        return hash(grid.tobytes())
    except Exception:
        return 0


def _is_novel(game_id: str, level: int, grid: np.ndarray) -> bool:
    """True when this state hasn't been seen before (or only once)."""
    try:
        key = (game_id, level)
        h = _state_hash(grid)
        seen = _STATE_SEEN.setdefault(key, {})
        count = seen.get(h, 0)
        if count >= _MAX_STATE_REPEAT:
            return False
        seen[h] = count + 1
        # Bounded: keep only last 64 states per (game, level)
        while len(seen) > 64:
            seen.pop(next(iter(seen)))
        return True
    except Exception:
        return True


def _skills_enabled() -> bool:
    """Kill-switch: ARC3_SKILLS=0 disables."""
    try:
        return os.environ.get("ARC3_SKILLS", "1") != "0"
    except Exception:
        return True


def reset_skill_state() -> None:
    """Clear per-(game,level) repeat tracker (call on game/level change)."""
    global _STATE_SEEN
    _STATE_SEEN = {}


def evaluate_skills(
    grid: np.ndarray,
    bg: int,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Evaluate all registered skills and return prioritized proposals.

    Returns list of (action_id, x, y, source, confidence) tuples sorted by
    confidence descending. Empty list on any failure (fail-open).
    """
    if not _skills_enabled():
        return []

    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return []

        # Novelty gate: only fire on new states
        if not _is_novel(game_id, level, g):
            return []

        proposals: list[tuple[int, int | None, int | None, str, float]] = []

        # 1) SovereignSkillsMatrix — 11 grandmaster skills
        try:
            from .skills import SovereignSkillsMatrix
            matrix_proposals = SovereignSkillsMatrix.evaluate_skills(g, bg, available)
            for act, x, y, src, conf in matrix_proposals:
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, f"skill_matrix_{src}", conf))
        except Exception:
            pass

        # 2) AbstractionSkillRegistry — archetype prior (advisory, lower conf)
        try:
            from .abstraction_skill_registry import AbstractionSkillRegistry
            # Registry is advisory only; it doesn't propose actions directly
            # but can boost confidence of matrix proposals that match archetype.
            # For now, we just ensure it's importable (fail-open).
            pass
        except Exception:
            pass

        # 3) ARC-specific skills: color transform, object tracker, pattern completion
        try:
            from .arc_color_transform import evaluate_color_transform
            color_proposals = evaluate_color_transform(g, bg, available)
            for act, x, y, src, conf in color_proposals:
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        try:
            from .arc_object_tracker import evaluate_object_tracker
            tracker_proposals = evaluate_object_tracker(
                g, bg, available, game_id, level)
            for act, x, y, src, conf in tracker_proposals:
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        try:
            from .arc_pattern_completion import evaluate_pattern_completion
            pattern_proposals = evaluate_pattern_completion(g, bg, available)
            for act, x, y, src, conf in pattern_proposals:
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        # Sort by confidence descending, deduplicate by (action, x, y)
        proposals.sort(key=lambda p: -p[4])
        seen: set[tuple[int, int | None, int | None]] = set()
        unique: list[tuple[int, int | None, int | None, str, float]] = []
        for p in proposals:
            key = (p[0], p[1], p[2])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique[:8]  # bounded: max 8 proposals per turn

    except Exception:
        return []


def evaluate_skills_safe(
    grid: np.ndarray,
    bg: int,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Fail-open wrapper — never raises, always returns a list."""
    try:
        return evaluate_skills(grid, bg, available, game_id, level)
    except Exception:
        return []


def get_skill_status() -> dict[str, Any]:
    """Bounded status dict for telemetry."""
    return {
        "enabled": _skills_enabled(),
        "gate": _SKILL_GATE,
        "max_repeat": _MAX_STATE_REPEAT,
        "states_tracked": len(_STATE_SEEN),
        "version": __version__,
    }
