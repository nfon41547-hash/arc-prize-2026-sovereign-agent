"""Automated Topological Grounded Interface Adapter (AutoInterface-Omega / ALIGN-ARC).

Transcendence over Baseline ALIGN (Liu et al., Tsinghua / AIR 2025):
Automates agent-environment alignment specifically for ARC-AGI-3 2D combinatorial grid worlds.
Provides static invariant extraction (INFER_ARC_INVARIANTS) and dynamic zero-latency
observation enhancement (WRAP_ARC_STEP), annihilating consecutive invalid actions,
collision blindness, and state-revisitation oscillations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


@dataclass
class StaticEnvironmentRules:
    """Static domain invariants inferred at level initialization."""
    grid_height: int
    grid_width: int
    active_colors: Set[int]
    foreground_colors: Set[int]
    background_color: int
    has_dihedral_symmetry: bool
    description: str


@dataclass
class AugmentedStepFeedback:
    """Dynamic observation enriched by WRAP_ARC_STEP."""
    grid: List[List[int]]
    raw_reward: float
    done: bool
    state_changed: bool
    is_collision: bool
    is_oscillation: bool
    player_coord: Optional[Tuple[int, int]]
    diagnostic_hint: str
    suggested_fallbacks: List[str]


class AutoAlignedInterfaceEngine:
    """Sovereign Auto-Aligned Interface for ARC-AGI-3 Environments."""

    def __init__(self, history_window: int = 8):
        self.history_window = history_window
        self.state_history: List[List[List[int]]] = []
        self.action_history: List[str] = []
        self.static_rules: Optional[StaticEnvironmentRules] = None
        self.kinematic_player_color: Optional[int] = None
        self.kinematic_player_coord: Optional[Tuple[int, int]] = None

    def infer_rules(self, initial_grid: List[List[int]]) -> StaticEnvironmentRules:
        """INFER_ARC_INVARIANTS: Static extraction of 2D grid rules and palette."""
        if not initial_grid or not initial_grid[0]:
            rules = StaticEnvironmentRules(0, 0, set(), set(), 0, False, "Empty Grid")
            self.static_rules = rules
            return rules

        h = len(initial_grid)
        w = len(initial_grid[0])
        color_counts: Dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                val = initial_grid[r][c]
                color_counts[val] = color_counts.get(val, 0) + 1

        # Background is typically the most frequent color (often 0)
        bg = max(color_counts.items(), key=lambda x: x[1])[0]
        active = set(color_counts.keys())
        fg = {c for c in active if c != bg}

        # Check horizontal / vertical symmetry
        h_sym = all(initial_grid[r][c] == initial_grid[r][w - 1 - c] for r in range(h) for c in range(w // 2))
        v_sym = all(initial_grid[r][c] == initial_grid[h - 1 - r][c] for r in range(h // 2) for c in range(w))

        desc = (
            f"Grid {h}x{w} | Background={bg} | Foreground Colors={sorted(list(fg))} | "
            f"Symmetry: H={h_sym}, V={v_sym}"
        )

        rules = StaticEnvironmentRules(
            grid_height=h,
            grid_width=w,
            active_colors=active,
            foreground_colors=fg,
            background_color=bg,
            has_dihedral_symmetry=(h_sym or v_sym),
            description=desc,
        )
        self.static_rules = rules
        self.state_history.clear()
        self.action_history.clear()
        self.state_history.append(initial_grid)
        return rules

    def wrap_step(
        self,
        prev_grid: List[List[int]],
        action: str,
        next_grid: List[List[int]],
        reward: float = 0.0,
        done: bool = False,
    ) -> AugmentedStepFeedback:
        """WRAP_ARC_STEP: Intercept raw observation and synthesize enriched diagnostics."""
        state_changed = (prev_grid != next_grid)
        is_collision = not state_changed and action in {"UP", "DOWN", "LEFT", "RIGHT", "1", "2", "3", "4"}

        # Detect cycle / oscillation: S_{t+1} == S_{t-1}
        is_oscillation = False
        if len(self.state_history) >= 2:
            if next_grid == self.state_history[-2]:
                is_oscillation = True

        # Kinematic coordinate discovery
        player_coord = None
        if state_changed and self.static_rules:
            # Find diff pixels between prev and next
            diffs = []
            h = len(prev_grid)
            w = len(prev_grid[0]) if h > 0 else 0
            for r in range(h):
                for c in range(w):
                    if prev_grid[r][c] != next_grid[r][c]:
                        diffs.append((r, c, prev_grid[r][c], next_grid[r][c]))

            if len(diffs) in {2, 4}:  # typical unit translation
                for r, c, p_val, n_val in diffs:
                    if n_val in self.static_rules.foreground_colors:
                        self.kinematic_player_color = n_val
                        self.kinematic_player_coord = (r, c)
                        player_coord = (r, c)
                        break

        # Synthesize clear diagnostic hint
        diagnostic = ""
        fallbacks: List[str] = []

        if is_collision:
            diagnostic = f"INVALID MOVE: Action '{action}' caused no state change (hit wall or impassable boundary)."
            # Suggest orthogonal directions
            if action in {"UP", "DOWN", "1", "2"}:
                fallbacks = ["LEFT", "RIGHT", "3", "4"]
            else:
                fallbacks = ["UP", "DOWN", "1", "2"]
        elif is_oscillation:
            diagnostic = f"LOOP WARNING: Action '{action}' returned to a previously visited state S_(t-1). Change direction."
            fallbacks = ["UP", "DOWN", "LEFT", "RIGHT"]
        elif state_changed:
            diagnostic = f"VALID: Action '{action}' progressed state successfully."
        else:
            diagnostic = f"NOOP: Action '{action}' produced identical state."

        # Maintain histories
        self.state_history.append(next_grid)
        self.action_history.append(action)
        if len(self.state_history) > self.history_window:
            self.state_history.pop(0)
            self.action_history.pop(0)

        return AugmentedStepFeedback(
            grid=next_grid,
            raw_reward=reward,
            done=done,
            state_changed=state_changed,
            is_collision=is_collision,
            is_oscillation=is_oscillation,
            player_coord=player_coord or self.kinematic_player_coord,
            diagnostic_hint=diagnostic,
            suggested_fallbacks=fallbacks,
        )
