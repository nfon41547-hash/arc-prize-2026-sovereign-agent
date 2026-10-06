"""Cognitive Action Utility Evaluator for ARC-AGI-3 (v1.0.0)

Implements the fundamental Action Utility Functional:
    U_i(E) = Delta_I_i^{(v)}(E) + sum_{j in E} S_ij^+
             - (lambda_R * R_i(E) + lambda_C * C_i + lambda_L * L_i + lambda_K * K_i + lambda_F * F_i)

Parameters:
- Delta_I_i^{(v)}: Epistemic Information Gain (novel grid entropy / unexplored frontier)
- sum S_ij^+: Synergy pull toward verified sub-goals (e.g. key -> door path alignment)
- R_i: Repetition / loop penalty (penalizes cyclic re-visitation)
- C_i: Cognitive / Action cost (1 step consumed in budget)
- L_i: Loss of critical state (risk of fatal transition / trap)
- K_i: Computational complexity cost
- F_i: Friction / boundary collision penalty (no-op wall collisions)
"""

from __future__ import annotations
import numpy as np
from typing import Any

class CognitiveUtilityEngine:
    def __init__(
        self,
        lambda_r: float = 1.8,   # Repetition penalty
        lambda_c: float = 0.1,   # Step consumption penalty
        lambda_l: float = 5.0,   # Fatal state penalty
        lambda_k: float = 0.05,  # Computation penalty
        lambda_f: float = 1.2,   # Wall bounce / No-op penalty
        tau: float = 0.45,       # Minimal utility threshold
    ):
        self.lambda_r = lambda_r
        self.lambda_c = lambda_c
        self.lambda_l = lambda_l
        self.lambda_k = lambda_k
        self.lambda_f = lambda_f
        self.tau = tau

    def compute_utility(
        self,
        action: Any,
        grid: np.ndarray,
        prev_grid: np.ndarray | None,
        fatal_state_actions: set[tuple[str, int]],
        fatal_clicks: set[tuple[int, int]],
        state_visit_counts: dict[str, int],
        state_sig: str,
        goal_coords: list[np.ndarray] | None = None,
        player_coords: list[np.ndarray] | None = None,
    ) -> float:
        """Evaluates U_i(E) for a candidate action."""
        act_id = action.get("action", 6) if isinstance(action, dict) else int(action)
        
        # 1. Fatal Risk Penalty L_i
        is_fatal = False
        if isinstance(action, int) and (state_sig, action) in fatal_state_actions:
            is_fatal = True
        elif isinstance(action, dict) and act_id == 6:
            cx = action.get("x", action.get("col"))
            cy = action.get("y", action.get("row"))
            if cx is not None and cy is not None and (int(cx), int(cy)) in fatal_clicks:
                is_fatal = True
                
        if is_fatal:
            return -999.0  # Immediate hard veto
            
        # 2. Epistemic Information Gain Delta_I
        # Prioritize actions that break symmetry or explore unvisited states
        visits = state_visit_counts.get(state_sig, 0)
        delta_i = 1.0 / (1.0 + np.log1p(visits))
        
        # 3. Synergy S_ij^+ toward Goal
        synergy = 0.0
        if player_coords is not None and len(player_coords) > 0 and goal_coords is not None and len(goal_coords) > 0:
            p_pos = player_coords[0]
            g_pos = goal_coords[0]
            curr_dist = float(np.sum(np.abs(p_pos - g_pos)))
            
            # Predict delta position
            delta_map = {1: np.array([-1, 0]), 2: np.array([1, 0]), 3: np.array([0, -1]), 4: np.array([0, 1])}
            if act_id in delta_map:
                new_pos = p_pos + delta_map[act_id]
                new_dist = float(np.sum(np.abs(new_pos - g_pos)))
                if new_dist < curr_dist:
                    synergy += 0.8  # Strong geodesic kinetic pull
                elif new_dist > curr_dist:
                    synergy -= 0.4
                    
        # 4. Repetition Penalty R_i
        r_i = float(visits) * 0.25
        
        # 5. Friction / No-op Penalty F_i
        f_i = 0.0
        if prev_grid is not None and np.array_equal(grid, prev_grid):
            # Last action caused a no-op; penalize repeating it
            f_i += 1.0
            
        # 6. Cognitive Step Cost C_i
        c_i = 1.0
        k_i = 0.1
        
        utility = delta_i + synergy - (
            self.lambda_r * r_i +
            self.lambda_c * c_i +
            self.lambda_k * k_i +
            self.lambda_f * f_i
        )
        return float(utility)

    def guard_cost_tradeoff(
        self,
        is_noop: bool,
        is_fatal: bool,
        retry_latency_s: float = 1.8,
        action_penalty_s: float = 0.5,
    ) -> tuple[bool, float]:
        """Calculates J_guard(a) = I[a in H_noop(s)] * (Delta_Latency_retry - Penalty_wasted_action).
        
        Returns (should_veto, cost_score).
        - Fatal state: unconditional hard veto (True, -999.0)
        - No-op: arbitrates whether to trigger LLM retry vs accepting next-frame feedback.
        """
        if is_fatal:
            return True, -999.0
        if not is_noop:
            return False, 0.0
        j_guard = retry_latency_s - action_penalty_s
        should_veto = (j_guard < 0.0)
        return should_veto, float(j_guard)

_GLOBAL_UTILITY = CognitiveUtilityEngine()

def get_cognitive_utility_engine() -> CognitiveUtilityEngine:
    return _GLOBAL_UTILITY
