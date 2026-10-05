import numpy as np
from arc3sdk.cognitive_utility import CognitiveUtilityEngine

def test_cognitive_utility_fatal_veto():
    engine = CognitiveUtilityEngine()
    grid = np.zeros((10, 10), dtype=np.uint8)
    fatal_states = {("state_123", 1)}
    fatal_clicks = {(5, 5)}
    
    u_fatal_act = engine.compute_utility(
        action=1,
        grid=grid,
        prev_grid=None,
        fatal_state_actions=fatal_states,
        fatal_clicks=fatal_clicks,
        state_visit_counts={},
        state_sig="state_123"
    )
    assert u_fatal_act < -500.0, "Fatal state action must have massive negative utility"

    u_fatal_click = engine.compute_utility(
        action={"action": 6, "x": 5, "y": 5},
        grid=grid,
        prev_grid=None,
        fatal_state_actions=fatal_states,
        fatal_clicks=fatal_clicks,
        state_visit_counts={},
        state_sig="state_123"
    )
    assert u_fatal_click < -500.0, "Fatal click must have massive negative utility"

def test_cognitive_utility_geodesic_pull():
    engine = CognitiveUtilityEngine()
    grid = np.zeros((10, 10), dtype=np.uint8)
    player = [np.array([5, 5])]
    goal = [np.array([4, 5])] # Up
    
    # Action 1 is UP (-1, 0) -> gets closer
    u_up = engine.compute_utility(
        action=1,
        grid=grid,
        prev_grid=None,
        fatal_state_actions=set(),
        fatal_clicks=set(),
        state_visit_counts={},
        state_sig="state_001",
        goal_coords=goal,
        player_coords=player
    )
    # Action 2 is DOWN (+1, 0) -> moves away
    u_down = engine.compute_utility(
        action=2,
        grid=grid,
        prev_grid=None,
        fatal_state_actions=set(),
        fatal_clicks=set(),
        state_visit_counts={},
        state_sig="state_001",
        goal_coords=goal,
        player_coords=player
    )
    assert u_up > u_down, f"Up utility ({u_up}) must exceed Down utility ({u_down})"
