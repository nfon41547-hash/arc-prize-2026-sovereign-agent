import numpy as np
from arc3sdk.sobu_v6 import Controller

def test_v6_falsification_checks():
    c = Controller()
    grid = np.zeros((10, 10), dtype=np.uint8)
    
    # 1. Novel action check (fail-open)
    res_novel = c.assess("game_test", 1, grid, 1, reason="test")
    u_novel = res_novel["utility"]
    print("U_novel:", u_novel)
    assert not res_novel["hard_negative"], "Novel action must NOT be hard-negative"
    assert abs(u_novel - 0.1458) < 0.05, f"Expected U_novel ~0.1458, got {u_novel}"

    # 2. Simulate 2 no-ops on exact state-action
    c.observe("game_test", 1, grid, grid, 1, reason="test")
    c.observe("game_test", 1, grid, grid, 1, reason="test")
    
    res_noop = c.assess("game_test", 1, grid, 1, reason="test")
    u_noop = res_noop["utility"]
    print("U_noop after 2 no-ops:", u_noop)
    assert res_noop["hard_negative"], "Action after 2 no-ops MUST be hard-negative"
    assert u_noop < 0.0, f"Expected negative utility, got {u_noop}"

def test_v6_arbitration_compare():
    c = Controller()
    grid = np.zeros((10, 10), dtype=np.uint8)
    
    # Observe good transition for tier action 2
    next_grid = np.ones((10, 10), dtype=np.uint8)
    c.observe("game_test2", 1, grid, next_grid, 2, reason="tier", score_delta=1.0)
    c.observe("game_test2", 1, grid, next_grid, 2, reason="tier", score_delta=1.0)
    
    # Observe noop for analyzer action 1
    c.observe("game_test2", 1, grid, grid, 1, reason="analyzer")
    c.observe("game_test2", 1, grid, grid, 1, reason="analyzer")
    
    # Compare
    cmp_res = c.compare("game_test2", 1, grid, (1, None, None), (2, None, None), "tier", 0.95)
    print("Compare result:", cmp_res)
    assert cmp_res["allow_tier"], "Tier with demonstrated score gain must be allowed over noop analyzer"
