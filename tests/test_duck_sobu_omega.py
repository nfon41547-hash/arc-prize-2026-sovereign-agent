import numpy as np
from arc3sdk.duck_sobu_omega import DuckSobuOmegaController

def test_duck_sobu_omega_modes():
    ctrl = DuckSobuOmegaController()
    grid = np.zeros((10, 10), dtype=np.uint8)
    game_id = "test_game"
    
    # 1. Initially in EXPLORE mode
    mem = ctrl.get_memory(game_id, 1)
    assert mem.mode == "EXPLORE", "Initial mode must be EXPLORE"
    
    # 2. Evaluate novel action
    u_novel = ctrl.evaluate_utility(game_id, 1, grid, 1)
    assert u_novel > 0.0, "Novel action in EXPLORE mode must have positive utility"
    
    # 3. Observe no-op transition -> failed action recorded
    ctrl.observe_transition(game_id, 1, grid, grid, 1, score_delta=0.0, level_completed=False, game_over=False)
    assert len(mem.failed_actions) == 1, "No-op transition must be recorded in failed_actions"
    
    # 4. Action 1 should now be hard-vetoed
    u_failed = ctrl.evaluate_utility(game_id, 1, grid, 1)
    assert u_failed < -500.0, "Failed action must be hard-negative vetoed"
    
    # 5. Observe score progress transition -> transitions to COMMIT mode
    next_grid = np.ones((10, 10), dtype=np.uint8)
    ctrl.observe_transition(game_id, 1, grid, next_grid, 2, score_delta=1.0, level_completed=True, game_over=False)
    assert mem.mode == "COMMIT", "Score progress must transition mode to COMMIT"
    
    # 6. Check summary formatting for prompt
    summary = mem.format_summary_for_prompt()
    print("M_t Summary:\n", summary)
    assert "[WORLD MEMORY M_t | Mode: COMMIT]" in summary

def test_install_tool_agent_sobu_hook():
    from arc3sdk.duck_sobu_omega import install_tool_agent_sobu_hook, get_sobu_omega
    
    class MockToolAgent:
        def __init__(self):
            self._session_runtime_dir = "test_game_01"
            self._last_step_summary = {"level": 2}
        
        def _summarized_knowledge_lines(self):
            return ["- World model: active", "- Goal model: target"]
            
    res = install_tool_agent_sobu_hook(MockToolAgent)
    assert res["installed"] is True
    
    agent = MockToolAgent()
    # Populate memory
    mem = get_sobu_omega().get_memory("test_game_01", 2)
    mem.rules.append("Rule: 1 leads to portal")
    
    lines = agent._summarized_knowledge_lines()
    assert any("[WORLD MEMORY M_t" in l for l in lines)
    assert any("Rule: 1 leads to portal" in l for l in lines)

