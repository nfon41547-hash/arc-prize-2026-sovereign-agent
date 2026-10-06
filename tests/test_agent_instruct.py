import pytest
import numpy as np
from arc3sdk.agent_instruct_reasoner import AgentInstructReasoner, get_agent_instruct

def test_agent_instruct_initialization():
    reasoner = get_agent_instruct()
    assert isinstance(reasoner, AgentInstructReasoner)
    assert reasoner.max_subgoals == 5

def test_generate_meta_instructions_grid():
    reasoner = AgentInstructReasoner()
    grid = np.zeros((10, 10), dtype=np.uint8)
    instructions = reasoner.generate_meta_instructions(
        task_type="grid_transformation",
        grid_state=grid,
        context_hints=["Preserve blue border"],
    )
    assert len(instructions) >= 5
    assert any("Phase 1" in s for s in instructions)
    assert any("Preserve blue border" in s for s in instructions)

def test_generate_meta_instructions_interactive():
    reasoner = AgentInstructReasoner()
    grid = np.zeros((10, 10), dtype=np.uint8)
    instructions = reasoner.generate_meta_instructions(
        task_type="interactive_game",
        grid_state=grid,
    )
    assert any("Counterfactual Action Pivoting" in s for s in instructions)

def test_steer_reasoning_step_stagnation():
    reasoner = AgentInstructReasoner()
    feedback = {
        "stagnant": True,
        "fallback_actions": [{"action": "ACTION2"}],
    }
    signal = reasoner.steer_reasoning_step(
        current_reasoning="Moving right again",
        predicted_action_or_grid={"action": "ACTION1"},
        grounding_feedback=feedback,
    )
    assert signal["valid"] is False
    assert signal["pivoted_action"] == {"action": "ACTION2"}
    assert "Stagnation loop detected" in signal["correction_directive"]

def test_format_agent_instruct_prompt():
    reasoner = AgentInstructReasoner()
    grid = np.ones((5, 5), dtype=np.uint8)
    prompt = reasoner.format_agent_instruct_prompt("task_001", "grid_transformation", grid)
    assert "AGENT-INSTRUCT: ZERO-SHOT REASONING SUPERVISOR" in prompt
    assert "Task: task_001" in prompt
