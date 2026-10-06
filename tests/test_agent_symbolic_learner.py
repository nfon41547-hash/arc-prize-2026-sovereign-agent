import pytest
from arc3sdk.agent_symbolic_learner import AgentSymbolicLearner, get_symbolic_learner

def test_symbolic_learner_initialization():
    learner = get_symbolic_learner()
    assert isinstance(learner, AgentSymbolicLearner)
    assert learner.learning_rate == 0.5
    assert "prompts" in learner.symbolic_weights

def test_forward_and_loss_computation():
    learner = AgentSymbolicLearner()
    learner.record_forward_step("perception", {"grid": [[1, 0]]}, {"objects": ["obj_0"]})
    learner.record_forward_step("action_selection", {"objects": ["obj_0"]}, {"action": "ACTION1"})
    
    assert len(learner.trajectory) == 2
    loss_score, critique = learner.compute_language_loss("reach_exit", "died", "won", False)
    assert loss_score == 1.0
    assert "Task failure detected" in critique

def test_backprop_and_weight_update():
    learner = AgentSymbolicLearner()
    learner.record_forward_step("node_1", "in_1", "out_1")
    learner.record_forward_step("node_2", "in_2", "out_2")
    
    loss_score, critique = learner.compute_language_loss("goal", "fail", "win", False)
    gradients = learner.backpropagate_language_gradients(critique)
    
    assert len(gradients) == 2
    assert "Node [node_2]" in gradients[1]
    
    updates = learner.update_symbolic_weights(gradients)
    assert "node_1" in updates
    assert "node_2" in updates
    assert len(learner.trajectory) == 0
