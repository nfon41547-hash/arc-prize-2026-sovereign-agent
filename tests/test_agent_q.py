"""Unit tests for Agent Q Guided MCTS, Self-Critique Process Supervision, and DPO Preference Pair Extraction."""

import pytest
import numpy as np
from arc3sdk.agent_q_mcts import AgentQNode, AgentQEngine, DPOPreferencePair


def test_agent_q_node_initialization_and_ucb1():
    grid = np.zeros((10, 10), dtype=np.int32)
    node = AgentQNode(state_grid=grid, history=(), depth=0)
    assert node.visits == 0
    assert node.get_blended_q(alpha=0.5) == 0.0

    child = AgentQNode(state_grid=grid, history=(1,), parent=node, action_taken=1, depth=1)
    child.q_critique = 0.8
    child.q_rollout = 0.4
    child.visits = 2
    node.visits = 5

    blended = child.get_blended_q(alpha=0.5)
    assert pytest.approx(blended, 0.01) == 0.6
    ucb1 = child.ucb1_score(c_puct=1.414, alpha=0.5)
    assert ucb1 > blended


def test_agent_q_search_and_plan_with_preference_extraction():
    engine = AgentQEngine(alpha=0.6, num_simulations=20, preference_margin_threshold=0.2)

    root_grid = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 2]
    ], dtype=np.int32)

    # Simple deterministic step function
    def step_fn(grid: np.ndarray, action: int):
        next_g = grid.copy()
        if action == 1: # Move right (towards goal 2)
            next_g[0, 2] = 1
            return next_g, 0.9, True, False
        elif action == 2: # Move left (away)
            next_g[0, 0] = 1
            return next_g, -0.5, False, False
        elif action == 3: # Fatal move
            return next_g, -1.0, False, True
        return next_g, 0.0, False, False

    best_act, conf, traj, pairs = engine.search_and_plan(
        root_grid=root_grid,
        candidate_actions=[1, 2, 3],
        step_fn=step_fn
    )

    assert best_act == 1
    assert conf >= 0.8
    assert traj == [1]


def test_agent_q_dpo_loss_computation():
    engine = AgentQEngine(beta_dpo=0.1)

    # When policy assigns higher probability to winner than reference relative to loser
    loss_favorable = engine.compute_dpo_loss(
        pi_winner_logprob=-0.2,
        pi_loser_logprob=-2.0,
        ref_winner_logprob=-1.0,
        ref_loser_logprob=-1.0
    )

    # When policy assigns higher probability to loser
    loss_unfavorable = engine.compute_dpo_loss(
        pi_winner_logprob=-2.0,
        pi_loser_logprob=-0.2,
        ref_winner_logprob=-1.0,
        ref_loser_logprob=-1.0
    )

    assert loss_favorable < loss_unfavorable
    assert loss_favorable > 0.0
