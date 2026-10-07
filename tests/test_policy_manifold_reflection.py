"""Tests for PolicyManifoldReflectionEngine (H-PMR / Sovereign Agent-Pro Omega)."""
from __future__ import annotations

import pytest
from arc3sdk.policy_manifold_reflection import (
    PolicyManifoldReflectionEngine,
    BeliefManifold,
    PolicyInvariantRule,
)


def test_compute_belief_manifold():
    engine = PolicyManifoldReflectionEngine()
    grid = [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    belief = engine.compute_belief_manifold(grid, player_color=1, goal_colors=[2])
    assert isinstance(belief, BeliefManifold)
    assert belief.kinematic_player_color == 1
    assert belief.goal_sink_colors == (2,)
    assert belief.epistemic_confidence > 0.0
    assert belief.mdl_complexity > 0.0


def test_reflect_trajectory_success_distillation():
    engine = PolicyManifoldReflectionEngine()
    traj = [
        {"action": "UP", "grid": [[0, 1], [0, 0]]},
        {"action": "RIGHT", "grid": [[0, 0], [0, 1]]},
    ]
    rules = engine.reflect_trajectory(traj, final_score=100.0, success=True)
    assert len(rules) > 0
    assert any(r.category == "behavioral_guideline" for r in rules)


def test_reflect_trajectory_failure_pruning():
    engine = PolicyManifoldReflectionEngine()
    traj = [
        {"action": "DOWN", "grid": [[1, 0], [0, 0]]},
        {"action": "LEFT", "grid": [[0, 0], [0, 0]]},
    ]
    rules = engine.reflect_trajectory(traj, final_score=0.0, success=False)
    assert len(rules) > 0
    assert any(r.category == "world_model" for r in rules)


def test_hamiltonian_action_selection():
    engine = PolicyManifoldReflectionEngine()
    grid = [[0, 1], [0, 0]]
    belief = engine.compute_belief_manifold(grid)

    # Add positive rule for 'RIGHT'
    rule_pos = PolicyInvariantRule(
        rule_id="r1",
        category="behavioral_guideline",
        precondition_chi=belief.euler_characteristic,
        precondition_player_color=None,
        recommended_action="RIGHT",
        avoid_action=None,
        weight=2.0,
    )
    engine.behavioral_guidelines["r1"] = rule_pos

    # Add negative rule for 'LEFT'
    rule_neg = PolicyInvariantRule(
        rule_id="r2",
        category="world_model",
        precondition_chi=belief.euler_characteristic,
        precondition_player_color=None,
        recommended_action=None,
        avoid_action="LEFT",
        weight=-2.0,
    )
    engine.world_model_invariants["r2"] = rule_neg

    best = engine.select_optimal_policy_action(["LEFT", "UP", "RIGHT"], belief)
    assert best == "RIGHT"
