"""Tests for AutoAlignedInterfaceEngine (ALIGN-ARC)."""
from __future__ import annotations

import pytest
from arc3sdk.auto_aligned_interface import (
    AutoAlignedInterfaceEngine,
    StaticEnvironmentRules,
    AugmentedStepFeedback,
)


def test_infer_rules():
    engine = AutoAlignedInterfaceEngine()
    grid = [
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    rules = engine.infer_rules(grid)
    assert isinstance(rules, StaticEnvironmentRules)
    assert rules.grid_height == 3
    assert rules.grid_width == 4
    assert rules.background_color == 0
    assert 1 in rules.foreground_colors
    assert rules.has_dihedral_symmetry is True


def test_wrap_step_collision_detection():
    engine = AutoAlignedInterfaceEngine()
    grid = [[0, 1], [0, 0]]
    engine.infer_rules(grid)

    feedback = engine.wrap_step(grid, "UP", grid)
    assert feedback.state_changed is False
    assert feedback.is_collision is True
    assert "INVALID MOVE" in feedback.diagnostic_hint
    assert len(feedback.suggested_fallbacks) > 0


def test_wrap_step_oscillation_detection():
    engine = AutoAlignedInterfaceEngine()
    g1 = [[1, 0], [0, 0]]
    g2 = [[0, 1], [0, 0]]
    engine.infer_rules(g1)

    engine.wrap_step(g1, "RIGHT", g2)
    # Move back to g1
    feedback = engine.wrap_step(g2, "LEFT", g1)
    assert feedback.is_oscillation is True
    assert "LOOP WARNING" in feedback.diagnostic_hint


def test_wrap_step_valid_move():
    engine = AutoAlignedInterfaceEngine()
    g1 = [[1, 0], [0, 0]]
    g2 = [[0, 1], [0, 0]]
    engine.infer_rules(g1)

    feedback = engine.wrap_step(g1, "RIGHT", g2)
    assert feedback.state_changed is True
    assert feedback.is_collision is False
    assert feedback.is_oscillation is False
    assert "VALID" in feedback.diagnostic_hint
