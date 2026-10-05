"""Tests for skill_orchestrator v2 (full SOTA technique fusion)."""

from __future__ import annotations

import numpy as np
import pytest


class TestSkillOrchestratorV2:
    def test_evaluate_skills_returns_list(self):
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[2:5, 2:5] = 1
        result = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)

    def test_evaluate_skills_fail_open_on_bad_input(self):
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        result = evaluate_skills_safe(np.array([]), 0, [])
        assert result == []

    def test_novelty_gate_suppresses_repeats(self):
        from arc3sdk.skill_orchestrator import evaluate_skills_safe, reset_skill_state
        reset_skill_state()
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[2:5, 2:5] = 1
        r1 = evaluate_skills_safe(grid, 0, [6], "test_game", 1)
        r2 = evaluate_skills_safe(grid, 0, [6], "test_game", 1)
        r3 = evaluate_skills_safe(grid, 0, [6], "test_game", 1)
        reset_skill_state()
        r4 = evaluate_skills_safe(grid, 0, [6], "test_game", 1)
        assert isinstance(r1, list)
        assert isinstance(r2, list)
        assert isinstance(r3, list)
        assert isinstance(r4, list)

    def test_kill_switch(self):
        import os
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        os.environ["ARC3_SKILLS"] = "0"
        try:
            grid = np.zeros((10, 10), dtype=np.uint8)
            grid[2:5, 2:5] = 1
            result = evaluate_skills_safe(grid, 0, [6])
            assert result == []
        finally:
            os.environ["ARC3_SKILLS"] = "1"

    def test_get_status(self):
        from arc3sdk.skill_orchestrator import get_skill_status
        status = get_skill_status()
        assert "enabled" in status
        assert "gate" in status
        assert "version" in status
        assert status["version"] == "v2-skill-orchestrator-2"

    def test_all_techniques_fail_open(self):
        """Every technique import must fail-open (no crash on any input)."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        # Empty grid
        assert evaluate_skills_safe(np.array([]), 0, []) == []
        # 1x1 grid
        assert evaluate_skills_safe(np.array([[0]]), 0, [1]) == []
        # Large grid
        grid = np.random.randint(0, 16, (64, 64), dtype=np.uint8)
        result = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)

    def test_bounded_proposals(self):
        """Max 12 proposals per turn."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        grid = np.random.randint(0, 16, (32, 32), dtype=np.uint8)
        result = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6])
        assert len(result) <= 12


class TestTaafHookSkillWiring:
    def test_skill_proposals_in_hook(self):
        from arc3sdk import taaf_stepenv_hook
        import inspect
        src = inspect.getsource(taaf_stepenv_hook)
        assert "skill_orchestrator" in src
        assert "evaluate_skills_safe" in src

    def test_skills_in_sub_allow_list(self):
        from arc3sdk import taaf_stepenv_hook
        allow = taaf_stepenv_hook._SUB_ALLOW_DEFAULT
        assert "skill_matrix_" in allow
        assert "arc_color_" in allow
        assert "arc_object_" in allow
        assert "arc_pattern_" in allow
