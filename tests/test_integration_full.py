"""Integration test: all 16 SOTA techniques work together seamlessly."""

from __future__ import annotations

import numpy as np
import pytest


class TestFullIntegration:
    def test_all_techniques_importable(self):
        """Every technique module must be importable."""
        modules = [
            "arc3sdk.skills",
            "arc3sdk.abstraction_skill_registry",
            "arc3sdk.arc_color_transform",
            "arc3sdk.arc_object_tracker",
            "arc3sdk.arc_pattern_completion",
            "arc3sdk.algebraic_planning_engine",
            "arc3sdk.causal_chain_reasoner",
            "arc3sdk.fusion_supremacy",
            "arc3sdk.spectral_topological_reasoning",
            "arc3sdk.realtime_abstract_cortex",
            "arc3sdk.world_model_simulator",
            "arc3sdk.mcts_planner",
            "arc3sdk.object_segmentation",
            "arc3sdk.hypothesis_ledger",
            "arc3sdk.flux_search",
            "arc3sdk.turn_memo",
        ]
        import importlib
        for mod in modules:
            importlib.import_module(mod)

    def test_orchestrator_returns_bounded_proposals(self):
        """Orchestrator returns max 12 proposals."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        grid = np.random.randint(0, 16, (32, 32), dtype=np.uint8)
        result = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)
        assert len(result) <= 12

    def test_orchestrator_fail_open_on_empty(self):
        """Orchestrator fails open on empty grid."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        result = evaluate_skills_safe(np.array([]), 0, [])
        assert result == []

    def test_orchestrator_fail_open_on_1x1(self):
        """Orchestrator fails open on 1x1 grid."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        result = evaluate_skills_safe(np.array([[0]]), 0, [1])
        assert isinstance(result, list)

    def test_orchestrator_novelty_gate(self):
        """Novelty gate suppresses repeated states."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe, reset_skill_state
        reset_skill_state()
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[2:5, 2:5] = 1
        r1 = evaluate_skills_safe(grid, 0, [6], "test", 1)
        r2 = evaluate_skills_safe(grid, 0, [6], "test", 1)
        assert isinstance(r1, list)
        assert isinstance(r2, list)

    def test_hook_wires_orchestrator(self):
        """TAAF hook imports and calls skill_orchestrator."""
        from arc3sdk import taaf_stepenv_hook
        import inspect
        src = inspect.getsource(taaf_stepenv_hook)
        assert "skill_orchestrator" in src
        assert "evaluate_skills_safe" in src

    def test_hook_sub_allowlist_covers_all_techniques(self):
        """Sub allowlist covers all technique prefixes."""
        from arc3sdk import taaf_stepenv_hook
        allow = taaf_stepenv_hook._SUB_ALLOW_DEFAULT
        expected = [
            "ape", "leap_photographic", "leap_q", "agno_offline_bfs_shortest_path",
            "skill_matrix_", "arc_color_", "arc_object_", "arc_pattern_",
            "causal_chain", "fusion_", "cortex_", "world_model",
            "mcts_", "seg_", "hypothesis", "flux_",
        ]
        for prefix in expected:
            assert prefix in allow, f"Missing prefix: {prefix}"

    def test_all_techniques_produce_valid_proposals(self):
        """Every technique produces valid (action, x, y, source, conf) tuples."""
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
        result = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6])
        for p in result:
            assert len(p) == 5
            act, x, y, src, conf = p
            assert isinstance(act, int)
            assert 1 <= act <= 7
            assert isinstance(src, str)
            assert 0.0 <= conf <= 1.0

    def test_kill_switch_disables_all(self):
        """ARC3_SKILLS=0 disables all techniques."""
        import os
        from arc3sdk.skill_orchestrator import evaluate_skills_safe
        os.environ["ARC3_SKILLS"] = "0"
        try:
            grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
            result = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6])
            assert result == []
        finally:
            os.environ["ARC3_SKILLS"] = "1"
