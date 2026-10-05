"""Integration test: Meta-Evolution + LLM feedback loop."""

from __future__ import annotations

import numpy as np
import pytest


class TestMetaEvolutionIntegration:
    def test_meta_evolution_in_orchestrator(self):
        """Orchestrator imports and uses meta_evolution."""
        from arc3sdk import skill_orchestrator
        import inspect
        src = inspect.getsource(skill_orchestrator)
        assert "meta_evolution" in src
        assert "apply_evolution" in src

    def test_meta_evolution_in_hook(self):
        """TAAF hook imports and uses meta_evolution."""
        from arc3sdk import taaf_stepenv_hook
        import inspect
        src = inspect.getsource(taaf_stepenv_hook)
        assert "meta_evolution" in src
        assert "observe_outcome" in src

    def test_full_feedback_loop(self):
        """Full loop: observe -> evolve -> apply."""
        from arc3sdk.meta_evolution import (
            observe_outcome, apply_evolution, get_evolution_summary, reset_evolution
        )
        reset_evolution()

        # Simulate 10 successful turns
        before = np.zeros((10, 10), dtype=np.uint8)
        for i in range(10):
            after = before.copy()
            after[i, i] = 1
            observe_outcome(before, after, 6, "test_tech", True)
            before = after

        # Apply evolution
        proposals = [(6, 5, 5, "test_tech", 0.8)]
        result = apply_evolution(proposals)
        assert result[0][4] > 0.8

        # Check summary
        summary = get_evolution_summary()
        assert summary["transitions"] == 10

    def test_llm_tier_coordination(self):
        """LLM and tier actions are both tracked."""
        from arc3sdk.meta_evolution import (
            evolve_from_llm_feedback, get_technique_weight, reset_evolution
        )
        reset_evolution()

        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        after[5, 5] = 1

        # LLM acts successfully
        evolve_from_llm_feedback(6, None, before, after)
        llm_weight = get_technique_weight("llm_analyzer")
        assert llm_weight > 1.0

        # Reset and let tier act successfully
        reset_evolution()
        evolve_from_llm_feedback(None, 6, before, after)
        tier_weight = get_technique_weight("tier_substitution")
        assert tier_weight > 1.0
