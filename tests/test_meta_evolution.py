"""Tests for Meta-Evolution — Self-improving system."""

from __future__ import annotations

import numpy as np
import pytest


class TestMetaEvolution:
    def test_observe_outcome_success(self):
        from arc3sdk.meta_evolution import observe_outcome, get_technique_weight, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        after[5, 5] = 1
        observe_outcome(before, after, 6, "test_tech", True)
        weight = get_technique_weight("test_tech")
        assert weight > 1.0

    def test_observe_outcome_failure(self):
        from arc3sdk.meta_evolution import observe_outcome, get_technique_weight, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        observe_outcome(before, after, 6, "test_tech", False)
        weight = get_technique_weight("test_tech")
        assert weight < 1.0

    def test_apply_evolution_boosts_successful(self):
        from arc3sdk.meta_evolution import observe_outcome, apply_evolution, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        after[5, 5] = 1
        observe_outcome(before, after, 6, "good_tech", True)
        proposals = [(6, 5, 5, "good_tech", 0.8)]
        result = apply_evolution(proposals)
        assert result[0][4] > 0.8

    def test_apply_evolution_suppresses_failed(self):
        from arc3sdk.meta_evolution import observe_outcome, apply_evolution, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        observe_outcome(before, after, 6, "bad_tech", False)
        proposals = [(6, 5, 5, "bad_tech", 0.8)]
        result = apply_evolution(proposals)
        assert result[0][4] < 0.8

    def test_get_evolution_summary(self):
        from arc3sdk.meta_evolution import get_evolution_summary, reset_evolution
        reset_evolution()
        summary = get_evolution_summary()
        assert "version" in summary
        assert "techniques" in summary
        assert "transitions" in summary

    def test_reset_evolution(self):
        from arc3sdk.meta_evolution import observe_outcome, get_evolution_summary, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        after[5, 5] = 1
        observe_outcome(before, after, 6, "test_tech", True)
        reset_evolution()
        summary = get_evolution_summary()
        assert summary["transitions"] == 0

    def test_evolve_from_llm_feedback(self):
        from arc3sdk.meta_evolution import evolve_from_llm_feedback, get_technique_weight, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        after[5, 5] = 1
        evolve_from_llm_feedback(6, None, before, after)
        weight = get_technique_weight("llm_analyzer")
        assert weight > 1.0

    def test_weight_bounds(self):
        from arc3sdk.meta_evolution import observe_outcome, get_technique_weight, reset_evolution
        reset_evolution()
        before = np.zeros((10, 10), dtype=np.uint8)
        after = before.copy()
        # Many successes — weight should cap at 2.0
        for _ in range(100):
            observe_outcome(before, after, 6, "test_tech", True)
        weight = get_technique_weight("test_tech")
        assert weight <= 2.0
        # Many failures — weight should floor at 0.1
        reset_evolution()
        for _ in range(100):
            observe_outcome(before, after, 6, "test_tech", False)
        weight = get_technique_weight("test_tech")
        assert weight >= 0.1
