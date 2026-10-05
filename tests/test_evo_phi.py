"""Tests for Φ-EVO (evo_phi) — MetricLiveness + HiddenDimensionMiner + OperatorGenome."""

from __future__ import annotations

import numpy as np
import pytest


class TestMetricLiveness:
    def test_register_and_live_weights(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        evo_phi.observe_metrics({"m1": 1.0, "m2": 2.0}, 0.5)
        weights = evo_phi.live_weights()
        assert "m1" in weights and "m2" in weights

    def test_flat_metric_retired_after_patience(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        for _ in range(8):
            evo_phi.observe_metrics({"flat": 1.0, "moving": float(np.random.rand())}, 0.5)
        retired = evo_phi.retired_metrics()
        assert "flat" in retired
        assert "flat" not in evo_phi.live_weights()  # excluded from votes

    def test_retired_metric_revives_when_informative(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        for _ in range(8):
            evo_phi.observe_metrics({"flat": 1.0}, 0.5)
        assert "flat" in evo_phi.retired_metrics()
        # Now make it strongly informative AND correlated with quality
        for i in range(12):
            q = 0.2 + 0.06 * i
            evo_phi.observe_metrics({"flat": q * 10.0}, q)
        assert "flat" not in evo_phi.retired_metrics()
        assert "flat" in evo_phi.live_weights()

    def test_uncorrelated_metric_retires(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        # Moving but anti-correlated with quality -> dies by corr
        for i in range(16):
            evo_phi.observe_metrics({"noise": float(np.random.rand())}, 0.9)
        retired = evo_phi.retired_metrics()
        assert "noise" in retired or evo_phi.live_weights().get("noise", 1.0) < 4.0

    def test_absent_metrics_count_as_flat(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        evo_phi.observe_metrics({"vanishing": 5.0}, 0.5)
        for _ in range(8):
            evo_phi.observe_metrics({"other": float(np.random.rand())}, 0.5)
        assert "vanishing" in evo_phi.retired_metrics()


class TestHiddenDimensionMiner:
    def test_power_iteration_deterministic(self):
        """LESSON: constant init dies on symmetric matrices — seeded init must
        produce identical results across calls."""
        from arc3sdk.evo_phi import _power_iteration_svd
        rng = np.random.RandomState(7)
        m = rng.rand(12, 6)
        v1, s1 = _power_iteration_svd(m)
        v2, s2 = _power_iteration_svd(m)
        assert np.allclose(s1, s2)
        assert np.allclose(np.abs(v1), np.abs(v2))

    def test_symmetric_matrix_does_not_collapse(self):
        """A symmetric matrix must still yield nonzero singular values
        (constant/zero init would die here)."""
        from arc3sdk.evo_phi import _power_iteration_svd
        base = np.random.RandomState(3).rand(8, 4)
        sym = base @ base.T  # symmetric, positive semidefinite
        v, s = _power_iteration_svd(sym, k=2)
        assert float(np.max(s)) > 1e-6

    def test_mine_finds_factors_when_variance_exists(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        rng = np.random.RandomState(11)
        for i in range(12):
            evo_phi.mine_hidden_dimensions(
                {"a": float(rng.rand()), "b": float(rng.rand())}, 0.5)
        result = evo_phi.mine_hidden_dimensions(
            {"a": float(rng.rand()), "b": float(rng.rand())}, 0.5)
        assert result["factors"]
        assert 0.0 <= result["explained"] <= 1.0

    def test_mine_fail_open_on_constant_matrix(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        for _ in range(8):
            result = evo_phi.mine_hidden_dimensions({"a": 1.0, "b": 1.0}, 0.5)
        assert result["factors"] == []
        assert result["residual"] == 0.0


class TestOperatorGenome:
    def test_select_operator_explores_unpulled(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        seen = set()
        for _ in range(10):
            seen.add(evo_phi.select_operator())
        assert len(seen) >= 3  # exploration must spread

    def test_note_generation_records_and_bandit_updates(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        for i in range(8):
            out = evo_phi.note_generation(0.3 + 0.05 * (i % 4))
            assert "generation" in out
        s = evo_phi.status()
        assert any(b["pulls"] > 0 for b in s["bandit"].values())

    def test_operator_self_evolves_every_4_generations(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        gens = []
        for i in range(12):
            out = evo_phi.note_generation(float(np.random.rand()))
            if out.get("evolved"):
                gens.append(out["generation"])
        assert gens  # at least one self-evolution happened
        # Operator changes after enough bandit evidence
        assert evo_phi.operator_genome()["operator"] in evo_phi._OPERATORS

    def test_genome_bounds_hold(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        for _ in range(40):
            evo_phi.note_generation(float(np.random.rand()))
        g = evo_phi.operator_genome()
        assert 2 <= g["tournament_k"] <= 6
        assert 1 <= g["elite_n"] <= 4
        assert 2 <= g["distinct_n"] <= 8
        assert 0.05 <= g["mutation_rate"] <= 0.60

    def test_generation_cap(self):
        from arc3sdk import evo_phi
        evo_phi.reset()
        for _ in range(80):
            evo_phi.note_generation(0.5)
        assert evo_phi.status()["generation"] <= 40
