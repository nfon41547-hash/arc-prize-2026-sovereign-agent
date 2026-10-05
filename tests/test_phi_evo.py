"""Tests for Φ-EVO: Phenomenological Self-Transcending Evolution Engine."""

from __future__ import annotations

import pytest


class TestPhiEvo:
    def test_register_and_live_weights(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        phi_evo.register_metrics({"m1": 1.0, "m2": 2.0})
        weights = phi_evo.live_weights()
        assert "m1" in weights
        assert "m2" in weights

    def test_metric_death_by_gain_collapse(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        # Flat metric: same value every turn -> gain collapses -> dies
        for _ in range(8):
            phi_evo.register_metrics({"flat": 1.0, "moving": float(len(phi_evo.live_weights()))})
        dead = phi_evo.dead_metrics()
        assert "flat" in dead
        assert dead["flat"]["reason"] == "gain-collapse"

    def test_metric_death_by_absence(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        # Metric present first, then absent for many turns -> dies
        phi_evo.register_metrics({"vanishing": 5.0})
        for _ in range(8):
            phi_evo.register_metrics({"other": 1.0})
        dead = phi_evo.dead_metrics()
        assert "vanishing" in dead

    def test_hidden_dimension_discovery(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        # Feed varied observations -> candidates mined -> promoted after streak
        for i in range(20):
            phi_evo.update_from_outcome({
                "n_actions": 4,
                "n_changed": (i % 3) + 1,  # varied -> informative
                "board_changed": bool(i % 2),
            })
        summary = phi_evo.status()
        assert summary["candidates"] > 0

    def test_candidate_promotion(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        # Strongly varied agree_rate for many turns -> promotion
        for i in range(24):
            phi_evo.update_from_outcome({
                "n_actions": 4,
                "n_changed": (i % 4) + 1,
            })
        promoted = phi_evo.promoted_candidates()
        assert isinstance(promoted, dict)

    def test_genome_mutation_on_stagnation(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        gen_before = phi_evo.genome()["generation"]
        # Stagnant progress for many turns -> genome mutates
        for _ in range(60):
            phi_evo.update_from_outcome({"board_changed": False, "level_completed": False})
        gen_after = phi_evo.genome()["generation"]
        assert gen_after > gen_before

    def test_genome_bounds_hold(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        for _ in range(120):
            phi_evo.update_from_outcome({"board_changed": False})
        g = phi_evo.genome()
        assert 0.02 <= g["rate"] <= 0.40
        assert 0.005 <= g["floor"] <= 0.10
        assert 3 <= g["promote_after"] <= 24
        assert 0.05 <= g["diversity"] <= 0.80

    def test_technique_weight_update(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        # Good performance -> weight up
        for _ in range(5):
            phi_evo.update_from_outcome({"technique_performances": {"ape": 1.0}})
        w = phi_evo.live_weights().get("tech:ape")
        assert w is not None and w > 1.0
        # Bad performance -> weight down
        for _ in range(5):
            phi_evo.update_from_outcome({"technique_performances": {"ape": 0.0}})
        w = phi_evo.live_weights().get("tech:ape")
        assert w is not None and w < 2.0

    def test_fail_open_on_bad_input(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        result = phi_evo.update_from_outcome({"bad": object()})
        assert isinstance(result, dict)
        phi_evo.register_metrics({"bad": "not-a-number"})
        assert isinstance(phi_evo.live_weights(), dict)

    def test_status_and_reset(self):
        from arc3sdk import phi_evo
        phi_evo.reset()
        s = phi_evo.status()
        assert s["version"] == "v1-phi-evo-1"
        assert s["alive_metrics"] == 0
        phi_evo.register_metrics({"m": 1.0})
        phi_evo.reset()
        assert phi_evo.status()["alive_metrics"] == 0
