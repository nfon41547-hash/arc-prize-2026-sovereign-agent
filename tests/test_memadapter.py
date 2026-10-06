import numpy as np
import pytest
from arc3sdk.memadapter_engine import (
    MemoryRecord,
    EvidenceObservation,
    CounterfactualInduction,
    ContextAwareReflection,
    EvidenceGroundedReasoning,
    MemAdapterController,
)


def test_counterfactual_risk_evaluation():
    ci = CounterfactualInduction()
    mem_invariant = MemoryRecord(
        id="mem_1",
        rule_type="structural_invariant",
        source_level=1,
        action_sequence=[1, 2],
        expected_delta_state={"grid_diff": True},
    )
    risk_inv = ci.evaluate_risk(mem_invariant, current_level=3, current_evidence=None)
    assert risk_inv <= 0.2

    mem_dynamic = MemoryRecord(
        id="mem_2",
        rule_type="local_dynamic",
        source_level=1,
        action_sequence=[3],
        expected_delta_state={"door_open": True},
    )
    risk_dyn = ci.evaluate_risk(mem_dynamic, current_level=4, current_evidence=None)
    assert risk_dyn >= 0.4


def test_context_aware_reflection():
    car = ContextAwareReflection(decay_rate=0.8)
    mem_dynamic = MemoryRecord(
        id="mem_dyn",
        rule_type="local_dynamic",
        source_level=1,
        action_sequence=[1],
        expected_delta_state={},
        confidence=1.0,
    )
    # Gap of 2 levels: 1.0 * (0.8^2) = 0.64
    calibrated = car.calibrate_confidence(mem_dynamic, current_level=3)
    assert pytest.approx(calibrated, 0.01) == 0.64


def test_evidence_grounded_death_blacklisting():
    egr = EvidenceGroundedReasoning()
    mem = MemoryRecord(
        id="mem_fatal",
        rule_type="local_dynamic",
        source_level=2,
        action_sequence=[4],
        expected_delta_state={},
    )
    obs = EvidenceObservation(
        level=2,
        turn=5,
        action_taken=4,
        fatal_death=True,
    )
    validated = egr.record_evidence(mem, obs)
    assert validated is False
    assert mem.is_blacklisted is True
    assert egr.is_action_fatal(2, 4) is True


def test_memadapter_controller_lifecycle():
    controller = MemAdapterController()
    mem = MemoryRecord(
        id="mem_gate",
        rule_type="local_dynamic",
        source_level=1,
        action_sequence=[2],
        expected_delta_state={},
        confidence=0.9,
    )
    controller.register_memory(mem)

    # In level 1: feasible
    allowed, weight = controller.query_action_feasibility("mem_gate", current_level=1, candidate_action=2)
    assert allowed is True
    assert weight > 0.5

    # Simulate 3 no-op failures on Level 2 (environment changed)
    for _ in range(3):
        obs = EvidenceObservation(level=2, turn=1, action_taken=2, state_changed=False)
        controller.ingest_turn_result("mem_gate", obs)

    # Now should be blacklisted to prevent sycophancy
    allowed_after, _ = controller.query_action_feasibility("mem_gate", current_level=2, candidate_action=2)
    assert allowed_after is False
