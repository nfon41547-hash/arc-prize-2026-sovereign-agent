"""Unit tests for AGENT KB Cross-Domain Experience Knowledge Base and Disagreement Gate."""

import pytest
import numpy as np
from arc3sdk.agent_kb_memory import AgentKBMemory, DisagreementGate, StructuredExperience


def test_agent_kb_add_deduplication_and_utility():
    kb = AgentKBMemory(dedup_threshold=0.85, max_capacity=3)

    added1 = kb.add_experience(
        experience_id="exp_01",
        task_description="Filter out background noise colors in ARC grid",
        action_reasoning_pairs=[(1, "Ignore color 0 and identify bounding boxes")]
    )
    assert added1 is True

    # Identical/highly similar description should be deduplicated
    added2 = kb.add_experience(
        experience_id="exp_02",
        task_description="Filter out background noise colors in ARC grid",
        action_reasoning_pairs=[(1, "Duplicate action")]
    )
    assert added2 is False

    # Check utility update: u_j <- u_j + eta * (r_j - u_j)
    initial_u = kb.experiences["exp_01"].utility_score
    kb.update_utility("exp_01", reward=2.0)
    assert kb.experiences["exp_01"].utility_score > initial_u


def test_agent_kb_hybrid_retrieval():
    kb = AgentKBMemory()
    kb.add_experience(
        experience_id="exp_flood_fill",
        task_description="Enclosed area flood fill algorithm using 4-connectivity",
        action_reasoning_pairs=[(4, "Traverse connected components")]
    )
    kb.add_experience(
        experience_id="exp_gravity",
        task_description="Simulate vertical gravity falling physics on blocks",
        action_reasoning_pairs=[(2, "Shift non-zero elements downward")]
    )

    results = kb.hybrid_retrieve("flood fill enclosed boundary", top_k=1)
    assert len(results) == 1
    assert results[0][0].experience_id == "exp_flood_fill"
    assert results[0][1] > 0.0


def test_agent_kb_disagreement_gate_and_reason_retrieve_refine():
    kb = AgentKBMemory(disagreement_beta=0.7)
    kb.add_experience(
        experience_id="exp_symmetry",
        task_description="Mirror reflection along vertical symmetry axis",
        action_reasoning_pairs=[(1, "Compute horizontal mirror coordinates")]
    )

    initial_plan = "Identify object contours and compute reflection symmetry."
    refined_plan, exps, passed = kb.reason_retrieve_refine_plan(
        task_query="Compute reflection symmetry across vertical line",
        initial_plan=initial_plan,
        top_k=1
    )

    assert len(exps) == 1
    assert passed is True
    assert "Guideline from [exp_symmetry]" in refined_plan
