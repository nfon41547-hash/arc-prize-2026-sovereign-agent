import pytest
from arc3sdk.agent_unanswerable_detector import AGentUnanswerableDetector, get_unanswerable_detector

def test_unanswerable_detector_initialization():
    detector = get_unanswerable_detector()
    assert isinstance(detector, AGentUnanswerableDetector)
    assert detector.alpha == 0.64
    assert detector.beta == 0.69

def test_compute_unanswerability_score():
    detector = AGentUnanswerableDetector(alpha=0.5, beta=0.5)
    # High confidence attempt vs 0 abstain
    score_high = detector.compute_unanswerability_score(1.0, 0.0, 1, 0)
    assert score_high == 0.5

    # 0 attempt vs high abstain
    score_low = detector.compute_unanswerability_score(0.0, 1.0, 0, 1)
    assert score_low == -0.5

def test_is_hypothesis_grounded():
    detector = AGentUnanswerableDetector()
    proposals_good = [
        {"attempt": True, "confidence": 0.9, "solution": "ACTION1"},
        {"attempt": True, "confidence": 0.85, "solution": "ACTION1"},
    ]
    is_grounded, v_score, reason = detector.is_hypothesis_grounded("hypo_01", proposals_good)
    assert is_grounded is True
    assert v_score > 0
    assert "empirically_grounded" in reason

    # Unanswerable / Abstaining majority
    proposals_bad = [
        {"attempt": False, "confidence": 0.95, "solution": None},
        {"attempt": False, "confidence": 0.90, "solution": None},
    ]
    is_grounded, v_score, reason = detector.is_hypothesis_grounded("hypo_02", proposals_bad)
    assert is_grounded is False
    assert v_score < 0
    assert "unanswerable_or_spurious" in reason

def test_filter_action_candidates():
    detector = AGentUnanswerableDetector()
    candidates = [
        {"id": "c1", "action": "ACTION1", "confidence": 0.9},
        {"id": "c2", "action": "ACTION2", "confidence": 0.8},
    ]
    filtered = detector.filter_action_candidates(candidates)
    assert len(filtered) == 2
