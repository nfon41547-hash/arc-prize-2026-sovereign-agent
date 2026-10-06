"""Test suite for Episodic Hindsight Pruner (ASCENT OaTTT in ARC-AGI-3)."""
import pytest
from arc3sdk.hindsight_pruner import EpisodicHindsightPruner

def test_prune_noop_and_cycles():
    # Simulate a run with a cycle (states returning to previous) and noops
    history = [
        {"action": "RIGHT", "grid": [[0, 1]], "level": 1, "score": 0.1, "changed": True},
        {"action": "UP", "grid": [[0, 1]], "level": 1, "score": 0.1, "changed": False}, # No-op wall hit
        {"action": "DOWN", "grid": [[1, 1]], "level": 1, "score": 0.2, "changed": True},
        {"action": "LEFT", "grid": [[0, 1]], "level": 1, "score": 0.2, "changed": True}, # Cycle back to step 0 grid
        {"action": "RIGHT", "grid": [[1, 1]], "level": 1, "score": 0.3, "changed": True},
        {"action": "RIGHT", "grid": [[2, 2]], "level": 1, "score": 1.0, "changed": True}, # Goal reached
    ]
    
    pruned = EpisodicHindsightPruner.prune_trajectory(history)
    assert len(pruned) < len(history)
    digest = EpisodicHindsightPruner.format_pruned_digest(pruned)
    assert "RIGHT" in digest
    assert "UP" not in digest  # Wall-hit noop eliminated

def test_empty_trajectory():
    assert EpisodicHindsightPruner.prune_trajectory([]) == []
    assert EpisodicHindsightPruner.format_pruned_digest([]) == ""
