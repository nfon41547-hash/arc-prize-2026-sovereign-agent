"""Tests for Transition Learner — Original Algorithm core."""

from __future__ import annotations

import numpy as np
import pytest


class TestTransitionLearner:
    def test_learn_identity(self):
        from arc3sdk.transition_learner import learn_transition
        grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
        result = learn_transition(grid, grid, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)

    def test_learn_rot90(self):
        from arc3sdk.transition_learner import learn_transition
        grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
        rotated = np.rot90(grid, 1)
        result = learn_transition(grid, rotated, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)

    def test_learn_flip(self):
        from arc3sdk.transition_learner import learn_transition
        grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
        flipped = np.fliplr(grid)
        result = learn_transition(grid, flipped, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)

    def test_learn_toggle(self):
        from arc3sdk.transition_learner import learn_transition
        grid = np.zeros((16, 16), dtype=np.uint8)
        grid[5, 5] = 1
        after = grid.copy()
        after[5, 5] = 0
        result = learn_transition(grid, after, [6])
        assert isinstance(result, list)

    def test_learn_fail_open(self):
        from arc3sdk.transition_learner import learn_transition
        result = learn_transition(np.array([]), np.array([]), [])
        assert result == []

    def test_predict_transition(self):
        from arc3sdk.transition_learner import predict_transition, learn_transition, reset
        reset()
        grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
        learn_transition(grid, grid, [1, 2, 3, 4, 5, 6])
        result = predict_transition(grid, [1, 2, 3, 4, 5, 6])
        assert isinstance(result, list)

    def test_get_status(self):
        from arc3sdk.transition_learner import get_status
        status = get_status()
        assert "version" in status
        assert "gate" in status
        assert "memory_size" in status

    def test_reset(self):
        from arc3sdk.transition_learner import reset, get_status
        reset()
        status = get_status()
        assert status["memory_size"] == 0

    def test_classify_transition_types(self):
        from arc3sdk.transition_learner import _classify_transition
        grid = np.random.randint(0, 16, (16, 16), dtype=np.uint8)
        # Identity
        t, c = _classify_transition(grid, grid)
        assert t == "identity"
        assert c == 1.0
        # Rot90
        t, c = _classify_transition(grid, np.rot90(grid, 1))
        assert t == "rot90"
        # Unknown: use a grid with completely different structure
        unknown_grid = np.zeros((16, 16), dtype=np.uint8)
        unknown_grid[0, 0] = 15
        unknown_grid[15, 15] = 14
        t, c = _classify_transition(grid, unknown_grid)
        assert t == "unknown"
