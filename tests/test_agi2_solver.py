import numpy as np
import pytest
from arc3sdk.agi2_solver import (
    solve_task,
    _learn_shape_rule,
    _apply_shape,
    _apply_d4,
    _learn_color_map,
)


def test_shape_rule_induction():
    # Tile test
    pairs = [
        (np.zeros((2, 2), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)),
        (np.zeros((3, 3), dtype=np.uint8), np.zeros((6, 6), dtype=np.uint8)),
    ]
    rule, params = _learn_shape_rule(pairs)
    assert rule == "tile"
    assert params == (2, 2)

    # Fixed shape test
    pairs_fixed = [
        (np.zeros((5, 5), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)),
        (np.zeros((8, 8), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)),
    ]
    rule_fix, params_fix = _learn_shape_rule(pairs_fixed)
    assert rule_fix == "fixed"
    assert params_fix == (3, 3)


def test_d4_dihedral_and_color_homomorphism():
    # Input -> Output rotated 90 deg and color shifted 1 -> 2
    b1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    a1 = np.array([[0, 2], [0, 0]], dtype=np.uint8)

    train_pairs = [([list(row) for row in b1], [list(row) for row in a1])]
    test_in = [[1, 0], [1, 0]]

    predictions = solve_task(train_pairs, test_in)
    assert len(predictions) == 2
    # Check top attempt matches exact rotation + color permutation
    top_attempt = np.array(predictions[0])
    expected = np.array([[2, 2], [0, 0]], dtype=np.uint8)
    assert np.array_equal(top_attempt, expected)


def test_fixed_crop_offset_morphism():
    # Training pair: 4x4 input cropped at top-left 2x2
    b1 = np.array([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)
    a1 = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.uint8)

    train_pairs = [([list(row) for row in b1], [list(row) for row in a1])]
    test_in = [
        [5, 6, 9, 9],
        [7, 8, 9, 9],
        [9, 9, 9, 9],
        [9, 9, 9, 9]
    ]

    predictions = solve_task(train_pairs, test_in)
    top_attempt = np.array(predictions[0])
    assert top_attempt.shape == (2, 2)
    assert np.array_equal(top_attempt, [[5, 6], [7, 8]])
