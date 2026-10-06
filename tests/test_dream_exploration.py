"""Unit tests for Dream Exploration Engine (NVIDIA Dream-Team SOTA Synthesis)."""

import pytest
import numpy as np
from arc3sdk.dream_exploration_engine import DreamExplorationEngine, BeamNode


def test_structured_click_extraction():
    engine = DreamExplorationEngine()
    grid = np.zeros((20, 20), dtype=np.int32)
    # Create an object of color 1 at (5..7, 5..7)
    grid[5:8, 5:8] = 1

    clicks = engine.extract_structured_clicks(grid, bg_color=0, max_clicks=6)
    assert len(clicks) > 0
    # Center of 5:8 is 6, 6
    assert (6, 6) in clicks


def test_productive_click_subdivision():
    engine = DreamExplorationEngine(click_subdivide=2)
    grid = np.zeros((10, 10), dtype=np.int32)
    engine.note_productive_click(5, 5)

    clicks = engine.extract_structured_clicks(grid, bg_color=0, max_clicks=10)
    # Subdivided clicks around (5,5) with delta +-2 should be included
    assert (3, 3) in clicks or (7, 7) in clicks or (3, 7) in clicks or (7, 3) in clicks


def test_beam_search_discovers_goal_path():
    engine = DreamExplorationEngine(beam_width=4, beam_depth=4)
    root = np.zeros((6, 6), dtype=np.int32)

    # Deterministic simulation where Action 1 -> Action 2 -> Win
    def mock_step(g: np.ndarray, action: int):
        next_g = g.copy()
        if action == 1:
            next_g[0, 0] = 1
            return next_g, 0.5, False, False
        elif action == 2 and g[0, 0] == 1:
            next_g[0, 1] = 2
            return next_g, 1.0, True, False  # Win!
        return next_g, 0.0, False, False

    best_act, conf, path = engine.beam_search(
        root_grid=root,
        available_actions=[1, 2, 3],
        step_fn=mock_step
    )

    assert best_act == 1
    assert conf >= 0.90
    assert path == [1, 2]
