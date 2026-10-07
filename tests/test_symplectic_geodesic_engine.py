"""Tests for SymplecticGeodesicWavefrontEngine (S-GWE)."""
from __future__ import annotations

import time
import pytest
from arc3sdk.symplectic_geodesic_engine import (
    SymplecticGeodesicWavefrontEngine,
    GeodesicPathResult,
)


def test_solve_simple_corridor():
    engine = SymplecticGeodesicWavefrontEngine()
    # 0 = empty, 1 = player, 2 = goal, 5 = wall
    grid = [
        [5, 5, 5, 5, 5],
        [5, 1, 0, 2, 5],
        [5, 5, 5, 5, 5],
    ]
    t0 = time.perf_counter()
    result = engine.solve_geodesic_flow(
        grid=grid,
        player_pos=(1, 1),
        wall_colors={5},
        goal_colors={2},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(result, GeodesicPathResult)
    assert result.actions == ["RIGHT", "RIGHT"]
    assert result.geodesic_length == 2
    assert result.confidence == 0.99
    assert elapsed_ms < 5.0  # sub-millisecond execution


def test_solve_obstacle_avoidance():
    engine = SymplecticGeodesicWavefrontEngine()
    # Path must detour around wall at (1, 2)
    grid = [
        [5, 5, 5, 5, 5],
        [5, 1, 5, 2, 5],
        [5, 0, 0, 0, 5],
        [5, 5, 5, 5, 5],
    ]
    result = engine.solve_geodesic_flow(
        grid=grid,
        player_pos=(1, 1),
        wall_colors={5},
        goal_colors={2},
    )
    assert len(result.actions) == 4
    assert result.actions == ["DOWN", "RIGHT", "RIGHT", "UP"]
    assert result.total_action_cost < 100.0  # did not hit wall
