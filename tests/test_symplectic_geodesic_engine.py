"""Tests for SymplecticGeodesicWavefrontEngine (S-GWE), Information Flux Tracker, and 5D Manifold Flow."""
from __future__ import annotations

import time
import pytest
from arc3sdk.symplectic_geodesic_engine import (
    SymplecticGeodesicWavefrontEngine,
    GeodesicPathResult,
    InformationFluxTracker,
    InformationFluxVector,
    ManifoldState5D,
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


def test_information_flux_tracker():
    tracker = InformationFluxTracker(divergence_epsilon=1e-4)

    # 1. Matching grids -> zero flux divergence, no disconfirmation
    actual = [[1, 2], [3, 4]]
    predicted_match = [[1, 2], [3, 4]]
    res_match = tracker.compute_flux(actual, predicted_match)
    assert isinstance(res_match, InformationFluxVector)
    assert res_match.kl_divergence == 0.0
    assert res_match.divergence == 0.0
    assert not res_match.is_disconfirmed

    # 2. Mismatching grid -> non-zero divergence and disconfirmation flag
    predicted_mismatch = [[1, 0], [3, 4]]
    res_mismatch = tracker.compute_flux(actual, predicted_mismatch)
    assert res_mismatch.kl_divergence > 0.0
    assert res_mismatch.divergence > 0.0
    assert res_mismatch.is_disconfirmed


def test_hamiltonian_potential_minimization():
    engine = SymplecticGeodesicWavefrontEngine()
    state = ManifoldState5D(
        r=5, c=5,
        shape_id=0, color_id=0, rotation_idx=0,
        energy_left=40
    )
    target_pos = (5, 8)  # Target is to the RIGHT (c=8)

    # Action "RIGHT" moves closer to target -> lower Hamiltonian potential
    h_right = engine.compute_hamiltonian_potential(
        action_name="RIGHT",
        current_state=state,
        target_pos=target_pos,
        target_shape=0,
        target_color=0,
        target_rotation=0,
    )

    # Action "LEFT" moves away from target -> higher Hamiltonian potential
    h_left = engine.compute_hamiltonian_potential(
        action_name="LEFT",
        current_state=state,
        target_pos=target_pos,
        target_shape=0,
        target_color=0,
        target_rotation=0,
    )

    assert h_right < h_left


def test_5d_manifold_solver():
    engine = SymplecticGeodesicWavefrontEngine()
    # 5x5 grid with modifiers:
    # (1, 1) = start (shape 0, color 0, rot 0)
    # (1, 2) = shape modifier (shape 0 -> 1)
    # (1, 3) = color modifier (color 0 -> 1)
    # (1, 4) = goal (requires shape 1, color 1, rot 0)
    grid = [[0 for _ in range(5)] for _ in range(5)]
    result = engine.solve_manifold_5d_flow(
        grid=grid,
        player_pos=(1, 1),
        current_shape=0,
        current_color=0,
        current_rotation=0,
        target_pos=(1, 4),
        target_shape=1,
        target_color=1,
        target_rotation=0,
        wall_coords=set(),
        rot_modifier_coords=set(),
        color_modifier_coords={(1, 3)},
        shape_modifier_coords={(1, 2)},
        battery_coords=set(),
        initial_energy=40,
        energy_decrement=2,
        num_shapes=6,
        num_colors=4,
    )

    assert result.reaches_goal_with_exact_attributes
    assert result.actions == ["RIGHT", "RIGHT", "RIGHT"]
    assert result.geodesic_length == 3
