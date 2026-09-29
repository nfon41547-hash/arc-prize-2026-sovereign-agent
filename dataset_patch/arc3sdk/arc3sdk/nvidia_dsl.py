# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ARC-AGI-3 Domain Specific Language (DSL)

Reusable analysis, temporal tracking, grid comparison, and planning
functions for ARC-AGI-3 interactive game solving.

Categories:
1. Static grid analysis — object detection, pathfinding, symmetry, patterns
2. Grid comparison metrics — pixel accuracy, structural similarity, IoU
3. Temporal/history analysis — object tracking, action effects, cycle detection
4. Planning/simulation — transition verification, Monte Carlo search

All functions operate on list[list[int]] grids (64x64, values 0-15).
History is list[dict] with keys: step, grid, action, levels_completed, reward.

FILE I/O NOTE: These functions are injected as stubs into the agent REPL
sandbox. Because they are defined in this module, bare ``open()`` resolves
to ``builtins.open`` (process CWD), NOT the REPL's sandboxed ``_safe_open``
(memory root). Functions that read data files must call ``_get_sandboxed_open()``
to walk the call stack and find the REPL's ``open`` override.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np
import contextlib

if TYPE_CHECKING:
    from arc_agi_3.search_methods.base import SearchBudget

__all__ = [
    # Display
    'print_grid', 'print_diff', 'print_event',
    # Static grid analysis (includes find_objects alias)
    'find_objects', 'find_connected_components', 'flood_fill_from_border', 'extract_objects',
    'split_grid_uniform', 'apply_gravity', 'crop_to_content',
    'extract_diagonals', 'sort_objects_by_position',
    'compute_distance_to_value', 'find_nearest_color',
    'detect_symmetry', 'find_borders', 'find_repeating_pattern',
    'compute_path', 'find_matching_regions',
    # Grid comparison metrics
    'pixel_accuracy', 'color_histogram_distance', 'structural_similarity',
    'object_iou', 'region_match',
    # Temporal / history analysis
    'track_objects', 'scan_clickable', 'diff_grids',
    'action_effect_summary', 'action_transition_matrix',
    'find_productive_actions', 'detect_cycles', 'progress_curve',
    'state_novelty', 'compute_velocity', 'detect_collision_events',
    'find_triggers', 'grid_entropy_over_time',
    # Planning / simulation
    'simulate_sequence', 'monte_carlo_search', 'verify_transition', 'search',
    # Team data loaders
    'load_constants', 'get_estimated_background',
    'load_z_encoded', 'load_h_encoded', 'load_z_predicted',
    'run_simulation', 'evaluate_sub_goals', 'simulate', 'rollout_actions',
    # Step / level log readers
    'read_step_log', 'read_level_log',
]


# ===================================================================
# Display
# ===================================================================

_HEX_MAP = {i: hex(i)[2:] for i in range(16)}


def print_grid(grid: list, step: int = 0, level: int = 0) -> None:
    """Print a 64×64 grid as compact, readable hex text.

    Wraps ``Grid.render_text`` and writes to stdout. Rendering format (sep,
    fmt) is controlled by ``Grid.configure()`` — set once at startup from
    solver config. Not configurable per-call.

    Prefer this over ``print(grid)`` — printing a raw list[list[int]] bloats
    context with thousands of numbers.

    Args:
        grid: 64x64 list of lists with values 0-15.
        step: Current episode step (for the header).
        level: Current level (for the header).
    """
    from arc_agi_3.grid import Grid
    data = grid.tolist() if hasattr(grid, 'tolist') else grid
    print(Grid(data=data, step=step, level=level).render_text(compact=True))


def print_diff(grid1: list, grid2: list, step: int = 0, level: int = 0) -> None:
    """Print the difference between two grids as compact text.

    Unchanged pixels show as '.', changed pixels show the NEW color value.
    Rendering format (sep, fmt) is controlled by ``Grid.configure()``.

    Args:
        grid1: Previous grid (list of lists).
        grid2: Current grid (list of lists).
        step: Current episode step.
        level: Current level.
    """
    from arc_agi_3.grid import Grid, DiffGrid
    data1 = grid1.tolist() if hasattr(grid1, 'tolist') else grid1
    data2 = grid2.tolist() if hasattr(grid2, 'tolist') else grid2
    prev = Grid(data=data1, step=step, level=level)
    curr = Grid(data=data2, step=step, level=level)
    print(DiffGrid.from_grids(prev, curr).render_text(compact=True))


def print_event(event) -> None:
    """Print an event frame sequence as stacked compact grids.

    Args:
        event: list of 64x64 grids — e.g. history[N]["event"] or the top-level
            `event` variable. Prints a short marker when None or empty.
    """
    if not event:
        print("(no event frames)")
        return
    from arc_agi_3.grid import Grid
    for i, frame_data in enumerate(event):
        if i > 0:
            print()
        print(f"Event Frame {i + 1}/{len(event)}:")
        data = frame_data.tolist() if hasattr(frame_data, 'tolist') else frame_data
        print(Grid(data=data).render_text(add_header=False))


def find_objects(grid: list, background_color: int) -> list:
    """Find connected components of non-background cells.

    Returns list of dicts sorted by size (largest first), each with:
        - 'color' (int): The color value of the object
        - 'cells' (list of (row, col)): All cells belonging to the object
        - 'size' (int): Number of cells
        - 'bbox' (tuple): (min_row, min_col, max_row, max_col)

    Args:
        grid: 64x64 list of lists with values 0-15.
        background_color: Background color to exclude (use np.bincount to find it).
    """
    comps = find_connected_components(grid, background=background_color)
    arr = np.array(grid)
    objects = []
    for cells in comps:
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        objects.append({
            "color": int(arr[rows[0], cols[0]]),
            "cells": list(cells),
            "size": len(cells),
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
        })
    return objects


# ===================================================================
# Static Grid Analysis
# ===================================================================

def find_connected_components(
    grid: list, target_value: int | None = None,
    connectivity: int = 4, background: int | None = None,
) -> list[set[tuple[int, int]]]:
    """Find connected components using BFS.

    Args:
        grid: 2D grid.
        target_value: Only find components of this value (None = all non-bg).
        connectivity: 4 (orthogonal) or 8 (include diagonals).
        background: Value to skip (None = auto-detect most frequent).

    Returns:
        List of sets of (row, col) positions, sorted largest first.
    """
    arr = np.array(grid)
    H, W = arr.shape
    if background is None and target_value is None:
        background = int(np.bincount(arr.flatten()).argmax())
    visited = np.zeros((H, W), dtype=bool)
    components = []

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(H):
        for c in range(W):
            if visited[r, c]:
                continue
            val = int(arr[r, c])
            if target_value is not None and val != target_value:
                continue
            if background is not None and val == background:
                continue
            cells = set()
            queue = deque([(r, c)])
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(arr[nr, nc]) == val:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)

    components.sort(key=len, reverse=True)
    return components


def flood_fill_from_border(
    grid: list, fill_value: int = -1, background: int | None = None,
) -> list[list[int]]:
    """Flood fill from grid borders to identify enclosed regions.

    Fills all border-connected background cells with fill_value.
    Remaining background cells are enclosed (interior rooms/panels).
    """
    arr = np.array(grid, dtype=int)
    H, W = arr.shape
    if background is None:
        background = int(np.bincount(arr.flatten()).argmax())
    result = arr.copy()
    visited = np.zeros((H, W), dtype=bool)
    queue = deque()

    for r in range(H):
        for c in [0, W - 1]:
            if result[r, c] == background and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True
    for c in range(W):
        for r in [0, H - 1]:
            if result[r, c] == background and not visited[r, c]:
                queue.append((r, c))
                visited[r, c] = True

    while queue:
        cr, cc = queue.popleft()
        result[cr, cc] = fill_value
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and result[nr, nc] == background:
                visited[nr, nc] = True
                queue.append((nr, nc))

    return result.tolist()


def extract_objects(
    grid: list, background: int | None = None,
) -> list[dict]:
    """Extract objects as cropped sub-grids with metadata.

    Returns list of dicts sorted by size (largest first):
        {color, cells, size, bbox: (r1,c1,r2,c2), mask, cropped_grid}
    """
    arr = np.array(grid)
    if background is None:
        background = int(np.bincount(arr.flatten()).argmax())
    components = find_connected_components(grid, background=background)
    objects = []
    for cells in components:
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        h, w = r2 - r1 + 1, c2 - c1 + 1
        color = int(arr[rows[0], cols[0]])
        mask = np.zeros((h, w), dtype=bool)
        cropped = np.full((h, w), background, dtype=int)
        for r, c in cells:
            mask[r - r1, c - c1] = True
            cropped[r - r1, c - c1] = arr[r, c]
        objects.append({
            "color": color, "cells": cells, "size": len(cells),
            "bbox": (r1, c1, r2, c2),
            "mask": mask.tolist(), "cropped_grid": cropped.tolist(),
        })
    return objects


def split_grid_uniform(grid: list, n_rows: int, n_cols: int) -> list[list]:
    """Split grid into n_rows x n_cols uniform blocks.

    Returns result[i][j] = sub-grid at block row i, col j.
    """
    arr = np.array(grid)
    H, W = arr.shape
    bh, bw = H // n_rows, W // n_cols
    result = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            block = arr[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            row.append(block.tolist())
        result.append(row)
    return result


def apply_gravity(
    grid: list, direction: str = "down",
    movable: set[int] | None = None, fixed: set[int] | None = None,
    background_color: int | None = None,
) -> list[list[int]]:
    """Simulate gravity: stack movable objects in the given direction.

    Args:
        direction: "down", "up", "left", "right".
        movable: Colors that fall (None = all non-background).
        fixed: Colors that block falling (None = none).
        background_color: Background color. When None, falls back to
            ``get_estimated_background()``. Raises ValueError if unknown.
    """
    arr = np.array(grid, dtype=int)
    H, W = arr.shape
    if background_color is None:
        background_color = get_estimated_background()
    if background_color is None:
        raise ValueError(
            "Background color unknown: Observer has not estimated bg yet. "
            "Pass background_color explicitly or wait for Observer."
        )
    bg = background_color
    if movable is None:
        movable = {int(v) for v in np.unique(arr) if v != bg}
    if fixed is None:
        fixed = set()
    result = arr.copy()

    vertical = direction in ("down", "up")
    reverse = direction in ("down", "right")
    size = H if vertical else W
    outer = W if vertical else H

    for outer_idx in range(outer):
        line = list(result[:, outer_idx]) if vertical else list(result[outer_idx, :])

        moved = [v for v in line if v in movable]
        fixed_pos = {i: v for i, v in enumerate(line) if v in fixed}
        new_line = [bg] * size
        for i, v in fixed_pos.items():
            new_line[i] = v
        free = [i for i in range(size) if i not in fixed_pos]
        if reverse:
            free = list(reversed(free))
        for i, val in enumerate(reversed(moved) if reverse else moved):
            if i < len(free):
                new_line[free[i]] = val

        if vertical:
            result[:, outer_idx] = new_line
        else:
            result[outer_idx, :] = new_line

    return result.tolist()


def crop_to_content(grid: list, background: int = 0) -> list[list[int]]:
    """Remove border rows/cols that contain only the background color."""
    arr = np.array(grid)
    mask = arr != background
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [[]]
    r1, r2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c1, c2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return arr[r1:r2 + 1, c1:c2 + 1].tolist()


def extract_diagonals(grid: list, main: bool = True) -> list[list[int]]:
    """Extract main or anti-diagonals from grid."""
    arr = np.array(grid)
    H, W = arr.shape
    diags = []
    if main:
        for offset in range(-(H - 1), W):
            diags.append(np.diag(arr, offset).tolist())
    else:
        flipped = np.fliplr(arr)
        for offset in range(-(H - 1), W):
            diags.append(np.diag(flipped, offset).tolist())
    return diags


def sort_objects_by_position(
    objects: list[dict], sort_by: str = "top_left",
) -> list[dict]:
    """Sort objects by position. sort_by: top_left/center/size."""
    def key_fn(obj):
        bbox = obj["bbox"]
        if sort_by == "top_left":
            return (bbox[0], bbox[1])
        elif sort_by == "center":
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        elif sort_by == "size":
            return -obj.get("size", len(obj.get("cells", [])))
        return (bbox[0], bbox[1])
    return sorted(objects, key=key_fn)


def compute_distance_to_value(
    grid: list, target: int, metric: str = "euclidean",
) -> list[list[float]]:
    """Compute distance from each cell to nearest cell of target value."""
    arr = np.array(grid)
    mask = (arr != target).astype(float)
    if metric == "manhattan":
        from scipy.ndimage import distance_transform_cdt
        dist = distance_transform_cdt(mask, metric="cityblock")
    else:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(mask)
    return dist.tolist()


def find_nearest_color(
    grid: list, position: tuple[int, int], target_color: int,
) -> tuple[int, int] | None:
    """BFS to find closest cell of target_color from position."""
    arr = np.array(grid)
    H, W = arr.shape
    visited = np.zeros((H, W), dtype=bool)
    queue = deque([position])
    visited[position[0], position[1]] = True
    while queue:
        r, c = queue.popleft()
        if int(arr[r, c]) == target_color and (r, c) != position:
            return (r, c)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return None


def detect_symmetry(grid: list, axis: str = "horizontal") -> dict:
    """Check grid symmetry. axis: horizontal/vertical/both.

    Returns {symmetric: bool, mismatches: int, total: int, score: float}.
    """
    arr = np.array(grid)
    results = {}
    if axis in ("horizontal", "both"):
        flipped = np.flipud(arr)
        mis = int(np.sum(arr != flipped))
        total = arr.size
        results["horizontal"] = {"symmetric": mis == 0, "mismatches": mis,
                                  "total": total, "score": 1 - mis / max(total, 1)}
    if axis in ("vertical", "both"):
        flipped = np.fliplr(arr)
        mis = int(np.sum(arr != flipped))
        total = arr.size
        results["vertical"] = {"symmetric": mis == 0, "mismatches": mis,
                                "total": total, "score": 1 - mis / max(total, 1)}
    if axis == "both":
        return results
    return results.get(axis, {})


def find_borders(grid: list, border_color: int | None = None,
                 background_color: int | None = None) -> list[dict]:
    """Detect rectangular bordered regions.

    Args:
        border_color: If given, only check this color for borders.
        background_color: Background color used to determine candidate border
            colors when ``border_color`` is None. Falls back to
            ``get_estimated_background()``. Raises ValueError if unknown.

    Returns list of {bbox, interior_bbox, border_color}.
    """
    arr = np.array(grid)
    H, W = arr.shape
    if border_color is None and background_color is None:
        background_color = get_estimated_background()
        if background_color is None:
            raise ValueError(
                "Background color unknown: Observer has not estimated bg yet. "
                "Pass background_color explicitly or wait for Observer."
            )
        candidates = [int(v) for v in np.unique(arr) if v != background_color]
    else:
        candidates = [border_color]

    borders = []
    for color in candidates:
        mask = (arr == color)
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue
        r1, r2 = int(rows[0]), int(rows[-1])
        c1, c2 = int(cols[0]), int(cols[-1])
        top = np.all(arr[r1, c1:c2 + 1] == color)
        bot = np.all(arr[r2, c1:c2 + 1] == color)
        left = np.all(arr[r1:r2 + 1, c1] == color)
        right = np.all(arr[r1:r2 + 1, c2] == color)
        if top and bot and left and right and (r2 - r1) > 1 and (c2 - c1) > 1:
            borders.append({
                "bbox": (r1, c1, r2, c2),
                "interior_bbox": (r1 + 1, c1 + 1, r2 - 1, c2 - 1),
                "border_color": color,
            })
    return borders


def find_repeating_pattern(grid: list, region: tuple | None = None) -> dict | None:
    """Detect if a region tiles a smaller pattern.

    Returns {pattern, tile_h, tile_w, repeats_h, repeats_w} or None.
    """
    arr = np.array(grid)
    if region:
        r1, c1, r2, c2 = region
        arr = arr[r1:r2 + 1, c1:c2 + 1]
    H, W = arr.shape
    for th in range(1, H // 2 + 1):
        if H % th != 0:
            continue
        for tw in range(1, W // 2 + 1):
            if W % tw != 0:
                continue
            tile = arr[:th, :tw]
            match = True
            for i in range(0, H, th):
                for j in range(0, W, tw):
                    if not np.array_equal(arr[i:i + th, j:j + tw], tile):
                        match = False
                        break
                if not match:
                    break
            if match:
                return {
                    "pattern": tile.tolist(), "tile_h": th, "tile_w": tw,
                    "repeats_h": H // th, "repeats_w": W // tw,
                }
    return None


def compute_path(
    grid: list, start: tuple[int, int], end: tuple[int, int],
    walkable: set[int] | None = None,
) -> list[tuple[int, int]] | None:
    """BFS shortest path. Returns list of (row, col) or None if unreachable."""
    arr = np.array(grid)
    H, W = arr.shape
    visited = np.zeros((H, W), dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start[0], start[1]] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            path = []
            cur = end
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            return list(reversed(path))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and walkable is None or int(arr[nr, nc]) in walkable:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    return None


def find_matching_regions(grid: list, template: list) -> list[tuple[int, int]]:
    """Find all positions where template matches within grid.

    Returns list of (row, col) for top-left corner of each match.
    """
    arr = np.array(grid)
    tmpl = np.array(template)
    th, tw = tmpl.shape
    H, W = arr.shape
    matches = []
    for r in range(H - th + 1):
        for c in range(W - tw + 1):
            if np.array_equal(arr[r:r + th, c:c + tw], tmpl):
                matches.append((r, c))
    return matches


# ===================================================================
# Grid Comparison Metrics
# ===================================================================

def _normalize_grid_for_metrics(
    grid: list,
    *,
    name: str,
    allowed_negative_values: set[int] | None = None,
    palette_size: int = 16,
    allow_out_of_palette: bool = False,
) -> np.ndarray:
    """Validate and coerce a metric grid to a 2D integer array.

    With ``allow_out_of_palette=True`` the palette-range checks are skipped:
    out-of-range cells are kept as-is so a comparison against a valid grid
    counts them as wrong rather than raising. Used for a rendered (agent
    output) grid checked against a valid ground-truth grid."""
    try:
        arr = np.asarray(grid)
    except ValueError as exc:
        raise ValueError(
            f"{name}: grid must be a rectangular 2D grid of integer colors "
            f"in [0, {palette_size}); got ragged/inhomogeneous rows"
        ) from exc
    if arr.ndim != 2:
        raise ValueError(
            f"{name}: grid must be a rectangular 2D grid of integer colors "
            f"in [0, {palette_size}); got shape {arr.shape}"
        )
    if 0 in arr.shape:
        raise ValueError(f"{name}: grid must be non-empty, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"{name}: grid contains non-integer values; expected integer colors "
            f"in [0, {palette_size}); got dtype {arr.dtype}"
        )

    allowed_negative_values = allowed_negative_values or set()
    if not allow_out_of_palette:
        bad_negative = (arr < 0) & ~np.isin(arr, list(allowed_negative_values)) if allowed_negative_values else arr < 0
        if np.any(bad_negative):
            values = sorted({int(v) for v in arr[bad_negative].flatten()})
            raise ValueError(
                f"{name}: grid contains negative color values {values}; "
                f"out-of-palette values {values}; expected integer colors in "
                f"[0, {palette_size}). Likely cause: agent render() copied "
                f"sentinels such as -1 from level constants without filtering."
            )
        bad_high = arr >= palette_size
        if np.any(bad_high):
            values = sorted({int(v) for v in arr[bad_high].flatten()})
            raise ValueError(
                f"{name}: grid contains out-of-palette color values {values}; "
                f"expected colors in [0, {palette_size})"
            )
    return arr.astype(int, copy=False)


def _require_same_shape(a: np.ndarray, b: np.ndarray, *, left: str, right: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{left} and {right} must have the same shape, "
            f"got {a.shape} and {b.shape}"
        )


def pixel_accuracy(
    grid: list, goal_grid: list, ignore: int = -1,
    *, allow_out_of_palette_grid: bool = False,
) -> float:
    """Cell-by-cell accuracy, ignoring cells marked with ignore value.

    ``allow_out_of_palette_grid=True`` keeps out-of-range cells in the
    rendered ``grid`` instead of raising, so they count as wrong against the
    valid ``goal_grid``."""
    a = _normalize_grid_for_metrics(
        grid, name="grid", allow_out_of_palette=allow_out_of_palette_grid,
    )
    b = _normalize_grid_for_metrics(
        goal_grid,
        name="goal_grid",
        allowed_negative_values={ignore},
    )
    _require_same_shape(a, b, left="grid", right="goal_grid")
    mask = b != ignore
    if not mask.any():
        return 1.0
    return float(np.sum((a == b) & mask) / np.sum(mask))


def color_histogram_distance(grid: list, goal_grid: list) -> float:
    """Normalized histogram distance between two grids. 0 = identical."""
    a_grid = _normalize_grid_for_metrics(
        grid, name="color_histogram_distance(grid)",
    )
    b_grid = _normalize_grid_for_metrics(
        goal_grid, name="color_histogram_distance(goal_grid)",
    )
    _require_same_shape(
        a_grid,
        b_grid,
        left="color_histogram_distance(grid)",
        right="color_histogram_distance(goal_grid)",
    )
    a = a_grid.flatten()
    b = b_grid.flatten()
    ha = np.bincount(a, minlength=16).astype(float)
    hb = np.bincount(b, minlength=16).astype(float)
    ha /= max(ha.sum(), 1)
    hb /= max(hb.sum(), 1)
    return float(np.sum(np.abs(ha - hb)) / 2)


def structural_similarity(grid: list, goal_grid: list) -> float:
    """Edge-based structural similarity. 1 = identical structure."""
    a = _normalize_grid_for_metrics(grid, name="structural_similarity(grid)")
    b = _normalize_grid_for_metrics(goal_grid, name="structural_similarity(goal_grid)")
    _require_same_shape(
        a,
        b,
        left="structural_similarity(grid)",
        right="structural_similarity(goal_grid)",
    )

    def edges(arr):
        e = np.zeros_like(arr, dtype=int)
        e[:, :-1] += (np.abs(np.diff(arr, axis=1)) > 0).astype(int)
        e[:-1, :] += (np.abs(np.diff(arr, axis=0)) > 0).astype(int)
        return (e > 0).astype(int)

    ea, eb = edges(a), edges(b)
    intersection = int(np.sum(ea & eb))
    union = int(np.sum(ea | eb))
    return float(intersection / max(union, 1))


def object_iou(grid: list, goal_grid: list, background: int | None = None) -> float:
    """Average IoU between matched objects in grid vs goal_grid."""
    a = _normalize_grid_for_metrics(grid, name="object_iou(grid)")
    b = _normalize_grid_for_metrics(goal_grid, name="object_iou(goal_grid)")
    _require_same_shape(a, b, left="object_iou(grid)", right="object_iou(goal_grid)")
    objs_a = extract_objects(a.tolist(), background)
    objs_b = extract_objects(b.tolist(), background)
    if not objs_a or not objs_b:
        return 1.0 if (not objs_a and not objs_b) else 0.0

    def iou(cells_a, cells_b):
        sa, sb = set(cells_a), set(cells_b)
        return len(sa & sb) / max(len(sa | sb), 1)

    used_b = set()
    total_iou = 0.0
    for oa in objs_a:
        best_iou, best_j = 0, -1
        for j, ob in enumerate(objs_b):
            if j in used_b:
                continue
            score = iou(oa["cells"], ob["cells"])
            if score > best_iou:
                best_iou, best_j = score, j
        if best_j >= 0:
            used_b.add(best_j)
        total_iou += best_iou
    return total_iou / max(len(objs_a), 1)


def region_match(
    grid: list, goal_grid: list, region_bbox: tuple[int, int, int, int],
) -> float:
    """Pixel accuracy within a specific region only."""
    r1, c1, r2, c2 = region_bbox
    grid_arr = _normalize_grid_for_metrics(grid, name="grid")
    goal_arr = _normalize_grid_for_metrics(goal_grid, name="goal_grid")
    _require_same_shape(grid_arr, goal_arr, left="grid", right="goal_grid")
    a = grid_arr[r1:r2 + 1, c1:c2 + 1]
    b = goal_arr[r1:r2 + 1, c1:c2 + 1]
    return float(np.sum(a == b) / max(a.size, 1))


# ===================================================================
# Temporal / History Analysis
# ===================================================================

def track_objects(history: list[dict], colors: list[int]) -> dict[int, list[tuple]]:
    """Track center position of colored objects across steps.

    Returns {color: [(step, center_row, center_col), ...]}.
    """
    result = {c: [] for c in colors}
    for rec in history:
        arr = np.array(rec["grid"])
        step = rec.get("step", 0)
        for color in colors:
            positions = np.argwhere(arr == color)
            if len(positions) > 0:
                center = positions.mean(axis=0)
                result[color].append((step, round(float(center[0]), 1),
                                      round(float(center[1]), 1)))
    return result


def scan_clickable(grid: list, history: list[dict]) -> list[tuple[int, int]]:
    """From history, find positions that caused grid changes when CLICKed."""
    clickable = set()
    for i in range(len(history) - 1):
        action = str(history[i].get("action", ""))
        if action.startswith("CLICK") and len(action.split()) == 3:
            try:
                parts = action.split()
                x, y = int(parts[1]), int(parts[2])
                g1 = np.array(history[i]["grid"])
                g2 = np.array(history[i + 1]["grid"])
                if not np.array_equal(g1, g2):
                    clickable.add((y, x))  # row=y, col=x
            except (ValueError, IndexError):
                pass
    return sorted(clickable)


def diff_grids(grid1: list, grid2: list) -> dict:
    """Structured diff of two grids. Returns {count,
    changed_pixels: [(r,c,old,new)], bbox, color_changes: {"old->new": count}}."""
    a, b = np.array(grid1), np.array(grid2)
    mask = a != b
    changed = np.argwhere(mask)
    pixels = [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in changed]
    bbox = None
    if len(changed) > 0:
        bbox = (int(changed[:, 0].min()), int(changed[:, 1].min()),
                int(changed[:, 0].max()), int(changed[:, 1].max()))

    color_changes = defaultdict(int)
    for _, _, old, new in pixels:
        color_changes[f"{old}->{new}"] += 1

    return {
        "count": len(pixels), "changed_pixels": pixels,
        "bbox": bbox, "color_changes": dict(color_changes),
    }


def action_effect_summary(history: list[dict]) -> dict[str, dict]:
    """Summarize what each action does across history.

    Returns {action: {count, avg_diff, max_diff, avg_reward, positions_affected}}.
    """
    effects = defaultdict(lambda: {"count": 0, "diffs": [], "rewards": [], "positions": set()})

    for i in range(len(history) - 1):
        action = str(history[i].get("action", "UNKNOWN"))
        action_base = action.split()[0] if action else "UNKNOWN"
        g1, g2 = np.array(history[i]["grid"]), np.array(history[i + 1]["grid"])
        diff_mask = g1 != g2
        diff_count = int(diff_mask.sum())
        reward = history[i + 1].get("reward", 0)
        effects[action_base]["count"] += 1
        effects[action_base]["diffs"].append(diff_count)
        effects[action_base]["rewards"].append(reward)
        for r, c in zip(*np.where(diff_mask), strict=False):
            effects[action_base]["positions"].add((int(r), int(c)))

    result = {}
    for action, data in effects.items():
        result[action] = {
            "count": data["count"],
            "avg_diff": round(sum(data["diffs"]) / max(len(data["diffs"]), 1), 1),
            "max_diff": max(data["diffs"]) if data["diffs"] else 0,
            "avg_reward": round(sum(data["rewards"]) / max(len(data["rewards"]), 1), 4),
            "positions_affected": len(data["positions"]),
        }
    return result


def action_transition_matrix(history: list[dict]) -> dict[str, dict[str, int]]:
    """Count action->action transitions to detect loops."""
    matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(history) - 1):
        a1 = str(history[i].get("action", "")).split()[0]
        a2 = str(history[i + 1].get("action", "")).split()[0]
        if a1 and a2:
            matrix[a1][a2] += 1
    return {k: dict(v) for k, v in matrix.items()}


def find_productive_actions(history: list[dict], min_diff: int = 10) -> list[dict]:
    """Filter history to steps where grid changed significantly."""
    productive = []
    for i in range(len(history) - 1):
        g1, g2 = np.array(history[i]["grid"]), np.array(history[i + 1]["grid"])
        diff = int(np.sum(g1 != g2))
        if diff >= min_diff:
            productive.append({
                "step": history[i].get("step", i),
                "action": history[i].get("action", ""),
                "diff_pixels": diff,
                "reward": history[i + 1].get("reward", 0),
            })
    return productive


def detect_cycles(history: list[dict], window: int = 10) -> dict | None:
    """Detect repeating grid states in recent history.

    Returns {cycle_length, start_step} or None.
    """
    recent = history[-window:] if len(history) > window else history
    grids = [tuple(map(tuple, h["grid"])) for h in recent]
    seen = {}
    for i, g in enumerate(grids):
        if g in seen:
            return {
                "cycle_length": i - seen[g],
                "start_step": recent[seen[g]].get("step", seen[g]),
            }
        seen[g] = i
    return None


def progress_curve(history: list[dict], goal_grid: list) -> list[dict]:
    """Track pixel_accuracy against goal_grid over time."""
    curve = []
    for h in history:
        acc = pixel_accuracy(h["grid"], goal_grid)
        curve.append({"step": h.get("step", 0), "accuracy": round(acc, 4)})
    return curve


def state_novelty(history: list[dict], grid: list) -> float:
    """How different is grid from all previously seen grids (0=duplicate, 1=novel)."""
    arr = np.array(grid)
    if not history:
        return 1.0
    total = max(arr.size, 1)
    min_diff = min(int(np.sum(np.array(h["grid"]) != arr)) for h in history)
    return min_diff / total


def compute_velocity(history: list[dict], color: int, window: int = 3) -> dict:
    """Average movement vector for a colored object over last N steps.

    Returns {dr, dc, speed, direction}.
    """
    positions = track_objects(history[-window - 1:], [color]).get(color, [])
    if len(positions) < 2:
        return {"dr": 0.0, "dc": 0.0, "speed": 0.0, "direction": "stationary"}
    drs = [positions[i][1] - positions[i - 1][1] for i in range(1, len(positions))]
    dcs = [positions[i][2] - positions[i - 1][2] for i in range(1, len(positions))]
    dr = sum(drs) / len(drs)
    dc = sum(dcs) / len(dcs)
    speed = (dr ** 2 + dc ** 2) ** 0.5
    if speed < 0.5:
        direction = "stationary"
    elif abs(dr) > abs(dc):
        direction = "down" if dr > 0 else "up"
    else:
        direction = "right" if dc > 0 else "left"
    return {"dr": round(dr, 1), "dc": round(dc, 1),
            "speed": round(speed, 1), "direction": direction}


def detect_collision_events(history: list[dict]) -> list[dict]:
    """Find steps where a directional action caused no grid change."""
    collisions = []
    directional = {"UP", "DOWN", "LEFT", "RIGHT"}
    for i in range(len(history) - 1):
        action = str(history[i].get("action", "")).split()[0]
        if action in directional:
            g1, g2 = np.array(history[i]["grid"]), np.array(history[i + 1]["grid"])
            if np.array_equal(g1, g2):
                collisions.append({
                    "step": history[i].get("step", i), "action": action,
                })
    return collisions


def find_triggers(history: list[dict], radius: int = 10) -> list[dict]:
    """Find CLICK actions that caused changes far from the click position."""
    triggers = []
    for i in range(len(history) - 1):
        action = str(history[i].get("action", ""))
        if not action.startswith("CLICK") or len(action.split()) != 3:
            continue
        try:
            parts = action.split()
            cx, cy = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        g1, g2 = np.array(history[i]["grid"]), np.array(history[i + 1]["grid"])
        changed = np.argwhere(g1 != g2)
        if len(changed) == 0:
            continue
        dists = np.sqrt((changed[:, 0] - cy) ** 2 + (changed[:, 1] - cx) ** 2)
        remote = changed[dists > radius]
        if len(remote) > 0:
            triggers.append({
                "step": history[i].get("step", i),
                "click": (cx, cy),
                "remote_changes": len(remote),
                "max_distance": round(float(dists.max()), 1),
            })
    return triggers


def grid_entropy_over_time(history: list[dict]) -> list[dict]:
    """Track grid entropy (color distribution disorder) over time.

    Higher entropy = more uniform color distribution = more chaotic.
    Lower entropy = fewer colors dominating = more ordered.
    """
    curve = []
    for h in history:
        arr = np.array(h["grid"]).flatten()
        counts = np.bincount(arr, minlength=16).astype(float)
        probs = counts / max(counts.sum(), 1)
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log2(probs)))
        curve.append({"step": h.get("step", 0), "entropy": round(entropy, 3)})
    return curve


# ===================================================================
# Planning / Simulation
# ===================================================================

def simulate_sequence(
    grid: list, transition_fn: Callable, actions: list[str],
) -> list[list]:
    """Apply transition function to a sequence of actions.

    Returns list of predicted grids (initial + one per action).
    """
    grids = [grid]
    current = grid
    for action in actions:
        try:
            current = transition_fn(current, action)
            grids.append(current)
        except Exception:
            break
    return grids


def monte_carlo_search(
    grid: list,
    transition_fn: Callable,
    goal_grid: list,
    available_actions: list[str],
    strategy_fn: Callable | None = None,
    n_rollouts: int = 50,
    depth: int = 10,
) -> list[dict]:
    """Run Monte Carlo rollouts and rank by goal proximity.

    Args:
        transition_fn: (grid, action) -> predicted_grid
        goal_grid: Target grid state.
        strategy_fn: (grid) -> action. If None, uses random actions.
        n_rollouts: Number of rollouts.
        depth: Max actions per rollout.

    Returns top 10 results sorted by final_accuracy (best first):
        [{actions, final_accuracy, steps}]
    """
    import random
    results = []
    for _ in range(n_rollouts):
        current = grid
        actions_taken = []
        for _ in range(depth):
            if strategy_fn:
                try:
                    action = strategy_fn(current)
                except Exception:
                    action = random.choice(available_actions)
            else:
                action = random.choice(available_actions)
            try:
                current = transition_fn(current, action)
                actions_taken.append(action)
            except Exception:
                break
        acc = pixel_accuracy(current, goal_grid)
        results.append({
            "actions": actions_taken,
            "final_accuracy": round(acc, 4),
            "steps": len(actions_taken),
        })
    results.sort(key=lambda x: -x["final_accuracy"])
    return results[:10]


def verify_transition(history: list[dict], transition_fn: Callable) -> dict:
    """Compare transition predictions against actual history.

    Returns {overall_accuracy, total, correct, per_action: {action: {correct, total, accuracy}}}.
    """
    per_action = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total = 0
    for i in range(len(history) - 1):
        action = str(history[i].get("action", ""))
        actual = np.array(history[i + 1]["grid"])
        try:
            predicted = np.array(transition_fn(history[i]["grid"], action))
            correct = bool(np.array_equal(predicted, actual))
        except Exception:
            correct = False
        action_base = action.split()[0] if action else "UNKNOWN"
        per_action[action_base]["total"] += 1
        total += 1
        if correct:
            per_action[action_base]["correct"] += 1
            total_correct += 1
    for k in per_action:
        d = per_action[k]
        d["accuracy"] = round(d["correct"] / max(d["total"], 1), 3)
    return {
        "overall_accuracy": round(total_correct / max(total, 1), 3),
        "total_predictions": total,
        "correct_predictions": total_correct,
        "per_action": dict(per_action),
    }


# ===================================================================
# Team Data Loaders
# ===================================================================

def _get_sandboxed_open(return_context: bool = False):
    """Return the agent REPL's path-aware ``open``, or ``builtins.open``.

    DSL helpers are defined in this module, so a bare ``open`` resolves to
    ``builtins.open`` (process CWD). When called from the REPL we need the
    REPL's patched ``open`` (resolves under the agent's memory_root).
    Walks caller frames to find a ``__builtins__["open"]`` that differs
    from the global ``open``; returns the global ``open`` as fallback.

    If ``return_context`` is true, also returns the caller's import hook and
    REPL memory root when available. These are structural sandbox handles,
    not agent-authored REPL variables.
    """
    import sys

    def _from_builtins(caller_builtins, name: str):
        if isinstance(caller_builtins, dict):
            return caller_builtins.get(name)
        return getattr(caller_builtins, name, None)

    def _memory_root_from_open(_open):
        owner = getattr(_open, "__self__", None)
        root = getattr(owner, "_memory_root", None)
        if root is None:
            return None
        try:
            from pathlib import Path
            return Path(str(root)).resolve()
        except OSError:
            return None

    try:
        frame = sys._getframe(2)  # skip _get_sandboxed_open + immediate caller
    except ValueError:
        frame = None
    while frame is not None:
        caller_builtins = frame.f_globals.get("__builtins__", {})
        _open = _from_builtins(caller_builtins, "open")
        if _open is not None and _open is not open and return_context:
            return (
                _open,
                _from_builtins(caller_builtins, "__import__"),
                _memory_root_from_open(_open),
            )
            return _open
        frame = frame.f_back
    if return_context:
        return open, __import__, None
    return open


def _load_jsonl(path_str: str) -> list[dict]:
    """Load a .jsonl file as a list of dicts. Returns [] if missing/empty.

    Uses ``_get_sandboxed_open()`` so that when invoked inside the REPL
    sandbox, paths resolve relative to the agent's memory root.
    """
    import json

    _open = _get_sandboxed_open()

    try:
        content = _open(path_str).read()
    except (FileNotFoundError, PermissionError, OSError):
        return []

    entries = []
    for line in content.strip().split("\n"):
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                entries.append(json.loads(line))
    return entries


def load_constants(level: int | None = None) -> dict | None:
    """Return the constants payload for a level from data/level_constants.jsonl.

    Each row in the file is {"level": <int>, "constants": <payload>}; this function returns
    <payload> (the inner dict, read its fields directly). level=None selects the
    latest row. Returns None when absent."""
    entries = _load_jsonl("data/level_constants.jsonl")
    if not entries:
        return None
    if level is not None:
        matches = [e for e in entries if e.get("level") == level]
        if matches:
            return _row_payload(matches[-1], "constants")
        return None
    return _row_payload(entries[-1], "constants")


def get_estimated_background(level: int | None = None) -> int | None:
    """Return the level's background color from level_constants.jsonl.

    Reads ``background_color`` from ``load_constants(level)``. Returns None
    when the file is missing, empty, or has no ``background_color`` key.
    """
    constants = load_constants(level)
    if not constants:
        return None
    bg = constants.get("background_color")
    if bg is None:
        return None
    return int(bg)


def _row_payload(row: dict, key: str) -> dict | None:
    """Return the dict payload stored in ``row[key]``."""
    if key not in row:
        return None
    payload = row[key]
    if not isinstance(payload, dict):
        return None
    if isinstance(row.get("step"), int) and _state_payload_step(payload) is None:
        payload = {**payload, "step": row["step"]}
    return payload


def _select_step_payload(
    entries: list[dict],
    key: str,
    step: int | None,
) -> dict | None:
    if not entries:
        return None

    rows = sorted(entries, key=lambda e: e.get("step", 0))
    if step is not None:
        rows = [e for e in rows if e.get("step") == step]
        if not rows:
            return None

    return _row_payload(rows[-1], key)


def load_z_encoded(step: int | None = None) -> dict | None:
    """Return the z_encoded payload for a step from data/z_encoded.jsonl.

    Each row in the file is {"step": <int>, "z_encoded": <payload>}; this function
    returns <payload> (read its fields directly). step=None selects the latest
    row. Returns None when absent."""
    entries = _load_jsonl("data/z_encoded.jsonl")
    return _select_step_payload(entries, "z_encoded", step)


def load_h_encoded(step: int | None = None) -> dict | None:
    """Return the h_encoded payload for a step from data/h_encoded.jsonl.

    Each row in the file is {"step": <int>, "h_encoded": <payload>}; this function returns
    <payload> (read its fields directly). step=None selects the latest row. Returns None when absent."""
    entries = _load_jsonl("data/h_encoded.jsonl")
    return _select_step_payload(entries, "h_encoded", step)


# Roles that own a {role}_step_log.md / {role}_level_log.md in the shared dir.
# Mirrors file_registry.py; a sync test guards against drift.
_LOG_ROLES = frozenset({
    "observer", "simulator", "transductive_explorer",
    "inductive_explorer", "critic", "team_leader",
})


def _validate_log_role(role: str) -> None:
    """Raise ValueError for an unrecognized role so a typo never reads nothing."""
    if role not in _LOG_ROLES:
        raise ValueError(
            f"unknown log role {role!r}; expected one of {sorted(_LOG_ROLES)}"
        )


def _read_log_text(filename: str) -> str:
    """Read a shared-dir log file via the sandboxed open. '' when missing.

    Uses ``_get_sandboxed_open()`` so paths resolve under the agent's memory
    root inside the REPL sandbox (mirrors ``_load_jsonl``).
    """
    _open = _get_sandboxed_open()
    try:
        return _open(filename).read()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def read_step_log(step: int, role: str) -> str | None:
    """Read one role's step-log entry covering a given step.

    Returns the markdown of the ``# {Role} — Step {step}`` entry from
    ``{role}_step_log.md`` in the shared team directory. Pass a role name —
    your own to revisit a prior step, or a teammate's to read theirs. A
    multi-action or policy entry that spans several steps is returned for any
    step it covers. Returns None when that role wrote no entry covering ``step``.
    """
    _validate_log_role(role)
    from arc_agi_3.arc_league.step_log_sections import (
        STEP_LOG_ENTRY_RE, iter_log_entry_spans, parse_step_log_entry_coverage,
    )
    text = _read_log_text(f"{role}_step_log.md")
    if not text:
        return None
    hits = []
    for match, body in iter_log_entry_spans(text, STEP_LOG_ENTRY_RE):
        start, end = parse_step_log_entry_coverage(match.group(0))
        if start is not None and start <= step <= end:
            hits.append(body.rstrip())
    return "\n\n".join(hits) if hits else None


def read_level_log(step: int, role: str) -> str | None:
    """Read one role's level-log sections covering a given step.

    Returns the markdown of the level-log sections from ``{role}_level_log.md``
    whose recorded step span includes ``step`` — level-completion sections that
    cover a step range, and reset-reflection sections recorded at that step. Pass
    a role name — your own or a teammate's. Returns None when no section covers
    ``step``.
    """
    _validate_log_role(role)
    from arc_agi_3.arc_league.step_log_sections import (
        LEVEL_LOG_ENTRY_RE, iter_log_entry_spans, level_log_entry_coverage,
    )
    text = _read_log_text(f"{role}_level_log.md")
    if not text:
        return None
    hits = []
    for match, body in iter_log_entry_spans(text, LEVEL_LOG_ENTRY_RE):
        start, end = level_log_entry_coverage(
            match.group("span"), match.group("kind"),
        )
        if end is not None and (start is None or start <= step) and step <= end:
            hits.append(body.rstrip())
    return "\n\n".join(hits) if hits else None


_ACTION_NAMES = {
    0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT",
    4: "RIGHT", 5: "USE", 6: "CLICK", 7: "UNDO",
}


def _normalise_action_name(action: Any) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, int):
        return _ACTION_NAMES.get(action, str(action))
    if isinstance(action, (list, tuple)) and action:
        return _normalise_action_name(action[0])
    return ""


def _normalise_action_names(actions: Any) -> list[str]:
    if not isinstance(actions, (list, tuple)):
        return []
    out = []
    for action in actions:
        name = _normalise_action_name(action)
        if name:
            out.append(name)
    return out


def _artifact_root_candidates(memory_root) -> list:
    from pathlib import Path

    base = Path(str(memory_root)).resolve() if memory_root else Path.cwd()
    roots = [base, *list(base.parents)[:8]]
    seen = set()
    out = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        out.append(root)
    return out


def _read_json_path(path) -> dict | None:
    import json

    try:
        with open(str(path), encoding="utf-8", errors="replace") as f:
            payload = json.load(f)
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_step_record_for_prediction(step: int, roots: list) -> dict | None:
    for root in roots:
        record = _read_json_path(root / "steps" / f"step_{step:04d}.json")
        if isinstance(record, dict):
            return record
    return None


def _load_available_actions_for_prediction(step: int, record: dict, roots: list) -> list[str]:
    actions = _normalise_action_names(record.get("available_actions"))
    if actions:
        return actions

    for root in roots:
        gameplay = _read_json_path(root / "gameplay.json")
        if not isinstance(gameplay, dict):
            continue
        log = gameplay.get("log")
        if not isinstance(log, list):
            continue
        for entry in log:
            if not isinstance(entry, dict):
                continue
            if entry.get("global_step") != step:
                continue
            actions = _normalise_action_names(entry.get("available_actions"))
            if actions:
                return actions
    return []


def _load_world_model_dynamics_for_prediction():
    _open, import_fn, _memory_root = _get_sandboxed_open(return_context=True)

    if callable(import_fn) and import_fn is not __import__:
        try:
            return import_fn(
                "world_model.dynamics",
                globals(),
                None,
                ("predict", "HYPOTHESES"),
                0,
            )
        except Exception:
            pass

    try:
        source = _open("world_model/dynamics.py").read()
    except (FileNotFoundError, PermissionError, OSError):
        source = None
    if source:
        import builtins
        import sys
        import types

        builtins_dict = dict(vars(builtins))
        builtins_dict["open"] = _open
        if callable(import_fn):
            builtins_dict["__import__"] = import_fn
        package = sys.modules.get("world_model")
        if package is None:
            package = types.ModuleType("world_model")
            package.__path__ = []
            sys.modules["world_model"] = package
        harness_module = None
        try:
            harness_source = _open("world_model/harness.py").read()
        except (FileNotFoundError, PermissionError, OSError):
            harness_source = None
        if harness_source:
            harness_module = types.ModuleType("world_model.harness")
            harness_module.__package__ = "world_model"
            harness_module.__file__ = "world_model/harness.py"
            harness_module.__dict__["__builtins__"] = builtins_dict
            sys.modules["world_model.harness"] = harness_module
            package.harness = harness_module
            try:
                exec(
                    compile(harness_source, "world_model/harness.py", "exec"),
                    harness_module.__dict__,
                )
            except Exception:
                sys.modules.pop("world_model.harness", None)
                if getattr(package, "harness", None) is harness_module:
                    delattr(package, "harness")
                harness_module = None
        module = types.ModuleType("world_model.dynamics")
        module.__package__ = "world_model"
        module.__file__ = "world_model/dynamics.py"
        module.__dict__["__builtins__"] = builtins_dict
        sys.modules["world_model.dynamics"] = module
        package.dynamics = module
        try:
            exec(compile(source, "world_model/dynamics.py", "exec"), module.__dict__)
        except Exception:
            sys.modules.pop("world_model.dynamics", None)
            if getattr(package, "dynamics", None) is module:
                delattr(package, "dynamics")
            if harness_module is not None:
                sys.modules.pop("world_model.harness", None)
                if getattr(package, "harness", None) is harness_module:
                    delattr(package, "harness")
            return None
        return module

    try:
        import importlib
        return importlib.import_module("world_model.dynamics")
    except Exception:
        return None


def _dynamics_attr(module_or_namespace, name: str):
    if isinstance(module_or_namespace, dict):
        return module_or_namespace.get(name)
    return getattr(module_or_namespace, name, None)


def _predict_hypotheses(predict_fn):
    globals_dict = getattr(predict_fn, "__globals__", None)
    if isinstance(globals_dict, dict):
        hypotheses = globals_dict.get("HYPOTHESES")
        if isinstance(hypotheses, dict):
            return hypotheses
    module = getattr(predict_fn, "__module__", None)
    if module:
        try:
            import sys
            module_obj = sys.modules.get(module)
            hypotheses = getattr(module_obj, "HYPOTHESES", None)
            if isinstance(hypotheses, dict):
                return hypotheses
        except Exception:
            return None
    return None


def select_primary_hypothesis(hypotheses: Mapping[str, Any] | None) -> str | None:
    """Return the highest-probability hypothesis name.

    Ties keep insertion order. Malformed entries are ignored; if no numeric
    probability is available, no hypothesis is selected.
    """
    if not hypotheses:
        return None

    best_name: str | None = None
    best_probability: float | None = None
    for name, probability in hypotheses.items():
        if not isinstance(name, str) or not name:
            continue
        try:
            score = float(probability)
        except (TypeError, ValueError):
            continue
        if best_name is None or score > best_probability:
            best_name = name
            best_probability = score
    return best_name


def resolve_hypothesis(
    hypothesis: str | None,
    hypotheses: Mapping[str, Any] | None,
) -> str | None:
    """Resolve an optional hypothesis against a named probability map."""
    if isinstance(hypothesis, str) and not hypothesis:
        return None
        if hypotheses is None:
            return hypothesis
        return hypothesis if hypothesis in hypotheses else None
    return select_primary_hypothesis(hypotheses)


def load_z_predicted(step: int, hypothesis: str | None = None) -> dict | None:
    """Compute ``predict(z_encoded[step-1], h_encoded[step], ...)``.

    ``hypothesis`` may be ``None``. When it is ``None`` or omitted, the helper
    selects the highest-probability entry from the dynamics module's
    ``HYPOTHESES`` map. Passing a hypothesis name computes that specific branch.

    Returns None when the required artifacts, metadata, hypothesis, or dynamics
    program are unavailable, or when predict(...) fails or returns a non-dict.
    """
    if not isinstance(step, int) or step <= 0:
        return None

    z_prev = load_z_encoded(step - 1)
    h_current = load_h_encoded(step)
    if not isinstance(z_prev, dict) or not isinstance(h_current, dict):
        return None

    _open, _import_fn, memory_root = _get_sandboxed_open(return_context=True)
    roots = _artifact_root_candidates(memory_root)
    current_record = _load_step_record_for_prediction(step, roots)
    if not isinstance(current_record, dict):
        return None
    from arc_agi_3.arc_league.step_index import parse_step_record
    step_record = parse_step_record(current_record)
    if step_record is None:
        return None

    level = step_record.level

    action = _normalise_action_name(step_record.action)
    if not action or action == "RESET":
        return None

    constants = load_constants(level)
    if not isinstance(constants, dict) or not constants:
        return None

    dynamics = _load_world_model_dynamics_for_prediction()
    if dynamics is None:
        return None
    hypotheses = _dynamics_attr(dynamics, "HYPOTHESES")
    if not isinstance(hypotheses, dict):
        return None
    selected_hypothesis = resolve_hypothesis(hypothesis, hypotheses)
    if selected_hypothesis is None:
        return None
    predict_fn = _dynamics_attr(dynamics, "predict")
    if not callable(predict_fn):
        return None

    import copy

    from arc_agi_3.arc_league.containers import build_metadata

    metadata = build_metadata(
        step,
        level,
        _load_available_actions_for_prediction(step, current_record, roots),
    )
    try:
        predicted = predict_fn(
            copy.deepcopy(z_prev),
            copy.deepcopy(h_current),
            action,
            constants,
            metadata,
            selected_hypothesis,
        )
    except Exception:
        return None
    if not isinstance(predicted, dict):
        return None
    z_schema = _load_z_schema_for_validation()
    if z_schema is None:
        return predicted
    from arc_agi_3.arc_league.scheme_loader import attach_drift
    return attach_drift(predicted, z_schema)


def _load_z_schema_for_validation():
    """Return the agent's Z_SCHEMA dict, or None when it cannot be loaded."""
    import sys
    obs = sys.modules.get("world_model.observable")
    if obs is not None and "Z_SCHEMA" in obs.__dict__:
        return obs.__dict__["Z_SCHEMA"]
    _open, _import_fn, _memory_root = _get_sandboxed_open(return_context=True)
    try:
        source = _open("world_model/observable.py").read()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    ns: dict = {}
    try:
        exec(compile(source, "world_model/observable.py", "exec"), ns)
    except Exception:
        return None
    return ns.get("Z_SCHEMA")


def evaluate_sub_goals(
    sub_goals: dict,
    z_encoded: dict,
    h_encoded: dict,
    constants: dict,
) -> list[dict]:
    """Run sub-goal checking functions and return per-sub-goal results.

    Call as evaluate_sub_goals(SUB_GOALS, z_encoded, h_encoded, constants).
    Each sub-goal is called as fn(z_encoded, h_encoded, constants).
    Sub-goal results can be (achieved, progress, reason) or (achieved, reason);
    the first docstring line becomes the description.
    Exceptions are reported in that sub-goal's reason as "ERROR: ...".
    Returns [{"name": str, "description": str, "achieved": bool,
    "progress": float, "reason": str}].
    """
    if not isinstance(sub_goals, dict):
        raise TypeError(
            "evaluate_sub_goals expected first argument SUB_GOALS to be a dict"
        )
    non_callable = [name for name, fn in sub_goals.items() if not callable(fn)]
    if non_callable:
        sample = ", ".join(map(str, non_callable[:5]))
        raise TypeError(
            "evaluate_sub_goals expected first argument SUB_GOALS to contain "
            f"callable values; non-callable entries: {sample}"
        )
    if isinstance(h_encoded, dict) and any(callable(fn) for fn in h_encoded.values()):
        raise TypeError(
            "evaluate_sub_goals expected "
            "(SUB_GOALS, z_encoded, h_encoded, constants); "
            "received SUB_GOALS in the third argument."
        )

    results = []
    for name, fn in sub_goals.items():
        description = (fn.__doc__ or "").strip().split("\n")[0]
        try:
            result = fn(z_encoded, h_encoded, constants)
            if len(result) == 3:
                achieved, progress, reason = result
            else:
                achieved, reason = result[0], result[1] if len(result) > 1 else ""
                progress = 1.0 if achieved else 0.0
            results.append({
                "name": name, "description": description, "achieved": achieved,
                "progress": progress, "reason": reason,
            })
        except Exception as e:
            results.append({
                "name": name, "description": description, "achieved": False,
                "progress": 0.0, "reason": f"ERROR: {e}",
            })
    return results


def search(sub_goal: Callable, *, method: str = "beam", budget: SearchBudget | None = None,
           hypothesis: str | None = None,
           predict_fn: Callable | None = None, history_fn: Callable | None = None,
           z_encoded: dict, h_encoded: dict, constants: dict, metadata: dict) -> dict:
    """Plan a verified action sequence toward a sub-goal from the current state.

    Searches the world model (``method``: "beam" | "astar") toward
    ``sub_goal`` — one detector function
    ``fn(z_encoded, h_encoded, constants) -> (achieved, progress, reason)`` — and
    replay-verifies the route. ``predict_fn`` / ``history_fn`` default to the ``predict`` /
    ``history`` in scope (the live simulator's); pass them to search over a model you
    choose. At each state a policy returns the route's first action.

    Returns a result — check ``found`` first:
        ok             bool        the search ran (state and model were usable)
        found          bool        a route reaching the sub-goal was located
        verified       bool|None   the route replays to the sub-goal (None when not found)
        actions        list[str]   the verified sequence from the current state
        depth          int         length of the route explored
        final_progress float       best progress toward the sub-goal reached
        reason         str         "goal_reached" | "frontier_empty" | "budget" | "timeout" | "not_ready"

    Raises when ``predict`` / ``history`` are neither passed nor in scope, and when
    ``method`` is not a known search method — never a guess.
    """
    import sys

    from arc_agi_3.search_methods import resolve_method
    from arc_agi_3.search_methods.base import WorldModelProblem

    if predict_fn is None or history_fn is None:
        scope = sys._getframe(1)
        names = {**scope.f_globals, **scope.f_locals}
        predict_fn = predict_fn or names.get("predict")
        history_fn = history_fn or names.get("history")
    if predict_fn is None or history_fn is None:
        raise ValueError("search needs predict/history: pass predict_fn/history_fn or "
                         "call from a scope that defines `predict` and `history`")

    name = sub_goal.__name__
    problem = WorldModelProblem(
        z_encoded=z_encoded, h_encoded=h_encoded, constants=constants, metadata=metadata,
        history=history_fn, predict=predict_fn, sub_goals={name: sub_goal},
        target=name, hypothesis=hypothesis)
    r = resolve_method(method)(problem, budget=budget)
    return {"ok": r.reason != "not_ready", "found": r.found, "verified": r.verified,
            "actions": list(r.actions), "depth": r.depth, "final_progress": r.final_progress,
            "reason": r.reason}


def simulate(
    history_fn,
    predict_fn,
    policy_fn,
    h_encoded: dict,
    z_prev: dict,
    constants: dict | None = None,
    metadata: dict | None = None,
    sub_goals: dict | None = None,
    depth: int = 10,
    hypothesis: str | None = None,
) -> dict:
    """Roll out one policy through the world model and score sub-goals.

    Each step calls ``policy(z_encoded[K-1], h_encoded[K-1], constants, metadata[K-1]) -> action_chosen[K-1]``,
    then ``history(h_encoded[K-1], z_encoded[K-1], action_chosen[K-1], constants, metadata[K]) -> h_encoded[K]``,
    then ``predict(z_encoded[K-1], h_encoded[K], action_chosen[K-1], constants, metadata[K], hypothesis) -> z_predicted[hypothesis][K]``;
    if ``predict`` exposes a ``hypothesis`` parameter and ``hypothesis`` is
    omitted, the helper selects the highest-probability entry from that
    predictor's ``HYPOTHESES`` map.
    ``z_predicted[hypothesis][K]`` is chained into the next iteration as ``z_encoded[K]``.
    Stops at ``depth`` or when all ``sub_goals`` achieve.
    Returns {
        "ok": bool, "error": str|None, "missing": list[str],
        "actions": list[str], "steps": int, "all_achieved": bool,
        "sub_goals": {name: {"achieved": bool, "step": int|None,
                             "progress": float, "reason": str,
                             "error": str|None}},
        "final_h_encoded": dict|None, "final_z_predicted": dict|None,
    }.
    """
    import copy
    import inspect

    selected_hypothesis = hypothesis

    def _result(
        *,
        ok: bool,
        error: str | None = None,
        missing: list[str] | None = None,
        actions: list | None = None,
        all_achieved: bool = False,
        sub_goals_status: dict | None = None,
        final_h_encoded: dict | None = None,
        final_z_predicted: dict | None = None,
    ) -> dict:
        actions = actions or []
        return {
            "ok": ok,
            "error": error,
            "missing": missing or [],
            "actions": actions,
            "steps": len(actions),
            "all_achieved": all_achieved,
            "sub_goals": sub_goals_status or {},
            "final_h_encoded": final_h_encoded,
            "final_z_predicted": final_z_predicted,
            "hypothesis": selected_hypothesis,
        }

    missing = []
    if metadata is None or not isinstance(metadata, dict):
        missing.append("metadata")
    else:
        if not isinstance(metadata.get("step"), int):
            missing.append("metadata.step")
        if not isinstance(metadata.get("level"), int):
            missing.append("metadata.level")
    if not isinstance(constants, dict) or not constants:
        missing.append("constants")
    if not isinstance(z_prev, dict) or not z_prev:
        missing.append("z_prev")
    if not isinstance(h_encoded, dict) or not h_encoded:
        missing.append("h_encoded")
    try:
        predict_signature = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        predict_signature = None
    predict_uses_hypothesis = (
        predict_signature is not None
        and "hypothesis" in predict_signature.parameters
    )
    if predict_uses_hypothesis and isinstance(hypothesis, str) and not hypothesis:
        selected_hypothesis = None
        missing.append("hypothesis")
    elif predict_uses_hypothesis and selected_hypothesis is None:
        selected_hypothesis = select_primary_hypothesis(
            _predict_hypotheses(predict_fn)
        )
        if selected_hypothesis is None:
            missing.append("hypothesis")
    if missing:
        return _result(
            ok=False,
            error=f"missing required inputs: {', '.join(missing)}",
            missing=missing,
        )

    h_encoded_k = copy.deepcopy(h_encoded)
    z_encoded_k = copy.deepcopy(z_prev)
    actions = []
    sg_status = {}
    all_achieved = False

    for i in range(depth):
        policy_metadata = {**metadata, "step": metadata["step"] + i}
        try:
            action = policy_fn(
                z_encoded_k, h_encoded_k, constants, policy_metadata
            )
        except Exception as e:
            return _result(
                ok=False,
                error=f"policy failed at rollout step {i + 1}: {type(e).__name__}: {e}",
                actions=actions,
                all_achieved=all_achieved,
                sub_goals_status=sg_status,
                final_h_encoded=h_encoded_k,
                final_z_predicted=z_encoded_k,
            )
        actions.append(action)
        step_metadata = {**metadata, "step": metadata["step"] + i + 1}
        try:
            h_encoded_k = history_fn(
                copy.deepcopy(h_encoded_k), copy.deepcopy(z_encoded_k),
                action, constants, step_metadata,
            )
        except Exception as e:
            return _result(
                ok=False,
                error=f"history failed at rollout step {i + 1}: {type(e).__name__}: {e}",
                actions=actions,
                all_achieved=all_achieved,
                sub_goals_status=sg_status,
                final_h_encoded=None,
                final_z_predicted=z_encoded_k,
            )
        if not isinstance(h_encoded_k, dict):
            return _result(
                ok=False,
                error=(
                    f"history returned {type(h_encoded_k).__name__} at "
                    f"rollout step {i + 1}; expected dict"
                ),
                actions=actions,
                all_achieved=all_achieved,
                sub_goals_status=sg_status,
                final_h_encoded=None,
                final_z_predicted=z_encoded_k,
            )
        try:
            predict_args = (
                z_encoded_k,
                h_encoded_k,
                action,
                constants,
                step_metadata,
            )
            if predict_uses_hypothesis:
                predict_args = (*predict_args, selected_hypothesis)
            z_predicted = predict_fn(*predict_args)
        except Exception as e:
            return _result(
                ok=False,
                error=f"predict failed at rollout step {i + 1}: {type(e).__name__}: {e}",
                actions=actions,
                all_achieved=all_achieved,
                sub_goals_status=sg_status,
                final_h_encoded=h_encoded_k,
                final_z_predicted=z_encoded_k,
            )
        if not isinstance(z_predicted, dict):
            return _result(
                ok=False,
                error=(
                    f"predict returned {type(z_predicted).__name__} at "
                    f"rollout step {i + 1}; expected dict"
                ),
                actions=actions,
                all_achieved=all_achieved,
                sub_goals_status=sg_status,
                final_h_encoded=h_encoded_k,
                final_z_predicted=z_encoded_k,
            )
        z_encoded_k = z_predicted  # chain forward
        if sub_goals:
            try:
                sg_results = evaluate_sub_goals(
                    sub_goals, z_encoded_k, h_encoded_k,
                    constants=constants,
                )
            except Exception as e:
                return _result(
                    ok=False,
                    error=(
                        f"sub-goal evaluation failed at rollout step {i + 1}: "
                        f"{type(e).__name__}: {e}"
                    ),
                    actions=actions,
                    all_achieved=all_achieved,
                    sub_goals_status=sg_status,
                    final_h_encoded=h_encoded_k,
                    final_z_predicted=z_encoded_k,
                )
            for r in sg_results:
                sg_name = r["name"]
                if sg_name not in sg_status or not sg_status[sg_name]["achieved"]:
                    reason = str(r.get("reason", ""))
                    error = reason[7:] if reason.startswith("ERROR: ") else None
                    sg_status[sg_name] = {
                        "achieved": r["achieved"],
                        "step": i + 1 if r["achieved"] else None,
                        "progress": r["progress"],
                        "reason": reason,
                        "error": error,
                    }
            if all(sg_status[r["name"]]["achieved"] for r in sg_results):
                all_achieved = True
                break

    return _result(
        ok=True,
        actions=actions,
        all_achieved=all_achieved,
        sub_goals_status=sg_status,
        final_h_encoded=h_encoded_k,
        final_z_predicted=z_encoded_k,
    )


def rollout_actions(
    actions: list[str] | tuple[str, ...],
    *,
    history_fn,
    predict_fn,
    h_start: dict,
    z_start: dict,
    constants: dict,
    metadata: dict,
    sub_goals: dict | None = None,
    hypothesis: str | None = None,
) -> dict:
    """Roll out explicit actions through the world model.

    Per action, calls ``history(h_encoded[K-1], z_encoded[K-1], action_chosen[K-1], constants, metadata[K]) -> h_encoded[K]``
    then ``predict(z_encoded[K-1], h_encoded[K], action_chosen[K-1], constants, metadata[K], hypothesis) -> z_predicted[hypothesis][K]``.
    Args: ``h_start = h_encoded[K0]``, ``z_start = z_encoded[K0]``, ``metadata.step = K0``.
    Hypothesis selection, h/z chaining, and sub-goal scoring follow ``simulate``.

    Invalid inputs return ``ok=False`` with ``missing`` populated before any
    world-model function is called. ``sub_goals=None`` skips sub-goal scoring. If
    ``hypothesis=None``, selects the highest-probability entry from
    ``predict_fn``'s ``HYPOTHESES`` map.

    Returns {
        "ok": bool, "error": str|None, "missing": list[str],
        "actions": list[str], "steps": int, "all_achieved": bool,
        "sub_goals": dict, "final_h_encoded": dict|None,
        "final_z_predicted": dict|None, "hypothesis": str|None,
    }.
    """

    selected_hypothesis = hypothesis

    def _result(
        *,
        ok: bool,
        error: str | None = None,
        missing: list[str] | None = None,
    ) -> dict:
        return {
            "ok": ok,
            "error": error,
            "missing": missing or [],
            "actions": [],
            "steps": 0,
            "all_achieved": False,
            "sub_goals": {},
            "final_h_encoded": None,
            "final_z_predicted": None,
            "hypothesis": selected_hypothesis,
        }

    missing = []
    if (
        actions is None
        or isinstance(actions, (str, bytes))
        or not isinstance(actions, (list, tuple))
    ):
        missing.append("actions")
    if not callable(history_fn):
        missing.append("history_fn")
    if not callable(predict_fn):
        missing.append("predict_fn")
    if not isinstance(h_start, dict) or not h_start:
        missing.append("h_start")
    if not isinstance(z_start, dict) or not z_start:
        missing.append("z_start")
    if not isinstance(constants, dict) or not constants:
        missing.append("constants")
    if metadata is None or not isinstance(metadata, dict):
        missing.append("metadata")
    else:
        if not isinstance(metadata.get("step"), int):
            missing.append("metadata.step")
        if not isinstance(metadata.get("level"), int):
            missing.append("metadata.level")
    if missing:
        return _result(
            ok=False,
            error=f"missing required inputs: {', '.join(missing)}",
            missing=missing,
        )

    action_sequence = tuple(actions)
    index = {"i": 0}

    def fixed_action_policy(_z_encoded, _h_encoded, _constants, _metadata):
        i = index["i"]
        if i >= len(action_sequence):
            raise IndexError("fixed action policy exhausted")
        index["i"] = i + 1
        return action_sequence[i]

    return simulate(
        history_fn,
        predict_fn,
        fixed_action_policy,
        h_encoded=h_start,
        z_prev=z_start,
        constants=constants,
        metadata=metadata,
        sub_goals=sub_goals,
        depth=len(action_sequence),
        hypothesis=hypothesis,
    )


def _state_payload_step(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("step", "_step", "source_step", "gameplay_step"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("step"), int):
        return metadata["step"]
    return None


def run_simulation(
    transition_fn,
    check_complete_fn,
    h_t_initial: dict,
    actions: list[str],
    z_t_initial: dict | None = None,
) -> list[dict]:
    """Roll out an action list through ``transition_fn`` and check completion.

    transition_fn is called as transition(h, z, action) -> h_next; pass it
    as a callable or as a Python source string (executed once to bind
    ``transition``). check_complete_fn (callable or source string, or None)
    is called as check_level_complete(h, z, level_index) -> bool.
    Returns one dict per action:
    [{"step": int, "action": str, "h_t": dict,
      "goal_complete": bool, "error": str|None}].
    A bound error in transition or check halts the rollout and the last
    entry carries the error string in ``error``.
    """
    import copy

    # Compile code strings to resolve their transition callable.
    if isinstance(transition_fn, str):
        ns = {"__builtins__": __builtins__, "copy": copy}
        try:
            exec(compile(transition_fn, "transition.py", "exec"), ns)
            transition_fn = ns.get("transition")
        except Exception as e:
            return [{"step": 0, "action": "", "h_t": {}, "goal_complete": False, "error": str(e)}]

    if transition_fn is None:
        return [{"step": 0, "action": "", "h_t": {}, "goal_complete": False, "error": "transition function is None"}]

    if isinstance(check_complete_fn, str):
        ns = {"__builtins__": __builtins__, "copy": copy}
        try:
            exec(compile(check_complete_fn, "check_complete.py", "exec"), ns)
            check_complete_fn = ns.get("check_level_complete")
        except Exception:
            check_complete_fn = None

    # Run simulation
    h = copy.deepcopy(h_t_initial)
    z = copy.deepcopy(z_t_initial) if z_t_initial else {}
    results = []

    for i, action in enumerate(actions):
        try:
            h = transition_fn(copy.deepcopy(h), copy.deepcopy(z), action)
            goal = False
            if check_complete_fn:
                with contextlib.suppress(Exception):
                    goal = bool(check_complete_fn(h, z, 0))
            results.append({
                "step": i + 1,
                "action": action,
                "h_t": h,
                "goal_complete": goal,
                "error": None,
            })
            if goal:
                break  # Goal reached, stop simulation
        except Exception as e:
            results.append({
                "step": i + 1,
                "action": action,
                "h_t": h,
                "goal_complete": False,
                "error": str(e),
            })
            break  # Error, stop simulation

    return results
