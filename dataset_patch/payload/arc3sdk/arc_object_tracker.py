"""ARC Object Tracker Skill — multi-object tracking + containment detection.

Analyzes the current grid for:
1. Multiple distinct objects (connected components).
2. Object containment (one object inside another).
3. Object movement patterns (velocity estimation from position deltas).
4. Small isolated objects (potential targets for interaction).

Returns (action_id, x, y, source, confidence) tuples for the skill orchestrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from collections import deque

__version__ = "v1-arc-object-tracker-1"

# Per-(game, level) object history: {(game_id, level): {object_id: deque([(x,y)])}}
_OBJ_HISTORY: dict[tuple[str, int], dict[int, deque]] = {}
_MAX_HISTORY = 8
_MAX_OBJECTS = 16


def _label_components(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    """Label connected components (4-connectivity). Returns (labels, count)."""
    g = np.asarray(grid, dtype=np.uint8)
    h, w = g.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    for y in range(h):
        for x in range(w):
            if g[y, x] != bg and labels[y, x] == 0:
                current_label += 1
                # BFS
                queue = deque([(y, x)])
                labels[y, x] = current_label
                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if g[ny, nx] != bg and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((ny, nx))

    return labels, current_label


def _object_centroids(labels: np.ndarray, n_objects: int) -> dict[int, tuple[float, float]]:
    """Compute centroid for each labeled object."""
    centroids: dict[int, tuple[float, float]] = {}
    for oid in range(1, n_objects + 1):
        ys, xs = np.where(labels == oid)
        if len(xs) > 0:
            centroids[oid] = (float(np.mean(xs)), float(np.mean(ys)))
    return centroids


def evaluate_object_tracker(
    grid: np.ndarray,
    bg: int,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Evaluate object tracking patterns on the current frame."""
    proposals: list[tuple[int, int | None, int | None, str, float]] = []
    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return []

        labels, n_objects = _label_components(g, bg)
        if n_objects == 0:
            return []

        centroids = _object_centroids(labels, n_objects)

        # Track history for movement detection
        key = (game_id, level)
        history = _OBJ_HISTORY.setdefault(key, {})

        # Update history
        for oid, (cx, cy) in centroids.items():
            if oid not in history:
                history[oid] = deque(maxlen=_MAX_HISTORY)
            history[oid].append((cx, cy))

        # Bounded: keep only recent objects
        while len(history) > _MAX_OBJECTS:
            history.pop(next(iter(history)))

        # Detect smallest object (potential target)
        obj_sizes = {}
        for oid in range(1, n_objects + 1):
            obj_sizes[oid] = int(np.sum(labels == oid))
        if obj_sizes:
            smallest_oid = min(obj_sizes, key=obj_sizes.get)
            if obj_sizes[smallest_oid] <= 4 and 6 in available:
                cx, cy = centroids[smallest_oid]
                proposals.append((6, int(cx), int(cy), "arc_object_smallest", 0.80))

        # Detect movement (object centroid shifted)
        for oid, hist in history.items():
            if len(hist) >= 2:
                dx = hist[-1][0] - hist[-2][0]
                dy = hist[-1][1] - hist[-2][1]
                if abs(dx) > 1 or abs(dy) > 1:
                    # Object moved: propose clicking ahead of movement
                    if 6 in available:
                        cx, cy = hist[-1]
                        nx = int(cx + dx * 2)
                        ny = int(cy + dy * 2)
                        if 0 <= nx < g.shape[1] and 0 <= ny < g.shape[0]:
                            proposals.append(
                                (6, nx, ny, "arc_object_movement_lead", 0.77))

        # Detect containment (object inside another)
        # Simplified: check if one object's centroid is inside another's bounding box
        for oid_a, (cax, cay) in centroids.items():
            for oid_b, (cbx, cby) in centroids.items():
                if oid_a == oid_b:
                    continue
                # Get bounding boxes
                ys_a, xs_a = np.where(labels == oid_a)
                ys_b, xs_b = np.where(labels == oid_b)
                if len(xs_a) == 0 or len(xs_b) == 0:
                    continue
                # Check if centroid of A is inside bbox of B
                if (xs_b.min() <= cax <= xs_b.max() and
                        ys_b.min() <= cay <= ys_b.max()):
                    if obj_sizes.get(oid_a, 0) < obj_sizes.get(oid_b, 0):
                        if 6 in available:
                            proposals.append(
                                (6, int(cax), int(cay), "arc_object_containment", 0.83))
                        break

    except Exception:
        pass

    return proposals


def reset_tracker() -> None:
    """Clear object history (call on game/level change)."""
    global _OBJ_HISTORY
    _OBJ_HISTORY = {}


def get_status() -> dict[str, Any]:
    return {"version": __version__, "enabled": True, "objects_tracked": len(_OBJ_HISTORY)}
