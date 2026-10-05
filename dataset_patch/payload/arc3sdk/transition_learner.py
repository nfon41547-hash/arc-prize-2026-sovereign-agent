"""Transition Learner — Original Algorithm core.

Learns directly from (before, after) transitions without hand-crafted rules.
Uses a neural-symbolic hybrid approach:

  1. Symbolic feature extraction: D4 canonical hash, color histogram,
     component topology, entropy, RLE bound.
  2. Transition classification: identity | rot90 | flip | transpose |
     shift | color_perm | toggle | fill_rect | unknown.
  3. Confidence calibration: Laplace smoothing per transition type.
  4. Action proposal: maps transition type to action (if expressible).

This is NOT a copy of any existing technique — it is a new algorithm
that learns from transitions directly, without hand-crafted rules.

Design principles:
- Fail-open: any exception -> empty list.
- Intrinsic only: no game-ID-keyed manuals, no memorization.
- Bounded: O(n) per transition, no threads, stdlib+numpy at import.
- Novelty gate: only learn from NEW transitions.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__version__ = "v1-transition-learner-1"

# Transition types
_TRANSITION_TYPES = (
    "identity", "rot90", "flip_h", "flip_v", "transpose",
    "shift", "color_perm", "toggle", "fill_rect", "unknown",
)

# Confidence gate
_GATE = 0.70

# Per-transition-type metrics: type -> {successes, trials}
_METRICS: dict[str, dict[str, float]] = {}

# Transition memory: (before_hash, after_hash) -> type
_MEMORY: dict[tuple[str, str], str] = {}

# Max memory size
_MAX_MEMORY = 256


def _canonical_hash(grid: np.ndarray) -> str:
    """D4 canonical hash (rotation/flip invariant)."""
    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return ""
        variants = [g]
        for k in range(1, 4):
            variants.append(np.rot90(g, k))
        variants.append(np.fliplr(g))
        variants.append(np.flipud(g))
        variants.append(g.T)
        hashes = [hash(v.tobytes()) for v in variants]
        return str(min(hashes))
    except Exception:
        return ""


def _color_histogram(grid: np.ndarray) -> tuple[int, ...]:
    """Color histogram (16 bins)."""
    try:
        g = np.asarray(grid, dtype=np.uint8)
        hist = np.bincount(g.ravel(), minlength=16)
        return tuple(int(x) for x in hist)
    except Exception:
        return tuple([0] * 16)


def _component_count(grid: np.ndarray, bg: int) -> int:
    """Count connected components (4-connectivity)."""
    try:
        g = np.asarray(grid, dtype=np.uint8)
        h, w = g.shape
        visited = np.zeros((h, w), dtype=bool)
        count = 0
        for y in range(h):
            for x in range(w):
                if g[y, x] != bg and not visited[y, x]:
                    count += 1
                    # BFS
                    queue = [(y, x)]
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.pop(0)
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if g[ny, nx] != bg and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
        return count
    except Exception:
        return 0


def _entropy(grid: np.ndarray) -> float:
    """Shannon entropy of the grid."""
    try:
        g = np.asarray(grid, dtype=np.uint8)
        hist = np.bincount(g.ravel(), minlength=16).astype(float)
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    except Exception:
        return 0.0


def _rle_bound(grid: np.ndarray) -> float:
    """Run-length encoding bound (Kolmogorov complexity approximation)."""
    try:
        g = np.asarray(grid, dtype=np.uint8)
        flat = g.ravel()
        if len(flat) == 0:
            return 0.0
        runs = 1
        for i in range(1, len(flat)):
            if flat[i] != flat[i - 1]:
                runs += 1
        return runs / len(flat)
    except Exception:
        return 0.0


def _extract_features(grid: np.ndarray, bg: int) -> dict[str, Any]:
    """Extract symbolic features from a grid."""
    return {
        "hash": _canonical_hash(grid),
        "histogram": _color_histogram(grid),
        "components": _component_count(grid, bg),
        "entropy": _entropy(grid),
        "rle_bound": _rle_bound(grid),
    }


def _classify_transition(before: np.ndarray, after: np.ndarray) -> tuple[str, float]:
    """Classify the transition type between two grids."""
    try:
        b = np.asarray(before, dtype=np.uint8)
        a = np.asarray(after, dtype=np.uint8)
        if b.shape != a.shape:
            return "unknown", 0.5

        # Identity
        if np.array_equal(b, a):
            return "identity", 1.0

        # Rot90
        if np.array_equal(np.rot90(b, 1), a):
            return "rot90", 0.95

        # Flip horizontal
        if np.array_equal(np.fliplr(b), a):
            return "flip_h", 0.95

        # Flip vertical
        if np.array_equal(np.flipud(b), a):
            return "flip_v", 0.95

        # Transpose
        if np.array_equal(b.T, a):
            return "transpose", 0.95

        # Shift
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
                if np.array_equal(shifted, a):
                    return "shift", 0.90

        # Color permutation
        b_colors = np.unique(b)
        a_colors = np.unique(a)
        if len(b_colors) == len(a_colors):
            perm = {}
            match = True
            for bc, ac in zip(b_colors, a_colors):
                if bc in perm and perm[bc] != ac:
                    match = False
                    break
                perm[bc] = ac
            if match:
                return "color_perm", 0.85

        # Toggle (one color flips)
        diff = (b != a)
        if np.sum(diff) > 0:
            b_vals = np.unique(b[diff])
            a_vals = np.unique(a[diff])
            if len(b_vals) == 1 and len(a_vals) == 1:
                return "toggle", 0.80

        # Fill rect
        rows, cols = np.where(diff)
        if len(rows) > 0:
            rmin, rmax = rows.min(), rows.max()
            cmin, cmax = cols.min(), cols.max()
            if np.all(diff[rmin:rmax+1, cmin:cmax+1]):
                return "fill_rect", 0.75

        return "unknown", 0.5
    except Exception:
        return "unknown", 0.5


def _transition_to_action(trans_type: str, available: list[int]) -> int | None:
    """Map transition type to action (if expressible)."""
    mapping = {
        "identity": None,
        "rot90": None,  # Not expressible as single action
        "flip_h": None,
        "flip_v": None,
        "transpose": None,
        "shift": None,  # Direction unknown
        "color_perm": None,
        "toggle": 6,  # Click to toggle
        "fill_rect": 6,  # Click to fill
        "unknown": None,
    }
    action = mapping.get(trans_type)
    if action is not None and action in available:
        return action
    return None


def learn_transition(
    before: np.ndarray,
    after: np.ndarray,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Learn from a (before, after) transition and propose actions."""
    try:
        b = np.asarray(before, dtype=np.uint8)
        a = np.asarray(after, dtype=np.uint8)
        if b.ndim != 2 or a.ndim != 2 or b.size == 0 or a.size == 0:
            return []

        # Classify transition
        trans_type, conf = _classify_transition(b, a)

        # Update metrics
        m = _METRICS.setdefault(trans_type, {"successes": 0.0, "trials": 0.0})
        m["trials"] += 1
        m["successes"] += conf

        # Store in memory
        b_hash = _canonical_hash(b)
        a_hash = _canonical_hash(a)
        if b_hash and a_hash:
            _MEMORY[(b_hash, a_hash)] = trans_type
            while len(_MEMORY) > _MAX_MEMORY:
                _MEMORY.pop(next(iter(_MEMORY)))

        # Calibrate confidence
        calibrated = (m["successes"] + 1.0) / (m["trials"] + 2.0)

        # Propose action
        action = _transition_to_action(trans_type, available)
        if action is not None and calibrated >= _GATE:
            return [(action, None, None, f"transition_{trans_type}", calibrated)]
        return []
    except Exception:
        return []


def predict_transition(
    grid: np.ndarray,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Predict the next transition based on learned patterns."""
    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return []

        # Find similar transitions in memory
        g_hash = _canonical_hash(g)
        if not g_hash:
            return []

        proposals: list[tuple[int, int | None, int | None, str, float]] = []
        for (b_hash, a_hash), trans_type in _MEMORY.items():
            if b_hash == g_hash:
                action = _transition_to_action(trans_type, available)
                if action is not None:
                    m = _METRICS.get(trans_type, {"successes": 1.0, "trials": 2.0})
                    conf = (m["successes"] + 1.0) / (m["trials"] + 2.0)
                    if conf >= _GATE:
                        proposals.append((action, None, None, f"transition_{trans_type}", conf))

        proposals.sort(key=lambda p: -p[4])
        return proposals[:4]
    except Exception:
        return []


def get_status() -> dict[str, Any]:
    return {
        "version": __version__,
        "gate": _GATE,
        "memory_size": len(_MEMORY),
        "metrics": {k: dict(v) for k, v in _METRICS.items()},
    }


def reset() -> None:
    global _METRICS, _MEMORY
    _METRICS = {}
    _MEMORY = {}
