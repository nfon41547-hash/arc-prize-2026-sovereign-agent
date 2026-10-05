"""ARC Color Transform Skill — detects and proposes color permutation/toggle actions.

Analyzes the current grid for:
1. Color swap patterns: two colors consistently exchange positions.
2. Color toggle: a color appears/disappears periodically.
3. Color completion: a partial color fill that should be extended.

Returns (action_id, x, y, source, confidence) tuples for the skill orchestrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__version__ = "v1-arc-color-transform-1"


def evaluate_color_transform(
    grid: np.ndarray,
    bg: int,
    available: list[int],
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Evaluate color transformation patterns on the current frame."""
    proposals: list[tuple[int, int | None, int | None, str, float]] = []
    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return []
        h, w = g.shape
        fg = g != bg
        if not np.any(fg):
            return []

        # Detect rare colors (potential targets for color completion)
        vals, counts = np.unique(g[fg], return_counts=True)
        if len(vals) < 2:
            return []

        # Sort by count ascending (rarest first)
        order = np.argsort(counts, kind="stable")
        rare_color = int(vals[order[0]])
        rare_count = int(counts[order[0]])

        # If rare color is very sparse, propose clicking its centroid
        if rare_count <= max(4, (h * w) // 100):
            ys, xs = np.where(g == rare_color)
            if len(xs) > 0:
                cx, cy = int(np.median(xs)), int(np.median(ys))
                if 6 in available:
                    proposals.append((6, cx, cy, "arc_color_rare_centroid", 0.78))

        # Detect color boundary (adjacent different colors -> potential swap)
        # Look for 2x2 blocks with diagonal color pattern
        for y in range(0, h - 1, 2):
            for x in range(0, w - 1, 2):
                block = g[y:y+2, x:x+2]
                if len(np.unique(block)) == 2:
                    # Diagonal pattern: [a,b] / [b,a]
                    if block[0, 0] == block[1, 1] and block[0, 1] == block[1, 0]:
                        if block[0, 0] != block[0, 1]:
                            # Click center of the block to trigger swap
                            cx, cy = x + 1, y + 1
                            if 6 in available:
                                proposals.append(
                                    (6, cx, cy, "arc_color_diagonal_swap", 0.82))
                            break
            if proposals:
                break

        # Detect color gradient (monotonic color change -> completion)
        # Check rows for monotonic non-bg sequences
        for y in range(h):
            row = g[y]
            fg_mask = row != bg
            if np.sum(fg_mask) >= 3:
                fg_vals = row[fg_mask]
                if len(np.unique(fg_vals)) >= 2:
                    # Check if values are monotonically increasing/decreasing
                    diffs = np.diff(fg_vals)
                    if np.all(diffs >= 0) or np.all(diffs <= 0):
                        # Click at the end of the gradient
                        last_fg_x = np.where(fg_mask)[0][-1]
                        if 6 in available:
                            proposals.append(
                                (6, int(last_fg_x), y, "arc_color_gradient_end", 0.76))
                        break

    except Exception:
        pass

    return proposals


def get_status() -> dict[str, Any]:
    return {"version": __version__, "enabled": True}
