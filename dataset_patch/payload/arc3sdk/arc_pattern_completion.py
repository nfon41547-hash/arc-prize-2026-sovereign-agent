"""ARC Pattern Completion Skill — row/column completion + symmetry repair.

Analyzes the current grid for:
1. Row completion: a row with a gap that should be filled.
2. Column completion: a column with a gap that should be filled.
3. Horizontal symmetry: mirror-asymmetry that should be repaired.
4. Vertical symmetry: top-bottom asymmetry that should be repaired.
5. Diagonal symmetry: diagonal mirror-asymmetry.

Returns (action_id, x, y, source, confidence) tuples for the skill orchestrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__version__ = "v1-arc-pattern-completion-1"


def evaluate_pattern_completion(
    grid: np.ndarray,
    bg: int,
    available: list[int],
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Evaluate pattern completion on the current frame."""
    proposals: list[tuple[int, int | None, int | None, str, float]] = []
    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return []
        h, w = g.shape
        fg = g != bg
        if not np.any(fg):
            return []

        # 1) Row completion: find rows with internal gaps
        for y in range(h):
            row = g[y]
            fg_mask = row != bg
            if np.sum(fg_mask) >= 2:
                fg_xs = np.where(fg_mask)[0]
                if len(fg_xs) >= 2:
                    gap_start = fg_xs[0]
                    gap_end = fg_xs[-1]
                    gap_size = gap_end - gap_start + 1 - len(fg_xs)
                    if 0 < gap_size <= 3:
                        # Click in the middle of the gap
                        gap_center = (gap_start + gap_end) // 2
                        if 6 in available:
                            proposals.append(
                                (6, gap_center, y, "arc_pattern_row_gap", 0.84))
                        break

        # 2) Column completion: find columns with internal gaps
        for x in range(w):
            col = g[:, x]
            fg_mask = col != bg
            if np.sum(fg_mask) >= 2:
                fg_ys = np.where(fg_mask)[0]
                if len(fg_ys) >= 2:
                    gap_start = fg_ys[0]
                    gap_end = fg_ys[-1]
                    gap_size = gap_end - gap_start + 1 - len(fg_ys)
                    if 0 < gap_size <= 3:
                        gap_center = (gap_start + gap_end) // 2
                        if 6 in available:
                            proposals.append(
                                (6, x, gap_center, "arc_pattern_col_gap", 0.84))
                        break

        # 3) Horizontal symmetry repair
        flipped_h = np.fliplr(g)
        asym_h = (g == bg) & (flipped_h != bg)
        asym_h_idx = np.argwhere(asym_h)
        if len(asym_h_idx) > 0 and len(asym_h_idx) <= 8:
            ay, ax = asym_h_idx[0]
            if 6 in available:
                proposals.append(
                    (6, int(ax), int(ay), "arc_pattern_h_symmetry", 0.86))

        # 4) Vertical symmetry repair
        flipped_v = np.flipud(g)
        asym_v = (g == bg) & (flipped_v != bg)
        asym_v_idx = np.argwhere(asym_v)
        if len(asym_v_idx) > 0 and len(asym_v_idx) <= 8:
            ay, ax = asym_v_idx[0]
            if 6 in available:
                proposals.append(
                    (6, int(ax), int(ay), "arc_pattern_v_symmetry", 0.86))

        # 5) Diagonal symmetry (transpose)
        if h == w:
            transposed = g.T
            asym_d = (g == bg) & (transposed != bg)
            asym_d_idx = np.argwhere(asym_d)
            if 0 < len(asym_d_idx) <= 8:
                ay, ax = asym_d_idx[0]
                if 6 in available:
                    proposals.append(
                        (6, int(ax), int(ay), "arc_pattern_d_symmetry", 0.85))

    except Exception:
        pass

    return proposals


def get_status() -> dict[str, Any]:
    return {"version": __version__, "enabled": True}
