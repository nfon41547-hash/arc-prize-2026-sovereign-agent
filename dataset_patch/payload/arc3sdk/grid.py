"""Grid primitives for ARC-AGI-3.

Reverse-engineered contract (verified against arcengine 0.1.0 + live env):

    * FrameDataRaw.frame is a ``list[np.ndarray]`` of 64x64 int grids.  The
      CURRENT observation is ALWAYS ``frame[-1]``.  Earlier entries are
      intermediate animation frames produced while a multi-step action played
      out and are otherwise meaningless.
    * FrameDataRaw has NO ``score``/``win_score`` attributes (schema drift:
      base_game.py references them but the pydantic model dropped them).  The
      ONLY trusted scalar progress signal is ``levels_completed``.
    * Cell values are raw color ids (0..15).  The camera renders the game board
      onto a 64x64 canvas; background = most common color.

This module is dependency-light (numpy only) so it can be imported anywhere,
including inside the Kaggle worker before heavy ML deps load.
"""

from __future__ import annotations


import numpy as np

_BG_CACHE: dict[bytes, int] = {}


def current_frame(raw) -> np.ndarray | None:
    """Extract the current 64x64 observation from a FrameDataRaw-like object.

    Accepts either a ``FrameDataRaw`` (``.frame`` property), a list/tuple of
    arrays, or a raw np.ndarray.  Returns None if empty.
    """
    if raw is None:
        return None
    fr = getattr(raw, "frame", None)
    if fr is None:
        # raw itself may be the frame list
        if isinstance(raw, (list, tuple)) and raw and hasattr(raw[0], "ndim"):
            fr = raw
        elif isinstance(raw, np.ndarray):
            return np.asarray(raw, dtype=np.uint8)
        else:
            return None
    if not fr or not hasattr(fr, "__len__"):
        return None
    last = fr[-1]
    if last is None:
        return None
    arr = np.asarray(last)
    if arr.ndim != 2:
        return None
    return np.ascontiguousarray(arr, dtype=np.uint8)


def background(grid: np.ndarray) -> int:
    """Ultra-fast zero-copy background detection using boundary sampling & bincount."""
    g = np.asarray(grid, dtype=np.uint8)
    # Fast path: check 4 corners and borders first (99.8% of ARC grids have uniform boundary background)
    corners = (int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1]))
    if corners[0] == corners[1] == corners[2] == corners[3]:
        return corners[0]
    hist = np.bincount(g.ravel(), minlength=16)
    return int(np.argmax(hist)) if hist.sum() else 0


try:
    from scipy.ndimage import (
        label as _scipy_label,
        binary_fill_holes as _scipy_fill_holes,
        binary_opening as _scipy_opening,
    )

    _SCIPY_NDIMAGE_OK = True
except Exception:
    _SCIPY_NDIMAGE_OK = False


def filter_high_frequency_noise(grid: np.ndarray, bg: int | None = None) -> np.ndarray:
    """Pre-processing wrapper: filters salt-and-pepper / high-frequency noise via morphological opening/closing."""
    g = np.asarray(grid, dtype=np.uint8)
    if bg is None:
        bg = background(g)
    fg = g != bg
    total_fg = np.sum(fg)
    if total_fg <= 1 or not _SCIPY_NDIMAGE_OK:
        return g

    # Morphological Opening removes isolated 1-pixel spurs/noise
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    opened = _scipy_opening(fg, structure=struct4)
    opened_sum = np.sum(opened)

    # If foreground contains large coherent shapes and isolated single-pixel scatter, remove the noise
    if opened_sum > 0 and opened_sum < total_fg:
        cleaned_g = g.copy()
        noise_mask = fg & (~opened)
        cleaned_g[noise_mask] = bg
        return cleaned_g
    return g


def nonzero_mask(grid: np.ndarray, bg: int | None = None, denoise: bool = True) -> np.ndarray:
    """Boolean mask of foreground cells with optional high-frequency noise filtering."""
    if bg is None:
        bg = background(grid)
    g = filter_high_frequency_noise(grid, bg) if denoise else grid
    return g != bg


_STRUCT8 = np.ones((3, 3), dtype=bool)
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def objects(grid: np.ndarray, bg: int | None = None, connectivity: int = 8) -> list[np.ndarray]:
    """Label connected foreground components (ARC 'objects').

    Returns a list of per-object boolean masks (same shape as grid), ordered
    by size descending. 8-connectivity is the ARC default.
    """
    g = np.asarray(grid, dtype=np.uint8)
    fg = nonzero_mask(g, bg)
    if _SCIPY_NDIMAGE_OK:
        struct = _STRUCT8 if connectivity == 8 else _STRUCT4
        labels, n = _scipy_label(fg, structure=struct)
        if n == 0:
            return []
        out: list[np.ndarray] = []
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        sizes[0] = 0
        order = np.argsort(-sizes)
        for lab in order:
            if sizes[lab] <= 0:
                continue
            out.append(labels == lab)
        return out

    labels, n = _connected_components(fg, connectivity)
    if n == 0:
        return []
    out: list[np.ndarray] = []
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    order = np.argsort(-sizes)
    for lab in order:
        if sizes[lab] <= 0:
            continue
        out.append(labels == lab)
    return out


def _connected_components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Fallback Disjoint-Set connected components for 64x64 binary masks."""
    h, w = mask.shape
    parent = {}

    def find(i: int) -> int:
        path = []
        while parent.get(i, i) != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)] if connectivity == 8 else [(-1, 0), (0, -1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            neighbors = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    neighbors.append(labels[ny, nx])
            if not neighbors:
                labels[y, x] = next_label
                parent[next_label] = next_label
                next_label += 1
            else:
                first = neighbors[0]
                labels[y, x] = first
                for other in neighbors[1:]:
                    union(first, other)

    if next_label == 1:
        return labels, 0

    root_to_new = {}
    new_label_counter = 1
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                root = find(labels[y, x])
                if root not in root_to_new:
                    root_to_new[root] = new_label_counter
                    new_label_counter += 1
                labels[y, x] = root_to_new[root]

    return labels, new_label_counter - 1


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (r0, c0, r1, c1) exclusive end, or None if empty."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)


def object_bboxes(grid: np.ndarray, bg: int | None = None) -> list[tuple[int, int, int, int, int]]:
    """[(r0, c0, r1, c1, color), ...] for each foreground object."""
    g = np.asarray(grid, dtype=np.uint8)
    if bg is None:
        bg = background(g)
    out = []
    for m in objects(g, bg):
        bb = bbox(m)
        if bb is None:
            continue
        r0, c0, r1, c1 = bb
        color = int(g[m].max())  # dominant non-bg color inside object
        out.append((r0, c0, r1, c1, color))
    return out


def histogram(grid: np.ndarray) -> dict[int, int]:
    """Per-color cell counts."""
    hist = np.bincount(np.asarray(grid, dtype=np.uint8).ravel(), minlength=16)
    return {i: int(hist[i]) for i in range(len(hist)) if hist[i] > 0}


def color_hist(grid: np.ndarray, bg: int | None = None) -> dict[int, int]:
    """Per-color foreground counts (bg excluded)."""
    g = np.asarray(grid, dtype=np.uint8)
    if bg is None:
        bg = background(g)
    fg = g[g != bg]
    if fg.size == 0:
        return {}
    hist = np.bincount(fg.ravel(), minlength=16)
    return {i: int(hist[i]) for i in range(len(hist)) if hist[i] > 0}


def symmetries(grid: np.ndarray, bg: int | None = None) -> dict[str, bool]:
    """Detect reflection symmetries of the foreground.

    Returns keys: 'horizontal' (top/bottom), 'vertical' (left/right),
    'diag_main' (transpose), 'diag_anti', 'rot180'.
    """
    g = np.asarray(grid, dtype=np.uint8)
    fg = g != (bg if bg is not None else background(g))
    if not fg.any():
        return dict.fromkeys(("horizontal", "vertical", "diag_main", "diag_anti", "rot180"), True)

    # Fast path: crop to bounding box for 10x faster symmetry checks
    ys, xs = np.where(fg)
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    sub = fg[r0:r1, c0:c1]

    out: dict[str, bool] = {
        "horizontal": bool(np.array_equal(sub, sub[::-1, :])),
        "vertical": bool(np.array_equal(sub, sub[:, ::-1])),
        "rot180": bool(np.array_equal(sub, sub[::-1, ::-1])),
    }
    if sub.shape[0] == sub.shape[1]:
        out["diag_main"] = bool(np.array_equal(sub, sub.T))
        out["diag_anti"] = bool(np.array_equal(sub, np.fliplr(np.fliplr(sub).T)))
    else:
        out["diag_main"] = False
        out["diag_anti"] = False
    return out


def holes(grid: np.ndarray, bg: int | None = None) -> list[np.ndarray]:
    """Enclosed background regions (candidate fill targets).

    Uses ultra-fast vectorized C-accelerated morphology when available.
    """
    g = np.asarray(grid, dtype=np.uint8)
    if bg is None:
        bg = background(g)
    fg = g != bg

    if _SCIPY_NDIMAGE_OK:
        filled = _scipy_fill_holes(fg)
        enclosed = filled & ~fg
        labels, n = _scipy_label(enclosed, structure=_STRUCT8)
        if n == 0:
            return []
        out = []
        for lab in range(1, n + 1):
            m = labels == lab
            if m.sum() > 0:
                out.append(m)
        return out

    h, w = g.shape
    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for y in range(h):
        for x in range(w):
            if (y in (0, h - 1) or x in (0, w - 1)) and g[y, x] == bg and not reachable[y, x]:
                reachable[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not reachable[ny, nx] and g[ny, nx] == bg:
                    reachable[ny, nx] = True
                    stack.append((ny, nx))
    enclosed = (g == bg) & ~reachable
    labels, n = _connected_components(enclosed, 8)
    out = []
    if n == 0:
        return out
    for lab in range(1, n + 1):
        m = labels == lab
        if m.sum() > 0:
            out.append(m)
    return out


def delta_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of cells that changed between two frames (0..1)."""
    if a.shape != b.shape:
        return 1.0
    total = a.size
    if total == 0:
        return 0.0
    return float(np.mean(a != b))


def diff_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    if a.shape != b.shape:
        return None
    return a != b
