"""Deterministic ARC-AGI-3 principle detectors.

Reverse-engineered from ``arc3solver.pt`` (stormchaser/arc3solver):

    principle_init = {
        'object':   0.40,
        'fill':     0.30,
        'color':    0.22,
        'symmetry': 0.18,
        'toggle':   0.12,
    }

    action_family_init = {           # (principle, action_family) -> prior
        ('object',   6): 0.60,
        ('fill',     6): 0.35,
        ('color',    6): 0.18,
        ('symmetry', 0): 0.25,
        ('symmetry', 1): 0.25,
        ('color',    4): 0.30,
        ('fill',     5): 0.34,
        ('object',   3): 0.16,
        ('toggle',   5): 0.20,
    }

The arc3solver architecture had a principle priors head; we replace the learned
priors with *deterministic detectors* that measure evidence for each principle
directly from the current 64x64 frame.  Each detector returns a confidence in
[0, 1] that the principle is the "active" reasoning mode of the level, which is
then fused with the model's priors to score candidate actions.

Design rules:
    * Numpy only, no torch — usable in the sandbox / notebook / worker alike.
    * Every detector must be cheap (< 1 ms typical).
    * Nothing here may raise.
"""

from __future__ import annotations


import numpy as np

from . import grid as _grid

# Priors reverse-engineered from arc3solver.pt.
PRINCIPLE_PRIOR: dict[str, float] = {
    "object": 0.40,
    "fill": 0.30,
    "color": 0.22,
    "symmetry": 0.18,
    "toggle": 0.12,
}

# (principle, action_family) -> prior.  action_family == GameAction id used by
# the arc3solver action model (0..6).  6 == complex click (x, y).
ACTION_FAMILY_PRIOR: dict[tuple[str, int], float] = {
    ("object", 6): 0.60,
    ("fill", 6): 0.35,
    ("color", 6): 0.18,
    ("symmetry", 0): 0.25,
    ("symmetry", 1): 0.25,
    ("color", 4): 0.30,
    ("fill", 5): 0.34,
    ("object", 3): 0.16,
    ("toggle", 5): 0.20,
}

# Map GameAction id -> "family bucket" used above.  6 == click with coords.
ACTION_FAMILY = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

PRINCIPLES = ("object", "fill", "color", "symmetry", "toggle")


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------


def detect_object(grid: np.ndarray) -> float:
    """Evidence the level is about discrete objects.

    Strong signal: several well-separated, medium-size monochrome blobs.
    """
    g = np.asarray(grid, dtype=np.uint8)
    bg = _grid.background(g)
    objs = _grid.objects(g, bg)
    if not objs:
        return 0.0
    # size distribution of objects
    sizes = np.array([m.sum() for m in objs], dtype=np.float64)
    total_fg = g.size - (g == bg).sum()
    if total_fg == 0:
        return 0.0
    frac = sizes.sum() / total_fg
    # prefer a handful of solid objects (1..6), not a noisy scatter
    n = len(sizes)
    if n > 24:
        return 0.15
    solid = float(np.mean(sizes > 4))
    var = float(np.std(sizes) / (np.mean(sizes) + 1e-9))
    score = 0.35 * frac + 0.45 * solid
    if 1 <= n <= 8:
        score += 0.2
    if var < 0.75:
        score += 0.1
    return float(np.clip(score, 0.0, 1.0))


def detect_fill(grid: np.ndarray) -> float:
    """Evidence the level wants interior filling.

    Strong signal: enclosed background holes surrounded by foreground.
    """
    g = np.asarray(grid, dtype=np.uint8)
    bg = _grid.background(g)
    hs = _grid.holes(g, bg)
    if not hs:
        # weak signal: large contiguous monochrome regions (candidate canvas)
        return 0.0
    hole_px = sum(int(h.sum()) for h in hs)
    total = g.size
    frac = hole_px / total
    score = min(1.0, frac * 3.0) * 0.8 + 0.2 * min(1.0, len(hs) / 4.0)
    return float(np.clip(score, 0.0, 1.0))


def detect_color(grid: np.ndarray) -> float:
    """Evidence the level is about color semantics (matching / recolor).

    Strong signal: many distinct colors with balanced use.
    """
    g = np.asarray(grid, dtype=np.uint8)
    bg = _grid.background(g)
    ch = _grid.color_hist(g, bg)
    if not ch:
        return 0.0
    n_colors = len(ch)
    vals = np.array(list(ch.values()), dtype=np.float64)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    entropy = -float((p * np.log(p + 1e-12)).sum())
    max_ent = np.log(max(2, n_colors))
    norm_ent = entropy / max_ent if max_ent > 0 else 0.0
    score = 0.5 * norm_ent
    if n_colors >= 3:
        score += 0.3
    if n_colors >= 5:
        score += 0.2
    return float(np.clip(score, 0.0, 1.0))


def detect_symmetry(grid: np.ndarray) -> float:
    """Evidence the level has symmetric structure to exploit."""
    g = np.asarray(grid, dtype=np.uint8)
    sym = _grid.symmetries(g)
    any_sym = any(sym.values())
    if not any_sym:
        return 0.0
    n = sum(1 for v in sym.values() if v)
    return float(np.clip(0.4 + 0.15 * (n - 1), 0.0, 1.0))


def detect_toggle(grid: np.ndarray, prev: np.ndarray | None = None) -> float:
    """Evidence the level toggles cell state (click flips color).

    Uses (a) checkerboard-like alternation in space, (b) large delta between
    consecutive frames (toggles cause big flips).
    """
    g = np.asarray(grid, dtype=np.uint8)
    _bg = _grid.background(g)
    score = 0.0
    # spatial alternation signal: fraction of orthogonal neighbors differing
    sub = g[::2, ::2]  # subsample for speed
    h, w = sub.shape
    if h > 1 and w > 1:
        horiz = float(np.mean(sub[:, :-1] != sub[:, 1:]))
        vert = float(np.mean(sub[:-1, :] != sub[1:, :]))
        alt = (horiz + vert) / 2.0
        # checkerboard-ish regions -> alt near 0.5..0.8 (not 0, not 1)
        score += 0.5 * float(np.clip(1.0 - abs(alt - 0.6) / 0.6, 0.0, 1.0))
    if prev is not None and prev.shape == g.shape:
        d = _grid.delta_ratio(prev, g)
        # a real toggle flips a moderate fraction of the canvas
        if 0.05 < d < 0.9:
            score += 0.5 * min(1.0, d * 3.0)
    return float(np.clip(score, 0.0, 1.0))


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def detect_all(grid: np.ndarray, prev: np.ndarray | None = None) -> dict[str, float]:
    """Run every detector; return raw confidence per principle."""
    out = {
        "object": detect_object(grid),
        "fill": detect_fill(grid),
        "color": detect_color(grid),
        "symmetry": detect_symmetry(grid),
        "toggle": detect_toggle(grid, prev),
    }
    return out


def principle_scores(grid: np.ndarray, prev: np.ndarray | None = None) -> dict[str, float]:
    """Posterior principle weights = prior x detector confidence, renormalised.

    Returns dict {principle: probability} summing to 1 (or zeros if no signal).
    """
    det = detect_all(grid, prev)
    raw = {p: PRINCIPLE_PRIOR[p] * det[p] for p in PRINCIPLES}
    s = sum(raw.values())
    if s <= 0:
        # fall back to pure priors when no detector fires
        return dict(PRINCIPLE_PRIOR)
    return {p: raw[p] / s for p in PRINCIPLES}


def action_scores(
    grid: np.ndarray,
    available: list[int],
    prev: np.ndarray | None = None,
) -> dict[int, float]:
    """Score each available GameAction id by principle compatibility.

    Combines (a) the arc3solver action-family priors given the active
    principle posterior, and (b) a small uniform floor so no legal action is
    excluded outright.
    """
    post = principle_scores(grid, prev)
    out: dict[int, float] = {}
    floor = 0.02
    for act in available:
        fam = ACTION_FAMILY.get(int(act), int(act))
        s = 0.0
        for p, w in post.items():
            s += w * ACTION_FAMILY_PRIOR.get((p, fam), 0.0)
        out[int(act)] = s + floor
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def best_action(grid: np.ndarray, available: list[int], prev: np.ndarray | None = None) -> tuple[int | None, float]:
    """Highest-scoring legal action id (or (None, 0.0))."""
    scores = action_scores(grid, available, prev)
    if not scores:
        return None, 0.0
    best = max(scores, key=scores.get)
    return best, scores[best]


def report(grid: np.ndarray, prev: np.ndarray | None = None) -> dict[str, object]:
    """One-call digest for the sandbox / agent."""
    g = np.asarray(grid, dtype=np.uint8)
    bg = _grid.background(g)
    det = detect_all(g, prev)
    post = principle_scores(g, prev)
    objs = _grid.objects(g, bg)
    holes = _grid.holes(g, bg)
    ch = _grid.color_hist(g, bg)
    sym = _grid.symmetries(g)
    return {
        "background": bg,
        "shape": list(g.shape),
        "n_objects": len(objs),
        "object_px": [int(o.sum()) for o in objs][:8],
        "n_holes": len(holes),
        "colors": ch,
        "symmetry": sym,
        "detector": det,
        "posterior": post,
        "top_principle": max(post, key=post.get),
    }
