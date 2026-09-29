"""Exact object segmentation (v24-seg-1): topological containment + adjacency.

Behavior spec reimplemented from the observed upstream contract
(nodes with translation-invariant hash, exact nesting children,
edge adjacency); no upstream text is vendored (upstream repo carries
no verifiable license file, so this module is clean-room code).

Differences from fusion_supremacy._type_signature: that hash folds D4
(rotations/flips share identity); seg hash here is translation-only
(rotated shapes are DIFFERENT objects). Both are kept: fusion `sig`
stays the stable ledger key, seg hash powers cross-frame matching and
exact containment.

Upstream uses ARC color chars; this module keeps integer colors (our
grids are uint8) to avoid importing foreign mappings.

stdlib + numpy at import. Bounded (64x64, _MAX_COMP). Never raises.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

__version__ = "v24-seg-1"

_MAX_COMP = 128
_MAX_DIM = 64

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
       (1, 1), (1, 0), (1, -1), (0, -1))
_CW_INDEX = {off: i for i, off in enumerate(_CW)}


def _trace_contour(cells: set, start: tuple[int, int]) -> list:
    """Moore-neighbour outer-perimeter trace, clockwise. Bounded."""
    try:
        if len(cells) == 1:
            return [start]
        contour = [start]
        b = start
        prev = (start[0], start[1] - 1)
        second = None
        for _ in range(8 * len(cells) + 16):
            idx = _CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
            nxt = None
            new_prev = prev
            for k in range(1, 9):
                off = _CW[(idx + k) % 8]
                cand = (b[0] + off[0], b[1] + off[1])
                if cand in cells:
                    nxt = cand
                    back = _CW[(idx + k - 1) % 8]
                    new_prev = (b[0] + back[0], b[1] + back[1])
                    break
            if nxt is None:
                break
            if second is None:
                second = nxt
            elif b == start and nxt == second:
                break
            contour.append(nxt)
            prev, b = new_prev, nxt
        if len(contour) > 1 and contour[-1] == contour[0]:
            contour.pop()
        return contour
    except Exception:
        return [start]


def _corners(contour: list) -> list:
    """Reduce a contour loop to direction-change vertices."""
    try:
        if len(contour) <= 2:
            return list(contour)
        m = len(contour)
        out = []
        for i in range(m):
            prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]
            if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
                out.append(cur)
        return out
    except Exception:
        return list(contour)[:4]


def _shape_hash(cells: set, color: int) -> str:
    """Translation-invariant identity: color + bbox-normalized shape."""
    try:
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        return hashlib.sha1(repr((int(color), norm)).encode()).hexdigest()[:16]
    except Exception:
        return "err"


def segment(grid: Any) -> dict[str, Any]:
    """Segment a grid into topologically-described objects. Never raises.

    Returns {"nodes": [{id,color,hash,pixels,boundary,children}],
    "adjacency_list": [[i,j],...]}. Node ids are reading order.
    children = exact topological nesting (complement flood-fill), NOT
    bbox approximation: a U-shape's concavity contents are NOT children.
    """
    empty: dict[str, Any] = {"nodes": [], "adjacency_list": []}
    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size == 0:
            return empty
        h, w = int(g.shape[0]), int(g.shape[1])
        if h > _MAX_DIM or w > _MAX_DIM:
            return empty
        comp_id = np.full((h, w), -1, dtype=np.int32)
        comps: list[dict[str, Any]] = []
        for sr in range(h):
            for sc in range(w):
                if comp_id[sr, sc] != -1:
                    continue
                if len(comps) >= _MAX_COMP:
                    break
                value = int(g[sr, sc])
                cells: set = set()
                stack = [(sr, sc)]
                comp_id[sr, sc] = len(comps)
                while stack:
                    r, c = stack.pop()
                    cells.add((r, c))
                    for dr, dc in _ORTH:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and comp_id[nr, nc] == -1
                                and int(g[nr, nc]) == value):
                            comp_id[nr, nc] = len(comps)
                            stack.append((nr, nc))
                comps.append({"value": value, "cells": cells,
                              "start": (sr, sc)})
        n = len(comps)
        pairs: set = set()
        for r in range(h):
            for c in range(w):
                cid = int(comp_id[r, c])
                if r + 1 < h and int(comp_id[r + 1, c]) != cid:
                    o = int(comp_id[r + 1, c])
                    pairs.add((min(cid, o), max(cid, o)))
                if c + 1 < w and int(comp_id[r, c + 1]) != cid:
                    o = int(comp_id[r, c + 1])
                    pairs.add((min(cid, o), max(cid, o)))
        adjacency = sorted([a, b] for a, b in pairs)
        enclosers: list[set] = [set() for _ in range(n)]
        for b in range(n):
            reached = np.zeros((h, w), dtype=bool)
            stack = []
            for r in range(h):
                for c in (0, w - 1):
                    if int(comp_id[r, c]) != b and not reached[r, c]:
                        reached[r, c] = True
                        stack.append((r, c))
            for c in range(w):
                for r in (0, h - 1):
                    if int(comp_id[r, c]) != b and not reached[r, c]:
                        reached[r, c] = True
                        stack.append((r, c))
            while stack:
                r, c = stack.pop()
                for dr, dc in _ORTH:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and not reached[nr, nc]
                            and int(comp_id[nr, nc]) != b):
                        reached[nr, nc] = True
                        stack.append((nr, nc))
            for a in range(n):
                if a != b:
                    ar, ac = comps[a]["start"]
                    if not reached[ar, ac]:
                        enclosers[a].add(b)
        children: list[list] = [[] for _ in range(n)]
        for a in range(n):
            if enclosers[a]:
                parent = max(enclosers[a],
                             key=lambda e: (len(enclosers[e]), -e))
                if len(children[parent]) < 16:
                    children[parent].append(a)
        for child_list in children:
            child_list.sort()
        nodes = []
        for cid in range(n):
            comp = comps[cid]
            nodes.append({
                "id": cid,
                "color": comp["value"],
                "hash": _shape_hash(comp["cells"], comp["value"]),
                "pixels": len(comp["cells"]),
                "boundary": [[r, c] for r, c in
                             _corners(_trace_contour(comp["cells"],
                                                     comp["start"]))],
                "children": children[cid],
                "start": [comp["start"][0], comp["start"][1]],
            })
        return {"nodes": nodes, "adjacency_list": adjacency}
    except Exception:
        return {"nodes": [], "adjacency_list": []}


def match(before: Any, after: Any) -> dict[str, Any]:
    """Cross-frame object matching by translation-invariant hash.

    Same contract shape as fusion_supremacy.track (appeared /
    disappeared / stationary / moved counts). Never raises.
    """
    try:
        nb = {nd["hash"]: nd for nd in segment(before)["nodes"]}
        na = {nd["hash"]: nd for nd in segment(after)["nodes"]}
        appeared = sorted(set(na) - set(nb))
        disappeared = sorted(set(nb) - set(na))
        stationary = moved = 0
        for key in set(nb) & set(na):
            cb = (nb[key]["boundary"][0] if nb[key]["boundary"] else None)
            ca = (na[key]["boundary"][0] if na[key]["boundary"] else None)
            if cb == ca:
                stationary += 1
            else:
                moved += 1
        return {"appeared": appeared, "disappeared": disappeared,
                "stationary": stationary, "moved": moved,
                "n_before": len(nb), "n_after": len(na)}
    except Exception:
        return {"appeared": [], "disappeared": [],
                "stationary": 0, "moved": 0, "n_before": 0, "n_after": 0}
