"""Per-turn shared memo (v28-fuse-1): one segmentation per unique grid.

fusion.objects() and flux novelty previously segmented the same grids
independently. This module memoizes the pure function segment(grid) by
(grid-shape, fingerprint) so every tier reads the same object graph for
the same turn. Entries are SHARED -- callers must treat results as
read-only. Bounded LRU, one RLock, never raises.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

__version__ = "v28-fuse-1"

_LOCK = threading.RLock()
_MAX_ENTRIES = 32
_MEMO: OrderedDict = OrderedDict()


def fingerprint(grid: Any) -> str:
    """Stable id for a grid's bytes+shape. Never raises."""
    try:
        g = np.ascontiguousarray(np.asarray(grid, dtype=np.uint8))
        h = hashlib.sha1(g.tobytes()).hexdigest()[:16]
        return "%dx%d:%s" % (int(g.shape[0]), int(g.shape[1]), h)
    except Exception:
        return "err"


def get_segment(grid: Any) -> dict[str, Any]:
    """Memoized object_segmentation.segment(grid). Never raises."""
    empty: dict[str, Any] = {"nodes": [], "adjacency_list": []}
    try:
        key = ("seg", fingerprint(grid))
        with _LOCK:
            hit = _MEMO.get(key)
            if hit is not None:
                _MEMO.move_to_end(key)
                return hit
        from .object_segmentation import segment as _seg
        val = _seg(grid)
        if not isinstance(val, dict):
            return empty
        if not val.get("nodes"):
            return empty  # garbage in: nothing worth memoizing
        with _LOCK:
            _MEMO[key] = val
            _MEMO.move_to_end(key)
            while len(_MEMO) > _MAX_ENTRIES:
                _MEMO.popitem(last=False)
        return val
    except Exception:
        return empty


def stats() -> dict[str, int]:
    """Cache size (audit). Never raises."""
    try:
        with _LOCK:
            return {"entries": len(_MEMO)}
    except Exception:
        return {"entries": 0}


def reset() -> None:
    """Clear (hermetic tests). Never raises."""
    try:
        with _LOCK:
            _MEMO.clear()
    except Exception:
        pass
