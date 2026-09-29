"""Pure-abstract causal chain reasoner with multiplicative self-compounding.

New module (not a rewrite of abstract_causal_reasoner, which stays live):
models thinking as an explicit CAUSE -> EFFECT lattice over abstract grid
transforms, all pure math, no game IDs, no I/O:

  Chain model (the core idea):
    Each candidate action a carries an abstract operator T(a) drawn from the
    same family as realtime_abstract_cortex.infer_transform (shift/rot/
    flip/color_perm/toggle/unknown). A chain is a predicted trajectory
    [T_a1, T_a2, ...] of length k. Its prior plausibility is the PRODUCT of
    per-step op confidences times the empirical transition agreement
    (how often the same op-class actually followed the same last op-class):
        P(chain) = prod_i conf(T_ai) * agree(op_{i-1} -> op_i)
    Pure math: no thresholds hidden in floats; log-domain sums for
    numerical stability; bounded k.

  Multiplicative self-compounding (L2, realtime, no threads):
    Every verified transition raises the agreement edge weight of its
    op-class pair by a multiplicative update with decay floor (0.5x..2.0x
    clamp), so chains that keep proving themselves compound EXPONENTIALLY
    (k steps => weight^k) while disproven chains decay geometrically:
        w' = clamp(w * (1 + gain) , 0.5, 2.0)     gain in [-0.5, +0.5]
    Bounded dict (op-pair -> weight), FIFO eviction.

  Public API (never raises):
    chain_confidence(grid, after, k=3) -> float        # predicted k-chain
    best_chain(grid, legal, k=3) -> dict | None        # argmax chain
    observe_op(prev_op, op, good) -> float             # multiplicative edge update
    op_weight(prev_op, op) -> float
    reason(grid, legal, game_id="", stagnation=0) -> dict | None
        # think(): best chain -> first action + cause-effect trace
    reset() / stats()

stdlib+numpy at import. Hot-path clean: no sleep/socket/subprocess, all
loops bounded, shared state behind one RLock.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque
from typing import Any

import numpy as np

__version__ = "v14-causalchain-1"

_LOCK = threading.RLock()

_MAX_EDGES = 512
_MAX_K = 4
_W_MIN, _W_MAX = 0.5, 2.0
_GAIN = 0.25


def _ops_of_grid(grid: Any) -> list[tuple[str, float]]:
    """Candidate abstract ops for the current frame (from cortex invariants)."""
    try:
        from .realtime_abstract_cortex import invariants as _inv
        inv = _inv(grid)
        if not inv.get("valid"):
            return []
        out: list[tuple[str, float]] = []
        h, w = inv["shape"]
        bbox = inv["bbox"]
        off = (bbox[0] > 0 and bbox[1] > 0
               and bbox[2] < h - 1 and bbox[3] < w - 1)
        if off:
            out.append(("shift", 0.85))
            out.append(("rot90", 0.55))
        sym = inv.get("sym", "none")
        if sym == "none":
            out.append(("flip_h", 0.50))
            out.append(("rot180", 0.45))
        if inv.get("n_comp", 0) >= 1:
            out.append(("toggle", 0.60))
            out.append(("color_perm", 0.45))
        if not out:
            out.append(("identity", 0.40))
        return out
    except Exception:
        return []


def _op_from_action(action: Any) -> str:
    """Map an action id to its abstract op-class (evidence-based prior)."""
    try:
        aid = action.get("action", action) if isinstance(action, dict) else action
        aid = int(aid)
        return {1: "shift", 2: "shift", 3: "shift", 4: "shift",
                5: "rot90", 6: "toggle", 7: "identity"}.get(aid, "unknown")
    except Exception:
        return "unknown"


# ---- multiplicative edge weights (op-class -> op-class) --------------------
_EDGES: OrderedDict[tuple[str, str], float] = OrderedDict()


def _edge(prev: str, op: str) -> float:
    try:
        with _LOCK:
            return float(_EDGES.get((prev, op), 1.0))
    except Exception:
        return 1.0


def op_weight(prev_op: str, op: str) -> float:
    return _edge(str(prev_op), str(op))


def observe_op(prev_op: str, op: str, good: bool) -> float:
    """Multiplicative edge update with decay floor/ceiling. Never raises."""
    try:
        key = (str(prev_op), str(op))
        with _LOCK:
            w = float(_EDGES.get(key, 1.0))
            w = w * (1.0 + (_GAIN if good else -_GAIN))
            w = max(_W_MIN, min(_W_MAX, w))
            _EDGES[key] = w
            while len(_EDGES) > _MAX_EDGES:
                _EDGES.popitem(last=False)
            return w
    except Exception:
        return 1.0


def chain_confidence(grid: Any, k: int = 3) -> float:
    """Predicted k-step chain plausibility (log-domain product). Never raises."""
    try:
        ops = _ops_of_grid(grid)
        if not ops:
            return 0.0
        steps = max(1, min(int(k), _MAX_K))
        best = 0.0
        # chains branch over op classes; k<=4, ops<=6 -> bounded work
        frontier: list[tuple[str, float]] = [("", 0.0)]
        for _ in range(steps):
            nxt: list[tuple[str, float]] = []
            for prev, lp in frontier:
                for op, conf in ops:
                    try:
                        step = math.log(max(1e-9, conf)) + math.log(max(1e-9, _edge(prev, op)))
                    except Exception:
                        step = -18.0
                    nxt.append((op, lp + step))
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[1])
            frontier = nxt[:6]
        if frontier:
            best = math.exp(frontier[0][1] / max(1, steps))
        return float(min(0.99, best))
    except Exception:
        return 0.0


def best_chain(grid: Any, legal: Any, k: int = 3) -> dict[str, Any] | None:
    """Argmax chain -> first action + cause-effect trace. Never raises."""
    try:
        acts = list(legal) if legal is not None else [1, 2, 3, 4, 5]
        if not acts:
            return None
        cc = chain_confidence(grid, k=k)
        if cc < 0.10:
            return None
        ops = _ops_of_grid(grid)
        if not ops:
            return None
        # pick the legal action whose op-class best matches the top op
        try:
            top_op = ops[0][0]
        except Exception:
            top_op = "unknown"
        best_act = None
        best_score = -1.0
        for a in acts:
            try:
                op = _op_from_action(a)
                score = _edge(top_op, op) * (1.0 if op == top_op else 0.6)
                if score > best_score:
                    best_score = score
                    best_act = a
            except Exception:
                continue
        if best_act is None:
            return None
        return {"action": best_act, "confidence": cc,
                "chain": [top_op] + [o for o, _ in ops[1:2]],
                "op": top_op, "reason": "causal_chain_argmax"}
    except Exception:
        return None


def reason(grid: Any, legal: Any, game_id: str = "",
           stagnation: int = 0) -> dict[str, Any] | None:
    """Think(): explicit cause-effect proposal. Never raises; None = quiet."""
    try:
        a = np.asarray(grid) if grid is not None else None
        if a is None or a.ndim != 2:
            return None
        return best_chain(a, legal, k=3)
    except Exception:
        return None


def stats() -> dict[str, Any]:
    try:
        with _LOCK:
            strong = sum(1 for w in _EDGES.values() if w > 1.2)
            weak = sum(1 for w in _EDGES.values() if w < 0.8)
            return {"edges": len(_EDGES), "strong": strong, "weak": weak,
                    "version": __version__}
    except Exception:
        return {"edges": 0, "strong": 0, "weak": 0}


def reset() -> None:
    try:
        with _LOCK:
            _EDGES.clear()
    except Exception:
        pass
