"""Realtime abstract cortex: pure-abstract invariants + empirical self-upgrade.

One module, fail-open, stdlib+numpy at import (worker-safe, vendor-shippable):

  L0 pure abstract (no learning, no game IDs — works on unseen games turn 1):
    - D4 canonical hash (rot/flip orbit minimum, cached)
    - per-color connected components via bounded BFS (Betti-b0 lite)
    - Shannon entropy, RLE Kolmogorov bound, centroid, bbox, bg color
    - rigid-shift vote (dx,dy in [-4,4]), color-permutation vote,
      symmetry class (h/v/rot180/d4/none), translation-invariant signature
  L1 one-shot rule induction from a single (before, after) pair:
    - ops: identity | rot90 | flip | transpose | shift | color_perm |
      toggle | fill_rect | unknown  (+ confidence)
  L2 empirical self-upgrade (realtime, bounded, no threads):
    - per-game hypothesis ledger (succ/fail Laplace beta), fatal (s,a) shield,
      stagnation breaker (repeat-state -> least-recent legal), RHAE economy
      (prefer proven, then untried, never zero-diff repeats)
    - priors EMA per game: shift vector votes, color-map votes, symmetry
      votes, best-action-family Q — capped dicts, FIFO eviction
    - rotation-invariant revisit sense (DSTS exact-match second key;
      defense-in-depth beside the D4-canonical visits counter; break
      confidence stays < consensus fast-path gate, so healthy movement
      is never overridden — direct consumers get early warning only)
    - optional spill to $WORKING_DIR/cortex_priors.json only when
      ARC3_CORTEX_PERSIST=1 (fail-open, lazy, no daemon)

Public API (never raises):
  invariants(grid) -> dict
  canonical_hash(grid) -> str
  infer_transform(before, after) -> dict
  observe(before, after, action, reward=0.0, terminal=False,
          level_up=False, game_id="") -> dict
  decide(grid, legal, game_id="", stagnation=0) -> dict | None
  propose_click(grid, legal) -> dict | None
  reset_episodic(game_id=None) / stats() / get_prior(game_id)

Hot-path discipline: no sleep/socket/subprocess, no unbounded loops,
all shared state behind one RLock, every store bounded.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict, deque
from typing import Any

import numpy as np

__version__ = "v12-cortex-1"

_LOCK = threading.RLock()

# Structural-identity threshold for the revisit sense: numerically
# identical structures differ by eig noise (~1e-2, measured 0.008
# across translated singletons); any genuine single-cell structural
# change moves beta0/chi by >=1 (weight 2 -> distance >= 2). 0.5 sits
# ~50x above noise and 4x below the smallest real change.
_SYM_TOL = 0.5

_MAX_GAMES = 64
_MAX_LEDGER = 512
_MAX_CACHE = 1024
_MAX_COMP = 256
_SHIFT_R = 4
_PERSIST_ENV = "ARC3_CORTEX_PERSIST"
_PERSIST_FILE = "cortex_priors.json"

_LAST_OP: OrderedDict[str, str] = OrderedDict()  # gid -> last op class


def _as_grid(g: Any) -> np.ndarray | None:
    try:
        a = np.asarray(g, dtype=np.uint8)
        if a.ndim != 2:
            return None
        h, w = a.shape
        if h < 1 or w < 1 or h > 64 or w > 64:
            return None
        return np.ascontiguousarray(a)
    except Exception:
        return None


def _bg_of(a: np.ndarray) -> int:
    try:
        return int(np.bincount(a.ravel(), minlength=16).argmax())
    except Exception:
        return 0


def _components(a: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Bounded BFS components -> [(color, size, cy, cx)]. Never raises."""
    try:
        h, w = a.shape
        seen = np.zeros((h, w), dtype=bool)
        out: list[tuple[int, int, int, int]] = []
        for i in range(h):
            for j in range(w):
                if a[i, j] == bg or seen[i, j]:
                    continue
                if len(out) >= _MAX_COMP:
                    return out
                color = int(a[i, j])
                stack = [(i, j)]
                seen[i, j] = True
                sy = sx = n = 0
                while stack:
                    ci, cj = stack.pop()
                    sy += ci
                    sx += cj
                    n += 1
                    if ci > 0 and not seen[ci - 1, cj] and a[ci - 1, cj] == color:
                        seen[ci - 1, cj] = True
                        stack.append((ci - 1, cj))
                    if ci + 1 < h and not seen[ci + 1, cj] and a[ci + 1, cj] == color:
                        seen[ci + 1, cj] = True
                        stack.append((ci + 1, cj))
                    if cj > 0 and not seen[ci, cj - 1] and a[ci, cj - 1] == color:
                        seen[ci, cj - 1] = True
                        stack.append((ci, cj - 1))
                    if cj + 1 < w and not seen[ci, cj + 1] and a[ci, cj + 1] == color:
                        seen[ci, cj + 1] = True
                        stack.append((ci, cj + 1))
                out.append((color, n, sy // max(1, n), sx // max(1, n)))
        return out
    except Exception:
        return []


def _entropy(a: np.ndarray) -> float:
    try:
        c = np.bincount(a.ravel(), minlength=16).astype(np.float64)
        p = c / max(1.0, float(c.sum()))
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
    except Exception:
        return 0.0


def _rle_bound(a: np.ndarray) -> float:
    """Kolmogorov upper bound via RLE runs / cells. Never raises."""
    try:
        flat = a.ravel()
        if flat.size == 0:
            return 1.0
        runs = 1 + int(np.count_nonzero(flat[1:] != flat[:-1]))
        return float(runs) / float(flat.size)
    except Exception:
        return 1.0


def _d4_variants(a: np.ndarray) -> list[np.ndarray]:
    try:
        return [
            a,
            np.rot90(a, 1),
            np.rot90(a, 2),
            np.rot90(a, 3),
            np.fliplr(a),
            np.flipud(a),
            a.T.copy(),
            np.fliplr(a).T.copy(),
        ]
    except Exception:
        return [a]


def _symmetry_class(a: np.ndarray) -> str:
    try:
        h = bool(np.array_equal(a, np.fliplr(a)))
        v = bool(np.array_equal(a, np.flipud(a)))
        r = bool(np.array_equal(a, np.rot90(a, 2)))
        if h and v:
            return "d4"
        if r:
            return "rot180"
        if h:
            return "h"
        if v:
            return "v"
        return "none"
    except Exception:
        return "none"


_CACHE: OrderedDict[str, str] = OrderedDict()


def canonical_hash(grid: Any) -> str:
    """D4-orbit minimum blake2b hash (cached, bounded). Never raises."""
    try:
        a = _as_grid(grid)
        if a is None:
            return "bad"
        key = hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=8).hexdigest()
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                _CACHE.move_to_end(key)
                return hit
        best: str | None = None
        for v in _d4_variants(a):
            try:
                dh = hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=8).hexdigest()
            except Exception:
                continue
            if best is None or dh < best:
                best = dh
        out = best or key
        with _LOCK:
            _CACHE[key] = out
            while len(_CACHE) > _MAX_CACHE:
                _CACHE.popitem(last=False)
        return out
    except Exception:
        return "error"


def invariants(grid: Any) -> dict[str, Any]:
    """Pure-abstract fingerprint of one frame. Never raises."""
    try:
        a = _as_grid(grid)
        if a is None:
            return {"valid": False}
        h, w = a.shape
        bg = _bg_of(a)
        comps = _components(a, bg)
        ys, xs = np.nonzero(a != bg)
        if ys.size:
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
            cy, cx = int(ys.mean()), int(xs.mean())
            fill = float(ys.size) / float(max(1, (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)))
        else:
            bbox = (0, 0, 0, 0)
            cy, cx, fill = 0, 0, 0.0
        return {
            "valid": True,
            "shape": (int(h), int(w)),
            "bg": bg,
            "n_comp": len(comps),
            "comps": comps[:16],
            "bbox": bbox,
            "centroid": (cy, cx),
            "fill": fill,
            "entropy": _entropy(a),
            "rle": _rle_bound(a),
            "sym": _symmetry_class(a),
            "canon": canonical_hash(a),
        }
    except Exception:
        return {"valid": False}


def _shift_vote(before: np.ndarray, after: np.ndarray) -> tuple[int, int, float]:
    """Best rigid shift in [-4,4]^2 by overlap-normalized match.

    Disambiguation (09-25 benchmark): pure-background overlaps score a
    vacuous 1.0, so a shift is accepted only if it also agrees on
    >=1 foreground (nonzero-source) cell — the shift must explain the
    object, not the void. Requires overlap >= 50% of the grid; ties
    break toward more foreground agreement, larger overlap, then
    smaller |dy|+|dx|. Never raises.
    """
    try:
        if before.shape != after.shape:
            return (0, 0, 0.0)
        h, w = before.shape
        full = float(h * w)
        best_key = (-1.0, -1.0, -1.0, 0.0)
        best = (0, 0, 0.0)
        for dy in range(-_SHIFT_R, _SHIFT_R + 1):
            for dx in range(-_SHIFT_R, _SHIFT_R + 1):
                try:
                    # after[i,j] == before[i-dy, j-dx]: overlap in after-coords
                    # is [max(0,dy), H+dy) x [max(0,dx), W+dx).
                    y0, y1 = max(0, dy), min(h, h + dy)
                    x0, x1 = max(0, dx), min(w, w + dx)
                    z0, z1 = y0 - dy, y1 - dy
                    w0, w1 = x0 - dx, x1 - dx
                    if y1 <= y0 or x1 <= x0:
                        continue
                    ov = float((y1 - y0) * (x1 - x0))
                    if ov < 0.25 * full:
                        continue
                    a_win = after[y0:y1, x0:x1]
                    b_win = before[z0:z1, w0:w1]
                    m = float(np.count_nonzero(a_win == b_win))
                    frac = m / ov
                    fg = float(np.count_nonzero((a_win == b_win) & (b_win != 0)))
                    if fg < 1.0 and float(np.count_nonzero(before)) > 0:
                        continue  # explains only void -> reject
                    key = (frac, fg, ov, float(-(abs(dy) + abs(dx))))
                    if key > best_key:
                        best_key = key
                        best = (dy, dx, frac)
                except Exception:
                    continue
        return best
    except Exception:
        return (0, 0, 0.0)


def _color_perm_vote(before: np.ndarray, after: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Greedy color remap by co-occurrence mass. Never raises."""
    try:
        if before.shape != after.shape:
            return (tuple(range(16)), 0.0)
        perm = list(range(16))
        hit = 0
        for c in range(16):
            try:
                mask = before == c
                n = int(np.count_nonzero(mask))
                if n == 0:
                    continue
                col = np.bincount(after[mask].ravel(), minlength=16)
                perm[c] = int(col.argmax())
                hit += int(col.max())
            except Exception:
                continue
        return (tuple(perm), float(hit) / float(max(1, before.size)))
    except Exception:
        return (tuple(range(16)), 0.0)


def infer_transform(before: Any, after: Any) -> dict[str, Any]:
    """One-shot op induction from a single pair. Never raises."""
    try:
        a = _as_grid(before)
        b = _as_grid(after)
        if a is None or b is None:
            return {"op": "unknown", "confidence": 0.0}
        if a.shape == b.shape and bool(np.array_equal(a, b)):
            return {"op": "identity", "confidence": 1.0, "params": {}}
        if a.shape == b.shape:
            for k, name in ((1, "rot90"), (2, "rot180"), (3, "rot270")):
                try:
                    if bool(np.array_equal(np.rot90(a, k), b)):
                        return {"op": name, "confidence": 0.99, "params": {"k": k}}
                except Exception:
                    continue
            try:
                if bool(np.array_equal(np.fliplr(a), b)):
                    return {"op": "flip_h", "confidence": 0.99, "params": {}}
                if bool(np.array_equal(np.flipud(a), b)):
                    return {"op": "flip_v", "confidence": 0.99, "params": {}}
                if a.shape[0] == a.shape[1] and bool(np.array_equal(a.T, b)):
                    return {"op": "transpose", "confidence": 0.99, "params": {}}
            except Exception:
                pass
            # Disambiguate shift vs recolor by histogram conservation:
            # a rigid shift preserves the color multiset, a recolor does not.
            try:
                hist_eq = bool(np.array_equal(np.bincount(a.ravel(), minlength=16),
                                              np.bincount(b.ravel(), minlength=16)))
            except Exception:
                hist_eq = False
            dy, dx, frac = _shift_vote(a, b)
            perm, pfrac = _color_perm_vote(a, b)
            perm_real = any(p != i for i, p in enumerate(perm))
            if hist_eq:
                if frac >= 0.90 and (dy or dx):
                    return {"op": "shift", "confidence": min(0.99, 0.60 + frac * 0.4),
                            "params": {"dy": dy, "dx": dx, "match": frac}}
            else:
                if pfrac >= 0.95 and perm_real:
                    return {"op": "color_perm", "confidence": min(0.99, pfrac),
                            "params": {"perm": list(perm)}}
                if frac >= 0.90 and (dy or dx):
                    return {"op": "shift", "confidence": min(0.99, 0.60 + frac * 0.4),
                            "params": {"dy": dy, "dx": dx, "match": frac}}
            try:
                n = int(np.count_nonzero(a != b))
                frac2 = float(n) / float(a.size)
                if 0 < frac2 <= 0.25:
                    return {"op": "toggle", "confidence": 0.70, "params": {"cells": n}}
            except Exception:
                pass
            return {"op": "unknown", "confidence": 0.20, "params": {}}
        # shape change -> fill/crop family
        try:
            if b.size >= a.size:
                return {"op": "fill_rect", "confidence": 0.45,
                        "params": {"from": list(a.shape), "to": list(b.shape)}}
            return {"op": "crop", "confidence": 0.45,
                    "params": {"from": list(a.shape), "to": list(b.shape)}}
        except Exception:
            return {"op": "unknown", "confidence": 0.20, "params": {}}
    except Exception:
        return {"op": "unknown", "confidence": 0.0}


class _GamePrior:
    __slots__ = ("succ", "fail", "fatal", "visits", "recent",
                 "shift_votes", "perm_votes", "sym_votes", "fam_q", "fam_n",
                 "dsts_recent", "seen")

    def __init__(self) -> None:
        self.succ: dict[Any, int] = {}
        self.fail: dict[Any, int] = {}
        self.fatal: set[tuple[str, Any]] = set()
        self.visits: dict[str, int] = {}
        self.recent: deque[Any] = deque(maxlen=8)
        self.dsts_recent: deque[dict[str, Any]] = deque(maxlen=32)
        self.shift_votes: dict[tuple[int, int], int] = {}
        self.perm_votes: dict[tuple[int, ...], int] = {}
        self.sym_votes: dict[str, int] = {}
        self.fam_q: dict[str, float] = {}
        self.fam_n: dict[str, int] = {}
        # novelty set for shaped intrinsic credit (bounded; cleared on overflow)
        self.seen: set[str] = set()


_PRIORS: OrderedDict[str, _GamePrior] = OrderedDict()
_LEDGER: deque[dict[str, Any]] = deque(maxlen=_MAX_LEDGER)


def _prior(game_id: str) -> _GamePrior:
    gid = str(game_id or "duck")
    with _LOCK:
        p = _PRIORS.get(gid)
        if p is None:
            p = _GamePrior()
            _PRIORS[gid] = p
            while len(_PRIORS) > _MAX_GAMES:
                _PRIORS.popitem(last=False)
        else:
            _PRIORS.move_to_end(gid)
        return p


def _act_key(action: Any) -> Any:
    try:
        if isinstance(action, dict):
            x = action.get("x")
            y = action.get("y")
            return (int(action.get("action", -1)), None if x is None else int(x),
                    None if y is None else int(y))
        return int(action)
    except Exception:
        return str(action)


def _act_family(action: Any) -> str:
    try:
        k = _act_key(action)
        aid = k[0] if isinstance(k, tuple) else k
        return {1: "A1", 2: "A2", 3: "A3", 4: "A4", 5: "A5", 6: "CLICK", 7: "UNDO"}.get(aid, "OTHER")
    except Exception:
        return "OTHER"


def observe(before: Any, after: Any, action: Any, reward: float = 0.0,
            terminal: bool = False, level_up: bool = False,
            game_id: str = "") -> dict[str, Any]:
    """Record one transition; update priors EMA. Never raises."""
    try:
        a = _as_grid(before)
        b = _as_grid(after)
        gid = str(game_id or "duck")
        p = _prior(gid)
        changed = True
        productive = True
        if a is not None and b is not None and a.shape == b.shape:
            try:
                changed = bool(np.any(a != b))
                # NVARC3 C-fix (top-5 source): border cells (2px frame) are
                # UI (score/timer) that drifts without game progress — the
                # productive signal counts interior-only change.
                h, w = a.shape
                iy1, iy2 = min(2, h), max(0, h - 2)
                ix1, ix2 = min(2, w), max(0, w - 2)
                if iy1 < iy2 and ix1 < ix2:
                    productive = bool(np.any(a[iy1:iy2, ix1:ix2] != b[iy1:iy2, ix1:ix2]))
                else:
                    productive = changed
            except Exception:
                changed = True
                productive = True
        good = bool(level_up or (reward and reward > 0) or (productive and not terminal))
        key = _act_key(action)
        fam = _act_family(action)
        sig = ""
        try:
            sig = canonical_hash(b if b is not None else after)
        except Exception:
            sig = "error"
        with _LOCK:
            if good:
                p.succ[key] = p.succ.get(key, 0) + 1
            else:
                p.fail[key] = p.fail.get(key, 0) + 1
            if terminal and not level_up and not changed:
                try:
                    p.fatal.add((sig, key if not isinstance(key, tuple) else key[0]))
                except Exception:
                    pass
            # family Q EMA with shaped INTRINSIC credit (NVARC3 _reward
            # weights, proven top-5 source — computed from own frames only):
            # novel state +1.5 / visited+productive +0.3 / visited+static -0.1
            # / any interior change +0.5 / object moved +0.3 each (cap 3).
            # succ/fail stay binary (Laplace ranking unchanged).
            n = p.fam_n.get(fam, 0) + 1
            old = p.fam_q.get(fam, 0.0)
            credit = 1.0 if good else 0.0
            try:
                shaped = 0.0
                if sig not in p.seen:
                    shaped += 1.5
                elif productive:
                    shaped += 0.3
                else:
                    shaped -= 0.1
                if productive:
                    shaped += 0.5
                try:
                    from .fusion_supremacy import track as _fs_track
                    if a is not None and b is not None:
                        _tr = _fs_track(a, b)
                        try:
                            shaped += 0.3 * min(int(_tr.get("moved", 0)), 3)
                        except Exception:
                            pass
                except Exception:
                    pass
                p.seen.add(sig)
                if len(p.seen) > 2048:
                    p.seen.clear()
                # blend: explicit good/bad dominates, shaping refines
                credit = (1.0 if good else 0.0) * 0.5 + max(0.0, min(1.0, shaped / 3.0)) * 0.5
            except Exception:
                pass
            p.fam_q[fam] = old + (credit - old) / max(1, n)
            p.fam_n[fam] = min(n, 256)
            # transform votes feed priors (self-upgrade signal)
            if a is not None and b is not None:
                try:
                    t = infer_transform(a, b)
                    # causal-chain edge update (multiplicative cause-effect compounding)
                    try:
                        from . import causal_chain_reasoner as _ccr
                        op_c = str(t.get("op", "unknown"))
                        prev_c = _LAST_OP.get(gid, "")
                        _ccr.observe_op(prev_c, op_c, good)
                        _LAST_OP[gid] = op_c
                        while len(_LAST_OP) > _MAX_GAMES:
                            _LAST_OP.popitem(last=False)
                    except Exception:
                        pass
                    if t.get("op") == "shift":
                        prm = t.get("params", {})
                        sv = (int(prm.get("dy", 0)), int(prm.get("dx", 0)))
                        p.shift_votes[sv] = p.shift_votes.get(sv, 0) + 1
                    elif t.get("op") == "color_perm":
                        pv = tuple(t.get("params", {}).get("perm", list(range(16))))
                        p.perm_votes[pv] = p.perm_votes.get(pv, 0) + 1
                except Exception:
                    pass
                try:
                    sv2 = _symmetry_class(b)
                    p.sym_votes[sv2] = p.sym_votes.get(sv2, 0) + 1
                except Exception:
                    pass
            try:
                _LEDGER.append({"game": gid, "action": str(key), "good": good,
                                "changed": changed, "sig": sig})
            except Exception:
                pass
        if os.environ.get(_PERSIST_ENV, "0") == "1":
            _spill(gid)
        return {"good": good, "changed": changed, "op": "recorded"}
    except Exception:
        return {"good": False, "changed": False, "op": "fail-open"}


def _spill(gid: str) -> None:
    """Best-effort persist of one game's priors. Fail-open, lazy."""
    try:
        root = os.environ.get("WORKING_DIR", "/kaggle/working")
        base = os.path.join(str(root), _PERSIST_FILE)
        p = _prior(gid)
        with _LOCK:
            blob = {gid: {
                "fam_q": dict(p.fam_q),
                "shift_votes": {f"{k[0]},{k[1]}": v for k, v in p.shift_votes.items()},
                "sym_votes": dict(p.sym_votes),
            }}
        try:
            old: dict[str, Any] = {}
            if os.path.exists(base):
                with open(base, encoding="utf-8") as fh:
                    old = json.load(fh)
        except Exception:
            old = {}
        try:
            old.update(blob)
            tmp = base + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(old, fh)
            os.replace(tmp, base)
        except Exception:
            pass
    except Exception:
        pass


def _beta(s: int, f: int) -> float:
    try:
        return (float(s) + 1.0) / (float(s) + float(f) + 2.0)
    except Exception:
        return 0.5


def propose_click(grid: Any, legal: Any) -> dict[str, Any] | None:
    """Centroid-snapped validated click. Never raises; None = fall through."""
    try:
        acts = list(legal) if legal else []
        has6 = any((a == 6 or (isinstance(a, dict) and a.get("action") == 6)) for a in acts)
        if not has6:
            return None
        a = _as_grid(grid)
        if a is None:
            return None
        inv = invariants(a)
        if not inv.get("valid"):
            return None
        h, w = inv["shape"]
        return _click_from_comps(a, inv["bg"], h, w, acts)
    except Exception:
        return None


def _click_from_comps(a: np.ndarray, bg: int, h: int, w: int, live: list,
                      comps: list[tuple[int, int, int, int]] | None = None) -> dict[str, Any] | None:
    """Shared click core: centroid-snapped validated click or None."""
    try:
        if comps is None:
            comps = [c for c in _components(a, bg)
                     if 3 <= c[1] <= int(0.25 * h * w)]
        else:
            comps = [c for c in comps if 3 <= c[1] <= int(0.25 * h * w)]
        if not comps:
            return None
        try:
            comps.sort(key=lambda c: -c[1])
            _, _, cy, cx = comps[0]
        except Exception:
            return None
        if not (0 <= cy < h and 0 <= cx < w):
            return None
        try:
            if int(a[cy, cx]) == bg:
                return None
        except Exception:
            return None
        return {"action": {"action": 6, "x": int(cx), "y": int(cy)},
                "confidence": 0.80, "reason": "cortex_centroid_click"}
    except Exception:
        return None


def decide(grid: Any, legal: Any, game_id: str = "",
           stagnation: int = 0) -> dict[str, Any] | None:
    """Fast abstract+empirical proposal. Never raises; None = fall through."""
    try:
        a = _as_grid(grid)
        if a is None:
            return None
        acts = list(legal) if legal else [1, 2, 3, 4, 5]
        gid = str(game_id or "duck")
        p = _prior(gid)
        sig = canonical_hash(a)
        # Rotation-invariant revisit count: hash detectors are blind to
        # symmetric cycles; DSTS sees them (fail-open, lazy).
        # Kill-switch ARC3_DSTS=0 (default 1) for ablation/operators.
        sym_repeat = 0
        try:
            import os as _os
            if _os.environ.get("ARC3_DSTS", "1") == "1":
                from .spectral_topological_reasoning import (
                    spectral_signature as _dsts_sig,
                    signature_distance as _dsts_dist,
                )
                _ds = _dsts_sig(a)
                if _ds.get("valid"):
                    with _LOCK:
                        _hist = list(p.dsts_recent)
                    try:
                        sym_repeat = sum(1 for _o in _hist
                                         if _dsts_dist(_o, _ds) <= _SYM_TOL)
                    except Exception:
                        sym_repeat = 0
                    with _LOCK:
                        p.dsts_recent.append(_ds)
        except Exception:
            sym_repeat = 0
        with _LOCK:
            visits = p.visits.get(sig, 0) + 1
            p.visits[sig] = visits
            while len(p.visits) > 2048:
                p.visits.pop(next(iter(p.visits)))
        # 1. fatal shield: drop known-fatal actions for this state
        live: list[Any] = []
        try:
            with _LOCK:
                fatal_here = {f for (s, f) in p.fatal if s == sig}
            for act in acts:
                k = _act_key(act)
                aid = k[0] if isinstance(k, tuple) else k
                if aid in fatal_here:
                    continue
                live.append(act)
        except Exception:
            live = acts
        if not live:
            return None
        # 2. stagnation breaker: least-recent legal, retire repeats
        # (sym_repeat>=2: third symmetric sighting — catches rotated cycles)
        if stagnation >= 3 or visits >= 3 or sym_repeat >= 2:
            try:
                # 2a. ACTION7 undo (NVARC3 M3, top-5 source): when the run is
                # deeply unproductive and 7 is legal, ACTION7 is an undo in
                # the observed games — worth one bet (fail-open: recorded).
                if stagnation >= 12:
                    for act in live:
                        k = _act_key(act)
                        aid = k[0] if isinstance(k, tuple) else k
                        if aid == 7:
                            return {"action": act, "confidence": 0.60,
                                    "reason": "cortex_undo_stuck"}
                with _LOCK:
                    recent = list(p.recent)
                for act in live:
                    if _act_key(act) not in recent:
                        return {"action": act, "confidence": 0.62,
                                "reason": "cortex_stagnation_break"}
                return {"action": live[0], "confidence": 0.55,
                        "reason": "cortex_stagnation_break"}
            except Exception:
                pass
        # 3. empirical ranking: Laplace beta per action + family Q
        best: dict[str, Any] | None = None
        best_q = -1.0
        try:
            with _LOCK:
                succ = dict(p.succ)
                fail = dict(p.fail)
                fam_q = dict(p.fam_q)
        except Exception:
            succ, fail, fam_q = {}, {}, {}
        for act in live:
            try:
                k = _act_key(act)
                q = _beta(succ.get(k, 0), fail.get(k, 0))
                fq = fam_q.get(_act_family(act), 0.5)
                score = 0.7 * q + 0.3 * (0.5 + 0.5 * fq)
                if score > best_q:
                    best_q = score
                    best = act
            except Exception:
                continue
        # 4. click affordance: large salient object + ACTION6 legal -> click.
        # Reuses the single invariants() pass (no second BFS).
        try:
            has6 = any((x == 6 or (isinstance(x, dict) and x.get("action") == 6))
                       for x in live)
            inv = invariants(a) if has6 else None
            if inv is not None and inv.get("valid") and inv.get("n_comp", 0) >= 1:
                ck = _click_from_comps(a, inv["bg"], inv["shape"][0], inv["shape"][1],
                                       live, comps=list(inv.get("comps", ())))
                if ck is not None and best_q < 0.80:
                    with _LOCK:
                        p.recent.append(ck["action"] if isinstance(ck["action"], dict)
                                        else _act_key(ck["action"]))
                    return ck
        except Exception:
            pass
        if best is None:
            return None
        if best_q <= 0.60:
            # no evidence yet on unseen game: stay quiet, let Duck explore
            return None
        try:
            with _LOCK:
                p.recent.append(_act_key(best))
        except Exception:
            pass
        return {"action": best, "confidence": min(0.92, max(0.60, best_q)),
                "reason": "cortex_empirical_rank"}
    except Exception:
        return None


def get_prior(game_id: str = "") -> dict[str, Any]:
    """Snapshot of learned priors (bounded copy). Never raises."""
    try:
        p = _prior(str(game_id or "duck"))
        with _LOCK:
            return {
                "fam_q": dict(p.fam_q),
                "shift_votes": dict(p.shift_votes),
                "sym_votes": dict(p.sym_votes),
                "n_fatal": len(p.fatal),
                "ledger": len(_LEDGER),
            }
    except Exception:
        return {}


def stats() -> dict[str, Any]:
    try:
        with _LOCK:
            return {"games": len(_PRIORS), "ledger": len(_LEDGER),
                    "cache": len(_CACHE), "version": __version__}
    except Exception:
        return {"games": 0, "ledger": 0, "cache": 0}


def reset_episodic(game_id: str | None = None) -> None:
    """Hermetic reset (tests): drop episodic visits/recent, keep priors."""
    try:
        with _LOCK:
            if game_id is None:
                for p in _PRIORS.values():
                    try:
                        p.visits.clear()
                        p.recent.clear()
                        p.dsts_recent.clear()
                        p.seen.clear()
                    except Exception:
                        continue
            else:
                p = _PRIORS.get(str(game_id))
                if p is not None:
                    p.visits.clear()
                    p.recent.clear()
                    try:
                        p.dsts_recent.clear()
                    except Exception:
                        pass
                    try:
                        p.seen.clear()
                    except Exception:
                        pass
    except Exception:
        pass
