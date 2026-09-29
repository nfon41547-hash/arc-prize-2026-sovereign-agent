"""Zero-waste coordination kernel: evidence loop + realtime self-learn + RHAE economy.

Single coordinated module for the five winning levers (all fail-open, bounded,
stdlib+numpy at import so it ships inside the notebook vendor cell AND the
team dataset patch with byte-identical bytes):

  L0 evidence loop  : act -> observe -> diff/delta -> ledger -> re-plan,
                      prune disproven hypotheses, macro-queue (k<=25), flush
                      on level/score change. No zero-diff repeats.
  L1 aTTT-lite      : token/n-gram novelty reweight — novel transitions train
                      at full weight, repeated ones decay (floor 0.10), so the
                      loop never amplifies drift when stuck.
  L1 S-TTT-lite     : evidence-span select — adapt only on changed cells
                      (diff fingerprint), never on full-grid noise.
  L2 RHAE economy   : exact level formula min((base/ai)^2*100,115), late-level
                      weighting, 115% cap-hunt, reset-only-when-exhausted.
  L2 ClickGuard     : centroid clicks validated — reject massive background
                      (>25%), 1-2px noise, out-of-bounds, coardless blind fire.

ZeroCopy discipline: no sockets, no subprocess, no sleep, no unbounded loops,
no raises out of any public API. All shared state behind one RLock (no
same-thread re-entry deadlock). Every store bounded with FIFO eviction.
Learners (rule/rune/registry/memory) feed this ledger record-only via lazy
import; nothing here imports them back (one-directional, no cycles).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict, deque
from typing import Any

import numpy as np

__version__ = "v9-zerowaste-1"

_LOCK = threading.RLock()

# ── bounds ──────────────────────────────────────────────────────────────────
_MAX_LEDGER = 512
_MAX_QUEUE = 25
_MAX_NGRAMS = 4096
_MAX_SPANS = 256
_BG_REJECT_FRAC = 0.25
_NOISE_NEIGHBORS = 2  # foreground cell with <2 filled 3x3 neighbours = noise
_RESET_STAGNATION = 8
_RESET_ZERODIFF = 5


# ── pure diff fingerprint (S-TTT-lite signal) ───────────────────────────────
def diff_fingerprint(before: Any, after: Any) -> dict[str, Any]:
    """Describe a transition. Never raises; bad shapes -> unchanged/unsafe."""
    try:
        lhs = np.asarray(before, dtype=np.uint8)
        rhs = np.asarray(after, dtype=np.uint8)
        if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape != rhs.shape:
            return {"changed": False, "identity": True, "shape_match": False,
                    "delta_nonzero": 0, "delta_frac": 0.0, "novel_hash": "bad"}
        diff = lhs != rhs
        n = int(np.count_nonzero(diff))
        changed = bool(n > 0)
        try:
            dh = hashlib.blake2b(np.ascontiguousarray(rhs).tobytes(),
                                 digest_size=8).hexdigest()
        except Exception:
            dh = "unknown"
        return {"changed": changed, "identity": not changed,
                "shape_match": True, "delta_nonzero": n,
                "delta_frac": float(n) / float(lhs.size) if lhs.size else 0.0,
                "novel_hash": dh}
    except Exception:
        return {"changed": False, "identity": True, "shape_match": False,
                "delta_nonzero": 0, "delta_frac": 0.0, "novel_hash": "error"}


def verified_gate(changed: bool, shape_match: bool, level_up: bool,
                  score_gain: bool) -> bool:
    """Promotion gate: (level_up OR score_gain) AND changed AND shape_match."""
    try:
        return bool((level_up or score_gain) and changed and shape_match)
    except Exception:
        return False


def select_evidence_spans(before: Any, after: Any,
                          limit: int = _MAX_SPANS) -> list[tuple[int, int, int]]:
    """Return capped [(row, col, new_val)] diff cells only. Never raises."""
    try:
        lhs = np.asarray(before, dtype=np.uint8)
        rhs = np.asarray(after, dtype=np.uint8)
        if lhs.shape != rhs.shape or lhs.ndim != 2:
            return []
        ys, xs = np.nonzero(lhs != rhs)
        out: list[tuple[int, int, int]] = []
        cap = max(0, int(limit))
        for y, x in zip(ys.tolist()[:cap], xs.tolist()[:cap]):
            try:
                out.append((int(y), int(x), int(rhs[y, x])))
            except Exception:
                continue
        return out
    except Exception:
        return []


# ── aTTT-lite novelty reweight ──────────────────────────────────────────────
class TokenReweight:
    """Repeat-n-gram downweight: novel=1.0, repeats decay, floor 0.10."""

    def __init__(self, max_keys: int = _MAX_NGRAMS) -> None:
        self.max_keys = max(1, int(max_keys))
        self._counts: OrderedDict[str, int] = OrderedDict()

    def weight(self, key: Any) -> float:
        try:
            k = str(key)
            with _LOCK:
                n = int(self._counts.get(k, 0))
                return max(0.10, 1.0 / (1.0 + n))
        except Exception:
            return 0.10

    def note(self, key: Any) -> float:
        try:
            k = str(key)
            with _LOCK:
                self._counts[k] = int(self._counts.get(k, 0)) + 1
                w = max(0.10, 1.0 / float(self._counts[k]))
                while len(self._counts) > self.max_keys:
                    self._counts.popitem(last=False)
                return w
        except Exception:
            return 0.10

    def reset(self) -> None:
        try:
            with _LOCK:
                self._counts.clear()
        except Exception:
            pass


# ── RHAE economy (exact formula) ────────────────────────────────────────────
def rhae_level_score(baseline: float, taken: float) -> float:
    """Exact per-level score: min(115, 100*(base/taken)^2). Never raises."""
    try:
        b, t = float(baseline), float(taken)
        if b <= 0 or t <= 0:
            return 0.0
        return min(115.0, 100.0 * (b / t) ** 2)
    except Exception:
        return 0.0


def rhae_level_weight(level_index_1based: int) -> int:
    try:
        return max(1, int(level_index_1based))
    except Exception:
        return 1


def cap_already_hit(baseline: float, taken: float) -> bool:
    """True when further actions cannot improve past the 115% cap."""
    try:
        return rhae_level_score(baseline, taken) >= 115.0
    except Exception:
        return False


def reset_allowed(failed_count: int, legal_count: int = 7, stagnation: int = 0,
                  zero_diff_streak: int = 0) -> bool:
    """Reset ONLY when the action pool is exhausted or deeply stalled."""
    try:
        try:
            n_legal = int(legal_count)
        except Exception:
            n_legal = len(list(legal_count or [])) or 7
        if int(zero_diff_streak) >= _RESET_ZERODIFF:
            return True
        if int(stagnation) >= _RESET_STAGNATION:
            return True
        return int(failed_count) >= max(1, n_legal - 1)
    except Exception:
        return False


# ── ClickGuard (ACTION6 precision) ──────────────────────────────────────────
def click_guard(grid: Any, x: Any, y: Any) -> tuple[bool, str]:
    """Validate a click. Returns (ok, reason). Never raises, never blind-fires.

    Rejects: non-int coords, out-of-bounds, clicks on massive background
    (>25% dominant color), 1-2px noise (foreground cell with <2 filled 3x3
    neighbours). Coardless (None) input is refused, not defaulted.
    """
    try:
        if x is None or y is None:
            return False, "coardless-refused"
        try:
            cx, cy = int(x), int(y)
        except Exception:
            return False, "non-int-coord"
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2:
            return False, "bad-grid"
        h, w = int(g.shape[0]), int(g.shape[1])
        if not (0 <= cx < w and 0 <= cy < h):
            return False, "out-of-bounds"
        if h <= 0 or w <= 0 or h > 64 or w > 64:
            return False, "bad-shape"
        try:
            hist = np.bincount(g.ravel(), minlength=16)
            bg = int(np.argmax(hist))
            bg_frac = float(hist[bg]) / float(g.size)
        except Exception:
            return True, "ok-unknown-bg"
        try:
            if int(g[cy, cx]) == bg and bg_frac > _BG_REJECT_FRAC:
                return False, "massive-background"
        except Exception:
            return False, "cell-unreadable"
        try:  # 1-2px noise: foreground speck with <=2 filled 3x3 cells
            if int(g[cy, cx]) != bg:
                y0, y1 = max(0, cy - 1), min(h, cy + 2)
                x0, x1 = max(0, cx - 1), min(w, cx + 2)
                filled = int(np.count_nonzero(g[y0:y1, x0:x1] != bg))
                if filled <= _NOISE_NEIGHBORS:
                    return False, "pixel-noise"
        except Exception:
            pass
        return True, "ok"
    except Exception:
        return False, "guard-error"


# ── EvidenceLedger (L0 loop, bounded) ───────────────────────────────────────
class EvidenceLedger:
    """Record-only loop: observe -> verify -> bias -> macro-queue.

    * verified promotions require the verified_gate (no identity/shape wins).
    * disproven actions (>=3 no-change) are suppressed, never executed blind.
    * macro-queue holds at most 25 winning steps; flushed on level/score move.
    * stats() exposes a TSV-ledger-friendly snapshot; reset() is hermetic.
    """

    def __init__(self, max_entries: int = _MAX_LEDGER,
                 max_queue: int = _MAX_QUEUE) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_queue = max(1, int(max_queue))
        self._entries: deque = deque(maxlen=self.max_entries)
        self._queue: deque = deque(maxlen=self.max_queue)
        self._nochange: dict[Any, int] = {}
        self._verified_wins: dict[Any, int] = {}
        self._level: Any = None
        self._score: Any = None
        self._reweight = TokenReweight()

    def observe(self, before: Any, action: Any, after: Any, *,
                level: Any = None, score: Any = None,
                level_up: bool = False, score_gain: bool = False) -> dict[str, Any]:
        """Record one transition; returns {verified, weight, spans_n}."""
        try:
            fp = diff_fingerprint(before, after)
            spans = select_evidence_spans(before, after)
            key = (str(action).upper(), fp["novel_hash"])
            w = self._reweight.note(key)
            verified = verified_gate(fp["changed"], fp["shape_match"],
                                     bool(level_up), bool(score_gain))
            with _LOCK:
                if level is not None and self._level is not None and level != self._level:
                    self._queue.clear()
                if score is not None and self._score is not None and score != self._score:
                    self._queue.clear()
                self._level, self._score = level, score
                akey = str(action).upper()
                if fp["changed"]:
                    self._nochange.pop(akey, None)
                    if verified:
                        self._verified_wins[akey] = self._verified_wins.get(akey, 0) + 1
                        if len(self._queue) < self.max_queue:
                            self._queue.append(akey)
                else:
                    self._nochange[akey] = self._nochange.get(akey, 0) + 1
                self._entries.append({"action": akey, "verified": verified,
                                      "delta": fp["delta_nonzero"]})
                while len(self._nochange) > 64:
                    self._nochange.pop(next(iter(self._nochange)))
            return {"verified": verified, "weight": w, "spans_n": len(spans),
                    "delta": fp["delta_nonzero"]}
        except Exception:
            return {"verified": False, "weight": 0.10, "spans_n": 0, "delta": 0}

    def bias(self, legal: list[Any]) -> dict[str, float]:
        """Advisory action bias: boost verified, suppress disproven. Never raises."""
        try:
            out: dict[str, float] = {}
            with _LOCK:
                nc = dict(self._nochange)
                vw = dict(self._verified_wins)
            for a in list(legal or []):
                try:
                    k = str(a).upper()
                except Exception:
                    continue
                b = 0.0
                if vw.get(k, 0) > 0:
                    b += min(1.0, 0.25 * vw[k])
                if nc.get(k, 0) >= 3:
                    b -= min(1.0, 0.20 * nc[k])
                if b:
                    out[k] = round(b, 4)
            return out
        except Exception:
            return {}

    def macro_queue(self) -> list[str]:
        try:
            with _LOCK:
                return list(self._queue)
        except Exception:
            return []

    def stats(self) -> dict[str, int]:
        try:
            with _LOCK:
                return {"entries": len(self._entries), "queued": len(self._queue),
                        "verified_kinds": len(self._verified_wins),
                        "suppressed": sum(1 for v in self._nochange.values() if v >= 3)}
        except Exception:
            return {"entries": 0, "queued": 0, "verified_kinds": 0, "suppressed": 0}

    def reset(self) -> None:
        try:
            with _LOCK:
                self._entries.clear()
                self._queue.clear()
                self._nochange.clear()
                self._verified_wins.clear()
                self._level = None
                self._score = None
                self._reweight.reset()
        except Exception:
            pass


_LEDGER = EvidenceLedger()


def ledger() -> EvidenceLedger:
    return _LEDGER


def reset_all() -> None:
    try:
        _LEDGER.reset()
    except Exception:
        pass
