"""Certified merge gates: exact dedup + Clopper-Pearson semantic certification.

Two merge paths (fail-open, bounded, stdlib-only at import):

  exact_dedup(a, b) — byte equality of state payloads merges immediately.
  SemanticMergeGate — semantic merge only when ALL hold:
      d_E <= tau_E, |V_i - V_j| <= tau_V, d_A <= tau_A, d_P <= tau_P
      AND UCB_0.95(P_wrong_merge) <= alpha over accumulated feedback.

The gate starts UNCERTIFIED (no history -> UCB = 1.0) and can prove itself
not safe enough: unsafe feedback drives UCB up and merges get refused.
Thresholds are never silently lowered to flatter a benchmark.

semantic_id() builds rename-stable identity (content hash, path-independent)
so search memory survives file renames/refactors: ID_before == ID_after.
"""

from __future__ import annotations

import hashlib
import threading
from math import comb

__version__ = "v10-cassspx-1"

_LOCK = threading.RLock()


def _binom_cdf_upto(k: int, n: int, p: float) -> float:
    try:
        k = max(-1, min(int(k), int(n)))
        n = int(n)
        p = min(1.0, max(0.0, float(p)))
        if k < 0:
            return 0.0
        if k >= n:
            return 1.0
        q = 1.0 - p
        return sum(comb(n, i) * (p ** i) * (q ** (n - i)) for i in range(k + 1))
    except Exception:
        return 1.0


def cp_upper(k: int, n: int, cl: float = 0.95) -> float:
    """Exact Clopper-Pearson upper bound for Binomial(n, p), k successes."""
    try:
        k, n = int(k), int(n)
        cl = min(0.9999, max(0.5, float(cl)))
        if n <= 0 or k >= n:
            return 1.0
        if k < 0:
            return 0.0
        lo, hi = 0.0, 1.0
        target = 1.0 - cl
        for _ in range(200):
            m = (lo + hi) / 2.0
            if _binom_cdf_upto(k, n, m) > target:
                lo = m
            else:
                hi = m
        return (lo + hi) / 2.0
    except Exception:
        return 1.0


def certify(accepted: int, wrong: int, alpha: float, cl: float = 0.95) -> dict:
    """Certification verdict over merge feedback. Never raises."""
    try:
        a, w = max(0, int(accepted)), max(0, int(wrong))
        ucb = cp_upper(w, a, cl)
        ok = bool(ucb <= float(alpha))
        return {"certified": ok, "ucb": round(ucb, 6), "n": a, "k": w,
                "alpha": float(alpha), "cl": float(cl)}
    except Exception:
        return {"certified": False, "ucb": 1.0, "n": 0, "k": 0,
                "alpha": 0.05, "cl": 0.95}


def exact_dedup(a: bytes, b: bytes) -> bool:
    """Byte-equality merge: immediate, no certification needed."""
    try:
        return bytes(a) == bytes(b)
    except Exception:
        return False


def semantic_id(payload: bytes, kind: str = "state") -> str:
    """Rename-stable identity: content hash only, no paths. Never raises."""
    try:
        h = hashlib.blake2b(bytes(payload), digest_size=8).hexdigest()
        return f"fib-{kind}-{h}"
    except Exception:
        return "fib-unknown-error"


class SemanticMergeGate:
    """Threshold gates + UCB certification over feedback. Never raises."""

    def __init__(self, tau_E: float = 0.05, tau_V: float = 0.02,
                 tau_A: float = 0.05, tau_P: float = 0.05,
                 alpha: float = 0.05, cl: float = 0.95) -> None:
        self.tau_E = float(tau_E)
        self.tau_V = float(tau_V)
        self.tau_A = float(tau_A)
        self.tau_P = float(tau_P)
        self.alpha = float(alpha)
        self.cl = float(cl)
        self._accepted = 0
        self._wrong = 0

    def attempt(self, d_E: float, d_V: float, d_A: float, d_P: float) -> tuple:
        """Request a merge. Returns (accepted: bool, reason: str)."""
        try:
            checks = (("E", float(d_E), self.tau_E), ("V", float(d_V), self.tau_V),
                      ("A", float(d_A), self.tau_A), ("P", float(d_P), self.tau_P))
            for name, d, tau in checks:
                if not (d <= tau):
                    return False, f"threshold-{name}-exceeded"
            with _LOCK:
                ucb = cp_upper(self._wrong, self._accepted, self.cl)
                if ucb > self.alpha:
                    return False, "uncertified-ucb"
                self._accepted += 1
            return True, "merged-certified"
        except Exception:
            return False, "gate-error"

    def feedback(self, was_wrong: bool) -> None:
        """Report a merged outcome. Never raises."""
        try:
            if bool(was_wrong):
                with _LOCK:
                    self._wrong += 1
        except Exception:
            pass

    def calibrate(self, accepted: int, wrong: int) -> None:
        """Ingest a controlled TRAIN/CERT run (known outcomes) to seed history."""
        try:
            with _LOCK:
                self._accepted += max(0, int(accepted))
                self._wrong += max(0, int(wrong))
        except Exception:
            pass

    def status(self) -> dict:
        try:
            with _LOCK:
                a, w = self._accepted, self._wrong
            rep = certify(a, w, self.alpha, self.cl)
            rep["thresholds"] = {"E": self.tau_E, "V": self.tau_V,
                                 "A": self.tau_A, "P": self.tau_P}
            return rep
        except Exception:
            return {"certified": False, "ucb": 1.0}

    def reset(self) -> None:
        try:
            with _LOCK:
                self._accepted = 0
                self._wrong = 0
        except Exception:
            pass
