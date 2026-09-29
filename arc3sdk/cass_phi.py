"""CASS-Phi execution law (v18-phi-1) — mandatory Predict -> Execute -> Audit.

Elevates the utility formula from a reward term to an execution-level
invariant (standing order):

    No execution without expected positive utility;
    no execution survives without reusable evidence.

Per turn the gate enforces, BEFORE acting::

    Execute(a) = 1[a in A_legal and U_pred > tau and Novelty(a|H) > nu_min
                   and not vetoed(a)]

and AFTER acting it scores the REALIZED utility from the true before/after
frames (s_t -> s_{t+1}), never hallucinated rewards::

    U_obs = change + 0.5*novel_state - 0.1*static        (locked by tests)
    D     = 1[U_obs <= 0]                                 (dead work)
    D_dup = 1[same (state,action) no-effect seen this level]
    D_stale = 1[no-effect and zero productive turns this level so far]
    r     = U_obs - eta*(D + alpha*D_dup + beta*D_stale) (eta=alpha=beta=1)

Negative memory M-: (game, level, state-fp, action) with strike counts.
3 strikes (ARC3_PHI_KAPPA) suppress the pair. Validity domain is the exact
(state, level): a different state or level auto-revives (no permanent
blacklists — ARC mechanics are state-dependent). Regret counterfactuals
and API-dollar accounting are NOT implemented (unmeasurable in-payload;
stated honestly, never fabricated).

Every executed turn leaves an artifact
  (fp, game, level, step, pre_fp, action, coords, post_fp, delta,
   U_pred, U_obs, dead flags, source, latency)
in a bounded ledger. Metrics (locked by tests):
  UAR = useful turns / audited turns    (U_obs>0 or knowledge reused)
  DCR = dead-flagged turns / audited    (turn-based proxy, documented)
  ARY = artifacts reused downstream / artifacts
  IPA = mean per-turn information gain
LPD (levels per API dollar) is OMITTED: API cost is unknowable in-payload.

Intrinsic only: fingerprints are within-run state hashes; nothing is
preloaded, nothing persists across runs. stdlib-only at import
(hashlib/os/threading/collections/time). Bounded stores. One RLock.
Never raises.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict, deque
from typing import Any

__version__ = "v18-phi-1"

_LOCK = threading.RLock()

_MAX_GAMES = 64
_MAX_ARTIFACTS = 512
_MAX_NEG = 1024
_MAX_RECENT = 64


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(float(os.environ.get(name, default))))
    except Exception:
        return default


def fingerprint(game_id: Any, level: Any, grid_bytes: bytes,
                action_id: Any, x: Any = None, y: Any = None) -> str:
    """Canonical action fingerprint. Never raises."""
    try:
        h = hashlib.blake2b(digest_size=12)
        h.update(str(game_id or "").encode("utf-8", "replace"))
        h.update(b"|")
        h.update(str(int(level or 0)).encode())
        h.update(b"|")
        h.update(bytes(grid_bytes or b""))
        h.update(b"|")
        h.update(str(action_id).encode())
        if x is not None and y is not None:
            h.update(("%s,%s" % (x, y)).encode())
        return h.hexdigest()
    except Exception:
        return "err"


def state_fp(grid_bytes: bytes) -> str:
    """Canonical pre/post-state fingerprint. Never raises."""
    try:
        return hashlib.blake2b(bytes(grid_bytes or b""),
                               digest_size=12).hexdigest()
    except Exception:
        return "err"


def interior_changed(before: bytes, after: bytes, h: int, w: int) -> bool:
    """Interior-only change (2px border HUD excluded). Never raises."""
    try:
        if not before or not after or len(before) != len(after):
            return bytes(before or b"") != bytes(after or b"")
        if h <= 4 or w <= 4 or h * w != len(before):
            return before != after
        for r in range(2, h - 2):
            base = r * w
            if before[base + 2:base + w - 2] != after[base + 2:base + w - 2]:
                return True
        return False
    except Exception:
        try:
            return bytes(before or b"") != bytes(after or b"")
        except Exception:
            return False


class PhiLedger:
    """Per-run execution ledger + negative memory. All methods never raise."""

    def __init__(self) -> None:
        self._artifacts: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._neg: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self._recent: dict[tuple, deque] = {}
        self._seen_state: dict[tuple, set] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._steps: dict[tuple, int] = {}
        self._productive: dict[tuple, int] = {}
        self._turns: dict[tuple, int] = {}

    # ---- internal helpers ----
    def _scope(self, game_id: Any, level: Any) -> tuple:
        try:
            return (str(game_id or "duck"), max(0, int(level or 0)))
        except Exception:
            return ("duck", 0)

    def _evict(self) -> None:
        try:
            while len(self._artifacts) > _MAX_ARTIFACTS:
                self._artifacts.popitem(last=False)
            while len(self._neg) > _MAX_NEG:
                self._neg.popitem(last=False)
        except Exception:
            pass

    # ---- Predict gate ----
    def check(self, game_id: Any, level: Any, pre_fp: str, action_id: Any,
              U_pred: Any, tau: float = 0.80) -> dict[str, Any]:
        """Decide Execute(a) in {0,1} with reasons. Never raises."""
        try:
            scope = self._scope(game_id, level)
            nu_min = _env_float("ARC3_PHI_NU", 1.0)
            kappa = _env_int("ARC3_PHI_KAPPA", 3)
            try:
                u = float(U_pred) if U_pred is not None else -1.0
            except Exception:
                u = -1.0
            reasons = []
            ok_pred = U_pred is not None and u > float(tau)
            if not ok_pred:
                reasons.append("U_pred<=tau" if U_pred is not None else "U_pred-missing")
            recent = self._recent.get(scope)
            nov_ok = True
            if nu_min > 0 and recent is not None:
                try:
                    nov_ok = not any(
                        r[0] == str(pre_fp) and str(r[1]) == str(action_id) and r[2]
                        for r in recent)
                except Exception:
                    nov_ok = True
            if not nov_ok:
                reasons.append("repeat-noeffect")
            vetoed = False
            try:
                key = (scope[0], scope[1], str(pre_fp), str(action_id))
                entry = self._neg.get(key)
                if entry and int(entry.get("strikes", 0)) >= max(2, kappa):
                    vetoed = True
                    reasons.append("negative-memory")
            except Exception:
                vetoed = False
            allow = bool(ok_pred and nov_ok and not vetoed)
            return {"allow": allow, "U_pred": u, "novelty_ok": nov_ok,
                    "vetoed": vetoed, "reasons": reasons}
        except Exception:
            return {"allow": False, "U_pred": -1.0, "novelty_ok": False,
                    "vetoed": False, "reasons": ["error"]}

    # ---- Execute record ----
    def observe_turn(self, game_id: Any, level: Any, pre_fp: str,
                     pre_bytes: bytes, h: int, w: int, action_id: Any,
                     x: Any = None, y: Any = None, U_pred: Any = None,
                     source: str = "", legal: Any = None, tau: Any = None,
                     decision: Any = None) -> dict[str, Any]:
        """Record an executed turn as pending audit. Never raises."""
        try:
            scope = self._scope(game_id, level)
            step = int(self._steps.get(scope, 0)) + 1
            self._steps[scope] = step
            self._turns[scope] = int(self._turns.get(scope, 0)) + 1
            fp = fingerprint(game_id, scope[1], pre_bytes, action_id, x, y)
            dedup = any(a.get("pre_fp") == pre_fp
                        and str(a.get("action")) == str(action_id)
                        for a in self._artifacts.values()
                        if a.get("game") == scope[0] and a.get("level") == scope[1])
            try:
                legal_snap = tuple(int(v) for v in (legal or []))[:9]
            except Exception:
                legal_snap = ()
            turn = {"fp": fp, "game": scope[0], "level": scope[1], "step": step,
                    "pre_fp": str(pre_fp), "action": action_id, "x": x, "y": y,
                    "U_pred": U_pred, "source": str(source or ""),
                    "pre_bytes": bytes(pre_bytes or b""), "h": int(h), "w": int(w),
                    "dedup_hit": bool(dedup), "latency_s": 0.0,
                    "legal": legal_snap, "tau": tau, "decision": decision}
            with _LOCK:
                self._pending[scope[0]] = turn
            return dict(turn)
        except Exception:
            return {}

    # ---- Audit ----
    def audit_turn(self, game_id: Any, level: Any, post_fp: str,
                   post_bytes: bytes) -> dict[str, Any] | None:
        """Score the pending turn against the true post frame. Never raises."""
        try:
            eta = _env_float("ARC3_PHI_ETA", 1.0)
            alpha = _env_float("ARC3_PHI_ALPHA", 1.0)
            beta = _env_float("ARC3_PHI_BETA", 1.0)
            with _LOCK:
                turn = self._pending.pop(str(game_id or "duck"), None)
            if not turn:
                return None
            scope = (turn.get("game", "duck"), int(turn.get("level", 0) or 0))
            cur = self._scope(game_id, level)
            if cur != scope:
                # level/game moved on: audit under the turn's own scope,
                # then reset the new scope's per-level state.
                self.on_level(cur[0], cur[1])
            try:
                changed = interior_changed(turn.get("pre_bytes", b""),
                                           bytes(post_bytes or b""),
                                           int(turn.get("h", 0)), int(turn.get("w", 0)))
            except Exception:
                changed = False
            seen = self._seen_state.setdefault(scope, set())
            novel = str(post_fp) not in seen
            if novel:
                seen.add(str(post_fp))
                while len(seen) > 2048:
                    seen.pop()
            static = not changed
            U_obs = (1.0 if changed else 0.0) + (0.5 if novel else 0.0) - (0.1 if static else 0.0)
            d = 1 if U_obs <= 0 else 0
            # dup: same (state,action) no-effect recorded before this level
            d_dup = 0
            try:
                for a in self._artifacts.values():
                    if (a.get("game") == scope[0] and a.get("level") == scope[1]
                            and a.get("pre_fp") == turn.get("pre_fp")
                            and str(a.get("action")) == str(turn.get("action"))
                            and a.get("delta") == "no-change"):
                        d_dup = 1
                        break
            except Exception:
                d_dup = 0
            # stale: no-effect, already-seen outcome, and the level shows
            # zero productive turns (led nowhere). First-time no-ops are
            # NOT stale: they teach (s,a)->nothing and feed negative memory.
            prod = int(self._productive.get(scope, 0))
            d_stale = 1 if (not changed and not novel and prod == 0) else 0
            r = U_obs - eta * (d + alpha * d_dup + beta * d_stale)
            if changed:
                self._productive[scope] = prod + 1
            delta = "changed" if changed else "no-change"
            consumers = ["audit"]
            # negative memory update
            try:
                key = (scope[0], scope[1], str(turn.get("pre_fp")), str(turn.get("action")))
                if not changed:
                    with _LOCK:
                        e = self._neg.get(key, {"strikes": 0})
                        e["strikes"] = int(e.get("strikes", 0)) + 1
                        self._neg[key] = e
                        self._neg.move_to_end(key)
                    consumers.append("negative-memory")
            except Exception:
                pass
            # recent no-effect window
            try:
                rec = self._recent.setdefault(scope, deque(maxlen=_MAX_RECENT))
                rec.append((str(turn.get("pre_fp")), str(turn.get("action")), (not changed)))
            except Exception:
                pass
            art = {"fp": turn.get("fp"), "game": scope[0], "level": scope[1],
                   "step": turn.get("step"), "pre_fp": turn.get("pre_fp"),
                   "action": turn.get("action"), "x": turn.get("x"), "y": turn.get("y"),
                   "legal": turn.get("legal", ()),
                   "post_fp": str(post_fp),
                   "post_level": int(level if level is not None else scope[1]),
                   "delta": delta,
                   "U_pred": turn.get("U_pred"), "tau": turn.get("tau"),
                   "execution": turn.get("decision", "executed"),
                   "U_obs": round(float(U_obs), 4),
                   "dead": d, "dead_dup": d_dup, "dead_stale": d_stale,
                   "reward": round(float(r), 4),
                   "provenance": {"source": turn.get("source", ""),
                                  "policy": "cass-phi/%s" % __version__,
                                  "latency_s": round(float(turn.get("latency_s", 0.0)), 3)},
                   "validity": {"scope": "game+level", "status": "observed"},
                   "consumers": consumers,
                   "reused": bool(turn.get("dedup_hit"))}
            with _LOCK:
                # Key includes step: same (state,action) repeats are distinct
                # turns (dedup is detected by VALUE scan, never by key loss).
                self._artifacts["%s#%d" % (art["fp"], int(art.get("step", 0)))] = art
                self._evict()
            return dict(art)
        except Exception:
            return None

    def note_latency(self, game_id: Any, latency_s: float) -> None:
        try:
            with _LOCK:
                t = self._pending.get(str(game_id or "duck"))
                if t is not None:
                    t["latency_s"] = max(0.0, float(latency_s))
        except Exception:
            pass

    # ---- scope hygiene ----
    def on_level(self, game_id: Any, level: Any) -> None:
        """Level change: drop this game's per-level negative/recent state."""
        try:
            gid = str(game_id or "duck")
            with _LOCK:
                for key in [k for k in self._neg if k[0] == gid]:
                    self._neg.pop(key, None)
                for scope in [s for s in self._recent if s[0] == gid]:
                    self._recent.pop(scope, None)
                for scope in [s for s in self._seen_state if s[0] == gid]:
                    self._seen_state.pop(scope, None)
        except Exception:
            pass

    def reset(self) -> None:
        try:
            with _LOCK:
                self._artifacts.clear()
                self._neg.clear()
                self._recent.clear()
                self._seen_state.clear()
                self._pending.clear()
                self._steps.clear()
                self._productive.clear()
                self._turns.clear()
        except Exception:
            pass

    # ---- metrics ----
    def metrics(self, game_id: Any = None) -> dict[str, Any]:
        """UAR/DCR/ARY/IPA + counters. Never raises. LPD omitted (no API$)."""
        try:
            with _LOCK:
                arts = [a for a in self._artifacts.values()
                        if game_id is None or a.get("game") == str(game_id)]
            n = len(arts)
            if n == 0:
                return {"n": 0, "UAR": 0.0, "DCR": 0.0, "ARY": 0.0, "IPA": 0.0,
                        "dead": 0, "productive": 0, "version": __version__}
            useful = sum(1 for a in arts
                         if float(a.get("U_obs", 0.0)) > 0)
            dead = sum(1 for a in arts if int(a.get("dead", 0)) == 1)
            reused = sum(1 for a in arts
                         if a.get("reused") and float(a.get("U_obs", 0.0)) > 0)
            # IPA counts change-driven information only. Novelty-at-audit is
            # the finer signal but is not stored per artifact — stated
            # honestly rather than reconstructed.
            info = sum(1.0 if a.get("delta") == "changed" else 0.0 for a in arts)
            prod = sum(1 for a in arts if a.get("delta") == "changed")
            return {"n": n, "UAR": round(useful / n, 4), "DCR": round(dead / n, 4),
                    "ARY": round(reused / n, 4), "IPA": round(info / n, 4),
                    "dead": dead, "productive": prod, "version": __version__}
        except Exception:
            return {"n": 0, "UAR": 0.0, "DCR": 0.0, "ARY": 0.0, "IPA": 0.0,
                    "dead": 0, "productive": 0}


# ---------- module-shared ledger (worker: one per process) ----------

_SHARED = PhiLedger()


def shared() -> PhiLedger:
    return _SHARED


def reset_shared() -> None:
    _SHARED.reset()
