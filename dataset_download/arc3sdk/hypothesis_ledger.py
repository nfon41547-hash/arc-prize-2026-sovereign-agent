"""Hypothesis ledger with retrodiction (v26-team-1).

Distilled control pattern from multi-agent team solvers (propose ->
predict -> observe -> confirm/refute), reimplemented as numpy-free
bookkeeping: no LLM, no threads beyond one RLock, never raises.

Contract:
- propose(): a tier (or shadow-denied tier) registers its predicted
  effect (confidence) for this turn. At most one pending entry kept
  per game; newer proposals expire older ones (bounded).
- observe(): the next turn's realized board-change resolves the pending
  entry of the same game/level. Entries tagged "shadow:*" (tiers the
  strict allowlist denied) are NEVER resolved -- they contribute
  proposed-volume + confidence mass only, so post-hoc analysis can
  compare tier confidence against analyzer outcomes without fake
  attribution.
- weight(): Laplace-smoothed confirm rate in (0, 1); 0.5 with no data
  (fail-open neutral). Internal score only -- this module never
  substitutes anything itself.

stdlib at import. Bounded (64 games, 64 tags). Never raises.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

__version__ = "v26-team-1"

_LOCK = threading.RLock()
_MAX_GAMES = 64
_MAX_TAGS = 64


class HypothesisLedger:
    """Per-game pending predictions + per-tag confirm/refute tallies."""

    def __init__(self) -> None:
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tags: OrderedDict[str, list] = OrderedDict()

    def _tag_entry(self, tag: str) -> list:
        try:
            t = str(tag or "untagged")[:48]
        except Exception:
            t = "untagged"
        ent = self._tags.get(t)
        if not isinstance(ent, list) or len(ent) != 4:
            ent = [0, 0, 0, 0.0]
            self._tags[t] = ent
            self._tags.move_to_end(t)
            while len(self._tags) > _MAX_TAGS:
                self._tags.popitem(last=False)
        return ent

    def propose(self, game_id: Any, level: Any, action: Any,
                conf: Any, tag: Any = "") -> None:
        """Register this turn's predicted effect. Never raises."""
        try:
            try:
                c = float(conf)  # type: ignore[arg-type]
            except Exception:
                return
            if not (0.0 <= c <= 1.0):
                return
            try:
                game = str(game_id or "duck")
                lvl = max(0, int(level))  # type: ignore[arg-type]
                aid = int(action)  # type: ignore[arg-type]
            except Exception:
                return
            with _LOCK:
                self._pending[game] = {
                    "level": lvl, "action": aid, "conf": c,
                    "tag": str(tag or "untagged")[:48],
                }
                self._pending.move_to_end(game)
                while len(self._pending) > _MAX_GAMES:
                    self._pending.popitem(last=False)
                self._tag_entry(str(tag or "untagged")[:48])[0] += 1
        except Exception:
            pass

    def observe(self, game_id: Any, level: Any, changed: Any) -> bool | None:
        """Resolve the pending entry with the realized outcome.

        Returns True (confirmed: change predicted, change seen), False
        (refuted or no-change), None (nothing pending / shadow entry /
        level moved on). Shadow-tagged entries are expired, never scored.
        """
        try:
            game = str(game_id or "duck")
            try:
                lvl = max(0, int(level))  # type: ignore[arg-type]
            except Exception:
                return None
            with _LOCK:
                pend = self._pending.get(game)
                if not pend:
                    return None
                if int(pend.get("level", -1)) != lvl:
                    self._pending.pop(game, None)
                    return None
                tag = str(pend.get("tag", "untagged"))
                if tag.startswith("shadow:"):
                    self._pending.pop(game, None)
                    return None
                self._pending.pop(game, None)
                ent = self._tag_entry(tag)
                ent[1] += 1
                hit = bool(changed)
                if hit:
                    ent[2] += 1
                try:
                    ent[3] += float(pend.get("conf", 0.0))
                except Exception:
                    pass
                return hit
        except Exception:
            return None

    def weight(self, tag: Any) -> float:
        """Laplace-smoothed confirm rate; 0.5 with no data. Never raises."""
        try:
            ent = self._tags.get(str(tag or "untagged")[:48])
            if not isinstance(ent, list) or int(ent[1]) <= 0:
                return 0.5
            return (float(ent[2]) + 1.0) / (float(ent[1]) + 2.0)
        except Exception:
            return 0.5

    def stats(self) -> dict[str, dict[str, float]]:
        """Per-tag {proposed, resolved, confirmed, mean_conf}. Never raises."""
        try:
            out: dict[str, dict[str, float]] = {}
            with _LOCK:
                items = list(self._tags.items())
            for t, ent in items:
                try:
                    n_res = int(ent[1])
                    out[t] = {
                        "proposed": float(ent[0]),
                        "resolved": float(n_res),
                        "confirmed": float(ent[2]),
                        "mean_conf": (float(ent[3]) / n_res) if n_res else 0.0,
                    }
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    def reset(self) -> None:
        """Clear all state (hermetic tests). Never raises."""
        try:
            with _LOCK:
                self._pending.clear()
                self._tags.clear()
        except Exception:
            pass


_SHARED = HypothesisLedger()


def shared_ledger() -> HypothesisLedger:
    """Module singleton. Never raises."""
    return _SHARED


def reset_shared() -> None:
    """Clear the singleton (tests). Never raises."""
    try:
        _SHARED.reset()
    except Exception:
        pass
