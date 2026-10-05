"""Deterministic long-horizon control primitives for ARC-AGI-3.

This module adapts publicly documented ideas from coding-agent harnesses:
bounded REPL-like observations, persistent transition memory, no-op guards,
and a supervisor that can recover from stalled hypotheses.  It deliberately
does not call an LLM or the competition API; the host agent supplies actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any
from collections.abc import Iterable


def state_hash(frame: Any) -> str:
    """Return a stable hash for nested grid-like data without numpy coupling."""
    if hasattr(frame, "tobytes") and hasattr(frame, "shape"):
        payload = f"{frame.shape}|{getattr(frame, 'dtype', '')}|".encode() + frame.tobytes()
    else:
        payload = repr(frame).encode("utf-8", "backslashreplace")
    return sha256(payload).hexdigest()[:24]


def abstraction_signature(frame: Any) -> tuple[Any, ...]:
    """Extract a compact, game-agnostic description of an observation.

    This is deliberately invariant-only: it gives the model useful structure
    without hard-coding a game id, pixel layout, or presumed rule.
    """
    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    if not rows:
        return (0, 0, 0, (), (), ())
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    colors: dict[str, int] = {}
    occupied_rows: list[int] = []
    occupied_cols: set[int] = set()
    for r, row in enumerate(rows):
        row_has_value = False
        for c, value in enumerate(row):
            key = str(value)
            colors[key] = colors.get(key, 0) + 1
            if value not in (0, "0", None):
                row_has_value = True
                occupied_cols.add(c)
        if row_has_value:
            occupied_rows.append(r)
    return (
        height,
        width,
        sum(colors.values()),
        tuple(sorted(colors.items())),
        tuple(occupied_rows),
        tuple(sorted(occupied_cols)),
    )


@dataclass(frozen=True)
class ActionKey:
    game_id: str
    level_id: str
    state: str
    action: int
    x: int | None = None
    y: int | None = None


@dataclass
class Transition:
    before: str
    after: str
    changed: bool
    reward: float = 0.0
    terminal: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    name: str
    score: float = 0.0
    failures: int = 0
    evidence: list[str] = field(default_factory=list)
    active: bool = True


class LongHorizonController:
    """Small, deterministic supervisor for interactive game agents.

    The controller does not choose semantic actions by itself.  It provides
    safe memory, context compaction, candidate ranking and recovery signals to
    a model or symbolic planner.
    """

    def __init__(self, context_limit: int = 48, max_transitions: int = 4096):
        if context_limit < 4 or max_transitions < 1:
            raise ValueError("limits must be positive and practical")
        self.context_limit = context_limit
        self.max_transitions = max_transitions
        self.transitions: dict[ActionKey, Transition] = {}
        self.events: list[dict[str, Any]] = []
        self.hypotheses: dict[str, Hypothesis] = {}
        self.abstract_memory: dict[tuple[Any, ...], dict[str, float]] = {}

    def blocked(self, key: ActionKey) -> bool:
        """Whether the exact action/state pair is known to be a no-op."""
        result = self.transitions.get(key)
        return result is not None and not result.changed and not result.terminal

    def record(self, key: ActionKey, transition: Transition) -> None:
        self.transitions[key] = transition
        self.events.append({"kind": "transition", "key": key, "value": transition})
        if len(self.transitions) > self.max_transitions:
            oldest = next(iter(self.transitions))
            del self.transitions[oldest]

    def register_hypothesis(self, name: str) -> Hypothesis:
        return self.hypotheses.setdefault(name, Hypothesis(name=name))

    def update_hypothesis(self, name: str, success: bool, evidence: str = "") -> None:
        hypothesis = self.register_hypothesis(name)
        if success:
            hypothesis.score += 1.0
        else:
            hypothesis.score -= 1.0
            hypothesis.failures += 1
        if evidence:
            hypothesis.evidence.append(evidence)
        if hypothesis.failures >= 3 and hypothesis.score < 0:
            hypothesis.active = False

    def learn_abstract_effect(self, before: Any, action: str, after: Any, reward: float) -> None:
        """Accumulate transferable action evidence without game identity."""
        before_sig = abstraction_signature(before)
        after_sig = abstraction_signature(after)
        key = (before_sig, str(action), after_sig)
        stats = self.abstract_memory.setdefault(key, {"count": 0.0, "reward": 0.0})
        stats["count"] += 1.0
        stats["reward"] += float(reward)

    def abstract_action_prior(self, before: Any, action: str) -> float:
        """Return a conservative prior learned from structurally similar states."""
        before_sig = abstraction_signature(before)
        matches = [
            stats for (seen, seen_action, _), stats in self.abstract_memory.items()
            if seen == before_sig and seen_action == str(action)
        ]
        if not matches:
            return 0.0
        count = sum(item["count"] for item in matches)
        return sum(item["reward"] for item in matches) / max(count, 1.0)

    @staticmethod
    def _coarse_key(sig: tuple[Any, ...]) -> tuple[Any, ...]:
        """Back-off key: (height, width, n_colors, fg_cells). Game-agnostic."""
        try:
            h, w, total, colors = sig[0], sig[1], sig[2], sig[3]
            counts = [int(c) for _, c in colors]
            n_colors = len(counts)
            fg = int(total) - (max(counts) if counts else 0)
            return (int(h), int(w), n_colors, fg)
        except Exception:
            return ("?",)

    def transfer_prior(self, before: Any, action: str) -> tuple[float, int]:
        """Cross-level prior: (mean_reward, evidence_count) for an action.

        Tiers (best non-empty wins): exact structural signature, then coarse
        (same dims/color-count/foreground mass), then action-global base
        rate. Exact-tier values are identical to abstract_action_prior.
        Never throws.
        """
        try:
            act = str(action)
            before_sig = abstraction_signature(before)
            if before_sig == (0, 0, 0, (), (), ()):
                return (0.0, 0)  # no structure observed -> no prior, fail closed
            exact = [
                stats for (seen, seen_action, _), stats in self.abstract_memory.items()
                if seen == before_sig and seen_action == act
            ]
            if exact:
                count = int(sum(item["count"] for item in exact))
                mean = sum(item["reward"] for item in exact) / max(count, 1)
                return (float(mean), count)
            coarse = self._coarse_key(before_sig)
            pool = [
                stats for (seen, seen_action, _), stats in self.abstract_memory.items()
                if seen_action == act and self._coarse_key(seen) == coarse
            ]
            if pool:
                count = int(sum(item["count"] for item in pool))
                mean = sum(item["reward"] for item in pool) / max(count, 1)
                return (float(mean), count)
            glob = [
                stats for (_, seen_action, _), stats in self.abstract_memory.items()
                if seen_action == act
            ]
            if glob:
                count = int(sum(item["count"] for item in glob))
                mean = sum(item["reward"] for item in glob) / max(count, 1)
                return (float(mean), count)
        except Exception:
            pass
        return (0.0, 0)

    def rank_actions(self, candidates: Iterable[tuple[ActionKey, float]]) -> list[tuple[ActionKey, float]]:
        """Filter proven no-ops and rank by utility, stably by action id."""
        ranked = [(key, utility) for key, utility in candidates if not self.blocked(key)]
        return sorted(ranked, key=lambda item: (-float(item[1]), item[0].action, item[0].x or -1, item[0].y or -1))

    def append_event(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))

    def compact_context(self) -> list[dict[str, Any]]:
        """Keep the system-relevant tail while bounding long-horizon memory."""
        if len(self.events) <= self.context_limit:
            return list(self.events)
        head = self.events[:1]
        tail = self.events[-(self.context_limit - 2):]
        return head + [{"kind": "context_evicted", "count": len(self.events) - len(head) - len(tail)}] + tail

    def supervisor_snapshot(self) -> dict[str, Any]:
        active = sorted((h.name, h.score, h.failures) for h in self.hypotheses.values() if h.active)
        return {
            "transitions": len(self.transitions),
            "events": len(self.events),
            "active_hypotheses": active,
            "no_op_transitions": sum(not t.changed for t in self.transitions.values()),
            "abstract_rules": len(self.abstract_memory),
        }
