"""Evidence-Gated Causal Discovery (EGCD) action policy.

ARC-AGI-3 has sparse rewards and expensive irreversible mistakes.  EGCD does
not bootstrap a value function from imagined rewards; it maintains compact,
game-isolated causal evidence and spends exploration only when it can reduce
hypothesis uncertainty.  The policy is deterministic for reproducible replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import numpy as np


def _sig(grid: Any) -> str:
    arr = np.asarray(grid, dtype=np.uint8)
    return sha256(arr.tobytes()).hexdigest()[:16]


@dataclass
class _Evidence:
    trials: int = 0
    changes: int = 0
    terminal: int = 0
    noops: int = 0

    @property
    def p_change(self) -> float:
        return (self.changes + 1.0) / (self.trials + 2.0)

    @property
    def risk(self) -> float:
        return (self.terminal + 0.5 * self.noops) / max(1, self.trials)


@dataclass
class EGCDPolicy:
    """Causal action selector with bounded exploration and no game lookup."""

    seed: int = 0
    _tables: dict[str, dict[int, _Evidence]] = field(default_factory=dict)
    _last: dict[str, tuple[str, int]] = field(default_factory=dict)

    def reset(self, game_id: str | None = None) -> None:
        if game_id is None:
            self._tables.clear()
            self._last.clear()
        else:
            self._tables.pop(str(game_id), None)
            self._last.pop(str(game_id), None)

    def observe(self, game_id: str, grid: Any, action: int, next_grid: Any, *, game_over: bool = False) -> None:
        gid = str(game_id)
        prev = self._last.get(gid)
        if prev is not None:
            state_sig, prev_action = prev
            if state_sig == _sig(grid):
                ev = self._tables.setdefault(state_sig, {}).setdefault(int(prev_action), _Evidence())
                ev.trials += 1
                changed = _sig(next_grid) != state_sig
                ev.changes += int(changed)
                ev.noops += int(not changed)
                ev.terminal += int(game_over)
        self._last[gid] = (_sig(next_grid), int(action))

    def choose(self, game_id: str, grid: Any, legal: list[int] | tuple[int, ...], *, stagnation: int = 0) -> tuple[int, float, str]:
        actions = sorted({int(a) for a in legal if int(a) in range(1, 8)})
        if not actions:
            return 1, 0.0, "egcd:no-legal-action"
        state = _sig(grid)
        evidence = self._tables.setdefault(state, {})
        # Unknown actions get a deterministic information bonus.  Once the
        # board stagnates, prefer the least-tested safe action to break loops.
        scored = []
        for a in actions:
            ev = evidence.setdefault(a, _Evidence())
            novelty = 1.0 / (1.0 + ev.trials)
            score = 1.8 * ev.p_change - 2.4 * ev.risk + (0.55 * novelty if stagnation else 0.15 * novelty)
            if a == 6:
                score -= 0.08  # click only when evidence supports it
            scored.append((score, -ev.trials, -a, a))
        _, _, _, action = max(scored)
        ev = evidence[action]
        confidence = max(0.0, min(1.0, ev.p_change * (1.0 - min(1.0, ev.risk))))
        return action, confidence, "egcd:causal-evidence" if ev.trials else "egcd:information-gain"
