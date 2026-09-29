"""ARC-AGI-3 replay buffer for real online trajectories.

This is an inference-time learning buffer, not a synthetic simulator.  It is
designed for sparse rewards and expensive environment actions:

* deduplicates exact (game, level, state, action) decisions;
* prioritizes surprising state changes, terminal outcomes and failures;
* samples stratified by game/level so one long game cannot dominate memory;
* keeps compact uint8 grids and bounded memory with deterministic sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import blake2b
import json
from pathlib import Path
from typing import Any

import numpy as np


def _hash_grid(grid: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.uint8))
    return blake2b(arr.tobytes(), digest_size=12).hexdigest()


@dataclass(frozen=True)
class ReplayTransition:
    game_id: str
    level: int
    state_hash: str
    action: str
    next_state_hash: str
    state: np.ndarray
    next_state: np.ndarray
    reward: float
    terminal: bool
    changed: bool
    priority: float
    visits: int = 1


class ArcRLReplayBuffer:
    """Bounded prioritized replay for real ARC environment observations."""

    def __init__(self, capacity: int = 4096, seed: int = 0) -> None:
        if capacity < 32:
            raise ValueError("capacity must be at least 32")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._items: dict[tuple[str, int, str, str], ReplayTransition] = {}
        self._strata: dict[tuple[str, int], set[tuple[str, int, str, str]]] = {}

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _priority(reward: float, terminal: bool, changed: bool, novelty: float, visits: int) -> float:
        # Terminal evidence and failures are more valuable than repeated no-ops.
        terminal_bonus = 3.0 if terminal else 0.0
        failure_bonus = 1.5 if reward < 0 else 0.0
        change_bonus = 1.0 if changed else 0.0
        return max(1e-3, abs(float(reward)) + terminal_bonus + failure_bonus + change_bonus + float(novelty) + 1.0 / visits)

    def add(
        self,
        game_id: str,
        level: int,
        state: Any,
        action: str,
        next_state: Any,
        reward: float,
        terminal: bool = False,
    ) -> bool:
        s = np.asarray(state, dtype=np.uint8)
        ns = np.asarray(next_state, dtype=np.uint8)
        if s.shape != ns.shape or s.ndim != 2 or s.shape != (64, 64):
            return False
        sh, nsh = _hash_grid(s), _hash_grid(ns)
        changed = not np.array_equal(s, ns)
        key = (str(game_id), int(level), sh, str(action).upper())
        previous = self._items.get(key)
        if previous is not None:
            # Update evidence without duplicating the same decision.
            visits = previous.visits + 1
            merged_reward = max(previous.reward, float(reward))
            item = ReplayTransition(
                previous.game_id, previous.level, previous.state_hash,
                previous.action, nsh, previous.state, ns, merged_reward,
                bool(previous.terminal or terminal), bool(previous.changed or changed),
                self._priority(merged_reward, previous.terminal or terminal, previous.changed or changed, 0.0, visits), visits,
            )
            self._items[key] = item
            return False
        novelty = 1.0 if nsh != sh else 0.0
        item = ReplayTransition(
            str(game_id), int(level), sh, str(action).upper(), nsh,
            s.copy(), ns.copy(), float(reward), bool(terminal), changed,
            self._priority(reward, terminal, changed, novelty, 1), 1,
        )
        self._items[key] = item
        self._strata.setdefault((item.game_id, item.level), set()).add(key)
        self._trim()
        return True

    def _trim(self) -> None:
        while len(self._items) > self.capacity:
            # Preserve terminal/failure evidence; evict the weakest nonterminal item.
            candidates = [x for x in self._items.values() if not x.terminal]
            victim = min(candidates or list(self._items.values()), key=lambda x: x.priority)
            key = (victim.game_id, victim.level, victim.state_hash, victim.action)
            self._items.pop(key, None)
            self._strata.get((victim.game_id, victim.level), set()).discard(key)

    def sample(self, batch_size: int, *, beta: float = 0.4) -> list[ReplayTransition]:
        if batch_size <= 0 or not self._items:
            return []
        by_stratum = list(self._strata.items())
        selected: list[ReplayTransition] = []
        # Round-robin strata first, then prioritized fill.
        for _, keys in by_stratum:
            if len(selected) >= batch_size:
                break
            available = [self._items[k] for k in keys if k in self._items]
            if available:
                selected.append(max(available, key=lambda x: x.priority))
        selected_keys = {(x.game_id, x.level, x.state_hash, x.action) for x in selected}
        remaining = [
            x for key, x in self._items.items()
            if key not in selected_keys
        ]
        if len(selected) < batch_size and remaining:
            weights = np.asarray([max(x.priority, 1e-6) ** float(beta) for x in remaining], dtype=np.float64)
            weights /= weights.sum()
            count = min(batch_size - len(selected), len(remaining))
            indices = self._rng.choice(len(remaining), size=count, replace=False, p=weights)
            selected.extend(remaining[int(i)] for i in np.atleast_1d(indices))
        return selected

    def snapshot(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "size": len(self), "strata": len(self._strata), "terminal": sum(x.terminal for x in self._items.values())}

    def save(self, path: str | Path) -> None:
        payload = []
        for item in self._items.values():
            data = asdict(item)
            data["state"] = item.state.tolist()
            data["next_state"] = item.next_state.tolist()
            payload.append(data)
        Path(path).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
