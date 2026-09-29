"""Evidence-backed real-time planner.

Unlike a physics heuristic, this planner only traverses transitions observed
from the real environment (or an exact PerfectSimulator replay).  Unknown
edges are never invented, so a returned plan is auditable edge-by-edge.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import numpy as np


def grid_hash(grid: Any) -> str:
    return blake2b(np.ascontiguousarray(np.asarray(grid, dtype=np.uint8)).tobytes(), digest_size=12).hexdigest()


@dataclass(frozen=True)
class EvidenceEdge:
    before: str
    action: str
    after: str
    game_id: str
    level: int
    observations: int
    source: str
    terminal: bool


class EvidencePlanner:
    def __init__(self, max_edges: int = 8192) -> None:
        self.max_edges = max_edges
        self.edges: dict[tuple[str, int, str, str, str], EvidenceEdge] = {}
        self.frames: dict[str, np.ndarray] = {}

    def observe(self, before: Any, action: str, after: Any, *, game_id: str, level: int, source: str, terminal: bool = False) -> None:
        b, a = grid_hash(before), grid_hash(after)
        self.frames.setdefault(b, np.asarray(before, dtype=np.uint8).copy())
        self.frames.setdefault(a, np.asarray(after, dtype=np.uint8).copy())
        key = (str(game_id), int(level), b, str(action).upper(), a)
        old = self.edges.get(key)
        self.edges[key] = EvidenceEdge(b, str(action).upper(), a, str(game_id), int(level), (old.observations + 1 if old else 1), source, bool(terminal or (old.terminal if old else False)))
        if len(self.edges) > self.max_edges:
            self.edges.pop(next(iter(self.edges)))

    def plan_to_terminal(self, state: Any, *, game_id: str, level: int, max_depth: int = 4) -> tuple[list[str], list[EvidenceEdge]] | None:
        start = grid_hash(state)
        queue = deque([(start, [], [])])
        visited = {start}
        while queue:
            current, actions, evidence = queue.popleft()
            if evidence and evidence[-1].terminal:
                return actions, evidence
            if len(actions) >= max_depth:
                continue
            outgoing = [edge for edge in self.edges.values()
                        if edge.before == current and edge.game_id == str(game_id) and edge.level == int(level)]
            outgoing.sort(key=lambda edge: (-edge.observations, edge.action))
            for edge in outgoing:
                if edge.after in visited:
                    continue
                visited.add(edge.after)
                queue.append((edge.after, actions + [edge.action], evidence + [edge]))
        return None

    def status(self) -> dict[str, int]:
        return {"edges": len(self.edges), "states": len(self.frames), "terminal_edges": sum(e.terminal for e in self.edges.values())}
