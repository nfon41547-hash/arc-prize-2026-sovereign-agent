"""Type-safe contracts for the ARC-AGI-3 decision boundary.

This is deliberately small and dependency-free.  It turns malformed frames,
illegal actions and accidental game-ID lookup into explicit validation errors
before they reach the competition environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


POLICIES = ("duck", "interceptor", "bbk", "cass", "world_model", "fusion", "fallback")


@dataclass(frozen=True)
class SolverFeatures:
    interceptor: bool = True
    bbk: bool = True
    cass: bool = True
    world_model: bool = False
    transfer: bool = True
    game_id_lookup: bool = False

    @classmethod
    def from_env(cls) -> "SolverFeatures":
        def enabled(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            return default if raw is None else raw.strip().lower() not in {"0", "false", "off", "no"}
        return cls(
            interceptor=enabled("ARC3_ENABLE_INTERCEPTOR", True),
            bbk=enabled("ARC3_ENABLE_BBK", True),
            cass=enabled("ARC3_ENABLE_CASS", True),
            world_model=enabled("ARC3_ENABLE_WORLD_MODEL", False),
            transfer=enabled("ARC3_ENABLE_TRANSFER", True),
            game_id_lookup=enabled("ARC3_ENABLE_GAME_ID_LOOKUP", False),
        )


@dataclass(frozen=True)
class Observation:
    game_id: str
    level: int
    grid: np.ndarray
    legal_actions: tuple[int, ...]
    state: str = "NOT_FINISHED"

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid)
        if grid.ndim != 2 or grid.size == 0:
            raise ValueError("grid must be a non-empty 2-D array")
        if not np.issubdtype(grid.dtype, np.integer):
            raise TypeError("grid must contain integer colors")
        if any(a not in range(1, 8) for a in self.legal_actions):
            raise ValueError("legal actions must be ACTION1..ACTION7 ids")
        if self.level < 0:
            raise ValueError("level must be non-negative")


@dataclass(frozen=True)
class Decision:
    action: int
    policy: str
    confidence: float = 0.0
    x: int | None = None
    y: int | None = None

    def validate(self, obs: Observation) -> "Decision":
        if self.action not in obs.legal_actions:
            raise ValueError(f"illegal action {self.action}; legal={obs.legal_actions}")
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.action == 6:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires coordinates")
            h, w = obs.grid.shape
            if not (0 <= int(self.x) < w and 0 <= int(self.y) < h):
                raise ValueError("ACTION6 coordinates outside the observation frame")
        elif self.x is not None or self.y is not None:
            raise ValueError("coordinates are only valid for ACTION6")
        return self


def validate_split(train: set[str], development: set[str], held_out: set[str]) -> None:
    """Reject leakage between train/development/held-out game manifests."""
    if train & development or train & held_out or development & held_out:
        raise ValueError("game split leakage detected")
    if not held_out:
        raise ValueError("held-out split must not be empty")
