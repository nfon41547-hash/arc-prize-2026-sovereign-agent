"""Small, dependency-light safety contract for the competition runtime.

This borrows only the robust boundary ideas from the native v3 artifact.  It
does not start a model, open a socket, or choose an action; Duck/vLLM remains
the authority.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


LOCAL_VLLM_BASE_URL = "http://localhost:1234/v1"


def normalize_grid(value: Any) -> np.ndarray:
    """Return a bounded uint8 grid without changing its geometry."""
    grid = np.asarray(value, dtype=np.uint8)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"expected non-empty 2-D grid, got shape={grid.shape}")
    return np.ascontiguousarray(grid)


def normalize_action(payload: dict[str, Any], allowed: set[str] | None = None) -> dict[str, Any]:
    """Validate one action payload; never invent coordinates for ACTION6."""
    if not isinstance(payload, dict):
        raise ValueError("action payload must be a dict")
    name = str(payload.get("action", "")).upper()
    legal = {f"ACTION{i}" for i in range(1, 8)} | {"RESET"}
    if name not in legal or (allowed is not None and name not in allowed):
        raise ValueError(f"illegal action: {name}")
    out: dict[str, Any] = {"action": name}
    if name == "ACTION6":
        if payload.get("row", payload.get("y")) is None or payload.get("col", payload.get("x")) is None:
            raise ValueError("ACTION6 requires explicit row/col")
        out["row"] = max(0, min(63, int(payload.get("row", payload.get("y")))))
        out["col"] = max(0, min(63, int(payload.get("col", payload.get("x")))))
    return out


@dataclass
class RuntimeSafetyBudget:
    hard_seconds: float = 32400.0
    reserve_seconds: float = 600.0
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.monotonic()

    @property
    def soft_remaining(self) -> float:
        return max(0.0, self.hard_seconds - self.reserve_seconds - (time.monotonic() - self.started_at))

    def can_continue(self) -> bool:
        return self.soft_remaining > 0.0


def vram_snapshot() -> dict[str, Any]:
    """Best-effort telemetry; never makes a CPU-only test fail."""
    try:
        import subprocess
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip().split(",")
        used, total = int(float(raw[0])), int(float(raw[1]))
        fraction = used / total if total else 0.0
        return {"used_mib": used, "total_mib": total, "fraction": fraction,
                "level": "critical" if fraction >= 0.94 else "pressure" if fraction >= 0.88 else "normal"}
    except Exception:
        return {"used_mib": 0, "total_mib": 0, "fraction": 0.0, "level": "unknown"}


def configured_vllm_url() -> str:
    """Return the only permitted local analyzer endpoint."""
    configured = os.environ.get("VLLM_BASE_URL", LOCAL_VLLM_BASE_URL).rstrip("/")
    if configured != LOCAL_VLLM_BASE_URL:
        return LOCAL_VLLM_BASE_URL
    return configured
