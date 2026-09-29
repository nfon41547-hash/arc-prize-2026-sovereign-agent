"""Offline guard — single source of truth for competition-server socket discipline.

Real Kaggle rerun worker facts (production_config.json):
  offline_mode=true, competition_mode=true, enable_internet=false,
  no vLLM server on the worker => hot path must do ZERO socket I/O.

Kill switches (any one disables all LLM socket paths):
  ARC3_DISABLE_VLLM=1   (documented in production_config.json, now enforced)
  ARC3_OFFLINE=1
  KAGGLE_IS_COMPETITION_RERUN / KAGGLE_KERNEL_RUN_TYPE set (auto-detected)

Usage:
  from arc3sdk.offline_guard import vllm_enabled, assert_no_socket_in_hot_path
  if not vllm_enabled():
      return sovereign_fallback()  # zero socket, never touch requests
"""
from __future__ import annotations

import os




def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_kaggle_worker() -> bool:
    """True when running inside a Kaggle kernel (offline rerun or interactive)."""
    return bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE") or os.environ.get("KAGGLE_DOCKER_IMAGE")
    )


def is_competition_rerun() -> bool:
    """True on the scoring rerun worker (offline, no server, no internet)."""
    return _truthy("KAGGLE_IS_COMPETITION_RERUN") or (
        is_kaggle_worker() and os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() not in ("interactive", "")
        and not _truthy("KAGGLE_INTERACTIVE")
        and os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    )


def is_offline() -> bool:
    """True when offline operation is explicitly requested.

    NOTE (2026-09-09 correction): Kaggle-worker auto-detection was REMOVED as a
    behavior switch. The production notebook runs its OWN in-process sidecars
    (SovereignNativeOfflineEngine on localhost:1234) that the hot path is
    supposed to call — auto-disabling sockets on any Kaggle worker would break
    that tested flow. Only the explicit ARC3_OFFLINE=1 flag forces offline.
    Use describe() for environment diagnostics (no behavior).
    """
    return _truthy("ARC3_OFFLINE")


def vllm_enabled() -> bool:
    """False => hot path must skip ALL LLM socket I/O (swallow/proxy/health probes).

    ONLY the explicit ARC3_DISABLE_VLLM=1 kill-switch disables sockets.
    Rationale: localhost sidecars (:1234 native engine, :8000 vLLM) are part of
    the production design and are started by the kernel itself; nothing in the
    pushed notebook sets ARC3_DISABLE_VLLM, so default flow keeps them.
    """
    return not _truthy("ARC3_DISABLE_VLLM")


def llm_base_or_none(configured: str = "") -> str | None:
    """Return configured LLM base URL, or None when sockets are forbidden."""
    if not vllm_enabled():
        return None
    base = configured or os.environ.get("VLLM_BASE_URL", "http://localhost:1234/v1")
    return base or None


def describe() -> dict:
    """Fail-open diagnostics for logs (never throws, never touches network)."""
    try:
        return {
            "vllm_enabled": vllm_enabled(),
            "offline": is_offline(),
            "kaggle_worker": is_kaggle_worker(),
            "competition_rerun": is_competition_rerun(),
            "flags": {
                "ARC3_DISABLE_VLLM": os.environ.get("ARC3_DISABLE_VLLM", ""),
                "ARC3_OFFLINE": os.environ.get("ARC3_OFFLINE", ""),
                "KAGGLE_KERNEL_RUN_TYPE": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
            },
        }
    except Exception:
        return {"vllm_enabled": False, "offline": True}
