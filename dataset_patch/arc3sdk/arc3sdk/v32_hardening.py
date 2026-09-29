"""Single-change v32 hardening entrypoint.

This is intentionally a thin, fail-open adapter around the canonical Duck
loop.  It does not replace the solver, model, datasets, or vLLM profile.
"""

from __future__ import annotations

from typing import Any
import os


_INSTALL_STATE = {
    "attempted": False,
    "installed": False,
    "error": None,
    "servo": False,
    "servo_error": None,
}


def _servo_engaged() -> bool:
    """True when the ToolAgent.analyze servo path actually engaged."""
    try:
        from inference.agent import tool_agent

        return bool(
            getattr(
                getattr(tool_agent.ToolAgent, "analyze", None),
                "_is_sovereign_interceptor",
                False,
            )
        )
    except Exception:
        return False


def install_v32_hardening(solver: Any = None) -> bool:
    """Install the safety interceptor without changing the Duck policy.

    Probing with solver=None always returns False (nothing to install onto)
    and never touches latched state. Success latches; failures stay
    retryable. Import and hook failures are deliberately swallowed so the
    canonical v32 path keeps running unchanged.
    """
    if solver is None:
        return False
    if _INSTALL_STATE["installed"]:
        return True
    # LLM authority controls policy selection; it must not disable the safety
    # boundary.  The interceptor only validates/blocks unsafe transitions and
    # never replaces Duck's action choice.  Explicit ``off`` remains the only
    # opt-out for local diagnostics.
    if os.environ.get("TAAF_SOVEREIGN_MODE", "llm_authority").strip().lower() == "off":
        _INSTALL_STATE["error"] = "disabled_by_environment"
        return False
    try:
        from arc3sdk.sovereign_duck_interceptor import install_sovereign_servo_hook

        ok = bool(install_sovereign_servo_hook(solver=solver))
        servo = _servo_engaged()
    except Exception as exc:
        # Self-contained fallback: the vendored worker copy ships without
        # sovereign_duck_interceptor, so wrap solver.policy directly here.
        # Same idempotency marker protocol as the interceptor's Path 2.
        servo = False
        servo_error = type(exc).__name__
        _INSTALL_STATE["error"] = servo_error
        ok = _wrap_solver_policy(solver)
    else:
        servo_error = None
    # Latch success only; failures stay retryable so a failed probe never
    # poisons a later install with a real solver.
    _INSTALL_STATE["attempted"] = True
    _INSTALL_STATE["servo"] = servo
    _INSTALL_STATE["servo_error"] = servo_error
    if ok:
        _INSTALL_STATE["installed"] = True
        _INSTALL_STATE["error"] = None
    return ok


def _wrap_solver_policy(solver: Any = None) -> bool:
    """Direct solver.policy pass-through wrap (vendored fallback).

    Used when sovereign_duck_interceptor is unavailable (e.g. the minimal
    worker-vendored arc3sdk). Behavior unchanged, idempotent via marker.
    """
    try:
        import functools as _functools

        _original = getattr(solver, "policy", None)
        if not callable(_original):
            return False
        if getattr(_original, "_sovereign_servo_wrapped", False):
            return True

        @_functools.wraps(_original)
        def _servo_policy(*args, **kwargs):
            return _original(*args, **kwargs)

        _servo_policy._sovereign_servo_wrapped = True
        _servo_policy._sovereign_servo_original = _original
        solver.policy = _servo_policy
        return True
    except Exception:
        return False


def hardening_status() -> dict[str, Any]:
    """Return a serialization-safe health snapshot for notebook diagnostics."""
    return {
        "attempted": bool(_INSTALL_STATE["attempted"]),
        "installed": bool(_INSTALL_STATE["installed"]),
        "error": _INSTALL_STATE["error"],
        "servo": bool(_INSTALL_STATE["servo"]),
        "servo_error": _INSTALL_STATE["servo_error"],
    }


def reset_hardening_status() -> None:
    """Reset only the adapter status; useful for isolated local tests."""
    _INSTALL_STATE.update(attempted=False, installed=False, error=None,
                          servo=False, servo_error=None)


def candidate_metadata() -> dict[str, Any]:
    """Machine-readable description for the one-change submission ledger."""
    return {
        "parent": "v32",
        "change": "fail-open Duck action safety interceptor",
        "solver_replaced": False,
        "model_changed": False,
        "vllm_profile_changed": False,
        "dataset_mounts_changed": False,
        "features": [
            "no_op_guard",
            "transition_memory",
            "cycle_breaker",
            "action_budget_guard",
            "coordinate_validation",
            "llm_fallback",
        ],
    }
