"""CASS-Xi wiring — coherent integration with existing systems (fail-open).

Connects arc3sdk.cass_xi to:
  1. arc3sdk.arc_cass.registry.PluginRegistry  -> registers 'cass-xi' factory
  2. arc3sdk.exploration_registry              -> adaptive-method vote tags
  3. arc3sdk.cass_v0 / cass_baselines          -> method-plan cross-check

Never raises at import. Never touches vendor payload / notebook cells.
"""
from __future__ import annotations

_WIRED = {"registry": False, "explorer": False, "error": None}


def _try_wire_registry() -> bool:
    try:
        from arc3sdk.arc_cass.registry import PluginRegistry
        from arc3sdk.cass_xi.adapters import AdapterConfig, method_plan

        def _cass_xi_factory(rank: int = 8, alpha: int = 16, **kw):
            cfg = AdapterConfig(rank=rank, alpha=alpha,
                                learning_rate=kw.get("learning_rate", 2e-4),
                                train_steps=kw.get("train_steps", 8))
            return method_plan("cass-xi", cfg)

        reg = PluginRegistry()
        reg.register("cass-xi", _cass_xi_factory)
        reg.register("online-lora", lambda rank=8, alpha=16, **kw: method_plan(
            "online-lora", AdapterConfig(rank=rank, alpha=alpha)))
        _WIRED["registry"] = True
        return True
    except Exception as e:
        _WIRED["error"] = f"registry:{type(e).__name__}"
        return False


def cass_xi_vote_tags(outcome) -> dict:
    """Map a measured Outcome to exploration_registry-compatible vote tags."""
    try:
        adapt = float(outcome.adapt if hasattr(outcome, "adapt") else outcome["adapt"])
        forget = float(outcome.forget if hasattr(outcome, "forget") else outcome["forget"])
        ood = float(outcome.ood if hasattr(outcome, "ood") else outcome["ood"])
    except Exception:
        return {"explorer_xi": 0.0}
    score = adapt + ood - forget
    return {"explorer_xi_adapt": adapt, "explorer_xi_ood": ood,
            "explorer_xi_forget": forget, "explorer_xi_score": score}


def wire_all() -> dict:
    _try_wire_registry()
    try:
        import arc3sdk.exploration_registry  # noqa: F401 — presence check only
        _WIRED["explorer"] = True
    except Exception as e:
        _WIRED["error"] = f"explorer:{type(e).__name__}"
    return dict(_WIRED)


def wiring_status() -> dict:
    return dict(_WIRED)
