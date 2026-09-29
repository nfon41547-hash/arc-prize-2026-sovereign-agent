"""CASS-Xi benchmark primitives (promoted from cass_xi_dev_v4, 2026-09-23).

Single source of truth: arc3sdk/cass_xi/*.py (stdlib-only at import).
Dev tree cass_xi_dev_v4/ is the reference snapshot; identity is locked
by tests/test_cass_xi_coherence.py.
"""
from arc3sdk.cass_xi.active import ResultStore, Outcome, derive_outcome, select_next, CONFIG_KEYS
from arc3sdk.cass_xi.adapters import AdapterConfig, MethodPlan, CausalGate, ReversibleFastMemory, method_plan
from arc3sdk.cass_xi.intelligence import Observation, Prediction, IntelligenceModel, normalize_adaptation, forgetting, ood_gain
from arc3sdk.cass_xi.llm_benchmark import ExactMatchTask, score_exact, balanced_split, aggregate_scores
from arc3sdk.cass_xi.pareto import pareto_front, uncertainty_aware_pareto
from arc3sdk.cass_xi.surrogate import HardwareConfig, Estimate, estimate_update
from arc3sdk.cass_xi.profile import SERVER_PROFILE, get as get_server_profile, operating_config

__all__ = [
    "ResultStore", "Outcome", "derive_outcome", "select_next", "CONFIG_KEYS",
    "AdapterConfig", "MethodPlan", "CausalGate", "ReversibleFastMemory", "method_plan",
    "Observation", "Prediction", "IntelligenceModel",
    "normalize_adaptation", "forgetting", "ood_gain",
    "ExactMatchTask", "score_exact", "balanced_split", "aggregate_scores",
    "pareto_front", "uncertainty_aware_pareto",
    "HardwareConfig", "Estimate", "estimate_update",
    "SERVER_PROFILE", "get_server_profile", "operating_config",
]
