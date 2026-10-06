"""MemAdapter: Counterfactual Adaptation Against Memory-Induced Sycophancy.

Implements the three core pillars of MemAdapter:
1. Counterfactual Induction (Risk estimation of past memory under current state distribution)
2. Context-Aware Reflection (Calibrated weighting separating structural invariants from local dynamics)
3. Evidence-Grounded Reasoning (Physical observation & frame-diff gating overriding stale memory priors)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np


@dataclass
class MemoryRecord:
    id: str
    rule_type: str  # 'structural_invariant' or 'local_dynamic'
    source_level: int
    action_sequence: List[int]
    expected_delta_state: Dict[str, Any]
    confidence: float = 1.0
    empirical_validations: int = 0
    empirical_violations: int = 0
    is_blacklisted: bool = False


@dataclass
class EvidenceObservation:
    level: int
    turn: int
    action_taken: int
    grid_before: Optional[np.ndarray] = None
    grid_after: Optional[np.ndarray] = None
    delta_energy: float = 0.0
    state_changed: bool = False
    fatal_death: bool = False


class CounterfactualInduction:
    """Probes the risk of applying a retrieved memory under current state hypothesis."""
    
    def __init__(self, conflict_threshold: float = 0.5):
        self.conflict_threshold = conflict_threshold

    def evaluate_risk(self, memory: MemoryRecord, current_level: int, current_evidence: Optional[EvidenceObservation]) -> float:
        """Returns risk score in [0.0, 1.0]. Higher risk indicates memory is likely stale/sycophantic."""
        if memory.is_blacklisted:
            return 1.0
        
        # Cross-level domain shift penalty for local dynamics
        if memory.rule_type == 'local_dynamic' and memory.source_level != current_level:
            base_risk = 0.4
        else:
            base_risk = 0.1

        # Check empirical violation ratio
        total_trials = memory.empirical_validations + memory.empirical_violations
        if total_trials > 0:
            violation_rate = memory.empirical_violations / total_trials
            base_risk = max(base_risk, violation_rate)
            
        return min(1.0, max(0.0, base_risk))


class ContextAwareReflection:
    """Calibrates memory influence based on invariant vs dynamic context."""

    def __init__(self, decay_rate: float = 0.85):
        self.decay_rate = decay_rate

    def calibrate_confidence(self, memory: MemoryRecord, current_level: int) -> float:
        if memory.is_blacklisted:
            return 0.0
        
        if memory.rule_type == 'structural_invariant':
            # Invariants retain high confidence across levels
            return memory.confidence * 0.98
        else:
            # Local dynamics decay across level boundaries until re-verified
            level_gap = abs(current_level - memory.source_level)
            calibrated = memory.confidence * (self.decay_rate ** level_gap)
            return max(0.05, calibrated)


class EvidenceGroundedReasoning:
    """Enforces empirical observation supremacy over retrieved memory beliefs."""

    def __init__(self):
        self.death_action_ledger: Dict[int, Set[int]] = {}  # level -> set of dead actions

    def record_evidence(self, memory: MemoryRecord, obs: EvidenceObservation) -> bool:
        """Updates memory record based on empirical observation. Returns True if validated."""
        if obs.fatal_death:
            memory.empirical_violations += 3
            memory.is_blacklisted = True
            if obs.level not in self.death_action_ledger:
                self.death_action_ledger[obs.level] = set()
            self.death_action_ledger[obs.level].add(obs.action_taken)
            return False

        if not obs.state_changed:
            # Action had no effect -> mild violation / potential stale rule
            memory.empirical_violations += 1
            if memory.empirical_violations >= 3 and memory.rule_type == 'local_dynamic':
                memory.is_blacklisted = True
            return False

        memory.empirical_validations += 1
        return True

    def is_action_fatal(self, level: int, action: int) -> bool:
        return action in self.death_action_ledger.get(level, set())


class MemAdapterController:
    """Unified MemAdapter Orchestrator eliminating Memory-Induced Sycophancy."""

    def __init__(self):
        self.memories: Dict[str, MemoryRecord] = {}
        self.counterfactual = CounterfactualInduction()
        self.reflection = ContextAwareReflection()
        self.evidence = EvidenceGroundedReasoning()

    def register_memory(self, memory: MemoryRecord) -> None:
        self.memories[memory.id] = memory

    def query_action_feasibility(self, memory_id: str, current_level: int, candidate_action: int) -> Tuple[bool, float]:
        """Returns (is_allowed, calibrated_weight)."""
        if candidate_action in self.evidence.death_action_ledger.get(current_level, set()):
            return False, 0.0

        if memory_id not in self.memories:
            return True, 0.5

        mem = self.memories[memory_id]
        if mem.is_blacklisted:
            return False, 0.0

        risk = self.counterfactual.evaluate_risk(mem, current_level, None)
        if risk > 0.8:
            return False, 0.0

        weight = self.reflection.calibrate_confidence(mem, current_level)
        return True, weight

    def ingest_turn_result(self, memory_id: Optional[str], obs: EvidenceObservation) -> None:
        if memory_id and memory_id in self.memories:
            self.evidence.record_evidence(self.memories[memory_id], obs)
