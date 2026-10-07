"""Agent KB: Cross-Domain Experience Knowledge Base and Shared Memory Infrastructure.

Based on:
"AGENT KB: LEVERAGING CROSS-DOMAIN EXPERIENCE FOR AGENTIC PROBLEM SOLVING"
Xiangru Tang, Tianrui Qin, Tianhao Peng, et al. (Yale University, OPPO, Stanford, Google DeepMind, Microsoft Research, DeepWisdom, 2025 - arXiv/GitHub: OPPO-PersonalAI/Agent-KB).

Key Architectural Components:
1. Structured Experience Representation:
   E = <pi, gamma, S, C>
   - pi: Task semantic embedding vector
   - gamma: Goal constraints and structured predicates
   - S = {(a_i, r_i)}: Action-reasoning pairs sequence
   - C: Cross-framework metadata & tool schema bindings
2. Self-Evolving Memory Management:
   - Dynamic Deduplication: max_{pi' in E} cos(pi, pi') > tau (default tau = 0.8)
   - Adaptive Utility Eviction: u_j <- u_j + eta * (r_j - u_j)
3. Two-Stage Reason-Retrieve-Refine Loop:
   - Planning Stage (seeds initial domain workflows)
   - Feedback Stage (applies diagnostic error fixes)
   - Calibrated Hybrid Retrieval: sigma_hyb = alpha * sigma_text + (1 - alpha) * sigma_sem
4. Disagreement Gate Mechanism:
   G(rho, rho') = 1[cos(phi(rho), phi(rho')) >= beta] (default beta = 0.8)
   Prevents knowledge interference and hallucination drift.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set


def _compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _simple_text_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Lightweight deterministic hashing-based semantic embedding for zero-dep environments."""
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    for i, word in enumerate(words):
        h = hash(word) % dim
        vec[h] += 1.0 / (1.0 + math.log(i + 1))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class StructuredExperience:
    """Experience tuple E = <pi, gamma, S, C>."""
    experience_id: str
    task_description: str
    pi_embedding: np.ndarray
    gamma_constraints: List[str] = field(default_factory=list)
    action_reasoning_pairs: List[Tuple[Any, str]] = field(default_factory=list)
    metadata_c: Dict[str, Any] = field(default_factory=dict)
    utility_score: float = 1.0
    access_count: int = 0
    domain: str = "general"


class DisagreementGate:
    """Disagreement Gate: G(rho, rho') = 1[cos(phi(rho), phi(rho')) >= beta]."""

    def __init__(self, beta: float = 0.8):
        self.beta = beta

    def evaluate(self, plan_original: str, plan_refined: str) -> bool:
        """Returns True if the refined plan is coherent and passes the stability gate."""
        if not plan_original or not plan_refined:
            return True
        phi_orig = _simple_text_embedding(plan_original)
        phi_ref = _simple_text_embedding(plan_refined)
        sim = _compute_cosine_similarity(phi_orig, phi_ref)
        return sim >= self.beta


class AgentKBMemory:
    """Cross-Domain Experience Knowledge Base with Hybrid Retrieval and Disagreement Gating."""

    def __init__(
        self,
        dedup_threshold: float = 0.80,
        disagreement_beta: float = 0.80,
        hybrid_alpha: float = 0.50,
        learning_rate_eta: float = 0.10,
        max_capacity: int = 1000
    ):
        self.dedup_threshold = dedup_threshold
        self.hybrid_alpha = hybrid_alpha
        self.learning_rate_eta = learning_rate_eta
        self.max_capacity = max_capacity
        self.experiences: Dict[str, StructuredExperience] = {}
        self.gate = DisagreementGate(beta=disagreement_beta)

    def add_experience(
        self,
        experience_id: str,
        task_description: str,
        action_reasoning_pairs: List[Tuple[Any, str]],
        gamma_constraints: Optional[List[str]] = None,
        metadata_c: Optional[Dict[str, Any]] = None,
        domain: str = "general"
    ) -> bool:
        """Adds or updates an experience with deduplication check."""
        emb = _simple_text_embedding(task_description)

        # Check deduplication against existing memory entries
        for existing in self.experiences.values():
            cos_sim = _compute_cosine_similarity(emb, existing.pi_embedding)
            if cos_sim >= self.dedup_threshold:
                # Merge / retain higher utility
                existing.access_count += 1
                return False

        # Evict if full
        if len(self.experiences) >= self.max_capacity:
            self._evict_lowest_utility()

        exp = StructuredExperience(
            experience_id=experience_id,
            task_description=task_description,
            pi_embedding=emb,
            gamma_constraints=gamma_constraints or [],
            action_reasoning_pairs=action_reasoning_pairs,
            metadata_c=metadata_c or {},
            utility_score=1.0,
            access_count=1,
            domain=domain
        )
        self.experiences[experience_id] = exp
        return True

    def _evict_lowest_utility(self) -> None:
        """Evicts the experience with the lowest learned utility score."""
        if not self.experiences:
            return
        lowest_id = min(self.experiences.keys(), key=lambda k: self.experiences[k].utility_score)
        del self.experiences[lowest_id]

    def update_utility(self, experience_id: str, reward: float) -> None:
        """Updates experience utility: u_j <- u_j + eta * (r_j - u_j)."""
        if experience_id in self.experiences:
            exp = self.experiences[experience_id]
            exp.utility_score += self.learning_rate_eta * (reward - exp.utility_score)

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None
    ) -> List[Tuple[StructuredExperience, float]]:
        """Two-Stage Hybrid Retrieval: sigma_hyb = alpha * sigma_text + (1 - alpha) * sigma_sem."""
        if not self.experiences:
            return []

        query_emb = _simple_text_embedding(query)
        query_words = set(query.lower().split())
        scored: List[Tuple[StructuredExperience, float]] = []

        for exp in self.experiences.values():
            if domain_filter and exp.domain != domain_filter:
                continue

            # 1. Lexical BM25 approximation
            doc_words = set(exp.task_description.lower().split())
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            lexical_sim = float(intersection / union) if union > 0 else 0.0

            # 2. Semantic Cosine similarity
            sem_sim = _compute_cosine_similarity(query_emb, exp.pi_embedding)

            # 3. Hybrid blend
            hybrid_score = self.hybrid_alpha * lexical_sim + (1.0 - self.hybrid_alpha) * sem_sim
            scored.append((exp, hybrid_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def reason_retrieve_refine_plan(
        self,
        task_query: str,
        initial_plan: str,
        top_k: int = 3
    ) -> Tuple[str, List[StructuredExperience], bool]:
        """Executes Reason-Retrieve-Refine cycle for the Planning Stage with Disagreement Gating.

        Returns: (final_plan, retrieved_experiences, gate_passed)
        """
        # 1. Retrieve relevant workflows
        retrieved = self.hybrid_retrieve(task_query, top_k=top_k)
        if not retrieved:
            return initial_plan, [], True

        exp_list = [item[0] for item in retrieved]

        # 2. Refine plan by incorporating past cross-domain workflows
        refinements = []
        for exp in exp_list:
            exp.access_count += 1
            if exp.action_reasoning_pairs:
                for act, reason in exp.action_reasoning_pairs:
                    refinements.append(f"Guideline from [{exp.experience_id}]: {reason}")

        refined_plan = initial_plan + "\n" + "\n".join(refinements)

        # 3. Disagreement Gate check
        gate_passed = self.gate.evaluate(initial_plan, refined_plan)
        final_plan = refined_plan if gate_passed else initial_plan

        return final_plan, exp_list, gate_passed

    def feedback_diagnose_refine(
        self,
        execution_trace: str,
        current_plan: str,
        error_context: str,
        top_k: int = 3
    ) -> Tuple[str, bool]:
        """Executes Reason-Retrieve-Refine cycle for the Feedback Stage with Disagreement Gating."""
        # Query memory using execution feedback
        retrieved = self.hybrid_retrieve(error_context, top_k=top_k)
        if not retrieved:
            return current_plan, True

        diagnostic_fixes = []
        for exp, score in retrieved:
            diagnostic_fixes.append(f"Diagnostic fix from [{exp.experience_id}]: {exp.task_description}")

        proposed_plan = current_plan + "\n[Feedback Diagnostic Fixes]:\n" + "\n".join(diagnostic_fixes)
        gate_passed = self.gate.evaluate(current_plan, proposed_plan)
        return (proposed_plan if gate_passed else current_plan), gate_passed
