"""AGent: Adversarial Unanswerable Hypothesis Detector & Hallucination Filter.

Based on the foundational research:
"AGent: A Novel Pipeline for Automatically Creating Unanswerable Questions"
(Tran, Do, Do, Kretchmar, Du, 2023).

In ARC-AGI reasoning, foundation models frequently hallucinate plausible-looking
transformations that lack empirical grounding in the demonstration pairs or game context.
This module uses multi-model adversarial filtering and confidence-weighted consensus:
    V(h) = c_a * alpha^(n_a) - c_u * beta^(n_u)
to detect under-determined, unanswerable, or spurious hypotheses and eliminate them
prior to action execution or grid emission.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional, Union
from collections import defaultdict


class AGentUnanswerableDetector:
    """Detects and prunes ungrounded / unanswerable hypotheses in ARC reasoning."""

    def __init__(
        self,
        alpha: float = 0.64,
        beta: float = 0.69,
        unanswerable_threshold: float = 0.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.unanswerable_threshold = unanswerable_threshold
        self.filtered_hypotheses_count: int = 0

    def compute_unanswerability_score(
        self,
        confidence_attempt: float,
        confidence_abstain: float,
        num_attempt: int,
        num_abstain: int,
    ) -> float:
        """Computes V(h) = c_a * alpha^(n_a) - c_u * beta^(n_u)."""
        score = (
            confidence_attempt * (self.alpha ** max(0, num_attempt))
            - confidence_abstain * (self.beta ** max(0, num_abstain))
        )
        return float(score)

    def is_hypothesis_grounded(
        self,
        hypothesis_id: str,
        proposals: list[dict[str, Any]],
        context_evidence: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, float, str]:
        """Evaluates whether a candidate hypothesis is sufficiently grounded or unanswerable.
        
        Args:
            hypothesis_id: Identifier of the hypothesis or rule.
            proposals: List of candidate predictions from different solvers / models / tiers.
                       Each dict contains: {'attempt': bool, 'confidence': float, 'solution': Any}
            context_evidence: Physical grid evidence or state-action transition trace.
            
        Returns:
            (is_grounded, v_score, reason)
        """
        if not proposals:
            return False, -1.0, "no_proposals"

        c_a = 0.0
        c_u = 0.0
        n_a = 0
        n_u = 0

        for p in proposals:
            conf = float(p.get("confidence", 0.5))
            if p.get("attempt", True) and p.get("solution") is not None:
                c_a += conf
                n_a += 1
            else:
                c_u += conf
                n_u += 1

        v_score = self.compute_unanswerability_score(c_a, c_u, n_a, n_u)

        # Evidence cross-check: if physical context contradicts proposal, penalize
        if context_evidence is not None:
            if context_evidence.get("contradiction", False):
                v_score -= 2.0

        is_grounded = v_score >= self.unanswerable_threshold

        if not is_grounded:
            self.filtered_hypotheses_count += 1
            reason = f"unanswerable_or_spurious (V={v_score:.3f} < {self.unanswerable_threshold})"
        else:
            reason = f"empirically_grounded (V={v_score:.3f})"

        return is_grounded, v_score, reason

    def filter_action_candidates(
        self,
        candidates: list[dict[str, Any]],
        context_evidence: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Filters out actions or grid solutions generated from unanswerable / ungrounded assumptions."""
        grounded_candidates = []
        for cand in candidates:
            mock_proposal = [{
                "attempt": True,
                "confidence": cand.get("confidence", 0.8),
                "solution": cand.get("action", cand.get("grid")),
            }]
            is_valid, _, _ = self.is_hypothesis_grounded(
                str(cand.get("id", "cand")),
                mock_proposal,
                context_evidence,
            )
            if is_valid:
                grounded_candidates.append(cand)

        return grounded_candidates if grounded_candidates else candidates


# Global singleton
_AGENT_UNANSWERABLE = AGentUnanswerableDetector()

def get_unanswerable_detector() -> AGentUnanswerableDetector:
    return _AGENT_UNANSWERABLE
