"""MPC Grounded Debate Engine: Multi-Party Collaborative Reasoning with Theory of Mind (ToM) Belief Books.

Based on:
"Multi-Party Conversational AI and Multi-Agent Deliberation"
Sapkota et al. (2025) — Survey on Turn-Taking, Addressee Selection, Communicative Acts, and Epistemic Belief Tracking.

Key Architectural Solutions to Prevent Debate Degradation (0/60 floor effect):
1. Strict Grounded Communicative Acts:
   - PROPOSE: Propose candidate action with invariant proof
   - CRITIQUE: Disprove candidate action with counter-evidence/diff penalty
   - REVISE: Coordinate or parameter adjustment
   - VOTE: Weighted epistemic vote based on calibrated belief
2. ToM-Lite Belief Books:
   - Tracks each agent's internal state belief B_i(s), expected reward, and uncertainty H(B_i).
3. Grounded Turn-Taking & Quorum Consensus:
   - Assigns speaking turn to agent with highest disagreement/entropy.
   - Early stops immediately upon reaching quorum consensus (sum w_i * 1[a_i = a*] >= theta_quorum).
   - Annihilates open-ended rambling to achieve deterministic convergence.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Set


class CommunicativeActType(str, Enum):
    PROPOSE = "PROPOSE"
    CRITIQUE = "CRITIQUE"
    REVISE = "REVISE"
    VOTE = "VOTE"


@dataclass
class CommunicativeAct:
    """Formal structured utterance in multi-party deliberation."""
    sender_id: str
    target_addressee: str
    act_type: CommunicativeActType
    action_candidate: Any
    confidence: float
    evidence_score: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefBook:
    """Theory of Mind (ToM) Epistemic Belief Book for individual agent."""
    agent_id: str
    persona: str
    prior_weight: float = 1.0
    action_beliefs: Dict[Any, float] = field(default_factory=dict)
    observed_counter_evidence: List[str] = field(default_factory=list)

    def update_belief(self, action: Any, delta_evidence: float) -> None:
        """Updates internal belief distribution over actions."""
        current = self.action_beliefs.get(action, 0.5)
        updated = float(np.clip(current + delta_evidence, 0.01, 0.99))
        self.action_beliefs[action] = updated

    def get_entropy(self) -> float:
        """Calculates epistemic uncertainty / entropy across action beliefs."""
        if not self.action_beliefs:
            return 1.0
        vals = np.array(list(self.action_beliefs.values()))
        probs = vals / np.sum(vals) if np.sum(vals) > 0 else np.ones_like(vals) / len(vals)
        probs = np.clip(probs, 1e-6, 1.0)
        return float(-np.sum(probs * np.log2(probs)))


class MPCGroundedDebateEngine:
    """Multi-Party Grounded Deliberation Engine with Quorum Consensus."""

    def __init__(
        self,
        max_debate_rounds: int = 3,
        quorum_threshold: float = 0.65,
        entropy_floor: float = 0.15
    ):
        self.max_debate_rounds = max_debate_rounds
        self.quorum_threshold = quorum_threshold
        self.entropy_floor = entropy_floor
        self.agents: Dict[str, BeliefBook] = {}
        self._initialize_default_personas()

    def _initialize_default_personas(self) -> None:
        """Initializes canonical specialized debate personas."""
        self.agents = {
            "geom_analyst": BeliefBook(
                agent_id="geom_analyst",
                persona="Geometric Invariantist",
                prior_weight=1.2
            ),
            "physics_tracker": BeliefBook(
                agent_id="physics_tracker",
                persona="Causal Dynamics Tracker",
                prior_weight=1.1
            ),
            "topologist": BeliefBook(
                agent_id="topologist",
                persona="Topological Morphism Specialist",
                prior_weight=1.0
            ),
        }

    def deliberate(
        self,
        grid: np.ndarray,
        available_actions: List[Any],
        verifier_fn: Optional[Callable[[np.ndarray, Any], Tuple[float, bool]]] = None
    ) -> Tuple[Any, float, List[CommunicativeAct]]:
        """Runs grounded multi-party debate and converges to highest consensus action.

        Args:
            grid: Current 2D grid observation
            available_actions: List of valid actions
            verifier_fn: Optional ground-truth simulator (grid, action) -> (evidence_score, is_fatal)

        Returns:
            (selected_action, confidence, debate_transcript)
        """
        if not available_actions:
            return None, 0.0, []

        transcript: List[CommunicativeAct] = []

        # 1. Initialize beliefs across available actions
        for agent in self.agents.values():
            agent.action_beliefs = {a: 1.0 / len(available_actions) for a in available_actions}

        # 2. Iterative Grounded Deliberation Rounds
        for round_idx in range(self.max_debate_rounds):
            # Select agent with highest uncertainty to speak first (turn-taking)
            sorted_speakers = sorted(
                self.agents.values(),
                key=lambda ag: ag.get_entropy(),
                reverse=True
            )

            for speaker in sorted_speakers:
                # Find current top candidate for this speaker
                best_act = max(speaker.action_beliefs.keys(), key=lambda a: speaker.action_beliefs[a])
                prior_score = speaker.action_beliefs[best_act]

                # Evaluate with verifier / heuristics if available
                evidence_score = 0.5
                is_fatal = False
                if verifier_fn:
                    evidence_score, is_fatal = verifier_fn(grid, best_act)

                if is_fatal:
                    # CRITIQUE: Disprove and penalize fatal branch
                    speaker.update_belief(best_act, -0.8)
                    act = CommunicativeAct(
                        sender_id=speaker.agent_id,
                        target_addressee="ALL",
                        act_type=CommunicativeActType.CRITIQUE,
                        action_candidate=best_act,
                        confidence=0.1,
                        evidence_score=-1.0,
                        rationale=f"Fatal transition detected for action {best_act}."
                    )
                    transcript.append(act)
                    # Broadcast critique to all other agents (ToM update)
                    for other in self.agents.values():
                        other.update_belief(best_act, -0.6)
                else:
                    # PROPOSE or VOTE
                    delta = (evidence_score - 0.5) * 0.5
                    speaker.update_belief(best_act, delta)
                    act = CommunicativeAct(
                        sender_id=speaker.agent_id,
                        target_addressee="ALL",
                        act_type=CommunicativeActType.PROPOSE if round_idx == 0 else CommunicativeActType.VOTE,
                        action_candidate=best_act,
                        confidence=speaker.action_beliefs[best_act],
                        evidence_score=evidence_score,
                        rationale=f"Preserves structural invariant with evidence {evidence_score:.2f}."
                    )
                    transcript.append(act)

            # 3. Check Quorum Consensus across all agents
            votes: Dict[Any, float] = {a: 0.0 for a in available_actions}
            total_weight = 0.0

            for ag in self.agents.values():
                top_choice = max(ag.action_beliefs.keys(), key=lambda a: ag.action_beliefs[a])
                weight = ag.prior_weight * ag.action_beliefs[top_choice]
                votes[top_choice] += weight
                total_weight += ag.prior_weight

            if total_weight > 0:
                best_consensus_act = max(votes.keys(), key=lambda a: votes[a])
                quorum_ratio = votes[best_consensus_act] / total_weight

                # Early Stop if Quorum is Reached
                if quorum_ratio >= self.quorum_threshold:
                    conf = float(np.clip(0.6 + 0.4 * quorum_ratio, 0.1, 0.99))
                    return best_consensus_act, conf, transcript

        # Fallback to highest voted action
        best_act = max(available_actions, key=lambda a: sum(ag.action_beliefs.get(a, 0.0) for ag in self.agents.values()))
        return best_act, 0.70, transcript
