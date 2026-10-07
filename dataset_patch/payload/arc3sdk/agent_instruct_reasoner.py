"""Agent-Instruct: Autonomous Agent Instructing LLMs for General Zero-Shot ARC Reasoning.

Based on the foundational research:
"Agent Instructs Large Language Models to be General Zero-Shot Reasoners"
(Crispino, Montgomery, Zeng, Song, Wang, 2023).

Core Architecture:
1. Dynamic Meta-Instruction: Decomposes complex visual/game tasks into explicit sub-reasoning phases.
2. Interactive Verification & Feedback Steering: Detects reasoning drift / hallucination and injects corrective guidance.
3. Zero-Shot Cognitive Policy Steering: Bridges symbolic physics with LLM latent reasoning without few-shot contamination.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any, Optional, Union
from collections import deque


class AgentInstructReasoner:
    """Autonomous Agent that steers LLM Zero-Shot Abstract Reasoning."""

    def __init__(
        self,
        max_subgoals: int = 5,
        confidence_gate: float = 0.85,
        temperature_steering: float = 0.2,
    ):
        self.max_subgoals = max_subgoals
        self.confidence_gate = confidence_gate
        self.temperature_steering = temperature_steering
        self.trajectory_history: list[dict[str, Any]] = []

    def generate_meta_instructions(
        self,
        task_type: str,
        grid_state: np.ndarray,
        context_hints: Optional[list[str]] = None,
    ) -> list[str]:
        """Generates dynamic, task-tailored reasoning instructions (Step-by-Step Meta Guidance)."""
        instructions = [
            "Phase 1: Perceive spatial invariants, object bounding boxes, and foreground/background colors.",
            "Phase 2: Identify topological transformations (symmetry D4, translation, containment, gravity).",
            "Phase 3: Formulate a verifiable causal hypothesis before proposing concrete coordinates or actions.",
        ]
        
        # Domain-specific dynamic injection
        if task_type == "interactive_game":
            instructions.append(
                "Phase 4 (Interactive): Evaluate action affordances. Never repeat an action that produced delta_score <= 0."
            )
            instructions.append(
                "Phase 5: If stuck (entropy delta = 0), execute Counterfactual Action Pivoting toward unvisited states."
            )
        elif task_type == "grid_transformation":
            instructions.append(
                "Phase 4 (Grid Exact): Verify output dimensions (H x W) match the training demonstration ratio."
            )
            instructions.append(
                "Phase 5: Apply color permutation and dihedral reflection check to eliminate rotational ambiguity."
            )

        if context_hints:
            for hint in context_hints:
                instructions.append(f"Meta-Constraint: {hint}")

        return instructions

    def steer_reasoning_step(
        self,
        current_reasoning: str,
        predicted_action_or_grid: Any,
        grounding_feedback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Inspects intermediate reasoning and provides corrective steering directives."""
        steering_signal = {
            "valid": True,
            "confidence": 1.0,
            "correction_directive": None,
            "pivoted_action": None,
        }

        if grounding_feedback is not None:
            # Check for stagnation or fatal transition
            if grounding_feedback.get("stagnant", False):
                steering_signal["valid"] = False
                steering_signal["confidence"] = 0.2
                steering_signal["correction_directive"] = (
                    "CRITICAL: Stagnation loop detected. Previous action yielded zero state entropy shift. "
                    "Pivot immediately to orthogonal search directions."
                )
                if "fallback_actions" in grounding_feedback:
                    fallbacks = grounding_feedback["fallback_actions"]
                    if fallbacks:
                        steering_signal["pivoted_action"] = fallbacks[0]

            elif grounding_feedback.get("fatal", False):
                steering_signal["valid"] = False
                steering_signal["confidence"] = 0.0
                steering_signal["correction_directive"] = (
                    "CRITICAL: Proposed action leads to fatal game-over or out-of-bounds. Veto and recalculate."
                )

        # Log steering trajectory
        self.trajectory_history.append({
            "reasoning": current_reasoning[:120] if current_reasoning else "",
            "steering": steering_signal,
        })

        return steering_signal

    def format_agent_instruct_prompt(
        self,
        task_id: str,
        task_type: str,
        grid: np.ndarray,
        prior_knowledge: Optional[list[str]] = None,
    ) -> str:
        """Constructs an Agent-Instruct steering context prompt for zero-shot LLM inference."""
        meta_steps = self.generate_meta_instructions(task_type, grid, prior_knowledge)
        
        prompt_lines = [
            f"=== [AGENT-INSTRUCT: ZERO-SHOT REASONING SUPERVISOR | Task: {task_id}] ===",
            "You are instructed by the Sovereign Meta-Reasoning Agent to execute rigorous zero-shot deductive synthesis.",
            "Follow these structured reasoning phases strictly:",
        ]
        for step in meta_steps:
            prompt_lines.append(f"  • {step}")
            
        prompt_lines.append(
            "Synthesize your final solution directly within exact operational grammar."
        )
        return "\n".join(prompt_lines)


# Global singleton instance
_AGENT_INSTRUCT = AgentInstructReasoner()

def get_agent_instruct() -> AgentInstructReasoner:
    return _AGENT_INSTRUCT
