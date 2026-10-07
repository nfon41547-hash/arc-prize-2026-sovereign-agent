"""Agent Symbolic Learning: Data-Centric Self-Evolving Agents via Language Back-Propagation.

Based on the foundational research:
"Symbolic Learning Enables Self-Evolving Agents"
(Zhou, Ou, Ding, Li, Wu, Wang, Chen, Wang, Xu, Zhang, Chen, Jiang, 2024 - AIWaves).

Analogy to Connectionist Deep Learning:
- Computational Graph <-> Agent Pipeline (A) with Nodes (N_1, ..., N_k)
- Neural Weights      <-> Prompts (P_n), Tools (T_n), and Node Connections
- Loss Function       <-> Language Loss L_lang = L(tau)
- Backpropagation     <-> Language Gradients nabla_lang^n = G(nabla_lang^(n+1), I_n, O_n, P_n, T_n, L_lang)
- Optimizer (SGD/Adam)<-> Symbolic Optimizers (PromptOptimizer, ToolOptimizer, PipelineOptimizer)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class NodeTrajectory:
    node_id: str
    node_input: Any
    node_output: Any
    prompt_config: dict[str, Any] = field(default_factory=dict)
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    gradient: Optional[str] = None


class AgentSymbolicLearner:
    """Orchestrates Language Loss Computation, Gradient Back-Propagation, and Symbolic Weight Updates."""

    def __init__(
        self,
        learning_rate: float = 0.5,
        max_retries: int = 3,
        rollback_on_degradation: bool = True,
    ):
        self.learning_rate = learning_rate
        self.max_retries = max_retries
        self.rollback_on_degradation = rollback_on_degradation
        self.trajectory: list[NodeTrajectory] = []
        self.symbolic_weights: dict[str, Any] = {
            "prompts": {},
            "tools": {},
            "pipeline": ["perception", "hypothesis", "action_selection", "verification"],
        }
        self.history_losses: list[float] = []

    def record_forward_step(
        self,
        node_id: str,
        node_input: Any,
        node_output: Any,
        prompt_config: Optional[dict[str, Any]] = None,
        tool_usage: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Records forward pass execution info into trajectory tau."""
        self.trajectory.append(
            NodeTrajectory(
                node_id=node_id,
                node_input=node_input,
                node_output=node_output,
                prompt_config=prompt_config or {},
                tool_usage=tool_usage or [],
            )
        )

    def compute_language_loss(
        self,
        task_goal: str,
        actual_outcome: Any,
        expected_outcome: Optional[Any] = None,
        success: bool = False,
    ) -> tuple[float, str]:
        """Computes holistic Language Loss L_lang = L(tau) with quantitative score and textual critique."""
        if success:
            loss_score = 0.0
            critique = "Execution successful. No gradient update needed."
        else:
            loss_score = 1.0
            critique = (
                f"Task failure detected for goal '{task_goal}'. "
                f"Outcome '{actual_outcome}' diverged from expected '{expected_outcome}'."
            )
        self.history_losses.append(loss_score)
        return loss_score, critique

    def backpropagate_language_gradients(self, language_loss_critique: str) -> list[str]:
        """Propagates language gradients backward from node N_k to node N_1."""
        gradients = []
        downstream_gradient = language_loss_critique

        # Iterate in reverse order
        for step in reversed(self.trajectory):
            node_grad = (
                f"Node [{step.node_id}] Gradient: Input had shape/type {type(step.node_input)}, "
                f"produced output {str(step.node_output)[:60]}. "
                f"Refined directive from downstream: '{downstream_gradient}'"
            )
            step.gradient = node_grad
            gradients.append(node_grad)
            downstream_gradient = f"Ensure output of [{step.node_id}] constrains search space better"

        return list(reversed(gradients))

    def update_symbolic_weights(self, gradients: list[str]) -> dict[str, Any]:
        """Updates prompts, tools, and pipeline via Symbolic Optimizers using language gradients."""
        updates_applied = {}
        for grad, step in zip(gradients, self.trajectory):
            node_id = step.node_id
            # Symbolic prompt optimizer
            old_prompt = self.symbolic_weights["prompts"].get(node_id, "default_policy")
            new_prompt = f"{old_prompt} | Guided by grad: {grad[:80]}"
            self.symbolic_weights["prompts"][node_id] = new_prompt
            updates_applied[node_id] = "prompt_updated"

        # Clear trajectory after successful gradient update
        self.trajectory.clear()
        return updates_applied


# Global singleton
_SYMBOLIC_LEARNER = AgentSymbolicLearner()

def get_symbolic_learner() -> AgentSymbolicLearner:
    return _SYMBOLIC_LEARNER
