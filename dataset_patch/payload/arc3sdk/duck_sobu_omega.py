"""Persistent World Memory (M_t) and Tri-Modal SOBU-Omega Controller.

M_t = {
    objects: list of persistent objects and spatial anchors,
    rules: inferred physics / interaction rules,
    goals: teleological target coords and conditions,
    tested_hypotheses: hypotheses with empirical validation count,
    failed_actions: exact state-action pairs proven fatal or ineffective,
    verified_transitions: (state_hash, action) -> (next_state_hash, delta_score),
    plan: queued shortest verified plan,
}

Tri-Modal Controller Modes:
    EXPLORE -> VERIFY -> COMMIT
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

def _grid_hash(grid: np.ndarray) -> str:
    try:
        a = np.asarray(grid, dtype=np.uint8)
        if a.ndim > 2:
            while a.ndim > 2 and a.shape[0] == 1: a = a[0]
        if a.ndim == 2 and min(a.shape) > 2:
            a = a[1:-1, 1:-1]
        return hashlib.blake2b(a.tobytes(), digest_size=12).hexdigest()
    except Exception:
        return hashlib.blake2b(repr(grid).encode("utf-8", "replace"), digest_size=12).hexdigest()

@dataclass
class WorldMemory:
    game_id: str
    level: int = 1
    objects: list[dict[str, Any]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    goals: list[list[int]] = field(default_factory=list)
    tested_hypotheses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_actions: set[tuple[str, Any]] = field(default_factory=set)
    verified_transitions: dict[tuple[str, Any], tuple[str, float]] = field(default_factory=dict)
    plan: deque[Any] = field(default_factory=deque)
    
    # Mode: "EXPLORE", "VERIFY", "COMMIT"
    mode: str = "EXPLORE"
    commit_threshold: float = 0.85

    def format_summary_for_prompt(self) -> str:
        """Render compact M_t summary into prompt context for ephemeral REPL."""
        lines = [f"[WORLD MEMORY M_t | Mode: {self.mode}]"]
        if self.goals:
            lines.append(f"- Goals: {self.goals[:3]}")
        if self.rules:
            lines.append(f"- Rules: {'; '.join(self.rules[:5])}")
        if self.failed_actions:
            lines.append(f"- Failed actions recorded: {len(self.failed_actions)}")
        if self.verified_transitions:
            lines.append(f"- Verified transitions: {len(self.verified_transitions)}")
        if self.plan:
            lines.append(f"- Active Plan ({len(self.plan)} steps): {list(self.plan)[:5]}")
        return "\n".join(lines)

class DuckSobuOmegaController:
    """The Tri-Modal Sovereign Controller for Duck Harness."""

    def __init__(
        self,
        beta_info: float = 0.40,
        eta_synergy: float = 0.25,
        gamma_value: float = 0.80,
        lambda_f: float = 0.85,
        lambda_r: float = 0.30,
        lambda_k: float = 0.20,
        lambda_l: float = 0.05,
    ):
        self.beta_info = beta_info
        self.eta_synergy = eta_synergy
        self.gamma_value = gamma_value
        self.lambda_f = lambda_f
        self.lambda_r = lambda_r
        self.lambda_k = lambda_k
        self.lambda_l = lambda_l
        
        self.memories: dict[str, WorldMemory] = {}
        self.state_visits: dict[str, int] = defaultdict(int)

    def get_memory(self, game_id: str, level: int = 1) -> WorldMemory:
        if game_id not in self.memories:
            self.memories[game_id] = WorldMemory(game_id=game_id, level=level)
        mem = self.memories[game_id]
        mem.level = level
        return mem

    def evaluate_utility(
        self,
        game_id: str,
        level: int,
        grid: np.ndarray,
        action: Any,
        candidate_goal: Optional[list[int]] = None,
    ) -> float:
        """Computes U_t(a) = Delta_Phi + beta*I* + eta*S + gamma*V - lambda_F*F - lambda_R*R - lambda_K*K - lambda_L*L."""
        mem = self.get_memory(game_id, level)
        sig = _grid_hash(grid)
        act_key = tuple(action.items()) if isinstance(action, dict) else action
        
        # 1. Failed action hard-negative veto
        if (sig, act_key) in mem.failed_actions:
            return -999.0
            
        # 2. Score progress expectation Delta_Phi (non-zero only if verified transition or goal progress)
        delta_phi = 0.0
        trans_key = (sig, act_key)
        if trans_key in mem.verified_transitions:
            _, delta_score = mem.verified_transitions[trans_key]
            delta_phi = delta_score * 1.5

        # 3. Decision Information Gain I*
        visits = self.state_visits[sig]
        info_gain = 1.0 / (1.0 + np.log1p(visits))
        
        # In COMMIT mode, information gain has 0 value (we only want verified path)
        if mem.mode == "COMMIT":
            info_gain = 0.0
            
        # 4. Synergy S toward goal
        synergy = 0.0
        if candidate_goal is not None and isinstance(action, int) and 1 <= action <= 4:
            # Check directional alignment
            synergy = 0.5

        # 5. Penalties
        f_risk = 0.0 # fatal risk
        r_repeat = min(1.0, visits / 4.0)
        k_cycle = 0.2 if visits > 2 else 0.0
        l_latency = 0.05
        
        utility = (
            delta_phi
            + self.beta_info * info_gain
            + self.eta_synergy * synergy
            - self.lambda_f * f_risk
            - self.lambda_r * r_repeat
            - self.lambda_k * k_cycle
            - self.lambda_l * l_latency
        )
        return float(utility)

    def observe_transition(
        self,
        game_id: str,
        level: int,
        before_grid: np.ndarray,
        after_grid: np.ndarray,
        action: Any,
        score_delta: float,
        level_completed: bool,
        game_over: bool,
    ) -> None:
        """Update M_t from realized transition."""
        mem = self.get_memory(game_id, level)
        sig_before = _grid_hash(before_grid)
        sig_after = _grid_hash(after_grid)
        act_key = tuple(action.items()) if isinstance(action, dict) else action
        
        changed = (sig_before != sig_after)
        self.state_visits[sig_before] += 1
        
        if game_over or (not changed and score_delta <= 0 and not level_completed):
            # Record failed action (no-op or fatal)
            mem.failed_actions.add((sig_before, act_key))
        else:
            # Verified transition
            mem.verified_transitions[(sig_before, act_key)] = (sig_after, score_delta)
            if score_delta > 0 or level_completed:
                mem.rules.append(f"Action {action} on {sig_before[:6]} yields progress")
                # Transition to COMMIT mode once progress plan is grounded
                mem.mode = "COMMIT"

_SOBU_OMEGA = DuckSobuOmegaController()

def get_sobu_omega() -> DuckSobuOmegaController:
    return _SOBU_OMEGA

def install_tool_agent_sobu_hook(tool_agent_cls: Any = None) -> dict[str, Any]:
    """Hook ToolAgent._summarized_knowledge_lines to inject persistent M_t into prompt context.
    
    Idempotent, fail-open, zero-dependency.
    """
    try:
        if tool_agent_cls is None:
            import sys
            mod = sys.modules.get("inference.agent.tool_agent")
            if mod is None:
                try:
                    mod = __import__("inference.agent.tool_agent", fromlist=["ToolAgent"])
                except Exception:
                    mod = None
            if mod is not None:
                tool_agent_cls = getattr(mod, "ToolAgent", None)
        
        if tool_agent_cls is None:
            return {"installed": False, "reason": "no-tool-agent-class"}
            
        if getattr(tool_agent_cls, "_sobu_omega_wrapped", False):
            return {"installed": True, "reason": "already-installed"}
            
        orig_knowledge = getattr(tool_agent_cls, "_summarized_knowledge_lines", None)
        if not callable(orig_knowledge):
            return {"installed": False, "reason": "no-knowledge-method"}
            
        def _wrapped_knowledge(self: Any) -> list[str]:
            lines = list(orig_knowledge(self) or [])
            try:
                # Derive game_id and level
                game_id = "active_game"
                rdir = getattr(self, "_session_runtime_dir", None)
                if rdir is not None:
                    game_id = str(rdir.stem if hasattr(rdir, "stem") else rdir)
                level = 1
                summary = getattr(self, "_last_step_summary", None)
                if isinstance(summary, dict) and summary.get("level") is not None:
                    try:
                        level = int(summary["level"])
                    except Exception:
                        level = 1
                mem = get_sobu_omega().get_memory(game_id, level)
                m_t_str = mem.format_summary_for_prompt()
                if m_t_str:
                    lines.append(m_t_str)
            except Exception:
                pass
            return lines
            
        tool_agent_cls._summarized_knowledge_lines = _wrapped_knowledge
        tool_agent_cls._sobu_omega_wrapped = True
        return {"installed": True, "reason": "success"}
    except Exception as exc:
        return {"installed": False, "reason": f"error:{type(exc).__name__}"}

