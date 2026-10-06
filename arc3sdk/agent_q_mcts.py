"""Agent Q: Guided Monte Carlo Tree Search (MCTS) with Self-Critique Process Supervision and Step-Level DPO.

Based on:
"Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents"
Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, Rafael Rafailov (Stanford University / MultiOn, 2024 - arXiv:2408.07199).

Key Capabilities:
1. Guided MCTS Tree Search over ARC-AGI-3 Grid and Decision Spaces.
2. Self-Critique Value Function with Dual Process-Supervision:
   Q(h_t, a_t) = alpha * Q_critique(h_t, a_t) + (1 - alpha) * Q_rollout(h_t, a_t)
3. Step-Level Preference Pair Extraction (h_t, a_w, a_l) for Direct Preference Optimization (DPO).
4. Sub-branch Pruning with Epistemic Confidence Calibration.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set


@dataclass
class DPOPreferencePair:
    """Step-level preference pair (x, y_w, y_l) for Direct Preference Optimization."""
    history_state: Any
    winning_action: Any
    losing_action: Any
    margin: float
    q_winner: float
    q_loser: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentQNode:
    """MCTS Node equipped with self-critique scores and process supervision."""
    __slots__ = (
        'state_grid', 'history', 'parent', 'action_taken', 'children',
        'visits', 'q_rollout', 'q_critique', 'prior_p', 'is_terminal', 'depth'
    )

    def __init__(
        self,
        state_grid: np.ndarray,
        history: Tuple[Any, ...] = (),
        parent: Optional[AgentQNode] = None,
        action_taken: Any = None,
        prior_p: float = 1.0,
        depth: int = 0
    ):
        self.state_grid = state_grid
        self.history = history
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[Any, AgentQNode] = {}
        self.visits = 0
        self.q_rollout = 0.0
        self.q_critique = 0.0
        self.prior_p = prior_p
        self.is_terminal = False
        self.depth = depth

    def get_blended_q(self, alpha: float = 0.5) -> float:
        """Computes blended Q value combining self-critique and Monte Carlo rollout."""
        return alpha * self.q_critique + (1.0 - alpha) * self.q_rollout

    def ucb1_score(self, c_puct: float = 1.414, alpha: float = 0.5) -> float:
        """UCB1 / PUCT score with exploration bonus and blended Q."""
        parent_visits = max(1, self.parent.visits if self.parent else 1)
        exploration = c_puct * self.prior_p * math.sqrt(math.log(parent_visits + 1) / (1 + self.visits))
        return self.get_blended_q(alpha) + exploration


class AgentQEngine:
    """Agent Q Sovereign MCTS and Preference Optimization Engine."""

    def __init__(
        self,
        alpha: float = 0.5,
        c_puct: float = 1.414,
        max_depth: int = 12,
        num_simulations: int = 32,
        preference_margin_threshold: float = 0.25,
        beta_dpo: float = 0.1
    ):
        self.alpha = alpha
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.preference_margin_threshold = preference_margin_threshold
        self.beta_dpo = beta_dpo
        self.preference_dataset: List[DPOPreferencePair] = []

    def evaluate_critique(
        self,
        grid_before: np.ndarray,
        action: Any,
        grid_after: np.ndarray,
        recent_hashes: Optional[List[int]] = None
    ) -> float:
        """Sovereign Self-Critique Value Function: Evaluates action quality using geometric, topological, and spatial feedback."""
        # 1. State difference magnitude (Wall collision & No-op detection)
        diff = int(np.sum(grid_before != grid_after))
        if diff == 0:
            # Hitting a wall, dead click, or ineffective move -> heavy penalty
            return -0.75

        # 2. Cycle & Oscillation Penalty (Detecting A -> B -> A loops)
        if recent_hashes is not None:
            after_hash = hash(grid_after.tobytes())
            if after_hash in recent_hashes[-6:]:
                # State revisited within last 6 steps -> severe oscillation penalty
                return -0.85

        # 3. Spatial Centroid Tracking & Target Distance Reduction
        # Identify foreground objects (excluding dominant background color)
        bg_color = int(np.argmax(np.bincount(grid_before.ravel(), minlength=16)))
        fg_coords_before = np.argwhere(grid_before != bg_color)
        fg_coords_after = np.argwhere(grid_after != bg_color)
        
        spatial_bonus = 0.0
        if len(fg_coords_before) > 0 and len(fg_coords_after) > 0:
            # Centroid delta
            c_before = np.mean(fg_coords_before, axis=0)
            c_after = np.mean(fg_coords_after, axis=0)
            centroid_shift = float(np.linalg.norm(c_after - c_before))
            # Meaningful controlled movement
            if 0.5 <= centroid_shift <= 5.0:
                spatial_bonus += 0.3

        # 4. Entropy decrease reward (moving towards ordered solved pattern)
        counts_before = np.bincount(grid_before.ravel(), minlength=16)
        p_b = counts_before[counts_before > 0] / float(grid_before.size)
        ent_before = -float(np.sum(p_b * np.log2(p_b)))

        counts_after = np.bincount(grid_after.ravel(), minlength=16)
        p_a = counts_after[counts_after > 0] / float(grid_after.size)
        ent_after = -float(np.sum(p_a * np.log2(p_a)))

        entropy_delta = ent_before - ent_after

        # 5. Combined Normalized Score in [-1.0, 1.0]
        raw_score = entropy_delta * 2.0 + spatial_bonus + (0.25 if diff > 0 else -0.5)
        return float(np.tanh(raw_score))

    def search_and_plan(
        self,
        root_grid: np.ndarray,
        candidate_actions: List[Any],
        step_fn: Callable[[np.ndarray, Any], Tuple[np.ndarray, float, bool, bool]],
        custom_critique_fn: Optional[Callable[[np.ndarray, Any, np.ndarray], float]] = None
    ) -> Tuple[Any, float, List[Any], List[DPOPreferencePair]]:
        """Runs Agent Q Guided MCTS with self-critique and extracts step-level DPO pairs.

        Args:
            root_grid: Initial 2D numpy grid
            candidate_actions: List of valid actions
            step_fn: Function (grid, action) -> (next_grid, reward, is_terminal, is_fatal)
            custom_critique_fn: Optional custom critique scorer

        Returns:
            (best_action, confidence, planned_trajectory, extracted_preference_pairs)
        """
        if not candidate_actions:
            return None, 0.0, [], []

        critique_fn = custom_critique_fn or self.evaluate_critique
        root = AgentQNode(state_grid=root_grid, history=(), depth=0)
        extracted_dpo_pairs: List[DPOPreferencePair] = []

        # Expand root children
        for act in candidate_actions:
            next_g, r, is_term, is_fatal = step_fn(root_grid, act)
            if is_fatal:
                continue

            critique_val = critique_fn(root_grid, act, next_g)
            child = AgentQNode(
                state_grid=next_g,
                history=(act,),
                parent=root,
                action_taken=act,
                prior_p=1.0 + max(0.0, critique_val),
                depth=1
            )
            child.q_critique = critique_val
            child.q_rollout = r
            child.is_terminal = is_term
            root.children[str(act) if isinstance(act, dict) else act] = child

            if is_term:
                # Immediate win
                return act, 1.0, [act], []

        if not root.children:
            return candidate_actions[0], 0.1, [candidate_actions[0]], []

        # Run MCTS Simulations with Self-Critique & Rollouts
        for _ in range(self.num_simulations):
            current = root
            search_path = [current]

            # 1. Selection
            while current.children and current.depth < self.max_depth and not current.is_terminal:
                best_act, best_child = max(
                    current.children.items(),
                    key=lambda it: it[1].ucb1_score(self.c_puct, self.alpha)
                )
                current = best_child
                search_path.append(current)

            # 2. Expansion
            if not current.is_terminal and current.depth < self.max_depth and not current.children:
                for sub_act in candidate_actions:
                    g_next, r_step, term_step, fatal_step = step_fn(current.state_grid, sub_act)
                    if fatal_step:
                        continue
                    sub_critique = critique_fn(current.state_grid, sub_act, g_next)
                    sub_child = AgentQNode(
                        state_grid=g_next,
                        history=current.history + (sub_act,),
                        parent=current,
                        action_taken=sub_act,
                        prior_p=1.0 + max(0.0, sub_critique),
                        depth=current.depth + 1
                    )
                    sub_child.q_critique = sub_critique
                    sub_child.q_rollout = r_step
                    sub_child.is_terminal = term_step
                    current.children[str(sub_act) if isinstance(sub_act, dict) else sub_act] = sub_child

            # 3. Backpropagation with Blended Q
            terminal_val = 1.0 if current.is_terminal else current.get_blended_q(self.alpha)
            for node in reversed(search_path):
                node.visits += 1
                node.q_rollout += (terminal_val - node.q_rollout) / node.visits

        # Extract Step-Level Preference Pairs from Root and Explored Branches
        for node in [root] + list(root.children.values()):
            if len(node.children) >= 2:
                ranked_children = sorted(
                    node.children.values(),
                    key=lambda ch: ch.get_blended_q(self.alpha),
                    reverse=True
                )
                best_ch = ranked_children[0]
                worst_ch = ranked_children[-1]
                q_diff = best_ch.get_blended_q(self.alpha) - worst_ch.get_blended_q(self.alpha)

                if q_diff >= self.preference_margin_threshold:
                    pair = DPOPreferencePair(
                        history_state=node.history,
                        winning_action=best_ch.action_taken,
                        losing_action=worst_ch.action_taken,
                        margin=q_diff,
                        q_winner=best_ch.get_blended_q(self.alpha),
                        q_loser=worst_ch.get_blended_q(self.alpha),
                        metadata={'depth': node.depth, 'visits': node.visits}
                    )
                    extracted_dpo_pairs.append(pair)
                    self.preference_dataset.append(pair)

        # Select Best Action
        best_child = max(
            root.children.values(),
            key=lambda node: (node.visits, node.get_blended_q(self.alpha))
        )
        trajectory = list(best_child.history)
        conf = float(np.clip(0.6 + 0.4 * best_child.get_blended_q(self.alpha), 0.1, 0.99))

        return best_child.action_taken, conf, trajectory, extracted_dpo_pairs

    def compute_dpo_loss(
        self,
        pi_winner_logprob: float,
        pi_loser_logprob: float,
        ref_winner_logprob: float,
        ref_loser_logprob: float
    ) -> float:
        """Calculates exact Agent Q Direct Preference Optimization (DPO) Loss."""
        # log (pi(y_w|x) / ref(y_w|x)) - log (pi(y_l|x) / ref(y_l|x))
        pi_ratio = pi_winner_logprob - ref_winner_logprob
        ref_ratio = pi_loser_logprob - ref_loser_logprob
        implicit_reward_diff = self.beta_dpo * (pi_ratio - ref_ratio)

        # Sigmoid binary cross-entropy: -log(sigmoid(diff)) = log(1 + exp(-diff))
        loss = float(np.log1p(np.exp(-np.clip(implicit_reward_diff, -30.0, 30.0))))
        return loss
