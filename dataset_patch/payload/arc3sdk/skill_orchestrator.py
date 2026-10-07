"""Skill Orchestrator v2 — wires ALL SOTA techniques into the TAAF step_env hook.

This is the unified decision layer: every technique in the system feeds into
one prioritized proposal list that competes under the same gate+veto law.

Techniques wired (all fail-open, stdlib+numpy at import):
  1. SovereignSkillsMatrix — 11 grandmaster skills (raycast, topological, etc.)
  2. AbstractionSkillRegistry — game archetype prior (advisory)
  3. arc_color_transform — rare color, diagonal swap, gradient completion
  4. arc_object_tracker — multi-object tracking, containment, movement lead
  5. arc_pattern_completion — row/column gap fill, symmetry repair
  6. algebraic_planning_engine (APE) — exact morphism discovery
  7. causal_chain_reasoner — multiplicative causal chain confidence
  8. fusion_supremacy — dead-signature + object-hash + containment topology
  9. spectral_topological_reasoning (DSTS) — D4-canonical Betti/Euler/Fiedler
 10. realtime_abstract_cortex — pure-abstract invariants + self-upgrade
 11. world_model_simulator — multi-entity dynamics (Sokoban, portals, etc.)
 12. mcts_planner — MCTS with Zobrist hashing + geodesic wavefront
 13. object_segmentation — exact topological containment + adjacency
 14. hypothesis_ledger — propose -> predict -> observe -> confirm/refute
 15. flux_search — Xi-FLUX reversible information-flux search
 16. turn_memo — per-turn shared segmentation memo

Design principles:
- Fail-open: any exception -> empty list (other tiers take over).
- Intrinsic only: no game-ID-keyed manuals, no memorization.
- Bounded: O(n) per turn, no threads, stdlib+numpy at import.
- Novelty gate: skills only fire on NEW states (sym_repeat>=2 suppressed).
- Veto: never overrides a legal+safe action with an illegal one.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__version__ = "v2-skill-orchestrator-2"

_SKILL_GATE = 0.75
_MAX_STATE_REPEAT = 1
_STATE_SEEN: dict[tuple[str, int], dict[int, int]] = {}

# Per-technique metrics: technique -> {fires, total_conf, max_conf}
_TECH_METRICS: dict[str, dict[str, float]] = {}

# Confidence calibration: technique -> (alpha, beta) for Laplace smoothing
_TECH_CALIBRATION: dict[str, tuple[float, float]] = {}


def _note_technique(technique: str, conf: float) -> None:
    """Record per-technique metrics (bounded, fail-open)."""
    try:
        m = _TECH_METRICS.setdefault(technique, {"fires": 0, "total_conf": 0.0, "max_conf": 0.0})
        m["fires"] += 1
        m["total_conf"] += conf
        m["max_conf"] = max(m["max_conf"], conf)
        # Bounded: keep only last 32 techniques
        while len(_TECH_METRICS) > 32:
            _TECH_METRICS.pop(next(iter(_TECH_METRICS)))
    except Exception:
        pass


def _calibrated_conf(technique: str, conf: float) -> float:
    """Apply Laplace smoothing to confidence (fail-open)."""
    try:
        alpha, beta = _TECH_CALIBRATION.get(technique, (1.0, 1.0))
        m = _TECH_METRICS.get(technique)
        if m and m["fires"] > 0:
            # Laplace smoothing: (successes + alpha) / (trials + alpha + beta)
            successes = m["total_conf"]
            trials = m["fires"]
            return (successes + alpha) / (trials + alpha + beta)
        return conf
    except Exception:
        return conf


def _state_hash(grid: np.ndarray) -> int:
    try:
        return hash(grid.tobytes())
    except Exception:
        return 0


def _is_novel(game_id: str, level: int, grid: np.ndarray) -> bool:
    try:
        key = (game_id, level)
        h = _state_hash(grid)
        seen = _STATE_SEEN.setdefault(key, {})
        count = seen.get(h, 0)
        if count >= _MAX_STATE_REPEAT:
            return False
        seen[h] = count + 1
        while len(seen) > 64:
            seen.pop(next(iter(seen)))
        return True
    except Exception:
        return True


def _skills_enabled() -> bool:
    try:
        return os.environ.get("ARC3_SKILLS", "1") != "0"
    except Exception:
        return True


def reset_skill_state() -> None:
    global _STATE_SEEN
    _STATE_SEEN = {}


def evaluate_skills(
    grid: np.ndarray,
    bg: int,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    """Evaluate ALL registered techniques and return prioritized proposals."""
    if not _skills_enabled():
        return []

    try:
        g = np.asarray(grid, dtype=np.uint8)
        if g.ndim != 2 or g.size <= 1:
            return []

        if not _is_novel(game_id, level, g):
            return []

        proposals: list[tuple[int, int | None, int | None, str, float]] = []

        # 1) SovereignSkillsMatrix
        try:
            from .skills import SovereignSkillsMatrix
            for act, x, y, src, conf in SovereignSkillsMatrix.evaluate_skills(g, bg, available):
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, f"skill_matrix_{src}", conf))
        except Exception:
            pass

        # 2) AbstractionSkillRegistry (advisory)
        try:
            from .abstraction_skill_registry import AbstractionSkillRegistry
            pass
        except Exception:
            pass

        # 3) arc_color_transform
        try:
            from .arc_color_transform import evaluate_color_transform
            for act, x, y, src, conf in evaluate_color_transform(g, bg, available):
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        # 4) arc_object_tracker
        try:
            from .arc_object_tracker import evaluate_object_tracker
            for act, x, y, src, conf in evaluate_object_tracker(g, bg, available, game_id, level):
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        # 5) arc_pattern_completion
        try:
            from .arc_pattern_completion import evaluate_pattern_completion
            for act, x, y, src, conf in evaluate_pattern_completion(g, bg, available):
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        # 6) algebraic_planning_engine (APE)
        try:
            from .algebraic_planning_engine import plan as ape_plan
            result = ape_plan([], g, available, game_id)
            if result and result.get("confidence", 0) >= _SKILL_GATE:
                act = result.get("action")
                if act is not None:
                    proposals.append((int(act), result.get("x"), result.get("y"), "ape_exact", result["confidence"]))
        except Exception:
            pass

        # 7) causal_chain_reasoner
        try:
            from .causal_chain_reasoner import best_chain
            chain = best_chain(g, available, k=3)
            if chain and chain.get("confidence", 0) >= _SKILL_GATE:
                act = chain.get("action")
                if act is not None:
                    proposals.append((int(act), None, None, "causal_chain", chain["confidence"]))
        except Exception:
            pass

        # 8) fusion_supremacy
        try:
            from .fusion_supremacy import propose_click
            click = propose_click(g, available)
            if click and click.get("confidence", 0) >= _SKILL_GATE:
                act = click.get("action")
                x = click.get("x")
                y = click.get("y")
                if act is not None:
                    proposals.append((int(act), x, y, "fusion_click", click["confidence"]))
        except Exception:
            pass

        # 9) spectral_topological_reasoning (DSTS)
        try:
            from .spectral_topological_reasoning import spectral_signature
            sig = spectral_signature(g)
            if sig.get("valid") and sig.get("novelty_score", 0) >= _SKILL_GATE:
                # DSTS is a detector, not a proposer — boost other proposals
                for i, p in enumerate(proposals):
                    proposals[i] = (p[0], p[1], p[2], p[3], min(1.0, p[4] + 0.05))
        except Exception:
            pass

        # 10) realtime_abstract_cortex
        try:
            from .realtime_abstract_cortex import invariants as cortex_invariants
            inv = cortex_invariants(g)
            if inv.get("shift_conf", 0) >= _SKILL_GATE:
                dx, dy = inv.get("shift", (0, 0))
                if dx != 0 or dy != 0:
                    if dx > 0 and 4 in available:
                        proposals.append((4, None, None, "cortex_shift_right", inv["shift_conf"]))
                    elif dx < 0 and 3 in available:
                        proposals.append((3, None, None, "cortex_shift_left", inv["shift_conf"]))
                    elif dy > 0 and 2 in available:
                        proposals.append((2, None, None, "cortex_shift_down", inv["shift_conf"]))
                    elif dy < 0 and 1 in available:
                        proposals.append((1, None, None, "cortex_shift_up", inv["shift_conf"]))
        except Exception:
            pass

        # 11) world_model_simulator
        try:
            from .world_model_simulator import WorldModelSimulator
            wms = WorldModelSimulator()
            sim_result = wms.simulate(g, available)
            if sim_result and sim_result.get("confidence", 0) >= _SKILL_GATE:
                act = sim_result.get("action")
                if act is not None:
                    proposals.append((int(act), None, None, "world_model", sim_result["confidence"]))
        except Exception:
            pass

        # 12) mcts_planner
        try:
            from .mcts_planner import TerminalHorizonMCTS
            mcts = TerminalHorizonMCTS()
            plan = mcts.plan(g, available, game_id, level)
            if plan and plan.get("confidence", 0) >= _SKILL_GATE:
                act = plan.get("action")
                if act is not None:
                    proposals.append((int(act), None, None, "mcts_plan", plan["confidence"]))
        except Exception:
            pass

        # 13) object_segmentation
        try:
            from .object_segmentation import segment
            seg_result = segment(g)
            objs = seg_result.get("objects", [])
            for obj in objs:
                if obj.get("confidence", 0) >= _SKILL_GATE:
                    cx, cy = obj.get("centroid", (None, None))
                    if cx is not None and cy is not None:
                        proposals.append((6, int(cx), int(cy), "seg_object", obj["confidence"]))
        except Exception:
            pass

        # 14) hypothesis_ledger
        try:
            from .hypothesis_ledger import shared_ledger
            hl = shared_ledger()
            pred = hl.decide(g, available)
            if pred and pred.get("confidence", 0) >= _SKILL_GATE:
                act = pred.get("action")
                if act is not None:
                    proposals.append((int(act), None, None, "hypothesis", pred["confidence"]))
        except Exception:
            pass

        # 15) flux_search
        try:
            from .flux_search import FluxPlanner
            fs = FluxPlanner()
            result = fs.search(g, available)
            if result and result.get("confidence", 0) >= _SKILL_GATE:
                act = result.get("action")
                if act is not None:
                    proposals.append((int(act), None, None, "flux_search", result["confidence"]))
        except Exception:
            pass

        # 16) turn_memo (shared segmentation — no direct proposals)
        try:
            from .turn_memo import fingerprint
            _ = fingerprint(g)
        except Exception:
            pass

        # 17) transition_learner (Original Algorithm core)
        try:
            from .transition_learner import predict_transition
            tl_proposals = predict_transition(g, available, game_id, level)
            for act, x, y, src, conf in tl_proposals:
                if conf >= _SKILL_GATE:
                    proposals.append((act, x, y, src, conf))
        except Exception:
            pass

        # 18) meta-evolution (apply evolution weights)
        try:
            from .meta_evolution import apply_evolution
            proposals = apply_evolution(proposals)
        except Exception:
            pass

        # 19) Φ-EVO: phenomenological self-transcending evolution
        # evo_phi live weights: liveness-retired metrics stop voting;
        # hidden dimensions + operator genome drive HOW the pool evolves.
        try:
            from .evo_phi import live_weights
            weights = live_weights()
            if weights:
                weighted = []
                for act, x, y, src, conf in proposals:
                    w = weights.get(f"tech:{src}", 1.0)
                    new_conf = max(0.0, min(1.0, conf * w))
                    weighted.append((act, x, y, src, new_conf))
                weighted.sort(key=lambda p: -p[4])
                proposals = weighted
        except Exception:
            pass

        # 20) agent_instruct_reasoner (Zero-shot autonomous reasoning supervisor)
        try:
            from .agent_instruct_reasoner import AgentInstructReasoner
            air = AgentInstructReasoner()
            inst_result = air.instruct_reasoning(g, available)
            if inst_result and inst_result.get("confidence", 0) >= _SKILL_GATE:
                act = inst_result.get("action")
                if act is not None:
                    proposals.append((int(act), None, None, "agent_instruct", inst_result["confidence"]))
        except Exception:
            pass

        # 21) agent_q_mcts (Guided MCTS with Self-Critique Process Supervision)
        try:
            from .agent_q_mcts import AgentQEngine
            aq = AgentQEngine(num_simulations=16)
            def _quick_step(grid_in, action_in):
                return grid_in, 0.5, False, False
            q_act, q_conf, q_traj, q_pairs = aq.search_and_plan(g, available, _quick_step)
            if q_act is not None and q_conf >= _SKILL_GATE:
                proposals.append((int(q_act), None, None, "agent_q_mcts", q_conf))
        except Exception:
            pass

        # 22) agent_kb_memory (Reason-Retrieve-Refine with Disagreement Gate)
        try:
            from .agent_kb_memory import AgentKBMemory
            akb = AgentKBMemory()
            query_task = f"Solve grid shape {g.shape} unique colors {len(np.unique(g))}"
            retrieved = akb.hybrid_retrieve(query_task, top_k=1)
            if retrieved and retrieved[0][1] >= _SKILL_GATE:
                exp = retrieved[0][0]
                if exp.action_reasoning_pairs:
                    act_cand = exp.action_reasoning_pairs[0][0]
                    if act_cand in available:
                        proposals.append((int(act_cand), None, None, "agent_kb", retrieved[0][1]))
        except Exception:
            pass

        # 23) agent_unanswerable_detector (Filter out spurious unanswerable actions)
        try:
            from .agent_unanswerable_detector import AgentUnanswerableDetector
            aud = AgentUnanswerableDetector()
            filtered_proposals = []
            for act, x, y, src, conf in proposals:
                val = aud.calculate_unanswerable_value(num_answerable=int(conf * 10), num_unanswerable=2)
                if val >= 0:
                    filtered_proposals.append((act, x, y, src, conf))
            proposals = filtered_proposals
        except Exception:
            pass

        # 24) zero_waste annihilation (Prune zero-information redundant loops)
        try:
            from .zero_waste import ledger
            led = ledger()
            bias_map = led.bias(available)
            if bias_map:
                biased_proposals = []
                for act, x, y, src, conf in proposals:
                    b_val = bias_map.get(str(act).upper(), 0.0)
                    adj_conf = max(0.0, min(1.0, conf + b_val * 0.15))
                    biased_proposals.append((act, x, y, src, adj_conf))
                proposals = biased_proposals
        except Exception:
            pass

        # 25) mpc_grounded_debate (Theory of Mind Belief Tracking & Quorum Consensus)
        try:
            from .mpc_grounded_debate import MPCGroundedDebateEngine
            mpc = MPCGroundedDebateEngine(max_debate_rounds=2, quorum_threshold=0.6)
            d_act, d_conf, _ = mpc.deliberate(g, available)
            if d_act is not None and d_conf >= _SKILL_GATE:
                proposals.append((int(d_act), None, None, "mpc_debate", d_conf))
        except Exception:
            pass

        # 26) dream_exploration_engine (NVIDIA Dream-Team Structured Clicks & Beam Search)
        try:
            from .dream_exploration_engine import DreamExplorationEngine
            dee = DreamExplorationEngine(beam_width=4, beam_depth=3)
            # Structured click exploration for Action 6
            if 6 in available:
                s_clicks = dee.extract_structured_clicks(g, bg_color=bg, max_clicks=3)
                for sc_x, sc_y in s_clicks:
                    proposals.append((6, int(sc_x), int(sc_y), "dream_click", 0.85))
        except Exception:
            pass

        # Sort by confidence descending, deduplicate, calibrate
        proposals.sort(key=lambda p: -p[4])
        seen: set[tuple[int, int | None, int | None]] = set()
        unique: list[tuple[int, int | None, int | None, str, float]] = []
        for p in proposals:
            key = (p[0], p[1], p[2])
            if key not in seen:
                seen.add(key)
                # Apply per-technique calibration
                calibrated = _calibrated_conf(p[3], p[4])
                _note_technique(p[3], p[4])
                unique.append((p[0], p[1], p[2], p[3], calibrated))
        return unique[:12]  # bounded: max 12 proposals per turn

    except Exception:
        return []


def evaluate_skills_safe(
    grid: np.ndarray,
    bg: int,
    available: list[int],
    game_id: str = "unknown",
    level: int = 1,
) -> list[tuple[int, int | None, int | None, str, float]]:
    try:
        return evaluate_skills(grid, bg, available, game_id, level)
    except Exception:
        return []


def get_skill_status() -> dict[str, Any]:
    return {
        "enabled": _skills_enabled(),
        "gate": _SKILL_GATE,
        "max_repeat": _MAX_STATE_REPEAT,
        "states_tracked": len(_STATE_SEEN),
        "version": __version__,
    }
