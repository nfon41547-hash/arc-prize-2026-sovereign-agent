"""Episodic Hindsight Pruner (ASCENT-inspired for ARC-AGI-3).

Extracts causal and goal-progressing actions from completed levels:
T_pruned = { a_t in T | DeltaPhi(a_t) > 0 or CausalStateChange(a_t) == True }
Prunes:
1. No-ops (board didn't change and no level/score advance)
2. Wall hits / fatal reverts
3. Cycles (returning to the exact same previous state without progress)
4. Redundant exploration loops
"""
from __future__ import annotations
import hashlib
from typing import Any, Sequence, Dict, List, Tuple

def _state_hash(grid: Any) -> str:
    try:
        import numpy as np
        arr = np.asarray(grid, dtype=np.uint8)
        return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()
    except Exception:
        return hashlib.blake2b(repr(grid).encode('utf-8', 'replace'), digest_size=8).hexdigest()

class EpisodicHindsightPruner:
    """Zero-overhead backward trace analyzer and trajectory pruner."""

    @staticmethod
    def prune_trajectory(history_entries: Sequence[Any]) -> List[Dict[str, Any]]:
        """Backward trace analysis: keep only causal, goal-progressing steps.
        
        history_entries are objects with attributes:
          .action (str or dict)
          .result (dict with 'action_display', 'board_changed', 'score', etc.)
          .frame (.grid, .level)
        or tuples/dicts with similar semantics.
        """
        if not history_entries:
            return []
            
        raw_steps: List[Dict[str, Any]] = []
        for i, entry in enumerate(history_entries):
            action_name = ""
            grid = None
            level = 0
            score = 0.0
            changed = False
            
            if hasattr(entry, 'frame') and entry.frame is not None:
                grid = getattr(entry.frame, 'grid', None)
                level = int(getattr(entry.frame, 'level', 0) or 0)
            if hasattr(entry, 'action'):
                action_name = str(entry.action or "")
            if hasattr(entry, 'result') and isinstance(entry.result, dict):
                action_name = entry.result.get('action_display') or action_name
                changed = bool(entry.result.get('board_changed', False))
                score = float(entry.result.get('score', 0.0) or 0.0)
            elif isinstance(entry, dict):
                grid = entry.get('grid')
                level = int(entry.get('level', 0) or 0)
                action_name = str(entry.get('action', ""))
                changed = bool(entry.get('changed', False))
                score = float(entry.get('score', 0.0) or 0.0)
                
            raw_steps.append({
                "idx": i,
                "action": action_name,
                "grid": grid,
                "level": level,
                "score": score,
                "changed": changed,
                "hash": _state_hash(grid) if grid is not None else ""
            })

        if not raw_steps:
            return []

        # 1. Forward Pass: Mark state changes and identify cycle loops
        pruned: List[Dict[str, Any]] = []
        seen_states: Dict[str, int] = {}
        
        for step in raw_steps:
            h = step["hash"]
            # Detect cycles: if we came back to a state seen earlier without level/score delta
            if h and h in seen_states:
                prev_idx = seen_states[h]
                # Rollback intermediate non-essential actions in cycle
                while pruned and pruned[-1]["idx"] >= prev_idx:
                    pruned.pop()
            
            seen_states[h] = step["idx"]
            pruned.append(step)

        # 2. Filter out explicit No-Ops (where grid didn't change and no score/level jump)
        final_causal: List[Dict[str, Any]] = []
        for i in range(len(pruned)):
            cur = pruned[i]
            prev = pruned[i-1] if i > 0 else None
            
            has_state_diff = (cur["hash"] != prev["hash"]) if (prev and cur["hash"] and prev["hash"]) else cur["changed"]
            has_score_diff = (cur["score"] > prev["score"]) if prev else (cur["score"] > 0)
            has_level_diff = (cur["level"] > prev["level"]) if prev else False
            
            # Action qualifies if CausalStateChange == True or DeltaPhi > 0
            if has_state_diff or has_score_diff or has_level_diff or cur["idx"] == 0:
                final_causal.append(cur)

        return final_causal

    @staticmethod
    def format_pruned_digest(pruned_steps: Sequence[Dict[str, Any]], max_steps: int = 15) -> str:
        """Format the hindsight-distilled sequence into high-density prompt working memory."""
        if not pruned_steps:
            return ""
        actions = [s["action"] for s in pruned_steps if s.get("action") and str(s.get("action")).upper() not in ("RESET", "")]
        if not actions:
            return ""
        recent = actions[-max_steps:]
        return " -> ".join(recent)
