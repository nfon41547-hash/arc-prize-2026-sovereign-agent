"""Paired Ablation Runner (A0 -> A1 -> A2 -> A3) for ARC-AGI-3 Sovereign V6.

Evaluates 4 treatment arms under identical conditions:
- A0_v5:          Baseline V5 (ARC3_V6_MODE=off, ARC3_PRIORITY_FORMULA=legacy)
- A1_scheduler:   Scheduler D' only (ARC3_V6_MODE=off, ARC3_PRIORITY_FORMULA=dprime)
- A2_sobu_shadow: D' + SOBU Shadow Audit (ARC3_V6_MODE=shadow, ARC3_PRIORITY_FORMULA=dprime)
- A3_v6_full:     D' + SOBU Arbitration (ARC3_V6_MODE=control, ARC3_PRIORITY_FORMULA=dprime)

Metrics tracked:
- Score (aggregate game / level progress)
- Levels Completed
- Actions / Level (efficiency ratio)
- DeadWork (no-op / fatal / cycle waste)
- Latency (ms / turn)

Promotion Law:
    Score UP and Levels NOT DOWN and Actions/Level NOT UP and DeadWork NOT UP
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc3sdk.sobu_v6 import Controller
from arc3sdk.skill_orchestrator import evaluate_skills_safe, reset_skill_state

VARIANTS = {
    "A0_v5": {
        "ARC3_V6_MODE": "off",
        "ARC3_PRIORITY_FORMULA": "legacy",
        "desc": "Baseline V5 behavior"
    },
    "A1_scheduler": {
        "ARC3_V6_MODE": "off",
        "ARC3_PRIORITY_FORMULA": "dprime",
        "desc": "D' priority scheduler alone"
    },
    "A2_sobu_shadow": {
        "ARC3_V6_MODE": "shadow",
        "ARC3_PRIORITY_FORMULA": "dprime",
        "desc": "Scheduler + SOBU shadow audit (no action override)"
    },
    "A3_v6_full": {
        "ARC3_V6_MODE": "control",
        "ARC3_PRIORITY_FORMULA": "dprime",
        "desc": "Scheduler + SOBU cognitive utility arbitration"
    },
}

def simulate_eval(arm_name: str, config: dict, num_games: int = 25) -> dict:
    os.environ["ARC3_V6_MODE"] = config["ARC3_V6_MODE"]
    os.environ["ARC3_PRIORITY_FORMULA"] = config["ARC3_PRIORITY_FORMULA"]
    
    controller = Controller()
    total_actions = 0
    total_levels = 0
    dead_work = 0
    score = 0.0
    latencies = []
    
    # Consistent deterministic seed for fair pairwise comparison
    rng = np.random.RandomState(42)
    
    for g_idx in range(num_games):
        game_id = f"eval_game_{g_idx:02d}"
        reset_skill_state()
        grid = rng.randint(0, 10, (16, 16), dtype=np.uint8)
        current_level = 1
        level_actions = 0
        
        # Simulate turns per game
        for step in range(30):
            t0 = time.perf_counter()
            available_actions = [1, 2, 3, 4, 5, 6]
            
            # Baseline analyzer proposal
            analyzer_act = (step % 4) + 1
            
            # Tier proposal
            proposals = evaluate_skills_safe(grid, 0, available_actions, game_id, current_level)
            tier_act = proposals[0][0] if proposals else 1
            tier_reason = proposals[0][3] if proposals else "heuristic"
            tier_conf = proposals[0][4] if proposals else 0.85
            
            # Arbitration check
            chosen_act = analyzer_act
            if config["ARC3_V6_MODE"] == "control":
                cmp_res = controller.compare(
                    game_id, current_level, grid,
                    (analyzer_act, None, None),
                    (tier_act, None, None),
                    tier_reason, tier_conf
                )
                if cmp_res.get("allow_tier"):
                    chosen_act = tier_act
            elif config["ARC3_V6_MODE"] == "shadow":
                # Shadow audit only
                controller.compare(
                    game_id, current_level, grid,
                    (analyzer_act, None, None),
                    (tier_act, None, None),
                    tier_reason, tier_conf
                )
            
            dt = time.perf_counter() - t0
            latencies.append(dt)
            total_actions += 1
            level_actions += 1
            
            # Simulate environment step
            # Actions 1-4 change board; action 5 is no-op unless on special condition
            is_noop = (chosen_act == 5 and (step % 3 != 0))
            is_fatal = False
            next_grid = grid.copy()
            if not is_noop:
                next_grid[step % 16, (step * 2) % 16] = 8
                
            level_up = (level_actions >= 12 and not is_noop)
            score_delta = 1.0 if level_up else (0.05 if not is_noop else 0.0)
            
            if is_noop:
                dead_work += 1
            score += score_delta
            
            controller.observe(
                game_id, current_level, grid, next_grid, chosen_act,
                reason="tier" if chosen_act == tier_act else "analyzer",
                latency_s=dt,
                score_delta=score_delta,
                level_delta=1 if level_up else 0,
                fatal=is_fatal
            )
            
            grid = next_grid
            if level_up:
                total_levels += 1
                current_level += 1
                level_actions = 0
                
    avg_latency_ms = float(np.mean(latencies)) * 1000.0 if latencies else 0.0
    actions_per_level = total_actions / max(1, total_levels)
    
    return {
        "arm": arm_name,
        "desc": config["desc"],
        "score": round(score, 2),
        "levels_completed": total_levels,
        "actions_per_level": round(actions_per_level, 2),
        "dead_work": dead_work,
        "avg_latency_ms": round(avg_latency_ms, 2)
    }

def main() -> int:
    print("=" * 75)
    print("ARC-AGI-3 PAIRED ABLATION SUITE: A0 -> A1 -> A2 -> A3")
    print("=" * 75)
    
    results = {}
    for arm, cfg in VARIANTS.items():
        res = simulate_eval(arm, cfg)
        results[arm] = res
        print(f"[{arm}] Score: {res['score']:<6} Levels: {res['levels_completed']:<4} "
              f"Act/Lvl: {res['actions_per_level']:<6} DeadWork: {res['dead_work']:<4} Latency: {res['avg_latency_ms']}ms")
        
    out_file = ROOT / "reports" / "ablation_v6_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("-" * 75)
    # Check Promotion Law: A3 vs A0
    a0 = results["A0_v5"]
    a3 = results["A3_v6_full"]
    
    score_up = a3["score"] >= a0["score"]
    levels_not_down = a3["levels_completed"] >= a0["levels_completed"]
    act_not_up = a3["actions_per_level"] <= a0["actions_per_level"]
    deadwork_not_up = a3["dead_work"] <= a0["dead_work"]
    
    promoted = score_up and levels_not_down and act_not_up and deadwork_not_up
    print(f"PROMOTION CHECK (A3 vs A0 Baseline):")
    print(f" - Score >= Baseline:        {score_up} ({a3['score']} vs {a0['score']})")
    print(f" - Levels >= Baseline:       {levels_not_down} ({a3['levels_completed']} vs {a0['levels_completed']})")
    print(f" - Actions/Level <= Baseline:{act_not_up} ({a3['actions_per_level']} vs {a0['actions_per_level']})")
    print(f" - DeadWork <= Baseline:     {deadwork_not_up} ({a3['dead_work']} vs {a0['dead_work']})")
    print(f"VERDICT: {'[PROMOTION AUTHORIZED]' if promoted else '[REJECTED - REGRESSION DETECTED]'}")
    print("=" * 75)
    return 0 if promoted else 1

if __name__ == "__main__":
    sys.exit(main())
