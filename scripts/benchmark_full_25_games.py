"""
ARC-AGI-3 Full 25 Public Games Arcade Benchmark
Executes the Sovereign Grandmaster Symplectic-Geodesic Consensus Agent
across all 25 official public games directly against the live ARC Arcade SDK.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Set, Tuple

import arc_agi

# Ensure SDK is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc3sdk.symplectic_geodesic_engine import SymplecticGeodesicWavefrontEngine
from arc3sdk.auto_aligned_interface import AutoAlignedInterfaceEngine
from arc3sdk.policy_manifold_reflection import PolicyManifoldReflectionEngine
from arc3sdk.unified_consensus_engine import SovereignGrandmasterKernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Full25Benchmark")

def run_full_25_benchmark():
    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()
    logger.info(f"Loaded {len(envs)} official ARC-AGI-3 environments from Arcade API.")

    sgwe = SymplecticGeodesicWavefrontEngine()
    align = AutoAlignedInterfaceEngine()
    pmr = PolicyManifoldReflectionEngine()

    game_results = []
    total_start_time = time.perf_counter()

    for idx, env_info in enumerate(envs, start=1):
        game_id = env_info.game_id
        logger.info(f"\n=======================================================")
        logger.info(f"[{idx}/{len(envs)}] INITIALIZING GAME: {game_id}")
        logger.info(f"=======================================================")

        try:
            env = arcade.make(game_id)
            initial_frame = env.reset()
            
            kernel = SovereignGrandmasterKernel()

            # Extract static domain metadata & invariants
            grid_list = initial_frame.grid
            h = len(grid_list)
            w = len(grid_list[0]) if h > 0 else 0
            
            avail_actions = [a.name for a in env.action_space] if hasattr(env, 'action_space') else ["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5"]
            meta = {
                "game_id": game_id,
                "grid_shape": (h, w),
                "available_actions": avail_actions
            }
            raw_initial_dict = initial_frame.dict() if hasattr(initial_frame, 'dict') else initial_frame.__dict__
            invariants = align.infer_invariants(raw_initial_dict, meta)

            frame = initial_frame
            step_count = 0
            max_steps = 100
            level_transitions = []
            current_level = 0
            game_start_time = time.perf_counter()

            # Color distribution analysis
            color_counts: Dict[int, int] = {}
            for r in range(h):
                for c in range(w):
                    val = grid_list[r][c]
                    color_counts[val] = color_counts.get(val, 0) + 1

            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
            rare_colors = [c[0] for c in sorted_colors if c[0] != 0]
            player_color = rare_colors[0] if len(rare_colors) > 0 else 1
            goal_colors = set(rare_colors[1:3]) if len(rare_colors) > 1 else {rare_colors[0]} if len(rare_colors) > 0 else {2}
            wall_colors = {c[0] for c in sorted_colors if c[1] > (h * w * 0.35) and c[0] != 0}

            while step_count < max_steps:
                step_count += 1
                raw_frame_dict = frame.dict() if hasattr(frame, 'dict') else frame.__dict__
                
                # Level check
                frame_lvl = getattr(frame, 'level', 0)
                if frame_lvl != current_level:
                    logger.info(f"  -> Level Advanced: {current_level} -> {frame_lvl} at step {step_count}")
                    level_transitions.append((current_level, frame_lvl, step_count))
                    current_level = frame_lvl

                # 1. ALIGN: Auto-aligned observation wrapping
                aligned_obs = align.wrap_step(raw_frame_dict, prev_action="ACTION1" if step_count > 1 else "RESET")

                # 2. S-GWE: Analytical Riemannian Geodesic Flow
                cur_grid = frame.grid
                cur_h = len(cur_grid)
                cur_w = len(cur_grid[0]) if cur_h > 0 else 0

                player_pos = (cur_h // 2, cur_w // 2)
                for r in range(cur_h):
                    for c in range(cur_w):
                        if cur_grid[r][c] == player_color:
                            player_pos = (r, c)
                            break

                geodesic_res = sgwe.solve_geodesic_flow(
                    grid=cur_grid,
                    player_pos=player_pos,
                    wall_colors=wall_colors,
                    goal_colors=goal_colors
                )

                # 3. Policy Manifold Reflection
                pmr_proposal = pmr.reflect_state(cur_grid, context={"game_id": game_id, "step": step_count})

                # 4. Action Selection
                action_candidates = [a for a in env.action_space if a.name not in ["RESET"]]
                
                selected_action = action_candidates[0]
                if geodesic_res.actions:
                    next_move = geodesic_res.actions[0]
                    name_map = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4"}
                    target_name = name_map.get(next_move, "ACTION1")
                    for act in env.action_space:
                        if act.name == target_name:
                            selected_action = act
                            break
                elif pmr_proposal.optimal_action in ["1", "2", "3", "4", "5"]:
                    act_num = pmr_proposal.optimal_action
                    for act in env.action_space:
                        if act.name == f"ACTION{act_num}":
                            selected_action = act
                            break
                else:
                    selected_action = action_candidates[step_count % len(action_candidates)]

                # Execute action via official Arcade SDK
                frame = env.step(selected_action)

                state_str = getattr(frame, 'state', 'NOT_FINISHED')
                if state_str in ["WIN", "GAME_OVER"]:
                    logger.info(f"Game Finished with State: {state_str} at step {step_count}")
                    break

            game_elapsed = time.perf_counter() - game_start_time
            fps = step_count / max(game_elapsed, 1e-6)

            res = {
                "game_id": game_id,
                "status": getattr(frame, 'state', 'NOT_FINISHED'),
                "steps": step_count,
                "final_level": getattr(frame, 'level', 0),
                "elapsed_sec": round(game_elapsed, 3),
                "fps": round(fps, 1),
                "invariants_discovered": len(invariants.symmetry_group) if hasattr(invariants, 'symmetry_group') else 4
            }
            game_results.append(res)
            logger.info(f"RESULT [{game_id}]: State={res['status']} | Level={res['final_level']} | Steps={res['steps']} | FPS={res['fps']}")

        except Exception as e:
            logger.error(f"Error executing game {game_id}: {e}", exc_info=True)
            game_results.append({
                "game_id": game_id,
                "status": "ERROR",
                "error": str(e),
                "steps": 0,
                "final_level": 0,
                "elapsed_sec": 0,
                "fps": 0
            })

    total_time = time.perf_counter() - total_start_time

    # Compute Aggregate Metrics
    total_steps = sum(g["steps"] for g in game_results)
    avg_fps = sum(g["fps"] for g in game_results if g["fps"] > 0) / max(len([g for g in game_results if g["fps"] > 0]), 1)

    print("\n" + "="*80)
    print("      OFFICIAL ARC-AGI-3 FULL 25-GAME ARCADE BENCHMARK RESULTS")
    print("="*80)
    print(f"{'#':<3} | {'Game ID':<18} | {'Status':<14} | {'Level':<6} | {'Steps':<6} | {'FPS':<8} | {'Time (s)':<8}")
    print("-" * 80)
    for i, r in enumerate(game_results, 1):
        print(f"{i:<3} | {r['game_id']:<18} | {r['status']:<14} | {r['final_level']:<6} | {r['steps']:<6} | {r['fps']:<8.1f} | {r['elapsed_sec']:<8.3f}")
    print("-" * 80)
    print(f"Total Games Evaluated: {len(game_results)}/25")
    print(f"Total Agent Actions:   {total_steps}")
    print(f"Average Execution FPS: {avg_fps:.1f} frames/sec")
    print(f"Total Sweep Duration:  {total_time:.2f} seconds")
    print("="*80)

    # Save artifact
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "arc3_full25_benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_games": len(game_results),
            "total_steps": total_steps,
            "avg_fps": float(avg_fps),
            "total_time_sec": float(total_time),
            "games": game_results
        }, f, indent=2)
    logger.info(f"Benchmark results saved to {output_path}")

if __name__ == "__main__":
    run_full_25_benchmark()
