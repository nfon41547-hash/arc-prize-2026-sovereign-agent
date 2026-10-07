"""
Sovereign Hyper-Cortex Full 25-Game Online Live Arcade Benchmark.
Executes official ARC-AGI-3 environments via the live ARC Prize Online API.
Integrates:
- Symplectic Geodesic Wavefront Engine (S-GWE)
- ALIGN Auto-Aligned Interface Engine (Rules + Step Wrapping)
- Policy Manifold Reflection Engine (H-PMR Hamiltonian Potential)
- Sovereign Grandmaster Consensus Kernel

Outputs: Verified Official Scorecard ID and Live 25-Game Table.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Set, Tuple

import numpy as np
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

# Ensure SDK path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc3sdk.symplectic_geodesic_engine import SymplecticGeodesicWavefrontEngine
from arc3sdk.auto_aligned_interface import AutoAlignedInterfaceEngine
from arc3sdk.policy_manifold_reflection import PolicyManifoldReflectionEngine
from arc3sdk.unified_consensus_engine import SovereignGrandmasterKernel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ARC3OnlineFull25")

def get_grid_from_obs(obs: Any) -> List[List[int]]:
    if hasattr(obs, 'frame') and obs.frame:
        raw_frame = obs.frame[0] if isinstance(obs.frame, list) else obs.frame
        if hasattr(raw_frame, 'tolist'):
            return raw_frame.tolist()
        elif isinstance(raw_frame, list):
            return raw_frame
    return [[0]*64 for _ in range(64)]

def run_online_full_25():
    logger.info("Connecting to ARC-AGI-3 Live Online API...")
    arc = Arcade(operation_mode=OperationMode.ONLINE)
    envs = arc.get_environments()
    logger.info(f"Retrieved {len(envs)} live online environments from ARC Prize Foundation.")

    sgwe = SymplecticGeodesicWavefrontEngine()
    align = AutoAlignedInterfaceEngine()
    pmr = PolicyManifoldReflectionEngine()

    game_results = []
    total_start_time = time.perf_counter()

    for idx, env_info in enumerate(envs, start=1):
        game_slug = env_info.game_id.split("-")[0] if "-" in env_info.game_id else env_info.game_id
        full_game_id = env_info.game_id
        
        logger.info(f"\n" + "="*70)
        logger.info(f"[{idx}/{len(envs)}] ONLINE SESSION START: {full_game_id} (slug: {game_slug})")
        logger.info("="*70)

        try:
            env = arc.make(game_slug)
            obs = env.reset()
            kernel = SovereignGrandmasterKernel()

            grid_list = get_grid_from_obs(obs)
            h = len(grid_list)
            w = len(grid_list[0]) if h > 0 else 0

            # 1. ALIGN: Infer static rules
            static_rules = align.infer_rules(grid_list)

            # Available actions
            avail_action_ints = obs.available_actions if hasattr(obs, 'available_actions') else [1, 2, 3, 4]
            avail_action_names = [f"ACTION{a}" for a in avail_action_ints]

            step_count = 0
            max_steps = 25  # Smooth, fast online execution within 600 RPM limit
            current_levels_completed = getattr(obs, 'levels_completed', 0)
            game_start_time = time.perf_counter()

            # Dynamic color parsing
            player_color = list(static_rules.foreground_colors)[0] if static_rules.foreground_colors else 1
            goal_colors = set(list(static_rules.foreground_colors)[1:3]) if len(static_rules.foreground_colors) > 1 else {player_color}
            wall_colors = {static_rules.background_color} if static_rules.background_color != 0 else set()

            final_state = getattr(obs, 'state', 'NOT_FINISHED')
            prev_grid = grid_list
            prev_action_str = "RESET"

            while step_count < max_steps:
                step_count += 1

                # Check level progression
                completed_now = getattr(obs, 'levels_completed', 0)
                if completed_now != current_levels_completed:
                    logger.info(f"  [!] LEVEL ADVANCED: {current_levels_completed} -> {completed_now} at step {step_count}")
                    current_levels_completed = completed_now

                cur_grid = get_grid_from_obs(obs)
                cur_h = len(cur_grid)
                cur_w = len(cur_grid[0]) if cur_h > 0 else 0

                # 1. ALIGN: Wrap observation
                feedback = align.wrap_step(
                    prev_grid=prev_grid,
                    action=prev_action_str,
                    next_grid=cur_grid,
                    reward=1.0 if completed_now > 0 else 0.0,
                    done=(final_state in ["WIN", "GAME_OVER"])
                )

                # 2. S-GWE: Analytical Riemannian Geodesic Flow
                player_pos = feedback.player_coord or (cur_h // 2, cur_w // 2)
                geodesic_res = sgwe.solve_geodesic_flow(
                    grid=cur_grid,
                    player_pos=player_pos,
                    wall_colors=wall_colors,
                    goal_colors=goal_colors
                )

                # 3. Policy Manifold Reflection (Hamiltonian Action Selection)
                belief = pmr.compute_belief_manifold(cur_grid, player_color=player_color)
                pmr_best_action = pmr.select_optimal_policy_action(avail_action_names, belief)

                # 4. Action Mapping
                action_to_send = GameAction.ACTION1
                chosen_name = "ACTION1"

                if geodesic_res.actions and not feedback.is_collision:
                    move_str = geodesic_res.actions[0]
                    act_map = {
                        "UP": (GameAction.ACTION1, "ACTION1"),
                        "DOWN": (GameAction.ACTION2, "ACTION2"),
                        "LEFT": (GameAction.ACTION3, "ACTION3"),
                        "RIGHT": (GameAction.ACTION4, "ACTION4")
                    }
                    action_to_send, chosen_name = act_map.get(move_str, (GameAction.ACTION1, "ACTION1"))
                elif pmr_best_action:
                    chosen_name = pmr_best_action
                    act_num = pmr_best_action.replace("ACTION", "")
                    if act_num == "1": action_to_send = GameAction.ACTION1
                    elif act_num == "2": action_to_send = GameAction.ACTION2
                    elif act_num == "3": action_to_send = GameAction.ACTION3
                    elif act_num == "4": action_to_send = GameAction.ACTION4
                    elif act_num == "5": action_to_send = GameAction.ACTION5
                else:
                    action_pool = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
                    action_to_send = action_pool[step_count % len(action_pool)]
                    chosen_name = f"ACTION{(step_count % len(action_pool)) + 1}"

                # Step the Online Environment
                prev_grid = cur_grid
                prev_action_str = chosen_name

                obs = env.step(action_to_send)
                final_state = getattr(obs, 'state', 'NOT_FINISHED')

                if final_state in ["WIN", "GAME_OVER"]:
                    logger.info(f"Online Game Terminated: State={final_state} at step {step_count}")
                    break

            game_elapsed = time.perf_counter() - game_start_time
            fps = step_count / max(game_elapsed, 1e-6)

            res = {
                "game_id": full_game_id,
                "guid": getattr(obs, "guid", ""),
                "status": final_state,
                "levels_completed": getattr(obs, 'levels_completed', 0),
                "win_levels": getattr(obs, 'win_levels', 0),
                "steps": step_count,
                "elapsed_sec": round(game_elapsed, 3),
                "fps": round(fps, 1)
            }
            game_results.append(res)
            logger.info(f"RESULT [{full_game_id}]: State={res['status']} | Levels={res['levels_completed']}/{res['win_levels']} | Steps={res['steps']} | FPS={res['fps']}")

        except Exception as e:
            logger.error(f"Error on online game {full_game_id}: {e}", exc_info=True)
            game_results.append({
                "game_id": full_game_id,
                "guid": "",
                "status": "ERROR",
                "levels_completed": 0,
                "win_levels": 0,
                "steps": 0,
                "elapsed_sec": 0,
                "fps": 0,
                "error": str(e)
            })

    total_duration = time.perf_counter() - total_start_time

    # Fetch Official Scorecard from Arcade API
    official_scorecard = None
    try:
        official_scorecard = arc.get_scorecard()
        logger.info(f"Successfully retrieved official Scorecard: {official_scorecard}")
    except Exception as e:
        logger.warning(f"Scorecard fetch notice: {e}")

    # Summary table
    print("\n" + "="*86)
    print("       OFFICIAL ARC-AGI-3 ONLINE 25-GAME LIVE ARCADE BENCHMARK RESULTS")
    print("="*86)
    print(f"{'#':<3} | {'Game ID':<18} | {'Status':<14} | {'Levels':<8} | {'Steps':<6} | {'FPS':<7} | {'Time (s)':<8}")
    print("-" * 86)
    for i, r in enumerate(game_results, 1):
        levels_str = f"{r['levels_completed']}/{r['win_levels']}"
        print(f"{i:<3} | {r['game_id']:<18} | {r['status']:<14} | {levels_str:<8} | {r['steps']:<6} | {r['fps']:<7.1f} | {r['elapsed_sec']:<8.3f}")
    print("-" * 86)
    print(f"Total Games Played Online: {len(game_results)}/25")
    print(f"Total Live Agent Steps:    {sum(r['steps'] for r in game_results)}")
    print(f"Total Execution Duration:  {total_duration:.2f}s")
    if official_scorecard:
        print(f"Official Scorecard:        {official_scorecard}")
    print("="*86)

    # Save artifact
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "arc3_online_full25_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "ONLINE",
            "scorecard": str(official_scorecard),
            "total_games": len(game_results),
            "total_steps": sum(r['steps'] for r in game_results),
            "total_duration_sec": total_duration,
            "games": game_results
        }, f, indent=2)
    logger.info(f"Online results persisted to {output_path}")

if __name__ == "__main__":
    run_online_full_25()
