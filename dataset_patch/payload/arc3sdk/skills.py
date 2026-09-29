"""Sovereign Grandmaster Cognitive Skills Matrix (SovereignSkills)

Specialized Grandmaster Cognitive Skills for ARC-AGI-3 (currently 11:
8 perception + goal-acquisition + lattice-vertical + FullAuto online runes):
1. SpatialRaycastNavigationSkill: multi-directional raycasts (L1) for walls,
   corridors, goal sockets.
2. TopologicalTunnelingSkill: 1-pixel bottlenecks / choke-points.
3. TetrisGravityPhysicsSkill: falling-block dynamics under gravity.
4. ChromaticParityTransformationSkill: chess-like parity lattices.
5. InvariantPatternExtrapolationSkill: horizontal periodicity continuation.
6. GeodesicTopologicalCentroidTargeter: object center-of-mass clicks.
7. DihedralGroupD4SymmetryCompleter: mirror-asymmetry repair clicks.
8. BoundaryEnclosedVoidInfill: enclosed-cavity clicks.
9. DynamicOnlineEvolvedSkills: FullAuto runtime rune learner (lazy).
10. GoalAcquisitionRarestProbe: unknown-unknown salient probe (rarest color).
11. LatticeVerticalContinuation: vertical periodicity continuation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import contextlib

with contextlib.suppress(Exception):
    pass


class SovereignSkillsMatrix:
    """Grandmaster Autonomous Skills Evaluator."""

    @staticmethod
    def evaluate_skills(
        grid: np.ndarray, bg: int, available: list[int]
    ) -> list[tuple[int, int | None, int | None, str, float]]:
        """Evaluate all 5 Grandmaster Skills on the current frame and return prioritized proposals."""
        g = np.asarray(grid, dtype=np.uint8)
        h, w = g.shape
        proposals: list[tuple[int, int | None, int | None, str, float]] = []

        # ----------------------------------------------------------------------
        # SKILL 1: Spatial Raycast & Corridor Navigation (Maze & Sokoban solver)
        # ----------------------------------------------------------------------
        fg = g != bg
        fg_idx = np.argwhere(fg)
        if len(fg_idx) > 0 and any(a in available for a in (1, 2, 3, 4)):
            cy, cx = np.mean(fg_idx, axis=0)
            cy, cx = int(round(cy)), int(round(cx))

            # Cast 4 cardinal rays from centroid
            # 1: Up, 2: Right, 3: Down, 4: Left
            ray_up = int(np.sum(fg[:cy, cx])) if cy > 0 else 0
            ray_down = int(np.sum(fg[cy + 1 :, cx])) if cy < h - 1 else 0
            ray_left = int(np.sum(fg[cy, :cx])) if cx > 0 else 0
            ray_right = int(np.sum(fg[cy, cx + 1 :])) if cx < w - 1 else 0

            # Target least congested or open corridor
            ray_scores = {
                1: ray_up,
                2: ray_right,
                3: ray_down,
                4: ray_left,
            }
            # Find open corridor direction
            open_dir = min(ray_scores.keys(), key=lambda d: ray_scores[d])
            if open_dir in available:
                proposals.append((open_dir, None, None, "skill_raycast_open_corridor", 0.86))

        # ----------------------------------------------------------------------
        # SKILL 2: Topological Medial Axis & Choke-Point Breaker (Narrow Passages)
        # ----------------------------------------------------------------------
        if 6 in available:
            # Detect narrow bottlenecks (points with exactly 2 opposing neighbors)
            for y in range(2, h - 2, 2):
                for x in range(2, w - 2, 2):
                    # Check horizontal or vertical choke-point
                    if (g[y, x] == bg and ((g[y - 1, x] != bg and g[y + 1, x] != bg and g[y, x - 1] == bg and g[y, x + 1] == bg) or (
                        g[y, x - 1] != bg and g[y, x + 1] != bg and g[y - 1, x] == bg and g[y + 1, x] == bg
                    ))):
                        proposals.append((6, x, y, "skill_topological_chokepoint_bridge", 0.91))
                        break

        # ----------------------------------------------------------------------
        # SKILL 3: Tetris Gravity & Grounding Alignment (Physics & Stacking)
        # ----------------------------------------------------------------------
        if 3 in available:  # Down action under gravity
            # If floating mass exists above empty floor
            bottom_fg = np.max(fg_idx[:, 0]) if len(fg_idx) > 0 else 0
            if bottom_fg < h - 4:
                proposals.append((3, None, None, "skill_tetris_gravity_descent", 0.88))

        # ----------------------------------------------------------------------
        # SKILL 4: Chromatic Parity & Checkerboard Lattice Alternation
        # ----------------------------------------------------------------------
        if 6 in available:
            # Check if grid exhibits parity structure (y + x) % 2
            colors = np.unique(g[fg]) if np.any(fg) else []
            if len(colors) >= 2:
                # Find parity discrepancy
                for y in range(4, h - 4, 4):
                    for x in range(4, w - 4, 4):
                        expected_parity = (y + x) % 2
                        if g[y, x] != bg and g[y, x] % 2 != expected_parity:
                            proposals.append((6, x, y, "skill_chromatic_parity_repair", 0.89))
                            break

        # ----------------------------------------------------------------------
        # SKILL 5: Invariant Periodicity & Arithmetic Pattern Extrapolation
        # ----------------------------------------------------------------------
        if 6 in available and len(fg_idx) >= 4:
            # Check periodic horizontal or vertical spacing
            xs_sorted = np.sort(np.unique(fg_idx[:, 1]))
            if len(xs_sorted) >= 3:
                diffs = np.diff(xs_sorted)
                if len(diffs) > 0 and np.std(diffs) < 1.0:
                    period = int(np.round(np.mean(diffs)))
                    next_x = int(xs_sorted[-1] + period)
                    if 0 <= next_x < w:
                        cy_m = int(round(np.mean(fg_idx[:, 0])))
                        proposals.append((6, next_x, cy_m, "skill_periodic_extrapolation", 0.93))

        # ----------------------------------------------------------------------
        # SKILL 6: Geodesic Topological Centroid Targeter (Sokoban / Core Anchor)
        # ----------------------------------------------------------------------
        if 6 in available and len(fg_idx) > 0:
            # Detect isolated distinct objects and target topological center of mass
            cy_m, cx_m = int(round(np.mean(fg_idx[:, 0]))), int(round(np.mean(fg_idx[:, 1])))
            if g[cy_m, cx_m] != bg:
                proposals.append((6, cx_m, cy_m, "skill_geodesic_centroid_targeter", 0.92))

        # ----------------------------------------------------------------------
        # SKILL 7: Bilateral & Dihedral Group D4 Symmetry Completer
        # ----------------------------------------------------------------------
        if 6 in available:
            # Check horizontal reflection asymmetry
            flipped_h = np.fliplr(g)
            asym_h = (g != flipped_h) & (g == bg) & (flipped_h != bg)
            asym_idx = np.argwhere(asym_h)
            if len(asym_idx) > 0:
                ay, ax = asym_idx[0]
                proposals.append((6, int(ax), int(ay), "skill_d4_bilateral_symmetry_repair", 0.94))

        # ----------------------------------------------------------------------
        # SKILL 8: Boundary Enclosed Void / Pocket Flood Infill
        # ----------------------------------------------------------------------
        if 6 in available:
            # Fast detection of 1-pixel enclosed empty cavities
            for y in range(1, h - 1, 2):
                for x in range(1, w - 1, 2):
                    if g[y, x] == bg:
                        neighbors = [g[y - 1, x], g[y + 1, x], g[y, x - 1], g[y, x + 1]]
                        if all(n != bg for n in neighbors):
                            proposals.append((6, x, y, "skill_enclosed_void_infill", 0.95))
                            break

        # ----------------------------------------------------------------------
        # SKILL 10: Goal-Acquisition Rarest-Color Probe (unknown unknowns)
        # ARC-AGI-3 never names the goal: the rarest foreground color is the
        # most salient unexplored affordance. Deterministic centroid click.
        # ----------------------------------------------------------------------
        if 6 in available and len(fg_idx) > 0:
            try:
                vals, counts = np.unique(g[fg], return_counts=True)
                order = np.argsort(counts, kind="stable")
                for idx in order:
                    c = int(vals[idx])
                    if c == bg:
                        continue
                    ys, xs = np.where(g == c)
                    if len(xs):
                        proposals.append((6, int(xs[len(xs) // 2]),
                                          int(ys[len(ys) // 2]),
                                          "skill_goal_rarest_probe", 0.77))
                        break
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # SKILL 11: Lattice Vertical Continuation (periodic vertical waves)
        # Vertical complement of SKILL 5: evenly spaced rows continue downward.
        # ----------------------------------------------------------------------
        if 6 in available and len(fg_idx) >= 4:
            try:
                ys_sorted = np.sort(np.unique(fg_idx[:, 0]))
                if len(ys_sorted) >= 3:
                    ydiffs = np.diff(ys_sorted)
                    if len(ydiffs) > 0 and float(np.std(ydiffs)) < 1.0:
                        yperiod = int(round(float(np.mean(ydiffs))))
                        next_y = int(ys_sorted[-1] + yperiod)
                        if 0 <= next_y < h and yperiod > 0:
                            cx_v = int(round(float(np.mean(fg_idx[:, 1]))))
                            proposals.append((6, cx_v, next_y,
                                              "skill_lattice_vertical", 0.90))
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # SKILL 9: Dynamic Online Evolved Skills (FullAuto Runtime Learner)
        # ----------------------------------------------------------------------
        try:
            from .rune import AutonomousOnlineRuneSynthesizer

            auto_runes = AutonomousOnlineRuneSynthesizer.get_instance().evaluate_live_runes(g, "unseen", available)
            for a_act, ax, ay, a_src, a_conf in auto_runes:
                proposals.append((a_act, ax, ay, f"fullauto_skill_{a_src}", a_conf * 1.05))
        except Exception:
            pass

        proposals.sort(key=lambda p: -p[4])
        return proposals

    @classmethod
    def solve_with_skills(
        cls, grid: np.ndarray, bg: int, available: list[int]
    ) -> tuple[int | None, int | None, int | None, str]:
        """Returns the highest-confidence Grandmaster Skill recommendation."""
        proposals = cls.evaluate_skills(grid, bg, available)
        if proposals:
            act, x, y, src, conf = proposals[0]
            return act, x, y, src
        return None, None, None, ""

    @staticmethod
    def _eval_python_repl(code_str: str, state_vars: dict[str, Any], timeout_sec: int = 5) -> Any:
        """
        Dynamically execute Python code within a restricted namespace.
        Matches the SOTA strategy (e.g. 'The Duck') of REPL-based hypothesis testing.
        """
        import io
        import contextlib
        import multiprocessing

        def run_code(code, vars_dict, result_dict):
            # Create a restricted namespace
            local_env = dict(vars_dict)
            local_env["__builtins__"] = {
                "print": print,
                "range": range,
                "len": len,
                "list": list,
                "dict": dict,
                "set": set,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "int": int,
                "float": float,
                "bool": bool,
                "str": str,
                "tuple": tuple,
                "enumerate": enumerate,
                "zip": zip,
                "Exception": Exception,
            }
            local_env["np"] = np

            output_capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_capture):
                    exec(code, local_env, local_env)
                result_dict["status"] = "success"
                # Extract potential action proposal variables if set
                result_dict["act"] = local_env.get("proposed_action")
                result_dict["x"] = local_env.get("proposed_x")
                result_dict["y"] = local_env.get("proposed_y")
                result_dict["stdout"] = output_capture.getvalue()
            except Exception as e:
                result_dict["status"] = "error"
                result_dict["stdout"] = output_capture.getvalue()
                result_dict["error_msg"] = str(e)

        manager = multiprocessing.Manager()
        result = manager.dict()
        p = multiprocessing.Process(target=run_code, args=(code_str, state_vars, result))
        p.start()
        p.join(timeout_sec)

        if p.is_alive():
            p.terminate()
            p.join()
            return {
                "status": "timeout",
                "act": None,
                "x": None,
                "y": None,
                "stdout": "",
                "error_msg": "Timeout reached",
            }

        return dict(result)
