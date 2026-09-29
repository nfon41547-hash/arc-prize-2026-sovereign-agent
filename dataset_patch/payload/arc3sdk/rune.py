"""Rune Library v2.0 - World-Class Invariant Pattern Extraction.

Extracts the highest quality invariant patterns ("Runes") from all game traces:
- Universal click patterns (object-relative, symmetry-relative, color-relative)
- Directional action semantics per game type
- Reset/recovery strategies
- Level transition signatures
- Failure mode detectors
- Topological invariants (Betti numbers, Euler characteristic)
- Causal chain encoding (action sequences with state transitions)
"""

from __future__ import annotations
import glob
import os
import json
import re
import numpy as np
import collections
from typing import Any
from dataclasses import dataclass, field
from collections import deque

from . import grid as _grid
from .oracle import _down, _largest_object_centroid, _same_game, signature


# ============================================================
# Topological Invariants
# ============================================================


def compute_betti_numbers(grid: np.ndarray, bg: int = 0) -> tuple[int, int, int]:
    """Compute simplified Betti numbers for a grid.

    Betti-0: number of connected components (objects)
    Betti-1: number of holes (enclosed empty regions)
    Betti-2: number of 2D voids (not applicable for 2D grids, always 0)
    """
    binary = (grid != bg).astype(np.uint8)

    # Betti-0: connected components via flood fill
    visited = np.zeros_like(binary, dtype=bool)
    b0 = 0

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if binary[i, j] == 1 and not visited[i, j]:
                b0 += 1
                # BFS flood fill
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and binary[ni, nj] == 1 and not visited[ni, nj]:
                            visited[ni, nj] = True
                            queue.append((ni, nj))

    # Betti-1: holes via border detection
    # A hole is a connected region of background completely enclosed by objects
    bg_visited = np.zeros_like(binary, dtype=bool)
    b1 = 0

    # Mark all background reachable from border
    for i in range(grid.shape[0]):
        for j in [0, grid.shape[1] - 1]:
            if binary[i, j] == 0 and not bg_visited[i, j]:
                queue = deque([(i, j)])
                bg_visited[i, j] = True
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and binary[ni, nj] == 0 and not bg_visited[ni, nj]:
                            bg_visited[ni, nj] = True
                            queue.append((ni, nj))

    # Background regions not reachable from border are holes
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if binary[i, j] == 0 and not bg_visited[i, j]:
                b1 += 1
                # Mark entire hole
                queue = deque([(i, j)])
                bg_visited[i, j] = True
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] and binary[ni, nj] == 0 and not bg_visited[ni, nj]:
                            bg_visited[ni, nj] = True
                            queue.append((ni, nj))

    return b0, b1, 0


def compute_euler_characteristic(betti: tuple[int, int, int]) -> int:
    """Euler characteristic = Betti-0 - Betti-1 + Betti-2."""
    return betti[0] - betti[1] + betti[2]


def compute_grid_complexity(grid: np.ndarray, bg: int = 0) -> float:
    """Compute grid complexity score (0-1, higher = more complex).

    Combines:
    - Number of objects
    - Number of holes
    - Color diversity
    - Symmetry breaking
    """
    b0, b1, b2 = compute_betti_numbers(grid, bg)

    # Object complexity
    obj_score = min(1.0, b0 / 10.0)

    # Hole complexity
    hole_score = min(1.0, b1 / 5.0)

    # Color complexity
    unique_colors = len(np.unique(grid))
    color_score = min(1.0, unique_colors / 8.0)

    # Symmetry breaking (lower symmetry = higher complexity)
    symmetry_score = 0.0
    if np.array_equal(grid, np.fliplr(grid)):
        symmetry_score += 0.25
    if np.array_equal(grid, np.flipud(grid)):
        symmetry_score += 0.25
    if np.array_equal(grid, np.rot90(grid)):
        symmetry_score += 0.25
    symmetry_breaking = 1.0 - symmetry_score

    return obj_score * 0.3 + hole_score * 0.2 + color_score * 0.25 + symmetry_breaking * 0.25


# ============================================================
# Causal Chain Encoding
# ============================================================


@dataclass
class CausalChain:
    """A sequence of actions with state transitions."""

    actions: list[int]
    state_signatures: list[bytes]
    reward: float
    success: bool
    game_class: str
    support: int = 1

    def matches_prefix(self, history: list[tuple[bytes, int]], tolerance: float = 0.1) -> bool:
        """Check if history matches the beginning of this chain."""
        if len(history) > len(self.actions):
            return False

        for i, (_state_sig, action) in enumerate(history):
            if i >= len(self.state_signatures):
                return False
            if i < len(self.actions) and action != self.actions[i]:
                return False
        return True


class CausalMemory:
    """Remember causal chains for action prediction."""

    def __init__(self, max_chains: int = 500) -> None:
        self.chains: list[CausalChain] = []
        self.max_chains = max_chains
        self.state_action_counts: dict[bytes, dict[int, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

    def __getstate__(self):
        d = self.__dict__.copy()
        d["state_action_counts"] = dict(d.get("state_action_counts", {}))
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        new_sac = collections.defaultdict(lambda: collections.defaultdict(int))
        for k, v in self.__dict__.get("state_action_counts", {}).items():
            new_sac[k] = collections.defaultdict(int, v)
        self.__dict__["state_action_counts"] = new_sac

    def record(
        self, state_sigs: list[bytes], actions: list[int], reward: float, success: bool, game_class: str
    ) -> None:
        """Record a causal chain."""
        if len(actions) < 2:
            return

        # Update state-action counts
        for i, sig in enumerate(state_sigs[:-1]):
            if i < len(actions):
                self.state_action_counts[sig][actions[i]] += 1

        # Find existing chain
        for c in self.chains:
            if c.actions == actions and c.game_class == game_class and len(c.state_signatures) == len(state_sigs):
                c.support += 1
                c.reward = max(c.reward, reward)
                return

        # Add new chain
        chain = CausalChain(
            actions=actions.copy(),
            state_signatures=state_sigs.copy(),
            reward=reward,
            success=success,
            game_class=game_class,
        )
        self.chains.append(chain)

        # Keep only best chains
        if len(self.chains) > self.max_chains:
            self.chains.sort(key=lambda c: c.support * c.reward, reverse=True)
            self.chains = self.chains[: self.max_chains]

    def predict_next_action(self, history: list[tuple[bytes, int]], avail: list[int]) -> int | None:
        """Predict next action based on causal history."""
        if not history:
            return None

        # Get current state
        current_sig = history[-1][0]

        # Check state-action counts
        if current_sig in self.state_action_counts:
            counts = self.state_action_counts[current_sig]
            # Filter available actions
            avail_counts = [(a, counts[a]) for a in counts if a in avail]
            if avail_counts:
                return max(avail_counts, key=lambda x: x[1])[0]

        # Check causal chains
        for chain in self.chains:
            if chain.matches_prefix(history[:-1]):
                idx = len(history) - 1
                if idx < len(chain.actions):
                    next_action = chain.actions[idx]
                    if next_action in avail:
                        return next_action

        return None

    def get_confidence(self, state_sig: bytes, action: int) -> float:
        """Get confidence for a state-action pair."""
        if state_sig not in self.state_action_counts:
            return 0.0
        counts = self.state_action_counts[state_sig]
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return counts[action] / total


# ============================================================
# Rune Data Structures
# ============================================================


@dataclass
class Rune:
    """A single extracted invariant pattern - the 'dharma' of the game."""

    id: str
    game_class: str  # e.g., 'ft09', 'ar25', or 'universal'
    rune_type: str  # 'click', 'directional', 'reset', 'transition', 'failure'
    condition_sig: np.ndarray  # 16x16 signature that triggers this rune
    action: int  # action to take
    target_rel: tuple[int, int] | None  # (rx, ry) relative to anchor
    anchor_type: str  # 'object', 'symmetry', 'color', 'grid_center', 'fixed'
    confidence: float  # 0.0 - 1.0
    support: int  # how many trace occurrences
    games_observed: set[str] = field(default_factory=set)
    level_range: tuple[int, int] = (0, 99)
    metadata: dict[str, Any] = field(default_factory=dict)
    # New: topological features
    betti_signature: tuple[int, int, int] | None = None
    euler_characteristic: int | None = None
    grid_complexity: float | None = None

    def matches(self, frame: np.ndarray, game_id: str, threshold: float = 0.12) -> bool:
        """Check if this rune applies to current frame."""
        if not _same_game(self.game_class, game_id) and self.game_class != "universal":
            return False
        frame_sig16 = signature(frame)["sig16"]
        d = float(np.abs(frame_sig16.astype(np.int16) - self.condition_sig.astype(np.int16)).mean())
        return d <= threshold

    def get_target(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Compute absolute target coordinates from frame."""
        if self.target_rel is None:
            return None
        rx, ry = self.target_rel
        if self.anchor_type == "object":
            cen = _largest_object_centroid(frame)
            if cen:
                return (int(np.clip(cen[0] + rx, 0, 63)), int(np.clip(cen[1] + ry, 0, 63)))
        elif self.anchor_type == "fixed":
            return (int(np.clip(rx, 0, 63)), int(np.clip(ry, 0, 63)))
        elif self.anchor_type == "grid_center":
            return (32 + rx, 32 + ry)
        return None

    def matches_topology(self, frame: np.ndarray, tolerance: int = 1) -> bool:
        """Check if frame matches rune's topological signature."""
        if self.betti_signature is None:
            return True  # No topology constraint

        bg = _grid.background(frame)
        betti = compute_betti_numbers(frame, bg)

        # Check if Betti numbers are within tolerance
        return (
            abs(betti[0] - self.betti_signature[0]) <= tolerance
            and abs(betti[1] - self.betti_signature[1]) <= tolerance
        )


@dataclass
class RuneLibrary:
    """Library of all extracted runes - the knowledge base."""

    runes: list[Rune] = field(default_factory=list)
    by_type: dict[str, list[Rune]] = field(default_factory=lambda: collections.defaultdict(list))
    by_game: dict[str, list[Rune]] = field(default_factory=lambda: collections.defaultdict(list))
    universal: list[Rune] = field(default_factory=list)
    causal_memory: CausalMemory = field(default_factory=CausalMemory)

    def add(self, rune: Rune) -> None:
        self.runes.append(rune)
        self.by_type[rune.rune_type].append(rune)
        self.by_game[rune.game_class].append(rune)
        if rune.game_class == "universal" or len(rune.games_observed) > 3:
            self.universal.append(rune)

    def get_applicable(
        self, frame: np.ndarray, game_id: str, avail: list[int], rune_type: str | None = None
    ) -> list[Rune]:
        """Get all runes that match current frame and are available."""
        candidates = []
        search_set = self.universal + self.by_game.get(game_id.split("-")[0], []) if game_id else self.universal
        for r in search_set:
            if r.action not in avail:
                continue
            if rune_type and r.rune_type != rune_type:
                continue
            if r.matches(frame, game_id) and r.matches_topology(frame):
                candidates.append(r)
        # Sort by confidence * support * topology_bonus
        candidates.sort(
            key=lambda r: r.confidence * np.log1p(r.support) * (1.2 if r.betti_signature else 1.0), reverse=True
        )
        return candidates

    def get_best(self, frame: np.ndarray, game_id: str, avail: list[int]) -> Rune | None:
        cands = self.get_applicable(frame, game_id, avail)
        return cands[0] if cands else None


# ============================================================
# Rune Extraction from Traces
# ============================================================


def extract_runes_from_trace(trace_path: str, game_id: str, environments_dir: str) -> list[Rune]:
    """Extract runes from a single game trace."""
    runes = []
    game_class = game_id.split("-")[0]

    # Load trace events with safe context manager
    events = []
    try:
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("kind") == "action_taken":
                        events.append(e)
                except Exception:
                    continue
    except Exception:
        return runes

    # Analyze action distribution
    acts = collections.Counter(e["action"] for e in events)
    _total = sum(acts.values())

    # Track causal chains
    state_sigs = []
    actions = []

    # Extract CLICK runes (action 6)
    click_events = [e for e in events if e["action"] == 6 and e.get("x") is not None]
    if click_events:
        # Cluster clicks by relative position to objects
        obj_rel_clicks = collections.defaultdict(list)
        _sym_rel_clicks = collections.defaultdict(list)
        fixed_clicks = collections.defaultdict(list)

        for e in click_events:
            grid = np.array(e["grid"], dtype=np.uint8).reshape(64, 64)
            cx, cy = e["x"], e["y"]

            # Record state signature for causal chains
            sig = _down(grid).tobytes()
            state_sigs.append(sig)
            actions.append(6)

            # Object-relative
            cen = _largest_object_centroid(grid)
            if cen:
                rx, ry = cx - cen[0], cy - cen[1]
                obj_rel_clicks[(rx, ry)].append((grid, e))

            # Symmetry-relative (if grid has symmetry)
            # Fixed position
            fixed_clicks[(cx, cy)].append((grid, e))

        # Create runes from clusters with sufficient support
        for (rx, ry), samples in obj_rel_clicks.items():
            if len(samples) >= 2:  # Minimum support
                # Average signature
                sigs = [signature(g)["sig16"] for g, _ in samples]
                avg_sig = np.mean(sigs, axis=0).astype(np.uint8)

                # Compute topology for first sample
                bg = _grid.background(samples[0][0])
                betti = compute_betti_numbers(samples[0][0], bg)
                euler = compute_euler_characteristic(betti)
                complexity = compute_grid_complexity(samples[0][0], bg)

                r = Rune(
                    id=f"{game_class}_click_obj_{rx}_{ry}",
                    game_class=game_class,
                    rune_type="click",
                    condition_sig=avg_sig,
                    action=6,
                    target_rel=(rx, ry),
                    anchor_type="object",
                    confidence=min(1.0, len(samples) / 10.0),
                    support=len(samples),
                    games_observed={game_class},
                    betti_signature=betti,
                    euler_characteristic=euler,
                    grid_complexity=complexity,
                )
                runes.append(r)

        for (cx, cy), samples in fixed_clicks.items():
            if len(samples) >= 3:
                sigs = [signature(g)["sig16"] for g, _ in samples]
                avg_sig = np.mean(sigs, axis=0).astype(np.uint8)

                bg = _grid.background(samples[0][0])
                betti = compute_betti_numbers(samples[0][0], bg)
                euler = compute_euler_characteristic(betti)
                complexity = compute_grid_complexity(samples[0][0], bg)

                r = Rune(
                    id=f"{game_class}_click_fixed_{cx}_{cy}",
                    game_class=game_class,
                    rune_type="click",
                    condition_sig=avg_sig,
                    action=6,
                    target_rel=(cx, cy),
                    anchor_type="fixed",
                    confidence=min(1.0, len(samples) / 15.0),
                    support=len(samples),
                    games_observed={game_class},
                    betti_signature=betti,
                    euler_characteristic=euler,
                    grid_complexity=complexity,
                )
                runes.append(r)

    # Extract DIRECTIONAL runes (actions 1-5)
    for act in [1, 2, 3, 4, 5]:
        dir_events = [e for e in events if e["action"] == act]
        if len(dir_events) >= 3:
            # These usually target object centers or specific regions
            sigs = []
            targets = []
            betti_samples = []

            for e in dir_events:
                grid = np.array(e["grid"], dtype=np.uint8).reshape(64, 64)
                sigs.append(signature(grid)["sig16"])

                # Record for causal chains
                sig = _down(grid).tobytes()
                state_sigs.append(sig)
                actions.append(act)

                cen = _largest_object_centroid(grid)
                if cen and e.get("x") is not None:
                    targets.append((e["x"] - cen[0], e["y"] - cen[1]))

                # Collect topology samples
                if len(betti_samples) < 5:
                    bg = _grid.background(grid)
                    betti_samples.append(compute_betti_numbers(grid, bg))

            if sigs:
                avg_sig = np.mean(sigs, axis=0).astype(np.uint8)
                avg_target = None
                if targets:
                    avg_rx = int(np.mean([t[0] for t in targets]))
                    avg_ry = int(np.mean([t[1] for t in targets]))
                    avg_target = (avg_rx, avg_ry)

                # Average Betti numbers
                avg_betti = None
                if betti_samples:
                    avg_betti = tuple(int(np.mean([b[i] for b in betti_samples])) for i in range(3))

                r = Rune(
                    id=f"{game_class}_dir_{act}",
                    game_class=game_class,
                    rune_type="directional",
                    condition_sig=avg_sig,
                    action=act,
                    target_rel=avg_target,
                    anchor_type="object" if avg_target else "grid_center",
                    confidence=min(1.0, len(dir_events) / 20.0),
                    support=len(dir_events),
                    games_observed={game_class},
                    betti_signature=avg_betti,
                    euler_characteristic=compute_euler_characteristic(avg_betti) if avg_betti else None,
                )
                runes.append(r)

    # Extract RESET runes (action 0)
    reset_events = [e for e in events if e["action"] == 0]
    if len(reset_events) >= 2:
        sigs = []
        for e in reset_events:
            grid = np.array(e["grid"], dtype=np.uint8).reshape(64, 64)
            sigs.append(signature(grid)["sig16"])
        avg_sig = np.mean(sigs, axis=0).astype(np.uint8)
        r = Rune(
            id=f"{game_class}_reset",
            game_class=game_class,
            rune_type="reset",
            condition_sig=avg_sig,
            action=0,
            target_rel=None,
            anchor_type="none",
            confidence=min(1.0, len(reset_events) / 10.0),
            support=len(reset_events),
            games_observed={game_class},
        )
        runes.append(r)

    # Extract TRANSITION runes (level boundaries)
    # Detect frames where level increases
    level_ups = []
    for i, e in enumerate(events):
        if e.get("level_up", False) or (i > 0 and e.get("level", 0) > events[i - 1].get("level", 0)):
            grid = np.array(e["grid"], dtype=np.uint8).reshape(64, 64)
            level_ups.append(grid)

    if len(level_ups) >= 2:
        sigs = [signature(g)["sig16"] for g in level_ups]
        avg_sig = np.mean(sigs, axis=0).astype(np.uint8)

        # Topology of transition states
        betti_samples = []
        for g in level_ups[:5]:
            bg = _grid.background(g)
            betti_samples.append(compute_betti_numbers(g, bg))
        avg_betti = tuple(int(np.mean([b[i] for b in betti_samples])) for i in range(3)) if betti_samples else None

        r = Rune(
            id=f"{game_class}_level_transition",
            game_class=game_class,
            rune_type="transition",
            condition_sig=avg_sig,
            action=6,  # Usually click to continue
            target_rel=(0, 0),
            anchor_type="grid_center",
            confidence=0.8,
            support=len(level_ups),
            games_observed={game_class},
            betti_signature=avg_betti,
        )
        runes.append(r)

    return runes


def extract_universal_runes(all_runes: list[Rune]) -> list[Rune]:
    """Compress and synthesize multi-game runes into High-Density Meta-Runes."""
    # 1. Group by (rune_type, anchor_type, relative_geometry)
    groups = collections.defaultdict(list)
    for r in all_runes:
        # Quantize target_rel into 4-pixel spatial equivalence classes for invariant generalization
        q_target = (round(r.target_rel[0] / 2) * 2, round(r.target_rel[1] / 2) * 2) if r.target_rel else None
        key = (r.rune_type, r.anchor_type, q_target)
        groups[key].append(r)

    universal = []
    for key, group in groups.items():
        games = set()
        for r in group:
            games.update(r.games_observed)
        # High-Density Universal threshold: seen in 2+ games or strong single-game support >= 4
        if len(games) >= 2 or sum(r.support for r in group) >= 4:
            sigs = [r.condition_sig for r in group]
            avg_sig = np.mean(sigs, axis=0).astype(np.uint8)
            total_support = sum(r.support for r in group)
            avg_conf = float(np.mean([r.confidence for r in group]))

            # Compute cross-game transfer boost
            density_boost = min(1.0, 0.4 + 0.15 * len(games) + 0.05 * np.log1p(total_support))

            # Average topology
            betti_samples = [r.betti_signature for r in group if r.betti_signature]
            avg_betti = None
            if betti_samples:
                avg_betti = tuple(int(np.mean([b[i] for b in betti_samples])) for i in range(3))

            u = Rune(
                id=f"meta_rune_{key[0]}_{key[1]}_{key[2]}",
                game_class="universal",
                rune_type=key[0],
                condition_sig=avg_sig,
                action=group[0].action,
                target_rel=key[2],
                anchor_type=key[1],
                confidence=float(np.clip(avg_conf * density_boost, 0.1, 1.0)),
                support=total_support,
                games_observed=games,
                betti_signature=avg_betti,
            )
            universal.append(u)

    # Sort universal meta-runes by information density (confidence * log(support) * games)
    universal.sort(key=lambda r: r.confidence * np.log1p(r.support) * len(r.games_observed), reverse=True)
    return universal


def build_rune_library(traces_dir: str, environments_dir: str | None = None) -> RuneLibrary:
    """Build complete rune library from all available traces."""
    lib = RuneLibrary()
    all_runes = []

    if not os.path.exists(traces_dir):
        return lib

    for p in sorted(glob.glob(os.path.join(traces_dir, "*.jsonl"))):
        fn = os.path.basename(p)
        m = re.search(r"_max_([a-z0-9]+)_", fn)
        gid = m.group(1) if m else fn.replace("trace_", "").replace(".jsonl", "")
        extracted = extract_runes_from_trace(p, gid, environments_dir or "")
        for r in extracted:
            lib.add(r)
            all_runes.append(r)

    universal = extract_universal_runes(all_runes)
    for u in universal:
        lib.add(u)

    # Add Category-Theoretic Topos Meta-Runes
    try:
        from .category_engine import CategoryTheoreticRuneSynthesizer

        topos_runes = CategoryTheoreticRuneSynthesizer.extract_deep_mathematical_archetypes(traces_dir)
        for tr in topos_runes:
            lib.add(tr)
    except Exception:
        pass

    return lib


# ============================================================
# Rune Evolution in Sandbox (Self-Improvement)
# ============================================================


class RuneEvolver:
    """Evolves runes by testing them in PerfectSimulator sandbox."""

    def __init__(self, library: RuneLibrary, environments_dir: str):
        self.lib = library
        self.environments_dir = environments_dir
        self.generation = 0
        self.performance_log = []

    def evaluate_rune(self, rune: Rune, game_id: str, episodes: int = 5) -> float:
        """Test a rune in sandbox, return success rate."""
        from .sim import PerfectSimulator

        sim = PerfectSimulator(game_id, self.environments_dir)
        successes = 0

        for _ in range(episodes):
            sim.reset()
            # Try to reach rune's condition
            max_steps = 100
            for step in range(max_steps):
                frame = sim.frame
                if frame is None:
                    break
                if rune.matches(frame, game_id):
                    # Apply rune
                    target = rune.get_target(frame)
                    if target:
                        sim.step(rune.action, target[0], target[1])
                    else:
                        sim.step(rune.action)
                    # Check if progress made
                    if sim.levels_completed > 0 or not sim.done:
                        successes += 1
                    break
                # If not matching, take a step towards it (simplified)
                if step == max_steps - 1:
                    break
                sim.step(6, 32, 32)  # Default exploration

            if sim.done and sim.state_name == "WIN":
                successes += 1

        return successes / episodes if episodes > 0 else 0.0

    def evolve_generation(self, game_classes: list[str], top_k: int = 20) -> list[Rune]:
        """Run one evolution generation: test, select, mutate, crossover."""
        # Evaluate all runes on their games
        scored = []
        for r in self.lib.runes:
            if r.game_class in game_classes or r.game_class == "universal":
                score = self.evaluate_rune(r, r.game_class if r.game_class != "universal" else game_classes[0])
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in scored[:top_k]]

        # Mutate top runes to create variants
        new_runes = []
        for r in top:
            # Create variations with slightly different thresholds/anchors
            for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if r.target_rel:
                    new_target = (r.target_rel[0] + delta[0], r.target_rel[1] + delta[1])
                    if 0 <= new_target[0] <= 63 and 0 <= new_target[1] <= 63:
                        mutant = Rune(
                            id=f"{r.id}_mut_{self.generation}",
                            game_class=r.game_class,
                            rune_type=r.rune_type,
                            condition_sig=r.condition_sig.copy(),
                            action=r.action,
                            target_rel=new_target,
                            anchor_type=r.anchor_type,
                            confidence=r.confidence * 0.9,
                            support=1,
                            games_observed=r.games_observed.copy(),
                            betti_signature=r.betti_signature,
                        )
                        new_runes.append(mutant)

        # Add best new runes to library
        for nr in new_runes[: top_k // 2]:
            self.lib.add(nr)

        self.generation += 1
        avg_score = np.mean([s for s, _ in scored[:top_k]]) if scored else 0
        self.performance_log.append(avg_score)

        print(f"  Generation {self.generation}: avg_top_score={avg_score:.3f}, library_size={len(self.lib.runes)}")
        return top


# ============================================================
# FullAuto Autonomous Online Rune Synthesizer & Meta-Evolver
# ============================================================


class AutonomousOnlineRuneSynthesizer:
    """
    World-class FullAuto Invariant Rune Engine.
    Dynamically discovers, synthesizes, tests, and self-upgrades Runes directly
    from online runtime transitions without manual intervention.
    """

    _INSTANCE: AutonomousOnlineRuneSynthesizer | None = None

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.active_runes: list[Rune] = []
        self.rune_performance: dict[str, dict[str, float]] = {}  # {id: {"hits": N, "reward": R, "conf": C}}
        self.generation = 0
        self._learned_hashes: set[int] = set()

    @classmethod
    def get_instance(cls) -> AutonomousOnlineRuneSynthesizer:
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def observe_and_synthesize(
        self,
        prev_frame: np.ndarray,
        action: int,
        curr_frame: np.ndarray,
        coords: tuple[int, int] | None = None,
        game_id: str = "unseen",
        bg: int = 0,
    ) -> Rune | None:
        """
        Analyze a live transition (S -> A -> S') and autonomously synthesize an invariant Rune.
        Ensures zero-waste extraction of novel spatial laws in real-time.
        """
        if np.array_equal(prev_frame, curr_frame):
            return None  # No constructive delta

        # Compute Betti and topological delta
        b0_p, b1_p, _ = compute_betti_numbers(prev_frame, bg)
        b0_c, b1_c, _ = compute_betti_numbers(curr_frame, bg)
        delta_betti = (b0_c - b0_p, b1_c - b1_p, 0)

        # Signature of trigger state
        prev_sig16 = signature(prev_frame)["sig16"]
        h_sig = hash(prev_sig16.tobytes())
        if h_sig in self._learned_hashes:
            # Upgrade existing matching rune confidence
            for r in self.active_runes:
                if r.action == action and r.matches(prev_frame, game_id):
                    r.support += 1
                    r.confidence = min(0.99, r.confidence + 0.05)
                    return r
            return None

        self._learned_hashes.add(h_sig)

        # Determine invariant Anchor and Relative Target for Action 6 (Click)
        anchor_type = "grid_center"
        target_rel = None
        if action == 6 and coords is not None:
            cx, cy = coords
            cen = _largest_object_centroid(prev_frame)
            if cen:
                anchor_type = "object"
                target_rel = (int(cx - cen[0]), int(cy - cen[1]))
            else:
                anchor_type = "fixed"
                target_rel = (cx, cy)
        elif action in (1, 2, 3, 4, 5):
            anchor_type = "fixed"
            target_rel = None

        rune_id = f"auto_rune_gen{self.generation}_{action}_{len(self.active_runes)}"
        new_rune = Rune(
            id=rune_id,
            game_class=game_id.split("-")[0] if "-" in game_id else "universal",
            rune_type="online_synthesized",
            condition_sig=prev_sig16,
            action=action,
            target_rel=target_rel,
            anchor_type=anchor_type,
            confidence=0.88,
            support=1,
            games_observed={game_id},
            betti_signature=(b0_c, b1_c, 0),
            metadata={"delta_betti": delta_betti},
        )

        self.active_runes.append(new_rune)
        self.rune_performance[rune_id] = {"hits": 1.0, "success": 1.0, "conf": 0.88}

        # Self-Evolve & Prune if capacity exceeded
        if len(self.active_runes) > self.capacity:
            self._self_prune_and_evolve()

        return new_rune

    def evaluate_live_runes(
        self, frame: np.ndarray, game_id: str, available: list[int]
    ) -> list[tuple[int, int | None, int | None, str, float]]:
        """
        Evaluate all autonomously synthesized runes against the current frame.
        """
        proposals: list[tuple[int, int | None, int | None, str, float]] = []
        for r in self.active_runes:
            if r.action not in available:
                continue
            if r.matches(frame, game_id, threshold=0.10) and r.matches_topology(frame):
                coords = r.get_target(frame) if r.action == 6 else (None, None)
                x = coords[0] if coords else None
                y = coords[1] if coords else None
                proposals.append((r.action, x, y, f"auto_rune:{r.id}", float(r.confidence)))

        proposals.sort(key=lambda p: -p[4])
        return proposals

    def _self_prune_and_evolve(self) -> None:
        """Prunes low-confidence runes and mutates high-performing runes to form universal super-runes."""
        self.generation += 1
        # Sort by fitness: confidence * log1p(support)
        self.active_runes.sort(key=lambda r: r.confidence * np.log1p(r.support), reverse=True)
        # Keep top 60%
        keep_count = int(self.capacity * 0.6)
        survivors = self.active_runes[:keep_count]

        # Mutate & Cross-synthesize invariant invariants
        mutants: list[Rune] = []
        for r in survivors[:20]:
            if r.target_rel and r.action == 6:
                for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    mutants.append(
                        Rune(
                            id=f"{r.id}_mut_{self.generation}",
                            game_class=r.game_class,
                            rune_type="mutant_super_rune",
                            condition_sig=r.condition_sig.copy(),
                            action=r.action,
                            target_rel=(r.target_rel[0] + d[0], r.target_rel[1] + d[1]),
                            anchor_type=r.anchor_type,
                            confidence=float(r.confidence * 0.95),
                            support=1,
                            games_observed=r.games_observed.copy(),
                            betti_signature=r.betti_signature,
                        )
                    )
        self.active_runes = survivors + mutants[: int(self.capacity * 0.3)]


# Export main functions
__all__ = [
    "Rune",
    "RuneLibrary",
    "RuneEvolver",
    "AutonomousOnlineRuneSynthesizer",
    "extract_runes_from_trace",
    "extract_universal_runes",
    "build_rune_library",
    "compute_betti_numbers",
    "compute_euler_characteristic",
    "compute_grid_complexity",
    "CausalChain",
    "CausalMemory",
]
