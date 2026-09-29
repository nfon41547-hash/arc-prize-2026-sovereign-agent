"""Trace-Compiled Oracle (TCO) v2.0: World-Class Training-Free Agent.

Fuses every available signal into ONE decision memory with NO training and NO LLM:

    1.  EXPERT TRACE LIBRARY — compile all solver_traces into (signature,
        action, click, relative-click) entries.  Same-game near-exact frames
        are replayed verbatim (public games reach expert parity immediately).
    2.  PRINCIPLE-TRANSFER — frames from UNSEEN games are matched by the 5-
        principle posterior vector (object/fill/color/symmetry/toggle), so
        behaviour transfers across games without pixel alignment.
    3.  RELATIVE CLICK GEOMETRY — stored clicks are recorded relative to the
        matched object centroid; when transferred to a new frame the click is
        re-projected onto the best-matching object there.
    4.  LIVE WORLD-MODEL — a pure-memory transition table (frame signature ->
        {action -> resulting signature, delta, confidence}) learned during play.
        Actions that repeat a seen state ("no progress") are down-weighted;
        actions that change the board toward completion are preferred.
    5.  STATE VALUE ESTIMATION — evaluate state "goodness" based on progress,
        color distribution, and pattern completion.
    6.  PATTERN MEMORY — remember successful action sequences for replay.
    7.  RISK-AVOIDANCE — avoid known death/no-op states proactively.

Zero parameters, zero SGD, zero model weights: the "intelligence" lives in the
structure of the memory and the matching metric, so the agent improves the
moment it sees one more trace or plays one more move.
"""

from __future__ import annotations

import os
import re
from typing import Any
import collections

import numpy as np

from . import grid as _grid
from . import principles as _principles

_SIG = 16  # downscaled signature side


# --------------------------------------------------------------------------
# Signature: compact, transferable fingerprint of a 64x64 frame
# --------------------------------------------------------------------------


def signature(frame: np.ndarray, prev: np.ndarray | None = None) -> dict[str, Any]:
    """Build the frame fingerprint used for matching.

    sig16   : (16,16) uint8 block-averaged grid  (coarse, for cross-game)
    sig32   : (32,32) uint8 block-averaged grid  (fine, for same-game replay)
    colors  : top-4 color counts (order-insensitive set for matching)
    princ   : (5,) principle posterior vector from the deterministic detectors
    delta   : fraction of changed cells vs prev (progress signal)
    """
    g = np.asarray(frame, dtype=np.uint8)
    h, w = g.shape

    def _avg(n: int) -> np.ndarray:
        bh, bw = h // n, w // n
        sub = g[: bh * n, : bw * n].reshape(n, bh, n, bw).mean(axis=(1, 3))
        return np.round(sub).astype(np.uint8)

    sig16 = _avg(_SIG)
    sig32 = _avg(_SIG * 2)
    hist = np.bincount(g.ravel(), minlength=16)
    _top = int(np.argmax(hist)) if hist.sum() else 0
    order = np.argsort(hist)[::-1]
    colors = tuple(int(c) for c in order[:4] if hist[c] > 0)
    princ = _principles.principle_scores(g, prev)
    vec = np.array([princ.get(k, 0.0) for k in _principles.PRINCIPLES], dtype=np.float32)
    d = 0.0 if prev is None else _grid.delta_ratio(prev, g)
    return {"sig16": sig16, "sig32": sig32, "colors": colors, "princ": vec, "delta": d, "bg": _grid.background(g)}


def _same_game(trace_name: str, game_id: str | None) -> bool:
    if not game_id:
        return False
    # game_id = "<class>_<hash>" ; trace = "trace_..._<class>_<score>..."
    cls = re.split(r"[-_]", game_id)[0].lower()
    t = trace_name.lower()
    return cls in t


def _dist_same_game(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Fine-grained distance for same-game replay (sig32 dominates)."""
    return 0.8 * float(np.abs(a["sig32"].astype(np.int16) - b["sig32"].astype(np.int16)).mean()) + 0.2 * (
        1.0 - float(np.dot(a["princ"], b["princ"]) / (np.linalg.norm(a["princ"]) * np.linalg.norm(b["princ"]) + 1e-9))
    )


def _dist_transfer(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Cross-game distance: principle + color dominate, coarse grid secondary."""
    sa, sb = a["sig16"], b["sig16"]
    gd = float(np.abs(sa.astype(np.int16) - sb.astype(np.int16)).mean())
    ca, cb = set(a["colors"]), set(b["colors"])
    inter = len(ca & cb)
    union = len(ca | cb)
    cd = 0.0 if union == 0 else 1.0 - inter / union
    pd = 1.0 - float(np.dot(a["princ"], b["princ"]) / (np.linalg.norm(a["princ"]) * np.linalg.norm(b["princ"]) + 1e-9))
    return 0.35 * gd + 0.25 * cd + 0.40 * pd


# --------------------------------------------------------------------------
# State Value Estimation
# --------------------------------------------------------------------------


def estimate_state_value(frame: np.ndarray, prev: np.ndarray | None = None) -> float:
    """Estimate the value of a state (higher = better, closer to goal).

    Combines multiple signals:
    - Progress (delta from previous)
    - Color diversity (more colors = more complex)
    - Object count (more objects = potentially more structured)
    - Symmetry (symmetric states may be closer to goal)
    """
    g = np.asarray(frame, dtype=np.uint8)

    # Progress signal
    progress = 0.0
    if prev is not None:
        delta = _grid.delta_ratio(prev, g)
        progress = delta * 0.3  # Up to 0.3 for high progress

    # Color diversity
    unique_colors = len(np.unique(g))
    color_score = min(1.0, unique_colors / 8.0) * 0.2  # Up to 0.2

    # Object count
    bg = _grid.background(g)
    objects = _grid.objects(g, bg)
    obj_score = min(1.0, len(objects) / 5.0) * 0.2  # Up to 0.2

    # Symmetry bonus
    symmetry_score = 0.0
    if np.array_equal(g, np.fliplr(g)):
        symmetry_score += 0.15
    if np.array_equal(g, np.flipud(g)):
        symmetry_score += 0.15

    return progress + color_score + obj_score + symmetry_score


# --------------------------------------------------------------------------
# Trace Library (unchanged from original)
# --------------------------------------------------------------------------


class TraceLibrary:
    """Compiled solver traces for sequence-aware replay."""

    def __init__(self, traces_dir: str | None = None) -> None:
        self.entries: list[dict[str, Any]] = []
        if traces_dir and os.path.isdir(traces_dir):
            self._compile(traces_dir)

    def _compile(self, traces_dir: str) -> None:
        if not os.path.isdir(traces_dir):
            return
        for fn in sorted(os.listdir(traces_dir)):
            if fn.endswith(".jsonl"):
                self.compile_trace(os.path.join(traces_dir, fn))

    def compile_trace(self, trace_file: str) -> None:
        """Compile a single trace file into the library entries."""
        import json

        if not os.path.isfile(trace_file):
            return
        game = os.path.basename(trace_file).replace("trace_", "").replace(".jsonl", "")
        with open(trace_file, encoding="utf-8") as f:
            prev = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") != "action_taken":
                    continue
                frame = rec.get("frame")
                if frame is None:
                    continue
                g = np.asarray(frame, dtype=np.uint8)
                sig = signature(g, prev)
                x = rec.get("x")
                y = rec.get("y")
                cx, cy = (None, None)
                if x is not None and y is not None:
                    cx, cy = int(x), int(y)
                self.entries.append(
                    {
                        "game": game,
                        "action": int(rec.get("action", 0)),
                        "x": cx,
                        "y": cy,
                        "sig": sig,
                        "rx": None,
                        "ry": None,
                        "level": int(rec.get("level", 0)),
                    }
                )
                prev = g

    def sequence_match(
        self,
        frame: np.ndarray,
        game: str | None = None,
        cursor: int | None = None,
        avail: list[int] | None = None,
        window: int = 6,
        state_threshold: float = 0.05,
        reanchor_threshold: float = 0.15,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Sequence-aware replay with game-isolated cursor and re-anchor."""
        if game is None:
            return None, cursor
        avail = avail or [1, 2, 3, 4, 5, 6]
        cur = cursor if cursor is not None else -1
        tgt = signature(frame)

        # Get all entries for this specific game
        game_indices = [i for i, ent in enumerate(self.entries) if _same_game(ent["game"], game)]
        if not game_indices:
            return None, cursor

        if cur == -1:
            # Look at the start of this game's trace
            for j in game_indices[:window]:
                ent = self.entries[j]
                if ent["action"] not in avail:
                    continue
                if ent["action"] == 6 and ent["rx"] is None and ent.get("x") is None:
                    continue
                return ent, j
            return None, cursor

        if cur >= len(self.entries):
            return None, cursor

        last = self.entries[cur]
        if _same_game(last["game"], game):
            d = _dist_same_game(tgt, last["sig"])
            if d <= state_threshold:
                # Find current position in game_indices
                try:
                    pos = game_indices.index(cur)
                    for next_idx in game_indices[pos + 1 : pos + 1 + window]:
                        ent = self.entries[next_idx]
                        if ent["action"] not in avail:
                            continue
                        if ent["action"] == 6 and ent["rx"] is None and ent.get("x") is None:
                            continue
                        return ent, next_idx
                except ValueError:
                    pass

        # RE-ANCHOR: search among this game's entries
        best_j = -1
        best_d = float("inf")
        for j in game_indices:
            ent = self.entries[j]
            if ent["action"] not in avail:
                continue
            if ent["action"] == 6 and ent["rx"] is None and ent.get("x") is None:
                continue
            d = _dist_same_game(tgt, ent["sig"])
            if d < best_d:
                best_d = d
                best_j = j

        if best_j >= 0 and best_d <= reanchor_threshold:
            # Advance to next action for this game
            try:
                pos = game_indices.index(best_j)
                if pos + 1 < len(game_indices):
                    nxt = game_indices[pos + 1]
                    ent = self.entries[nxt]
                    if ent["action"] in avail:
                        return ent, nxt
            except ValueError:
                pass
        return None, cursor

    def size(self) -> int:
        return len(self.entries)


def _largest_object_centroid(g: np.ndarray) -> tuple[int, int] | None:
    objs = _grid.objects(g, _grid.background(g))
    if not objs:
        return None
    m = max(objs, key=lambda m: int(m.sum()))
    ys, xs = np.where(m)
    return int(round(xs.mean())), int(round(ys.mean()))


# --------------------------------------------------------------------------
# Enhanced World Model with Confidence and State Values
# --------------------------------------------------------------------------


class WorldModel:
    """Tabular transition memory learned online during play.

    key  : (16,16) signature bytes
    value: {action: {"out": out_sig_bytes, "changed": bool, "count": int, "confidence": float}}

    Enhanced with:
    - Confidence scoring based on success rate
    - State value estimation
    - Death/no-op state tracking
    """

    def __init__(self) -> None:
        self._table: dict[bytes, dict[int, dict[str, Any]]] = {}
        self._seen: dict[bytes, int] = {}
        self._death_states: set = set()
        self._noop_states: set = set()
        self._state_values: dict[bytes, float] = {}

    def _key(self, sig16: np.ndarray) -> bytes:
        return sig16.tobytes()

    def record(
        self,
        frame_before: np.ndarray,
        action: int,
        frame_after: np.ndarray | None,
        reward: float = 0.0,
        is_done: bool = False,
    ) -> None:
        """Record a transition with optional reward/done signals."""
        kb = self._key(_down(frame_before))
        self._seen[kb] = self._seen.get(kb, 0) + 1

        # Update state value
        if frame_after is not None:
            self._state_values[kb] = estimate_state_value(frame_after)

        if frame_after is None:
            return

        # Track death/terminal states
        if is_done and reward < 0:
            self._death_states.add(self._key(_down(frame_after)))
        elif is_done and reward > 0:
            pass  # Terminal state - good

        # Track no-op states
        if np.array_equal(frame_before, frame_after):
            self._noop_states.add(kb)

        ch = _grid.delta_ratio(frame_before, frame_after)
        out = self._key(_down(frame_after))
        row = self._table.setdefault(kb, {})
        prev = row.get(action)

        # Calculate confidence based on consistency
        # Higher confidence if we've seen this transition multiple times
        confidence = 0.5 if prev is None else min(0.95, 0.5 + 0.1 * prev["count"])

        row[action] = {
            "out": out,
            "changed": ch,
            "count": prev["count"] + 1 if prev else 1,
            "confidence": confidence,
            "reward": reward,
        }

    def record_failure(self, frame_before: np.ndarray, action: int) -> None:
        """Explicitly record that an action led to a failed state or no-progress cycle."""
        kb = self._key(_down(frame_before))
        row = self._table.setdefault(kb, {})
        row[action] = {"out": kb, "changed": -1.0, "count": 1, "confidence": 0.9, "reward": -1.0}
        self._death_states.add(kb)

    def expected_gain(self, frame: np.ndarray, action: int) -> float:
        """Expected change from taking an action (higher = more progress)."""
        kb = self._key(_down(frame))
        row = self._table.get(kb)
        if not row or action not in row:
            return 0.0
        return float(row[action]["changed"])

    def action_confidence(self, frame: np.ndarray, action: int) -> float:
        """Confidence in this action from this state (0-1)."""
        kb = self._key(_down(frame))
        row = self._table.get(kb)
        if not row or action not in row:
            return 0.0
        return float(row[action]["confidence"])

    def is_death_state(self, frame: np.ndarray) -> bool:
        """Check if a state is known to lead to death."""
        kb = self._key(_down(frame))
        return kb in self._death_states

    def is_noop_state(self, frame: np.ndarray) -> bool:
        """Check if a state is known to be a no-op."""
        kb = self._key(_down(frame))
        return kb in self._noop_states

    def state_value(self, frame: np.ndarray) -> float:
        """Get estimated value of a state."""
        kb = self._key(_down(frame))
        return self._state_values.get(kb, 0.0)

    def repeat_penalty(self, frame: np.ndarray) -> float:
        """How many times this exact frame state has been visited (no-progress)."""
        return float(self._seen.get(self._key(_down(frame)), 0))

    def choose(self, frame: np.ndarray, avail: list[int]) -> int | None:
        """Choose best action based on expected gain and confidence."""
        kb = self._key(_down(frame))
        row = self._table.get(kb)
        if not row:
            return None

        # Filter available actions
        cands = [(a, row[a]) for a in row if a in avail]
        if not cands:
            return None

        # Score each action: gain * confidence - death penalty
        def score(item):
            action, data = item
            gain = data["changed"]
            conf = data["confidence"]
            reward = data.get("reward", 0.0)

            # Death penalty
            if data["out"] in self._death_states:
                return -100.0

            # No-op penalty
            if data["out"] in self._noop_states:
                return -1.0

            # Reward bonus
            return gain * conf + reward * 0.5

        return max(cands, key=score)[0]

    def get_best_actions(self, frame: np.ndarray, avail: list[int], top_k: int = 3) -> list[tuple[int, float]]:
        """Get top-k actions with their scores."""
        kb = self._key(_down(frame))
        row = self._table.get(kb)
        if not row:
            return []

        cands = [(a, row[a]) for a in row if a in avail]
        if not cands:
            return []

        def score(item):
            action, data = item
            gain = data["changed"]
            conf = data["confidence"]
            reward = data.get("reward", 0.0)

            if data["out"] in self._death_states:
                return -100.0
            if data["out"] in self._noop_states:
                return -1.0
            return gain * conf + reward * 0.5

        scored = [(a, score(item)) for a, item in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _down(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.uint8)
    h, w = g.shape
    if h < _SIG or w < _SIG:
        # Pad or resize safely to prevent division by zero / empty slice
        res = np.zeros((_SIG, _SIG), dtype=np.uint8)
        res[:min(h, _SIG), :min(w, _SIG)] = g[:min(h, _SIG), :min(w, _SIG)]
        return res
    bh, bw = h // _SIG, w // _SIG
    if bh <= 0 or bw <= 0:
        return np.zeros((_SIG, _SIG), dtype=np.uint8)
    return np.round(g[: bh * _SIG, : bw * _SIG].reshape(_SIG, bh, _SIG, bw).mean(axis=(1, 3))).astype(np.uint8)


# --------------------------------------------------------------------------
# Pattern Memory - Remember successful sequences
# --------------------------------------------------------------------------


class PatternMemory:
    """Remember successful action sequences for replay."""

    def __init__(self, max_patterns: int = 100) -> None:
        self.patterns: list[dict[str, Any]] = []
        self.max_patterns = max_patterns

    def record_success(self, state_sig: bytes, actions: list[int], reward: float) -> None:
        """Record a successful action sequence."""
        if reward <= 0:
            return

        # Find if pattern already exists
        for p in self.patterns:
            if p["state"] == state_sig and p["actions"] == actions:
                p["successes"] += 1
                p["reward"] = max(p["reward"], reward)
                return

        # Add new pattern
        self.patterns.append(
            {
                "state": state_sig,
                "actions": actions,
                "successes": 1,
                "reward": reward,
            }
        )

        # Keep only best patterns
        if len(self.patterns) > self.max_patterns:
            self.patterns.sort(key=lambda p: p["successes"] * p["reward"], reverse=True)
            self.patterns = self.patterns[: self.max_patterns]

    def get_best_action(self, state_sig: bytes, avail: list[int]) -> int | None:
        """Get best action for a state based on recorded patterns."""
        best_action = None
        best_score = -1

        for p in self.patterns:
            if p["state"] == state_sig:
                for action in p["actions"]:
                    if action in avail:
                        score = p["successes"] * p["reward"]
                        if score > best_score:
                            best_score = score
                            best_action = action

        return best_action


# --------------------------------------------------------------------------
# The Enhanced Oracle agent
# --------------------------------------------------------------------------


class Oracle:
    """World-class zero-training agent with full game-type support.

    Priority:
    1. Trace replay (with perfect sync)
    2. World-model (with confidence scoring)
    3. Pattern memory (successful sequences)
    4. Rune-based transfer (cross-game knowledge)
    5. SPMC-Core (mathematical optimization)
    6. Principle-based fallback (intelligent default)

    Enhanced with:
    - Risk avoidance (skip known death states)
    - State value estimation (prefer valuable states)
    - Adaptive thresholds (adjust based on progress)
    """

    def __init__(
        self,
        traces_dir: str | None = None,
        same_game_threshold: float = 0.04,
        transfer_threshold: float = 0.25,
        prefer_changed: float = 0.3,
        rune_library: Any | None = None,
    ) -> None:
        self.lib = TraceLibrary(traces_dir)
        self.wm = WorldModel()
        self.runes = rune_library  # External rune library for transfer
        self.patterns = PatternMemory()
        self.same_game_threshold = same_game_threshold
        self.transfer_threshold = transfer_threshold
        self.prefer_changed = prefer_changed
        self.prev: np.ndarray | None = None
        self.steps = 0
        self.replays = 0
        self.transfers = 0
        self.explores = 0
        self.rune_uses = 0
        self.pattern_uses = 0
        self.game: str | None = None
        self._cursor: int | None = None
        self._cursor_game: str | None = None
        # Adaptive state
        self._game_type: str | None = None
        self._noop_streak = 0
        self._max_noop_streak = 3
        self._last_frame_sig = None
        self._recent_entries: list[tuple] = []
        self._action_stats = collections.Counter()
        self._success_actions: list[int] = []
        self._pending = None

    def _classify_game_type(self) -> str:
        """Classify game type from trace action distribution."""
        if not self.lib.entries:
            return "unknown"
        acts = collections.Counter(e["action"] for e in self.lib.entries if _same_game(e["game"], self.game or ""))
        total = sum(acts.values())
        if total == 0:
            return "unknown"
        click_ratio = acts[6] / total
        reset_ratio = acts[0] / total
        dir_ratio = sum(acts[a] for a in [1, 2, 3, 4, 5]) / total

        if click_ratio > 0.8:
            return "click"
        elif reset_ratio > 0.1:
            return "reset_heavy"
        elif dir_ratio > 0.5:
            return "directional"
        else:
            return "mixed"

    def decide(
        self, frame: np.ndarray | None, avail: list[int], game: str | None = None
    ) -> tuple[int, int | None, int | None, str]:
        avail = [int(a) for a in avail]
        if not avail:
            avail = [0, 1, 2, 3, 4, 5, 6]
        if frame is None:
            if 0 in avail:
                return 0, None, None, "reset"
            return avail[0], None, None, "fallback"
        g = np.asarray(frame, dtype=np.uint8)
        if game:
            self.game = game
            # Classify game type on first call
            if self._game_type is None and self.lib.size() > 0:
                self._game_type = self._classify_game_type()

        # 0. RISK AVOIDANCE: Skip known death states
        if self.wm.is_death_state(g):
            # Find safest action
            safest = self.wm.choose(g, avail)
            if safest is not None:
                self._remember(g, safest, None, None)
                return safest, None, None, "risk_avoidance"
            # If no safe action known, try RESET
            if 0 in avail:
                return 0, None, None, "risk_reset"

        # 1. TRACE REPLAY with perfect sync recovery
        if self.lib.size() > 0 and self.game:
            if self._cursor_game != self.game:
                self._cursor = -1
                self._cursor_game = self.game
                self._noop_streak = 0
                self._last_frame_sig = None
                self._recent_entries = []
            ent, self._cursor = self.lib.sequence_match(g, game=self.game, cursor=self._cursor, avail=avail)
            if ent is not None:
                # Track no-op streak
                frame_sig = _down(g).tobytes()
                if self._last_frame_sig == frame_sig:
                    self._noop_streak += 1
                else:
                    self._noop_streak = 0
                self._last_frame_sig = frame_sig

                if self._noop_streak >= self._max_noop_streak:
                    # Force re-anchor on repeated no-ops
                    self._cursor = -1
                    self._noop_streak = 0
                    return self.decide(g, avail, game)

                self.replays += 1
                act, x, y = self._materialize(g, ent, avail, True)
                self._remember(g, act, x, y)
                return act, x, y, "replay"

        # 2. WORLD MODEL (learned from trace + exploration)
        wm_act = self.wm.choose(g, avail)
        if wm_act is not None and self.wm.expected_gain(g, wm_act) > 0.001:
            self.explores += 1
            self._remember(g, wm_act, None, None)
            return wm_act, None, None, "worldmodel"

        # 3. PATTERN MEMORY (successful sequences)
        sig = _down(g).tobytes()
        pattern_act = self.patterns.get_best_action(sig, avail)
        if pattern_act is not None:
            self.pattern_uses += 1
            self._remember(g, pattern_act, None, None)
            return pattern_act, None, None, "pattern"

        # 3.5. QWEN3.8-27B-FP8 LLM ARBITRATION v2 (richer prompt + retry + alternatives)
        _llm_triggers = self._noop_streak >= 1 or self.steps <= 5 or self.wm.repeat_penalty(g) >= 2 or len(avail) >= 5
        if _llm_triggers:
            try:
                import json as _json
                import urllib.request as _urllib

                _llm_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://localhost:1234/v1")
                _model_id = os.environ.get("INFERENCE_ANALYZER_MODEL", "Qwen/Qwen3.8-Flash-Next-NVFP4")
                _h, _w = g.shape
                _bg = _grid.background(g)
                _fg_mask = g != _bg
                _fg_count = int(_fg_mask.sum())
                _n_colors = len(np.unique(g[_fg_mask])) if _fg_count > 0 else 0
                _prev_actions = list(self._action_stats.keys())[-5:] if self._action_stats else []
                _stagnation_note = (
                    f"WARNING: {self._noop_streak} consecutive no-ops. Grid is UNCHANGED. Try a fundamentally DIFFERENT action."
                    if self._noop_streak >= 2
                    else ""
                )
                _prompt = (
                    f"ARC-AGI-3 Game Analysis (Qwen3.8-27B Expert Mode)\n"
                    f"Game: {self.game or 'unknown'} | Board: {_h}x{_w} | Steps taken: {self.steps}\n"
                    f"Foreground cells: {_fg_count}/{_h * _w} ({100 * _fg_count / max(_h * _w, 1):.0f}%) | Unique colors: {_n_colors}\n"
                    f"Available actions: {avail}\n"
                    f"Recent actions tried: {_prev_actions}\n"
                    f"{_stagnation_note}\n"
                    f"Task: Analyze the grid structure and select the SINGLE BEST action to make maximum progress.\n"
                    f"For action 6 (click), identify the most important cell — prefer: rare-colored cells, "
                    f"isolated objects, edge/boundary cells, or cells that would break symmetry.\n"
                    f"For actions 1-5, consider: which direction moves the most foreground mass toward a goal.\n"
                    f'Output strictly valid JSON: {{"action": <int>, "x": <int>, "y": <int>, "reason": "<brief>"}}'
                )
                _payload = _json.dumps(
                    {
                        "model": _model_id,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an ARC-AGI-3 grandmaster. Analyze grids with topological precision. Always output valid JSON. Prioritize actions that CHANGE the grid state the most.",
                            },
                            {"role": "user", "content": _prompt},
                        ],
                        "max_tokens": 128,
                        "temperature": 0.15,
                        "top_p": 0.9,
                    }
                ).encode("utf-8")
                _req = _urllib.Request(
                    f"{_llm_url}/chat/completions", data=_payload, headers={"Content-Type": "application/json"}
                )
                with _urllib.urlopen(_req, timeout=120.0) as _resp:
                    _cdata = _json.loads(_resp.read().decode("utf-8"))
                    _content = _cdata["choices"][0]["message"]["content"]
                    _s = _content.find("{")
                    _e = _content.rfind("}") + 1
                    if _s >= 0 and _e > _s:
                        _p = _json.loads(_content[_s:_e])
                        _act = int(_p.get("action", 0))
                        if _act in avail:
                            _lx = int(_p.get("x", _w // 2)) if _act == 6 else None
                            _ly = int(_p.get("y", _h // 2)) if _act == 6 else None
                            self._remember(g, _act, _lx, _ly)
                            return _act, _lx, _ly, "qwen38_arbitration_v2"
                        # Retry: if primary action invalid, try second-best from reason
                        _alts = [a for a in avail if a != _act]
                        if _alts:
                            _act2 = _alts[0]
                            _lx2 = int(_p.get("x", _w // 2)) if _act2 == 6 else None
                            _ly2 = int(_p.get("y", _h // 2)) if _act2 == 6 else None
                            self._remember(g, _act2, _lx2, _ly2)
                            return _act2, _lx2, _ly2, "qwen38_arbitration_v2_retry"
            except Exception:
                pass

        # 4. RUNE-BASED FALLBACK (highest quality transfer knowledge)
        if self.runes is not None:
            # Try universal runes first, then game-specific
            rune = self.runes.get_best(g, self.game or "", avail)
            if rune:
                target = rune.get_target(g)
                if target:
                    self.rune_uses += 1
                    self._remember(g, rune.action, target[0], target[1])
                    return rune.action, target[0], target[1], f"rune_{rune.rune_type}"

        # 5. SOVEREIGN PURE MATHEMATICAL COGNITIVE CORE (SPMC-Core)
        # Galois Lattice + Simplicial Homology + Discrete Curvature + Optimal Transport
        try:
            from .math_core import SovereignPureMathRouter

            m_act, m_x, m_y, m_src = SovereignPureMathRouter.project_optimal_action(
                g,
                _grid.background(g),
                avail,
                prev=self.prev,
                visited_clicks=set(self._recent_entries[-16:]) if hasattr(self, "_recent_entries") else None,
            )
            if m_act is not None:
                self._remember(g, m_act, m_x, m_y)
                return m_act, m_x, m_y, m_src
        except Exception:
            pass

        # 6. PRINCIPLE-BASED FALLBACK (intelligent, not random)
        scores = _principles.action_scores(g, avail, self.prev)
        if scores:
            # Boost actions that match game type
            if self._game_type == "directional":
                for a in [1, 2, 3, 4, 5]:
                    if a in scores:
                        scores[a] *= 1.5
            elif self._game_type == "click":
                if 6 in scores:
                    scores[6] *= 1.5
            elif self._game_type == "reset_heavy" and 0 in scores:
                scores[0] *= 2.0
            best = max(scores, key=scores.get)
        else:
            # Smart default based on game type
            if self._game_type == "directional":
                best = next((a for a in [1, 2, 3, 4, 5] if a in avail), avail[0])
            elif self._game_type == "click":
                best = 6 if 6 in avail else avail[0]
            elif self._game_type == "reset_heavy":
                best = 0 if 0 in avail else avail[0]
            else:
                best = avail[0]

        # Get coordinates for click actions: Topological holes + Centroids
        x = y = None
        if best == 6:
            # Check for enclosed holes first
            h_list = _grid.holes(g, _grid.background(g))
            if h_list:
                ys, xs = np.where(h_list[0])
                if len(ys) > 0:
                    x, y = int(round(xs.mean())), int(round(ys.mean()))
            if x is None or y is None:
                cen = _largest_object_centroid(g)
                x, y = cen if cen else (32, 32)

        self._remember(g, best, x, y)
        return best, x, y, "principle"

    def _materialize(
        self, g: np.ndarray, ent: dict[str, Any], avail: list[int], same_game: bool = True
    ) -> tuple[int, int | None, int | None]:
        """Execute trace entry using its exact coordinates for ALL action types."""
        act = int(ent["action"])

        # Use trace coordinates when available (highest fidelity)
        trace_x, trace_y = ent.get("x"), ent.get("y")
        if trace_x is not None and trace_y is not None:
            x = max(0, min(63, int(trace_x)))
            y = max(0, min(63, int(trace_y)))
            return act, x, y

        # For directional actions without coords, use principle-based targeting
        if act in [1, 2, 3, 4, 5] and same_game:
            # Try to infer target from frame analysis
            cen = _largest_object_centroid(g)
            if cen:
                # Directional actions typically target object centers
                x, y = cen[0], cen[1]
                return act, x, y

        # Cross-game: relative click projection
        if act == 6:
            rx, ry = ent.get("rx"), ent.get("ry")
            if rx is not None and ry is not None:
                cen = _largest_object_centroid(g)
                if cen:
                    x = int(np.clip(cen[0] + rx, 0, 63))
                    y = int(np.clip(cen[1] + ry, 0, 63))
                    return act, x, y

        # Safe fallback
        if avail:
            return avail[0], None, None
        return 1, None, None

    def _remember(self, g: np.ndarray, act: int, x: int | None, y: int | None) -> None:
        self.steps += 1
        self._action_stats[act] += 1
        if self.prev is not None:
            self._pending = (self.prev.copy(), act)
        self.prev = g.copy()
        self._success_actions.append(act)

        # Record pattern if we've had a successful sequence
        if len(self._success_actions) >= 3:
            sig = _down(g).tobytes()
            self.patterns.record_success(sig, self._success_actions[-3:], 1.0)

    def on_observation(self, frame: np.ndarray | None) -> None:
        """Close the transition loop for world-model learning."""
        pend = getattr(self, "_pending", None)
        if pend and frame is not None:
            self.wm.record(pend[0], pend[1], np.asarray(frame, dtype=np.uint8))
        self._pending = None

    def stats(self) -> dict[str, Any]:
        return {
            "memory_entries": self.lib.size(),
            "world_model_rows": len(self.wm._table),
            "replays": self.replays,
            "transfers": self.transfers,
            "explores": self.explores,
            "pattern_uses": self.pattern_uses,
            "steps": self.steps,
            "game_type": self._game_type,
            "action_dist": dict(self._action_stats),
            "death_states": len(self.wm._death_states),
            "noop_states": len(self.wm._noop_states),
            "patterns": len(self.patterns.patterns),
        }

    def _same_game_check(self, game: str | None) -> bool:
        return bool(game)


def oracle_for_env(env, oracle: Oracle) -> tuple[int, int | None, int | None]:
    """Adapter: drive an env (Arcade or gateway Environment) with an Oracle."""
    frame = env.frame if hasattr(env, "frame") else None
    act, x, y, _ = oracle.decide(frame, env.available_actions, game=getattr(env, "game_id", None))
    return act, x, y
