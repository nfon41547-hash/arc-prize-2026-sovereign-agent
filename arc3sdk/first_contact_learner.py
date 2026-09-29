"""First-contact learner: few-shot system identification for unseen games.

On a never-seen game the only honest policy is: probe systematically,
record CAUSAL effects (state-class x action -> pixel change / level-up),
ban proven no-ops, and exploit the first signal found. Everything here
runs in microseconds — no LLM, no network, no weight loads.

Concepts:
- state_class: coarse, translation-tolerant signature (bg + 8x8 mosaic +
  color count). Similar screens share statistics (transfer across levels).
- effect model: (game, state_class, action) -> [tries, diff_sum, wins].
  Laplace-smoothed progress estimates; RESET is recorded but never
  promoted (its whole-grid diff would otherwise dominate).
- probe-first rule: explore while the game is unknown (< cold_steps
  observations) or while recent actions fail (stagnation) and untried
  probes remain; never interrupt a queued winning macro.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

_COLD_STEPS = 8
_MAX_STATES = 4096
_BAN_TRIES = 2


def state_class(grid: np.ndarray) -> str:
    """Coarse screen signature: bg + 8x8 mosaic + color count."""
    g = np.ascontiguousarray(grid, dtype=np.uint8)
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    h, w = g.shape
    mosaic = g[(np.arange(8) * h // 8)[:, None], (np.arange(8) * w // 8)]
    digest = hashlib.blake2b(
        mosaic.astype(np.uint8).tobytes() + bytes([bg, len(vals) & 0xFF]),
        digest_size=8).hexdigest()
    return f"{bg}:{len(vals)}:{digest}"


class FirstContactLearner:
    """Per-game causal effect memory + probe policy. All methods fail-open."""

    def __init__(self, cold_steps: int = _COLD_STEPS, max_states: int = _MAX_STATES):
        self.cold_steps = cold_steps
        self.max_states = max_states
        self.obs_count: dict = {}
        self.eff: OrderedDict[tuple, list] = OrderedDict()

    # -- learning ----------------------------------------------------------
    def observe(self, game_id: str, grid_before: np.ndarray, action_key: str,
                grid_after: np.ndarray, level_up: bool = False) -> None:
        try:
            sc = state_class(np.asarray(grid_before, dtype=np.uint8))
            after = np.asarray(grid_after, dtype=np.uint8)
            before = np.asarray(grid_before, dtype=np.uint8)
            diff = int((after.reshape(-1) != before.reshape(-1)).sum()) \
                if after.shape == before.shape else 0
            self.obs_count[game_id] = self.obs_count.get(game_id, 0) + 1
            key = (game_id, sc, str(action_key))
            rec = self.eff.get(key)
            if rec is None:
                if len(self.eff) >= self.max_states:
                    self.eff.popitem(last=False)
                rec = [0, 0, 0]
                self.eff[key] = rec
            else:
                self.eff.move_to_end(key)
            rec[0] += 1
            rec[1] += diff
            rec[2] += 1 if level_up else 0
        except Exception:
            pass

    # -- reading -------------------------------------------------------------
    def progress_rate(self, game_id: str, sc: str, action_key: str) -> float:
        rec = self.eff.get((game_id, sc, str(action_key)))
        if rec is None:
            return 0.5  # maximum uncertainty for the unknown
        n, d, _w = rec
        return (min(d, n * 64) / (n * 64) + 1.0) / (n + 2.0) if n else 0.5

    def banned(self, game_id: str, sc: str, action_key: str) -> bool:
        """Proven no-op (>=BAN_TRIES tries, zero effect, no wins)."""
        if str(action_key) == "RESET":
            return False
        rec = self.eff.get((game_id, sc, str(action_key)))
        return bool(rec is not None and rec[0] >= _BAN_TRIES
                    and rec[1] == 0 and rec[2] == 0)

    def best_known(self, game_id: str, sc: str,
                   candidates: list) -> str | None:
        """Highest expected-progress non-RESET candidate. None if all banned."""
        best, best_q = None, -1.0
        for cand in candidates:
            if cand == "RESET" or self.banned(game_id, sc, cand):
                continue
            q = self.progress_rate(game_id, sc, cand)
            if q > best_q:
                best, best_q = cand, q
        return best

    def should_probe_first(self, game_id: str, stagnation: int,
                           has_macro: bool, probe_available: bool) -> bool:
        """Explore on failure/ignorance; never interrupt a queued macro."""
        if has_macro or not probe_available:
            return False
        if self.obs_count.get(game_id, 0) < self.cold_steps:
            return True
        return stagnation >= 1
