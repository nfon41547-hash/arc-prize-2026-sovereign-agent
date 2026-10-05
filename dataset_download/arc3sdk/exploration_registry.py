from __future__ import annotations
import hashlib
import os
import threading
from typing import Any
import numpy as np
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES (policy inspiration, Apache-2.0).
# Reimplementation of dream-team exploration policies (E2/E4/E5/E7/E8) over our
# own visit/click memory. No upstream code copied; offline, numpy-only.

"""Exploration registry — structured novelty/coverage voters for unknown games.

Ports the *policies* (not code) of NVIDIA dream-team
``exploration_strategy.py`` to our fail-open single-step setting:

  E2 cycle-by-novelty  -> least-visited (state, action) pair first
  E4 novelty walk      -> same as E2 with stagnation boost
  E5 click group test  -> untried clicks, then bisect ±4 around productive ones
  E7 effect probe      -> actions with zero observed change get priority
  E8 coverage clicks   -> inverse-frequency cell coverage

All state is per-game, bounded, and never throws.

Cross-game self-development (2026-09-09): per-game memory used to die on
``reset_game``. Now the wiped game's no-change action counts fold into bounded
GLOBAL priors (``_g_nochange``), so invariant knowledge compounds across the 25
games: on first contact with an unseen state, globally-dead actions are gently
deprioritized (tag ``explorer_xfer_prior``) while per-game evidence keeps
dominant weight. ``ARC3_DISABLE_XFER=1`` ablates transfer for honest A/B.
``stats()`` exposes learning telemetry for the TSV ledger.
"""




_lock = threading.Lock()
_visits: dict[str, dict[tuple[str, int], int]] = {}
_click_payoff: dict[str, dict[tuple[int, int], int]] = {}
_productive_clicks: dict[str, list[tuple[int, int]]] = {}
_nochange_actions: dict[str, dict[int, int]] = {}

# ── Cross-game transfer stores (global priors, bounded, fail-open) ──
_g_nochange: dict[int, int] = {}  # action -> no-change count folded from past games
_g_games_folded: int = 0

# ── Positive transfer: productive action patterns across games ──
_g_action_delta: dict[int, float] = {}  # action -> avg change ratio when used
_g_action_uses: dict[int, int] = {}
_g_cell_efficiency: dict[tuple[int, int], float] = {}  # (x,y) -> change rate
_g_patterns: dict[str, dict] = {}  # grid-type hash -> strategy memory

_MAX_GAMES = 32
_MAX_CELLS = 512
_G_MAX_TOTAL = 4096


def _xfer_enabled() -> bool:
    try:
        return os.environ.get("ARC3_DISABLE_XFER", "").strip().lower() not in ("1", "true", "yes", "on")
    except Exception:
        return True


def _grid_pattern(grid: np.ndarray) -> str:
    """Coarse grid-type signature: color histogram + symmetry. Bounded, deterministic."""
    try:
        h, w = grid.shape
        hist = np.bincount(grid.ravel(), minlength=16).astype(int)
        sig = tuple(int(x) for x in hist[:8])
        sym_v = bool(np.array_equal(grid, np.flipud(grid)))
        sym_h = bool(np.array_equal(grid, np.fliplr(grid)))
        return f"{'v' if sym_v else '-'}{'h' if sym_h else '-'}{sig[:6]}"
    except Exception:
        return "unknown"


def _pattern_advice(grid: np.ndarray, acts: list[int], stagnation: int) -> tuple[int | None, str, float] | None:
    """If past games had this pattern, reuse its best action first.
    Works at stagnation=0 for perfect pattern matches; stale games need >=1."""
    try:
        if not _xfer_enabled():
            return None
        pat = _grid_pattern(grid)
        mem = _g_patterns.get(pat)
        if mem is None:
            return None
        best_a = mem.get("best_action")
        if best_a is not None and best_a in acts:
            rate = float(mem.get("rate", 0.5))
            # Require decent success rate unless stagnation is high.
            if rate >= 0.3 or stagnation >= 1:
                return best_a, f"pattern_{pat[:8]}", rate
        return None
    except Exception:
        return None


def _sequence_advice(history: list[int], acts: list[int]) -> tuple[int | None, str, float] | None:
    """If last action produced change, try its 'neighbor' action next.
    Simple Markov-1 chain, bounded."""
    try:
        if len(history) < 2:
            return None
        prev = int(history[-1])
        prev2 = int(history[-2])
        key = (prev2, prev)
        chain = _g_patterns.get(f"chain_{key}", {})
        if not chain:
            return None
        best_next = max(chain, key=chain.get)
        if best_next in acts:
            return best_next, f"chain_{prev}->{best_next}", float(chain[best_next])
        return None
    except Exception:
        return None


def _global_nochange_rate(a: int) -> float:
    """0..1 deadness prior for action ``a`` from past games. 0.0 when disabled/empty."""
    try:
        if not _xfer_enabled():
            return 0.0
        with _lock:
            total = sum(_g_nochange.values())
            if total <= 0:
                return 0.0
            return min(1.0, float(_g_nochange.get(int(a), 0)) / float(total))
    except Exception:
        return 0.0


def record_transition(game_id: str, action: int, changed: bool, next_grid: np.ndarray | None = None) -> None:
    """Record action→outcome for sequence Markov chain. Called from observe_transition."""
    try:
        gid = str(game_id or "unknown")
        a = int(action)
        with _lock:
            seq = _g_patterns.setdefault(f"seq_{gid}", [])
            seq.append((a, 1 if changed else 0))
            if len(seq) > 16:
                seq.pop(0)
            for i in range(1, len(seq)):
                ka = (seq[i - 1][0], a)
                ch = _g_patterns.setdefault(f"chain_{ka}", {})
                ch[seq[i][0]] = ch.get(seq[i][0], 0) + seq[i][1]
        if changed and next_grid is not None:
            pat = _grid_pattern(next_grid)
            with _lock:
                mem = _g_patterns.setdefault(pat, {"action_counts": {}, "wins": 0, "total": 0})
                ac = mem.setdefault("action_counts", {})
                ac[a] = ac.get(a, 0) + 1
                mem["total"] += 1
                if changed:
                    mem["wins"] += 1
                if mem["total"] >= 3:
                    mem["rate"] = mem["wins"] / mem["total"]
                    if mem["rate"] >= 0.3:
                        mem["best_action"] = a
    except Exception:
        pass


def _state_key(grid: np.ndarray) -> str:
    try:
        return hashlib.blake2b(np.ascontiguousarray(grid).tobytes(), digest_size=8).hexdigest()
    except Exception:
        return "unknown"


def _grid_delta(grid: np.ndarray) -> float:
    """Fraction of non-background cells. 0..1 proxy for 'richness'."""
    try:
        return float(np.count_nonzero(grid)) / float(grid.size)
    except Exception:
        return 0.0


def observe(game_id: str, grid: Any, act: int, x: int | None, y: int | None, changed: bool) -> None:
    """Record one transition outcome. Never throws."""
    try:
        gid = str(game_id or "unknown")
        sk = _state_key(np.asarray(grid, dtype=np.uint8)) if grid is not None else "unknown"
        a = int(act)
        with _lock:
            v = _visits.setdefault(gid, {})
            v[(sk, a)] = v.get((sk, a), 0) + 1
            if len(v) > 4096:
                for k in list(v)[:1024]:
                    v.pop(k, None)
            if not changed:
                nc = _nochange_actions.setdefault(gid, {})
                nc[a] = nc.get(a, 0) + 1
            else:
                # Positive signal: action produced visible change
                da = _grid_delta(grid) if grid is not None else 0.0
                _g_action_delta[a] = (
                    (_g_action_delta.get(a, 0.0) * _g_action_uses.get(a, 0) + da)
                    / max(1, _g_action_uses.get(a, 0) + 1)
                )
                _g_action_uses[a] = _g_action_uses.get(a, 0) + 1
            over = len(_visits) - _MAX_GAMES
            if over > 0:
                victims = [k for k in _visits if k != gid][:over]
                for _d in (_visits, _click_payoff, _productive_clicks, _nochange_actions):
                    for k in victims:
                        _d.pop(k, None)
        if a == 6 and x is not None and y is not None:
            try:
                cx, cy = int(x), int(y)
                with _lock:
                    cp = _click_payoff.setdefault(gid, {})
                    cp[(cx, cy)] = cp.get((cx, cy), 0) + (1 if changed else 0)
                    if len(cp) > _MAX_CELLS:
                        for k in list(cp)[:128]:
                            cp.pop(k, None)
                    if changed:
                        pl = _productive_clicks.setdefault(gid, [])
                        if (cx, cy) not in pl:
                            pl.append((cx, cy))
                            del pl[: max(0, len(pl) - 64)]
                    # Cell efficiency: fold into global cell_efficiency
                    if changed:
                        _g_cell_efficiency[(cx, cy)] = (
                            _g_cell_efficiency.get((cx, cy), 0.0) * 0.8 + 0.2
                        )
                    else:
                        _g_cell_efficiency[(cx, cy)] = (
                            _g_cell_efficiency.get((cx, cy), 0.0) * 0.9
                        )
            except Exception:
                pass
    except Exception:
        pass


def _bisect_around(clicks: list[tuple[int, int]], h: int, w: int, taken: set) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    try:
        for cx, cy in clicks:
            for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4), (2, 2), (-2, -2)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in taken:
                    out.append((nx, ny))
    except Exception:
        pass
    return out


def propose(
    game_id: str,
    grid: np.ndarray,
    legal: list[int],
    resonance_order: list[int] | None = None,
    failed_clicks: set | None = None,
    stagnation: int = 0,
) -> tuple[int, int | None, int | None, str, float] | None:
    """E2/E4/E5/E7/E8 vote. Returns (act,x,y,tag,conf) or None. Never throws."""
    try:
        gid = str(game_id or "unknown")
        acts = [int(a) for a in list(legal) if 1 <= int(a) <= 7]
        if not acts:
            return None
        h, w = int(grid.shape[0]), int(grid.shape[1])
        sk = _state_key(grid)
        failed = set(failed_clicks or set())
        with _lock:
            v = dict(_visits.get(gid, {}))
            cp = dict(_click_payoff.get(gid, {}))
            pl = list(_productive_clicks.get(gid, []))
            nc = dict(_nochange_actions.get(gid, {}))
        order = list(resonance_order) if resonance_order else list(acts)
        order = [a for a in order if a in acts] + [a for a in acts if a not in (resonance_order or [])]

        if 6 in acts:
            tried = {(cx, cy) for (cx, cy) in cp} | set(failed)
            cand = _bisect_around(pl, h, w, tried)
            if stagnation >= 2 and cand:
                nx, ny = cand[0]
                return 6, nx, ny, "explorer_e5_bisect", 0.52
            if not tried or stagnation >= 4:
                fg = np.argwhere(grid != int(np.bincount(grid.ravel(), minlength=16).argmax()))
                if len(fg):
                    cov = sorted((cp.get((int(x), int(y)), 0), (int(x), int(y))) for y, x in fg[:256])
                    for _, (nx, ny) in cov:
                        if (nx, ny) not in tried:
                            return 6, nx, ny, "explorer_e8_coverage", 0.50
                    if cand:
                        nx, ny = cand[0]
                        return 6, nx, ny, "explorer_e5_bisect", 0.50

        scored: list[tuple[float, int]] = []
        xfer_used = False

        # 1. Pattern memory: recognized grid type → reuse its best action.
        pat_adv = _pattern_advice(grid, acts, stagnation)
        if pat_adv is not None:
            a, tag, conf = pat_adv
            return a, None, None, tag, conf

        # 2. Sequence memory: Markov-1 chain from last action.
        seq_adv = None
        try:
            rec = _g_patterns.get(f"seq_{gid}", [])
            if len(rec) >= 2:
                prev_a = rec[-1][0]
                chain = _g_patterns.get(f"chain_{(rec[-2][0], prev_a)}", {})
                if chain:
                    nxt = max(chain, key=chain.get)
                    if nxt in acts:
                        seq_adv = (nxt, f"chain_{prev_a}->{nxt}", float(chain[nxt]))
        except Exception:
            pass

        for a in acts:
            if a == 6:
                continue
            visits = v.get((sk, a), 0)
            nochange = nc.get(a, 0)
            score = visits * 2.0 + (0.0 if nochange == 0 else -1.0 / (1.0 + nochange))
            # Positive transfer: actions that historically produced change get prioritized.
            delta = _g_action_delta.get(a, 0.0)
            if delta > 0.0:
                score -= 1.5 * delta
                xfer_used = True
            # Deadness prior (negative transfer): avoid actions that never changed anything.
            if visits == 0 and nochange == 0:
                gr = _global_nochange_rate(a)
                if gr > 0.0:
                    score += 1.0 * gr
                    xfer_used = True
            if stagnation >= 2 and visits == 0:
                score -= 5.0
            scored.append((score, a))
        if scored:
            scored.sort(key=lambda t: (t[0], order.index(t[1]) if t[1] in order else 99))
            best_score, best = scored[0]
            # Sequence chain overrides when it has high confidence.
            if seq_adv is not None and seq_adv[2] >= 0.6 and seq_adv[0] in acts:
                return seq_adv[0], None, None, seq_adv[1], seq_adv[2]
            if best_score <= 0.0 or stagnation >= 2:
                tag = "explorer_e4_novelty" if stagnation >= 2 else "explorer_e2_cycle"
                if xfer_used and stagnation < 2 and best_score > -5.0:
                    tag = "explorer_xfer_prior"
                return best, None, None, tag, 0.48
        if stagnation >= 3 and acts:
            return order[0], None, None, "explorer_e7_probe", 0.45
        return None
    except Exception:
        return None


def reset_game(game_id: str = "") -> None:
    """Clear per-game memory, folding no-change + productive counts into global priors."""
    try:
        gid = str(game_id or "unknown")
        with _lock:
            try:
                if _xfer_enabled():
                    global _g_games_folded
                    nc = _nochange_actions.get(gid, {})
                    for a, c in nc.items():
                        _g_nochange[int(a)] = _g_nochange.get(int(a), 0) + int(c)
                    if nc:
                        _g_games_folded += 1
                    total = sum(_g_nochange.values())
                    if total > _G_MAX_TOTAL:
                        scale = _G_MAX_TOTAL / float(total)
                        for a in list(_g_nochange):
                            _g_nochange[a] = int(_g_nochange[a] * scale)
            except Exception:
                pass
            _visits.pop(gid, None)
            _click_payoff.pop(gid, None)
            _productive_clicks.pop(gid, None)
            _nochange_actions.pop(gid, None)
    except Exception:
        pass


def stats(game_id: str = "") -> dict:
    """Learning telemetry for the TSV ledger. Never throws."""
    try:
        gid = str(game_id or "unknown")
        with _lock:
            v = _visits.get(gid, {})
            nc = _nochange_actions.get(gid, {})
            cp = _click_payoff.get(gid, {})
            pl = _productive_clicks.get(gid, [])
            g = dict(_g_nochange)
            folded = _g_games_folded
        return {
            "game": gid,
            "visits": len(v),
            "visit_mass": int(sum(v.values())),
            "nochange": dict(nc),
            "click_cells": len(cp),
            "productive_clicks": len(pl),
            "global_nochange": g,
            "global_games_folded": folded,
            "xfer_enabled": _xfer_enabled(),
        }
    except Exception:
        return {"game": str(game_id or "unknown"), "error": "stats failed open"}


def reset_all() -> None:
    try:
        with _lock:
            global _g_games_folded
            _visits.clear()
            _click_payoff.clear()
            _productive_clicks.clear()
            _nochange_actions.clear()
            _g_nochange.clear()  # full wipe includes transferred priors
            _g_games_folded = 0
    except Exception:
        pass
    try:  # hermetic: coordinated ledger resets with the registry
        from arc3sdk import zero_waste as _zw
        _zw.reset_all()
    except Exception:
        pass


def note_evidence(before: Any, action: Any, after: Any, *, level: Any = None,
                  score: Any = None, level_up: bool = False,
                  score_gain: bool = False) -> dict:
    """Record-only feed into the zero-waste ledger. Never throws, never votes."""
    try:
        from arc3sdk import zero_waste as _zw
        return _zw.ledger().observe(before, action, after, level=level,
                                    score=score, level_up=bool(level_up),
                                    score_gain=bool(score_gain))
    except Exception:
        return {"verified": False, "weight": 0.10, "spans_n": 0, "delta": 0}


def economy_bias(legal: list[int]) -> dict:
    """Advisory RHAE bias (boost verified, suppress disproven). Never throws."""
    try:
        from arc3sdk import zero_waste as _zw
        return _zw.ledger().bias(list(legal or []))
    except Exception:
        return {}


def propose_spx(game_id: str, grid: Any, failed_clicks: set | None = None,
                limit: int = 64, start: tuple = (0, 0)) -> list:
    """Exact-TSP-ordered click candidates (E8 pool, travel order). Never throws.

    Additive entry point: propose() keeps voting single actions; this returns
    the full ordered candidate list for planners. Falls back to [] when the
    SPX module is unavailable (fail-open, never a blind click).
    """
    try:
        from arc3sdk import cass_spx as _spx
        return _spx.coverage_order(grid, tried=failed_clicks, limit=limit, start=start)
    except Exception:
        return []
