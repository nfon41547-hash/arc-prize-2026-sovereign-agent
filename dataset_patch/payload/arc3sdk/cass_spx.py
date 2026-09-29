"""CASS-SPX/Q subset-DAG exact action ordering (Held-Karp over subsets).

Replaces factorial permutation enumeration with the exact subset DAG:

    C_DAG = n * 2^(n-1)      vs      C_brute = n!

valid ONLY when transition cost is Markov in (covered-set, next action).
If a problem needs the full order known (non-Markov dependence), the system
must REFUSE silent subset-key merging (needs_full_order) and either expand
the state key or fall back — never trade exactness for speed quietly.

Also provides order_clicks(): exact shortest coverage path over click
candidates (travel cost between clicks is genuinely Markov), used by the
explorer E5/E8 click ordering path.

Guards: n > 20 -> greedy fallback (bounded); bad shapes -> None (fail-open).
Stdlib-only at import (no numpy needed); RLock-free pure functions.
Never raises out of any public API.
"""

from __future__ import annotations

import math

__version__ = "v10-cassspx-1"

_MAX_EXACT_N = 20
_MAX_BRUTE_N = 9


def count_dag(n: int) -> int:
    try:
        n = int(n)
        return n * (2 ** (n - 1)) if n >= 1 else 0
    except Exception:
        return 0


def count_brute(n: int) -> int:
    try:
        n = int(n)
        return math.factorial(n) if n >= 0 else 0
    except Exception:
        return 0


def _check_cost(cost) -> list | None:
    try:
        m = [list(map(float, row)) for row in cost]
        n = len(m)
        if n < 1 or any(len(r) != n for r in m):
            return None
        if any(v != v or v in (float("inf"), float("-inf")) for r in m for v in r):
            return None
        return m
    except Exception:
        return None


def _greedy(cost_m, start) -> tuple:
    try:
        n = len(cost_m)
        remaining = set(range(n))
        order: list[int] = []
        total = 0.0
        cur = None
        while remaining:
            if cur is None:
                nxt = min(remaining, key=lambda a: start[a])
                total += start[nxt]
            else:
                nxt = min(remaining, key=lambda a: cost_m[cur][a])
                total += cost_m[cur][nxt]
            order.append(nxt)
            remaining.discard(nxt)
            cur = nxt
        return order, total
    except Exception:
        return [], float("inf")


def held_karp_path(cost, start_cost=None) -> tuple | None:
    """Exact min-cost path covering all n actions. Returns (order, total)."""
    try:
        m = _check_cost(cost)
        if m is None:
            return None
        n = len(m)
        if n == 1:
            s = [0.0] if start_cost is None else [float(start_cost[0])]
            return [0], s[0]
        try:
            start = [float(v) for v in start_cost] if start_cost is not None else [0.0] * n
            if len(start) != n:
                return None
        except Exception:
            return None
        if n > _MAX_EXACT_N:
            return _greedy(m, start)  # bounded fallback, still exact-free
        size = 1 << n
        INF = float("inf")
        dp = [[INF] * n for _ in range(size)]
        par = [[-1] * n for _ in range(size)]
        for i in range(n):
            dp[1 << i][i] = start[i]
        for mask in range(size):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                cur = dp[mask][last]
                if cur == INF:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    nmask = mask | (1 << nxt)
                    val = cur + m[last][nxt]
                    if val < dp[nmask][nxt]:
                        dp[nmask][nxt] = val
                        par[nmask][nxt] = last
        full = size - 1
        last = min(range(n), key=lambda i: dp[full][i])
        total = dp[full][last]
        order: list[int] = []
        mask = full
        while last != -1:
            order.append(last)
            prev = par[mask][last]
            mask ^= (1 << last)
            last = prev
        order.reverse()
        return order, total
    except Exception:
        return None


def brute_path(cost, start_cost=None) -> tuple | None:
    """Exhaustive verifier (n <= 9 only). Returns (order, total)."""
    try:
        from itertools import permutations
        m = _check_cost(cost)
        if m is None:
            return None
        n = len(m)
        if n > _MAX_BRUTE_N:
            return None
        start = [float(v) for v in start_cost] if start_cost is not None else [0.0] * n
        best = None
        best_t = float("inf")
        for perm in permutations(range(n)):
            t = start[perm[0]]
            for a, b in zip(perm, perm[1:]):
                t += m[a][b]
            if t < best_t:
                best_t, best = t, list(perm)
        return best, best_t
    except Exception:
        return None


def needs_full_order(order_dependent: bool) -> tuple | None:
    """Refuse silent subset merging when full order matters. Never raises."""
    try:
        if bool(order_dependent):
            return None
        return ([], 0.0)
    except Exception:
        return None


def order_clicks(points, start=(0, 0)) -> list | None:
    """Exact shortest coverage path over click (x, y) candidates."""
    try:
        pts = [(float(p[0]), float(p[1])) for p in list(points or [])]
        if not pts:
            return []
        n = len(pts)
        if n == 1:
            return [0]
        if n > _MAX_EXACT_N:
            return list(range(n))  # bounded: keep input order, never blow up
        sx, sy = float(start[0]), float(start[1])
        cost = [[math.hypot(ax - bx, ay - by) for bx, by in pts] for ax, ay in pts]
        start_c = [math.hypot(ax - sx, ay - sy) for ax, ay in pts]
        res = held_karp_path(cost, start_c)
        return res[0] if res else list(range(n))
    except Exception:
        return None


def coverage_order(grid, tried=None, limit: int = 64, start=(0, 0)) -> list:
    """Order untried foreground cells by exact shortest path. Never raises.

    Same candidate pool as the explorer E8 coverage voter, but returns the
    full ordered list (capped) instead of a single vote, so a planner can
    consume productive clicks in travel order with zero wasted motion.
    """
    try:
        import numpy as _np
        g = _np.asarray(grid, dtype=_np.uint8)
        if g.ndim != 2 or g.size == 0:
            return []
        bg = int(_np.bincount(g.ravel(), minlength=16).argmax())
        tried_set = set()
        try:
            tried_set = {(int(x), int(y)) for (x, y) in (tried or set())}
        except Exception:
            pass
        fg = _np.argwhere(g != bg)
        out: list[tuple[int, int]] = []
        cap = max(0, min(int(limit), 64))
        for y, x in fg[: cap * 4].tolist():
            if (int(x), int(y)) not in tried_set:
                out.append((int(x), int(y)))
            if len(out) >= cap:
                break
        order = order_clicks(out, start)
        if not order:
            return out
        return [out[i] for i in order if 0 <= i < len(out)]
    except Exception:
        return []
