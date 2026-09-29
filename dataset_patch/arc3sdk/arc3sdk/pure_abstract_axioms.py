"""Pure Abstract Axioms Codex — world-distilled reasoning priors, embedded.

Distilled from the global ARC state of the art (provenance in WORLD_SOURCES):
core knowledge priors (Chollet 2019; Spelke & Kinzler 2007), the six
reasoning categories + interactive demands of ARC-AGI-3 (Living Survey 2026,
ARC-AGI-3 paper 2026), and the winning mechanisms of ARC Prize 2024-2025:
refinement loops / test-time training (ARChitects, NVARC), evolutionary
program synthesis (Berman, Pang), zero-pretraining MDL search (TRM),
vector-symbolic binding (VSA), wake-sleep library learning (DreamCoder),
and reflective multi-agent refinement (ARCANA).

Design contract (competition-safe):
- Pure CPU/numpy, deterministic, no I/O, no network, no game-ID memory.
- Advisory only: propose() returns legal actions with capped confidence and
  NEVER throws; the caller (consensus) keeps full veto power.
- runes.json (team's real 25-game manifest) is untouched — these axioms carry
  their own provenance strings instead of polluting game evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Provenance — every axiom points at world evidence, never at a game ID.
# ---------------------------------------------------------------------------
WORLD_SOURCES: tuple[str, ...] = (
    "Chollet2019: On the Measure of Intelligence (core knowledge priors)",
    "Spelke-Kinzler2007: Core Knowledge (objectness/agentness/numerosity/geometry)",
    "ARC-Prize-2025-TR: refinement loops drive AGI progress (arXiv:2601.10904)",
    "Living-Survey-2026: 6 reasoning categories, compositional cliff (arXiv:2603.13372)",
    "ARC-AGI-3-paper-2026: interactive unknowns, efficiency scoring (arXiv:2603.24621)",
    "NVARC-2025: TTT + synthetic data + symmetry-aware scoring (1st place)",
    "TRM-2025: zero-pretraining recursive search, MDL principle (7M params)",
    "VSA-2025: vector-symbolic object binding (runner-up paper)",
    "DreamCoder: wake-sleep library learning via compression (Ellis et al.)",
    "ARCANA-2026: perceive-hypothesize-execute-reflect agents (arXiv:2607.09059)",
)


@dataclass(frozen=True)
class Axiom:
    """One frozen reasoning prior: what to detect, what it licenses."""

    id: str
    family: str  # objectness | symmetry | topology | numerosity | goalness | parsimony
    statement: str
    source: str
    action_hint: str


AXIOMS: tuple[Axiom, ...] = (
    # -- objectness (permanence, cohesion, background segregation) --
    Axiom("OBJ-PERMANENCE", "objectness",
          "Foreground objects persist; change concentrates on objects, not void.",
          "Spelke-Kinzler2007", "prefer clicks on/near the largest foreground object"),
    Axiom("OBJ-COHESION", "objectness",
          "4-connected same-color cells form one manipulable unit.",
          "Chollet2019", "treat connected components as click candidates"),
    Axiom("OBJ-BG-SEGREGATION", "objectness",
          "The majority color is inert background; agency lives in the minority.",
          "Living-Survey-2026", "never spend information budget on pure background"),
    # -- symmetry (D4 group; NVARC multi-perspective scoring) --
    Axiom("SYM-D4-INVARIANCE", "symmetry",
          " lawful scenes are invariant under the dihedral group D4 (8 views).",
          "NVARC-2025", "a cell breaking all 8-view consensus is the repair target"),
    Axiom("SYM-MULTI-VIEW", "symmetry",
          "A hypothesis must score consistently across augmented views.",
          "ARC-AGI-2-TR-2026", "corroborate a click across rotations/reflections"),
    # -- topology (containment, adjacency, holes) --
    Axiom("TOP-CONTAINMENT", "topology",
          "Enclosed voids (Betti-1) are intentional affordances awaiting fills.",
          "Chollet2019", "click smallest enclosed void first"),
    Axiom("TOP-ADJACENCY", "topology",
          "Contact graphs decide push/pull/sokoban feasibility, not pixels.",
          "Living-Survey-2026", "move toward contact, never into blockage"),
    Axiom("TOP-THINNING", "topology",
          "Skeletons preserve connectivity while discarding noise.",
          "DreamCoder", "aim for medial-axis corridors in mazes"),
    # -- numerosity (counting, parity, periodicity) --
    Axiom("NUM-COUNT-PARAM", "numerosity",
          "Counts parameterize transformations (N objects -> N steps).",
          "Living-Survey-2026", "let object count bound probe depth"),
    Axiom("NUM-PARITY", "numerosity",
          "Parity (odd/even) selects branches in toggling/permutation scenes.",
          "Chollet2019", "on parity scenes prefer the parity-consistent move"),
    Axiom("NUM-PERIODIC", "numerosity",
          "Repeating lattices extrapolate by wave-vector, not by memory.",
          "Living-Survey-2026", "continue the lattice, don't re-derive it"),
    # -- goalness (ARC-AGI-3 unknown unknowns: infer goals from interaction) --
    Axiom("GOAL-UNKNOWN", "goalness",
          "No instructions are given; goals are inferred from state change.",
          "ARC-AGI-3-paper-2026", "probe the least-understood salient cell first"),
    Axiom("GOAL-EFFICIENCY", "goalness",
          "Score is efficiency vs human baselines: every action must buy info.",
          "ARC-AGI-3-paper-2026", "information-gain per action is the currency"),
    Axiom("GOAL-REFINE-LOOP", "goalness",
          "Hypothesize -> probe -> verify -> compress, until demos agree.",
          "ARC-Prize-2025-TR", "one probe per turn; keep the loop tight"),
    # -- parsimony (TRM description-length principle) --
    Axiom("MDL-SHORTEST", "parsimony",
          "Among explanations, the shortest description wins (Occam/MDL).",
          "TRM-2025", "tie-break candidates by encoding cost, not confidence"),
    Axiom("MDL-REUSE", "parsimony",
          "Compressed libraries (DreamCoder) beat re-derivation every level.",
          "DreamCoder", "reuse proven sub-patterns across levels"),
    Axiom("VSA-BIND", "parsimony",
          "Bind role+object as one signature vector for compositional search.",
          "VSA-2025", "match (role, object) pairs, not raw pixels"),
    Axiom("REFLECT-VERIFY", "parsimony",
          "A reflector agent re-checks each plan step before execution.",
          "ARCANA-2026", "verify a click against all axiom families first"),
)

_FAMILY_OF = {a.id: a.family for a in AXIOMS}

# Confidence discipline: advisory tier sits between skills (>=0.85) and MCTS (0.80).
CODEX_CONF_CAP = 0.83


# ---------------------------------------------------------------------------
# Pure detectors (numpy only, bounded, deterministic).
# ---------------------------------------------------------------------------
def _as_grid(grid: Any) -> np.ndarray | None:
    try:
        g = np.asarray(grid)
        if g.ndim != 2 or g.size == 0:
            return None
        return g.astype(np.int64)
    except Exception:
        return None


def _bg_of(g: np.ndarray) -> int:
    try:
        return int(np.bincount(g.ravel()).argmax())
    except Exception:
        return 0


def _largest_component(g: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    """Return (cy, cx, size) of the largest 4-connected foreground component."""
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    best: tuple[int, int, int] | None = None
    for i in range(h):
        for j in range(w):
            if g[i, j] == bg or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            sy = sx = 0
            n = 0
            while stack:
                ci, cj = stack.pop()
                sy += ci
                sx += cj
                n += 1
                if ci > 0 and g[ci - 1, cj] != bg and not seen[ci - 1, cj]:
                    seen[ci - 1, cj] = True
                    stack.append((ci - 1, cj))
                if ci + 1 < h and g[ci + 1, cj] != bg and not seen[ci + 1, cj]:
                    seen[ci + 1, cj] = True
                    stack.append((ci + 1, cj))
                if cj > 0 and g[ci, cj - 1] != bg and not seen[ci, cj - 1]:
                    seen[ci, cj - 1] = True
                    stack.append((ci, cj - 1))
                if cj + 1 < w and g[ci, cj + 1] != bg and not seen[ci, cj + 1]:
                    seen[ci, cj + 1] = True
                    stack.append((ci, cj + 1))
            if n > 0 and (best is None or n > best[2]):
                best = (sy // n, sx // n, n)
    return best


def _enclosed_voids(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """Background regions unreachable from the border: (cy, cx, size)."""
    h, w = g.shape
    open_bg = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for i in range(h):
        for j in (0, w - 1):
            if g[i, j] == bg and not open_bg[i, j]:
                open_bg[i, j] = True
                stack.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i, j] == bg and not open_bg[i, j]:
                open_bg[i, j] = True
                stack.append((i, j))
    while stack:
        ci, cj = stack.pop()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < h and 0 <= nj < w and g[ni, nj] == bg and not open_bg[ni, nj]:
                open_bg[ni, nj] = True
                stack.append((ni, nj))
    seen = np.zeros((h, w), dtype=bool)
    voids: list[tuple[int, int, int]] = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != bg or open_bg[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            sy = sx = 0
            n = 0
            while stack:
                ci, cj = stack.pop()
                sy += ci
                sx += cj
                n += 1
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = ci + di, cj + dj
                    if (0 <= ni < h and 0 <= nj < w and g[ni, nj] == bg
                            and not open_bg[ni, nj] and not seen[ni, nj]):
                        seen[ni, nj] = True
                        stack.append((ni, nj))
            if n > 0:
                voids.append((sy // n, sx // n, n))
    voids.sort(key=lambda v: v[2])
    return voids


def d4_views(g: np.ndarray) -> list[np.ndarray]:
    """The 8 dihedral views (NVARC multi-perspective corroboration)."""
    try:
        return [np.rot90(g, k).copy() for k in range(4)] + [
            np.fliplr(np.rot90(g, k)).copy() for k in range(4)]
    except Exception:
        return [g]


def detect_triggers(grid: Any, bg: int | None = None) -> dict[str, float]:
    """Pure trigger strengths per axiom family in [0, 1]. Never throws."""
    out = {"objectness": 0.0, "symmetry": 0.0, "topology": 0.0,
           "numerosity": 0.0, "goalness": 0.5, "parsimony": 0.5}
    try:
        g = _as_grid(grid)
        if g is None:
            return out
        b = int(bg) if bg is not None else _bg_of(g)
        fg = g != b
        fg_ratio = float(fg.mean()) if g.size else 0.0
        if 0.0 < fg_ratio < 0.9:
            out["objectness"] = min(1.0, 0.4 + fg_ratio)
        comp = _largest_component(g, b)
        if comp is not None and comp[2] >= 2:
            out["objectness"] = max(out["objectness"], 0.75)
        voids = _enclosed_voids(g, b)
        if voids:
            out["topology"] = min(1.0, 0.5 + 0.1 * len(voids))
        views = d4_views(g)
        if len(views) == 8:
            base = views[0]
            same = 0
            for v in views[1:]:
                try:
                    if v.shape == base.shape and bool((v == base).all()):
                        same += 1
                except Exception:
                    continue
            if same >= 1:
                out["symmetry"] = min(1.0, 0.45 + 0.1 * same)
        try:
            n_colors = int(len(np.unique(g)))
            if n_colors >= 3:
                out["numerosity"] = min(1.0, 0.3 + 0.1 * n_colors)
        except Exception:
            pass
        return out
    except Exception:
        return out


def mdl_key(act: Any) -> tuple[int, int, int]:
    """Description-length proxy for tie-breaking (TRM MDL): directional < click."""
    try:
        if isinstance(act, dict):
            a = int(act.get("action", 0))
            if a == 6:
                return (1, int(act.get("x", 0)), int(act.get("y", 0)))
            return (0, a, 0)
        return (0, int(act), 0)
    except Exception:
        return (9, 0, 0)


class PureAbstractCodex:
    """Single unified entry: world axioms -> legal proposals. Fail-open."""

    @staticmethod
    def axioms(family: str | None = None) -> tuple[Axiom, ...]:
        if family is None:
            return AXIOMS
        return tuple(a for a in AXIOMS if a.family == family)

    @staticmethod
    def as_skill() -> dict[str, Any]:
        """Advisory skill descriptor; never carries action authority."""
        return {
            "codex": "pure_abstract_axioms",
            "axioms": len(AXIOMS),
            "families": sorted({_FAMILY_OF[a.id] for a in AXIOMS}),
            "sources": list(WORLD_SOURCES),
            "action_authority": "live_observation_only",
        }

    @staticmethod
    def propose(grid: Any, available: Any, *, stagnation: int = 0) -> list[tuple[Any, Any, Any, str, float]]:
        """Return legal (act, x, y, reason, conf) sorted desc. Never throws."""
        try:
            g = _as_grid(grid)
            if g is None:
                return []
            avail = list(available or [])
            if not avail:
                return []
            avail_set = set()
            for a in avail:
                try:
                    avail_set.add(int(a))
                except Exception:
                    continue
            if not avail_set:
                return []
            h, w = g.shape
            b = _bg_of(g)
            trig = detect_triggers(g, b)
            out: list[tuple[Any, Any, Any, str, float]] = []

            def _click(y: int, x: int, why: str, conf: float) -> None:
                if 6 not in avail_set:
                    return
                yy = max(0, min(h - 1, int(y)))
                xx = max(0, min(w - 1, int(x)))
                out.append((6, xx, yy, why, min(conf, CODEX_CONF_CAP)))

            def _move(a: int, why: str, conf: float) -> None:
                if a in avail_set:
                    out.append((a, None, None, why, min(conf, CODEX_CONF_CAP)))

            # OBJ: largest-object centroid (permanence + cohesion).
            if trig["objectness"] >= 0.4:
                comp = _largest_component(g, b)
                if comp is not None:
                    cy, cx, _n = comp
                    _click(cy, cx, f"codex:OBJ-PERMANENCE@{cx},{cy}", 0.80)
            # TOP: smallest enclosed void first (containment affordance).
            if trig["topology"] >= 0.5:
                voids = _enclosed_voids(g, b)
                if voids:
                    cy, cx, _n = voids[0]
                    _click(cy, cx, f"codex:TOP-CONTAINMENT@{cx},{cy}", 0.82)
            # SYM: corroborated repair cell (multi-view consensus).
            if trig["symmetry"] >= 0.45 and h == w:
                try:
                    cands: dict[tuple[int, int], int] = {}
                    base = g
                    for k in (1, 2, 3):
                        v = np.rot90(base, k)
                        if v.shape == base.shape:
                            diff = np.argwhere(v != base)
                            for dy, dx in diff[:64]:
                                cands[(int(dy), int(dx))] = cands.get((int(dy), int(dx)), 0) + 1
                    if cands:
                        (by, bx), _votes = max(cands.items(), key=lambda kv: kv[1])
                        _click(by, bx, f"codex:SYM-D4-INVARIANCE@{bx},{by}", 0.78)
                except Exception:
                    pass
            # GOAL: stagnation escalates to information probes (efficiency currency).
            try:
                stag = int(stagnation or 0)
            except Exception:
                stag = 0
            if stag >= 4:
                # Least-background salient probe: rarest color centroid.
                try:
                    vals, counts = np.unique(g, return_counts=True)
                    order = np.argsort(counts)
                    for idx in order:
                        c = int(vals[idx])
                        if c == b:
                            continue
                        ys, xs = np.where(g == c)
                        if len(xs):
                            _click(int(ys[len(ys) // 2]), int(xs[len(xs) // 2]),
                                   f"codex:GOAL-UNKNOWN@color{c}", 0.76)
                            break
                except Exception:
                    pass
            # NUM: parity scenes keep the parity-consistent directional option open.
            if trig["numerosity"] >= 0.5:
                for a in (1, 2, 3, 4):
                    if a in avail_set:
                        _move(a, "codex:NUM-COUNT-PARAM@probe", 0.70)
                        break
            # MDL tie-break: cheapest description first (stable order).
            out.sort(key=lambda p: (-p[4], mdl_key(p[0] if not isinstance(p[0], dict) else p[0])))
            seen: set[str] = set()
            uniq: list[tuple[Any, Any, Any, str, float]] = []
            for p in out:
                k = f"{p[0]}:{p[1]}:{p[2]}"
                if k not in seen:
                    seen.add(k)
                    uniq.append(p)
            return uniq[:4]
        except Exception:
            return []
