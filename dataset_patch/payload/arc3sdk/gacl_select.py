"""GACL joint search+hardware selection over MEASURED options.

Candidates are whole pairs (search x backend), scored by:

    o* = argmax_o ( E[dV_o] - lam * Risk(o) ) / C(o)

    C(o) = N_expanded * C_model/backend + C_merge + C_verify + C_semantic

Options below use measured latencies where measured (provenance strings);
unverified backends (UMSS native/GPU here) are EXCLUDED by default and only
enter when the caller explicitly passes verified_backends — the selector can
refuse to optimize on unproven hardware, same doctrine as merge_cert.

Stdlib-only at import. Pure functions. Never raises.
"""

from __future__ import annotations

__version__ = "v10-cassspx-1"

# Measured latencies (ms) with provenance. Consensus/explorer/MCTS numbers
# come from this repo's own locks (performance_test, serving_tune, audit);
# duck-llm is the field-typical round-trip (watchdog ceiling 90s).
OPTIONS = {
    "consensus-fast": {"ms": 30.0, "provenance": "consensus 13.6ms + verify margin",
                       "verified": True},
    "explorer": {"ms": 5.0, "provenance": "registry propose budget (typical <1ms)",
                 "verified": True},
    "mcts": {"ms": 25.0, "provenance": "performance_test MCTS max_time_ms=25",
             "verified": True},
    "duck-llm": {"ms": 2500.0, "provenance": "field-typical round-trip; ceiling 90s watchdog",
                 "verified": True},
}

BACKENDS = {
    "dense": {"factor": 1.0, "provenance": "live bf16/NVFP4 serving", "verified": True},
    "umss": {"factor": 0.4, "provenance": "UNMEASURED here (external UMSS claim only)",
             "verified": False},
}


def option_cost(search: str, backend: str = "dense", n_expanded: float = 1.0,
                c_merge: float = 0.0, c_verify: float = 0.0,
                c_semantic: float = 0.0) -> float | None:
    """C(o) in ms. Returns None for unknown/unverified-without-opt-in."""
    try:
        opt = OPTIONS.get(str(search))
        be = BACKENDS.get(str(backend))
        if opt is None or be is None:
            return None
        return (max(0.0, float(n_expanded)) * float(opt["ms"]) * float(be["factor"])
                + max(0.0, float(c_merge)) + max(0.0, float(c_verify))
                + max(0.0, float(c_semantic)))
    except Exception:
        return None


def select(e_dv: dict, risk: dict, lam: float = 1.0,
           backend: str = "dense", verified_only: bool = True,
           costs: dict | None = None) -> tuple | None:
    """Pick o* over search options. Returns (best, table) or None."""
    try:
        lam = max(0.0, float(lam))
        if verified_only and str(backend) in BACKENDS and not BACKENDS[str(backend)]["verified"]:
            return None  # refuse unproven hardware
        table: dict[str, float] = {}
        offered = set(e_dv or {}) | set(risk or {})
        try:
            offered |= {k for (k, _b) in (costs or {})}
        except Exception:
            pass
        names = [o for o in OPTIONS if o in offered]
        if not names:
            if offered:
                return None  # caller offered nothing usable: refuse, don't invent
            names = list(OPTIONS)  # no offers at all: degrade to cheapest known
        for name in names:
            try:
                dv = float((e_dv or {}).get(name, 0.0))
                rk = max(0.0, float((risk or {}).get(name, 1.0)))
                c = (costs or {}).get((name, str(backend)))
                if c is None:
                    c = option_cost(name, str(backend))
                if c is None or c <= 0:
                    continue
                table[name] = (dv - lam * rk) / c
            except Exception:
                continue
        if not table:
            return None
        best = max(table, key=table.get)
        return best, {k: round(v, 6) for k, v in table.items()}
    except Exception:
        return None
