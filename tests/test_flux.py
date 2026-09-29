"""Xi-FLUX locks: boundary commit, abstention, budget, scope, fail-open."""
import numpy as np
import pytest

from arc3sdk import flux_search as fx


@pytest.fixture(autouse=True)
def _clean():
    yield


def _sokoban():
    # Player 8 left of goal 9: RIGHT yields surrogate reward + terminal.
    g = np.zeros((8, 8), dtype=np.uint8)
    g[4, 2] = 8
    g[4, 3] = 9
    return g


def _flat():
    return np.zeros((8, 8), dtype=np.uint8)


def test_commit_clear_winner_with_certificate():
    p = fx.FluxPlanner()
    out = p.plan(_sokoban(), legal=[1, 2, 3, 4, 5], game_id="t", level=0)
    assert out is not None
    assert out["action"] == 4 and out["reason"] == "flux_interval_commit"
    cert = out["certificate"]
    assert cert["verifier"] == "PASS" and cert["open_edges"] == 0
    assert cert["lcb"] > cert["best_rival_ucb"]
    assert out["stats"]["real_sims"] <= 8
    assert 0.80 <= out["confidence"] <= 0.95


def test_abstain_on_flat_grid():
    p = fx.FluxPlanner()
    assert p.plan(_flat(), legal=[1, 2, 3, 4, 5], game_id="t") is None
    assert isinstance(p.last_stats, dict) and p.last_stats["candidates"] == 5


def test_no_candidates_or_bad_input():
    p = fx.FluxPlanner()
    assert p.plan(_sokoban(), legal=[6], game_id="t") is None
    assert p.plan(None, legal=[1], game_id="t") is None
    assert p.plan("garbage", legal=[1], game_id="t") is None
    assert p.plan(np.zeros((100, 100), dtype=np.uint8), legal=[1]) is None
    assert p.plan(_sokoban(), legal=[], game_id="t") is None


def test_branch_memory_cache_and_scope():
    p = fx.FluxPlanner()
    p.plan(_sokoban(), legal=[1, 2, 3, 4, 5], game_id="t", level=0)
    first_sims = p.last_stats["real_sims"]
    p.plan(_sokoban(), legal=[1, 2, 3, 4, 5], game_id="t", level=0)
    assert p.last_stats["cache_hits"] > 0
    assert p.last_stats["real_sims"] <= first_sims
    p.plan(_sokoban(), legal=[1, 2, 3, 4, 5], game_id="t", level=1)
    assert len(p._mem) <= 5  # scope change cleared stale intervals


def test_fatal_action_never_committed():
    p = fx.FluxPlanner()
    out = p.plan(_sokoban(), legal=[1, 2, 3, 4, 5], game_id="t",
                 fatal_states={("move", 4)})
    if out is not None:
        assert out["action"] != 4


def test_budget_bound_and_mem_bound():
    p = fx.FluxPlanner(max_real_sims=2)
    for i in range(40):
        g = _flat()
        g[1, 1] = 8
        g[6, 6] = 9
        p.plan(g, legal=[1, 2, 3, 4, 5], game_id=f"g{i % 3}", level=0)
    assert len(p._mem) <= 512
    assert p.last_stats["real_sims"] <= 2


def test_version():
    assert fx.__version__ == "v27-flux-1"
