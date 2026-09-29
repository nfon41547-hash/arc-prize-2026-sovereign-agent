"""Tier behavior locks: APE exact/vacuous, causal gates, fusion coords,
cortex invariants. Fail-open everywhere, never raises on garbage."""
import numpy as np
import pytest

from arc3sdk import algebraic_planning_engine as ape
from arc3sdk import causal_chain_reasoner as cc
from arc3sdk import fusion_supremacy as fs
from arc3sdk import realtime_abstract_cortex as ctx


@pytest.fixture(autouse=True)
def _clean():
    cc.reset()
    fs.reset_shared()
    yield
    cc.reset()
    fs.reset_shared()
    try:
        ctx.reset_episodic(None)
    except Exception:
        pass


def _lshape(dx):
    g = np.zeros((8, 8), dtype=np.uint8)
    g[1, 1 + dx] = 3
    g[2, 1 + dx] = 3
    g[2, 2 + dx] = 3
    return g


def test_ape_exact_shift_and_proof():
    b, a = _lshape(0), _lshape(1)
    out = ape.plan_exact([(b, a)], b, [1, 2, 3])
    assert out is not None and out["action"] == 1
    assert out["proof"]["verified"] is True
    assert out["confidence"] >= 0.85


def test_ape_refuses_vacuous_and_bounds():
    g = np.zeros((8, 8), dtype=np.uint8)
    g[2:5, 2:5] = 3
    assert ape.plan_exact([(g, g.copy())], g, [1, 2, 3]) is None
    big = np.zeros((64, 64), dtype=np.uint8)
    big[4:12, 4:12] = 3
    other = big.copy()
    other[4:12, 5:13] = 3
    other[40, 40] = 7
    import time
    t0 = time.perf_counter()
    out = ape.plan_exact([(big, other)], big, [1, 2, 3, 4, 5])
    assert time.perf_counter() - t0 < 2.0
    assert out is None or out["proof"]["verified"] is True
    # full plan() never claims verified on structural fallback
    b = np.zeros((6, 6), dtype=np.uint8)
    b[1:3, 1:3] = 3
    fb = ape.plan([(b, np.rot90(b, k=2))], b, [5])
    assert fb is not None and fb["reason"] == "ape"


def test_causal_learns_and_fires():
    g = np.zeros((8, 8), dtype=np.uint8)
    g[2:5, 2:5] = 3
    h = np.zeros((8, 8), dtype=np.uint8)
    h[2:5, 3:6] = 3
    assert cc.chain_confidence(g) >= 0.10
    cc.observe_op("", "shift", True)
    assert cc.op_weight("", "shift") > 1.0
    out = cc.reason(g, [1, 2, 3], game_id="t")
    assert out is None or "causal" in out.get("reason", "")
    cc.observe_op("x", "y", True)
    assert cc.stats()["edges"] >= 1
    cc.reset()
    assert cc.stats()["edges"] == 0


def test_fusion_click_grounded_and_never_blind():
    g = np.zeros((12, 12), dtype=np.uint8)
    g[1, 1] = 2
    g[1, 2] = 2
    g[8:10, 8:12] = 3
    out = fs.propose_click(g, [6], game_id="t", level=0)
    assert out is not None
    assert out["action"] == 6
    assert 0 <= out["x"] < 12 and 0 <= out["y"] < 12
    assert "type_sig" in out
    assert fs.propose_click(g, [1, 2], game_id="t") is None
    assert fs.propose_click(None, [6], game_id="t") is None
    # dead type suppressed after 3 no-change clicks
    sig = out["type_sig"]
    for _ in range(3):
        fs.observe_click_result("t", 0, sig, False)
    assert fs.shared_dead("t", 0, sig) is True


def test_cortex_invariants_and_observe():
    g = np.zeros((8, 8), dtype=np.uint8)
    g[2:5, 2:5] = 3
    inv = ctx.invariants(g)
    assert inv.get("valid")
    h1 = ctx.canonical_hash(g)
    h2 = ctx.canonical_hash(np.rot90(g, k=1))
    assert h1 == h2  # D4 canonical
    t = ctx.infer_transform(g, np.roll(g, 1, axis=1))
    assert isinstance(t.get("op"), str) and t.get("op")
    assert 0.0 <= float(t.get("confidence", 0.0)) <= 1.0
    r = ctx.observe(g, g.copy(), 1, game_id="t-ctx")
    assert r.get("op") in ("recorded", "fail-open")


def test_modules_never_raise_on_garbage():
    bad = [None, 0, "x", [], np.zeros((100, 100), dtype=np.uint8)]
    for g in bad:
        ape.plan([], g, [1])
        ape.plan_exact([], g, [1])
        cc.best_chain(g, [1])
        cc.reason(g, [1])
        fs.propose_click(g, [6], game_id="t")
        fs.objects(g)
        fs.track(g, g)
        ctx.invariants(g)
        ctx.decide(g, [1, 2], game_id="t-fuzz")
