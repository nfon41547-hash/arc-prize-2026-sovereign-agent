"""CASS-Phi execution law locks: gate math, audit numbers, M-, metrics."""
import os

import numpy as np
import pytest

from arc3sdk import cass_phi as phi


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {}
    for k in ("ARC3_PHI", "ARC3_PHI_NU", "ARC3_PHI_KAPPA", "ARC3_PHI_ETA",
              "ARC3_PHI_ALPHA", "ARC3_PHI_BETA"):
        if k in os.environ:
            keep[k] = os.environ.pop(k)
    yield
    for k in ("ARC3_PHI", "ARC3_PHI_NU", "ARC3_PHI_KAPPA", "ARC3_PHI_ETA",
              "ARC3_PHI_ALPHA", "ARC3_PHI_BETA"):
        os.environ.pop(k, None)
    os.environ.update(keep)
    phi.reset_shared()


def _grid(v=3):
    g = np.zeros((10, 10), dtype=np.uint8)
    g[2:5, 2:5] = v
    return g


def _turn(pl, grid, aid=1, upred=0.9, gid="g", level=0):
    b = np.ascontiguousarray(grid).tobytes()
    h, w = grid.shape
    pre = phi.state_fp(b)
    pl.observe_turn(gid, level, pre, b, h, w, aid, None, None, upred, "t")
    return pre, b, h, w


def test_gate_allow_and_blocks():
    pl = phi.PhiLedger()
    assert pl.check("g", 0, "s", 1, 0.9, 0.8)["allow"] is True
    assert pl.check("g", 0, "s", 1, 0.8, 0.8)["allow"] is False
    assert pl.check("g", 0, "s", 1, None, 0.8)["allow"] is False


def test_audit_productive_numbers():
    pl = phi.PhiLedger()
    g = _grid()
    _turn(pl, g, aid=1, upred=0.9)
    post = g.copy()
    post[4, 4] = 7
    art = pl.audit_turn("g", 0, phi.state_fp(post.tobytes()), post.tobytes())
    assert art["U_obs"] == 1.5 and art["dead"] == 0
    assert art["reward"] == 1.5 and art["delta"] == "changed"


def test_audit_static_then_repeat():
    pl = phi.PhiLedger()
    g = _grid()
    _turn(pl, g, aid=2, upred=0.9)
    a1 = pl.audit_turn("g", 0, phi.state_fp(g.tobytes()), g.tobytes())
    assert a1["U_obs"] == 0.4 and a1["dead"] == 0  # first no-op informs
    _turn(pl, g, aid=2, upred=0.9)
    a2 = pl.audit_turn("g", 0, phi.state_fp(g.tobytes()), g.tobytes())
    assert a2["U_obs"] == -0.1 and a2["dead"] == 1
    assert a2["dead_dup"] == 1 and a2["reward"] == -3.1


def test_negative_memory_suppress_revive():
    pl = phi.PhiLedger()
    g = _grid()
    for _ in range(3):
        _turn(pl, g, aid=4)
        pl.audit_turn("g", 0, phi.state_fp(g.tobytes()), g.tobytes())
    pre = phi.state_fp(g.tobytes())
    assert pl.check("g", 0, pre, 4, 0.99, 0.8)["vetoed"] is True
    assert pl.check("g", 0, "other", 4, 0.99, 0.8)["vetoed"] is False
    pl.on_level("g", 1)
    assert pl.check("g", 0, pre, 4, 0.99, 0.8)["vetoed"] is False


def test_metrics_and_bounds():
    pl = phi.PhiLedger()
    g = _grid()
    for i in range(700):
        _turn(pl, g, aid=i % 7, gid="zz")
        pl.audit_turn("zz", 0, phi.state_fp(g.tobytes()), g.tobytes())
    assert len(pl._artifacts) <= 512 and len(pl._neg) <= 1024
    m = pl.metrics("zz")
    assert m["n"] == 512 and m["UAR"] < 1.0
    phi.fingerprint(None, None, None, None)
    pl.check(None, None, None, None, None)
    assert pl.audit_turn("nope", 0, "x", b"y") is None
    pl.reset()
    assert pl.metrics()["n"] == 0
