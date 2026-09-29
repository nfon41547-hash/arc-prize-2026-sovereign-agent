"""Turn-memo locks: sharing, identity, bounds, fail-open."""
import numpy as np

from arc3sdk import turn_memo as tm


def _grid():
    g = np.zeros((10, 10), dtype=np.uint8)
    g[2:5, 2:5] = 3
    return g


def test_same_grid_shares_one_object():
    tm.reset()
    a = tm.get_segment(_grid())
    b = tm.get_segment(_grid().copy())
    assert a is b
    assert tm.stats()["entries"] == 1
    assert {n["color"] for n in a["nodes"]} >= {0, 3}


def test_different_grid_new_entry_and_bound():
    tm.reset()
    tm.get_segment(_grid())
    g2 = _grid()
    g2[0, 0] = 7
    assert tm.get_segment(g2) is not tm.get_segment(_grid())
    for i in range(60):
        g = _grid()
        g[0, 0] = i % 16
        tm.get_segment(g)
    assert tm.stats()["entries"] <= 32
    tm.reset()
    assert tm.stats() == {"entries": 0}


def test_fingerprint_stable_and_shaped():
    tm.reset()
    assert tm.fingerprint(_grid()) == tm.fingerprint(_grid().copy())
    assert tm.fingerprint(_grid()) != tm.fingerprint(np.zeros((9, 9), dtype=np.uint8))
    assert tm.fingerprint(None) == "err"


def test_garbage_never_raises():
    tm.reset()
    assert tm.get_segment(None) == {"nodes": [], "adjacency_list": []}
    assert tm.get_segment("x")["nodes"] == []
    assert tm.stats()["entries"] == 0
    assert tm.__version__ == "v28-fuse-1"
