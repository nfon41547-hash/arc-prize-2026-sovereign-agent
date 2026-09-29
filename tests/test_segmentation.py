"""Exact segmentation locks: topology, invariance, wiring, fail-open."""
import numpy as np
import pytest

from arc3sdk import object_segmentation as seg
from arc3sdk import fusion_supremacy as fs


@pytest.fixture(autouse=True)
def _clean():
    fs.reset_shared()
    yield
    fs.reset_shared()


def _lshape(dx=0, dy=0):
    g = np.zeros((8, 8), dtype=np.uint8)
    g[1 + dy, 1 + dx] = 3
    g[2 + dy, 1 + dx] = 3
    g[2 + dy, 2 + dx] = 3
    return g


def _donut():
    g = np.zeros((8, 8), dtype=np.uint8)
    g[2:6, 2:6] = 3
    g[3:5, 3:5] = 0
    g[3, 4] = 2
    return g


def _ushape():
    # U open at the top: concavity cell (2,3) is NOT enclosed.
    g = np.zeros((8, 8), dtype=np.uint8)
    g[2:6, 2] = 3
    g[5, 2:5] = 3
    g[2:6, 4] = 3
    g[3, 3] = 2
    return g


def test_translation_invariant_hash():
    ha = {n["hash"] for n in seg.segment(_lshape())["nodes"] if n["color"] == 3}
    hb = {n["hash"] for n in seg.segment(_lshape(dx=3, dy=2))["nodes"] if n["color"] == 3}
    assert ha and ha == hb


def test_rotation_is_distinct_object():
    # Translation-only identity (documented): rotation changes the hash.
    ha = {n["hash"] for n in seg.segment(_lshape())["nodes"] if n["color"] == 3}
    hr = {n["hash"] for n in seg.segment(np.rot90(_lshape(), k=1))["nodes"] if n["color"] == 3}
    assert ha and hr and ha != hr


def test_exact_containment_donut_vs_u():
    d = seg.segment(_donut())
    ring = next(n for n in d["nodes"] if n["color"] == 3 and n["pixels"] > 4)
    inner = next(n for n in d["nodes"] if n["color"] == 2)
    assert inner["id"] in ring["children"]
    u = seg.segment(_ushape())
    wall = next(n for n in u["nodes"] if n["color"] == 3 and n["pixels"] > 4)
    concavity = next(n for n in u["nodes"] if n["color"] == 2)
    assert concavity["id"] not in wall["children"]
    # bbox approximation would have failed the U case (strict bbox nesting
    # holds while no topological enclosure exists).


def test_adjacency_pairs():
    g = np.zeros((6, 6), dtype=np.uint8)
    g[1:3, 1:3] = 3
    g[1:3, 3:5] = 4
    s = seg.segment(g)
    ids = {n["color"]: n["id"] for n in s["nodes"]}
    assert [ids[3], ids[4]] in s["adjacency_list"] or \
        [ids[4], ids[3]] in s["adjacency_list"]


def test_fusion_objects_exact_children():
    objs = fs.objects(_donut())
    ring = next(o for o in objs if o["color"] == 3 and o["pixels"] > 4)
    inner_idx = next(i for i, o in enumerate(objs) if o["color"] == 2)
    assert inner_idx in ring["children"]
    objs_u = fs.objects(_ushape())
    wall = next(o for o in objs_u if o["color"] == 3 and o["pixels"] > 4)
    conc_idx = next(i for i, o in enumerate(objs_u) if o["color"] == 2)
    assert conc_idx not in wall["children"]
    # stable contract keys preserved
    assert set(("sig", "color", "pixels", "centroid", "bbox", "children")) \
        <= set(objs[0].keys())


def test_match_contract():
    # Foreground shift: L keeps its hash (moved via boundary), while the
    # background component legitimately changes shape -> appears/disappears.
    m = seg.match(_lshape(), _lshape(dx=1))
    assert m["n_before"] == m["n_after"] == 2
    assert m["moved"] == 1
    assert m["appeared"] and m["disappeared"]
    m2 = seg.match(_lshape(), _lshape())
    assert m2["moved"] == 0 and m2["stationary"] == m2["n_before"] == 2
    assert m2["appeared"] == [] and m2["disappeared"] == []


def test_bounds_and_garbage():
    assert seg.segment(None) == {"nodes": [], "adjacency_list": []}
    assert seg.segment(np.zeros((100, 100), dtype=np.uint8))["nodes"] == []
    assert seg.segment("garbage")["nodes"] == []
    assert seg.match(None, None)["n_before"] == 0
    assert seg.segment(np.zeros((4, 4), dtype=np.uint8))["nodes"]
    assert seg.__version__ == "v24-seg-1"
