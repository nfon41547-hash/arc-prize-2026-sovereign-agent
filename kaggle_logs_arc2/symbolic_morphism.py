import numpy as np
from typing import Any, List, Optional, Set, Tuple

_MAX_DIM = 64

def to_grid(data: Any) -> Optional[np.ndarray]:
    try:
        if not isinstance(data, list) or not data or not isinstance(data[0], list):
            return None
        h, w = len(data), len(data[0])
        if h == 0 or w == 0 or h > _MAX_DIM or w > _MAX_DIM:
            return None
        g = np.zeros((h, w), dtype=np.uint8)
        for y, row in enumerate(data):
            for x, v in enumerate(row):
                g[y, x] = max(0, min(15, int(v)))
        return g
    except Exception:
        return None

def to_json(g: np.ndarray) -> List[List[int]]:
    return [[int(v) for v in row] for row in g]

D4_OPS = ['identity', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'transpose', 'anti_transpose']

def apply_d4(g: np.ndarray, op: str) -> np.ndarray:
    if op == 'rot90': return np.rot90(g, -1)
    if op == 'rot180': return np.rot90(g, 2)
    if op == 'rot270': return np.rot90(g, 1)
    if op == 'flip_h': return np.fliplr(g)
    if op == 'flip_v': return np.flipud(g)
    if op == 'transpose': return g.T
    if op == 'anti_transpose': return np.rot90(np.fliplr(g), 1)
    return g

def learn_shape_rule(pairs):
    if not pairs: return 'unknown', None
    if all(b.shape == a.shape for b, a in pairs): return 'same', None
    ratios = {(a.shape[0] // b.shape[0], a.shape[1] // b.shape[1]) for b, a in pairs if a.shape[0] % b.shape[0] == 0 and a.shape[1] % b.shape[1] == 0}
    if len(ratios) == 1: return 'tile', ratios.pop()
    outs = {a.shape for _, a in pairs}
    if len(outs) == 1: return 'fixed', outs.pop()
    return 'unknown', None

def apply_shape(rule, params, tin):
    if rule == 'same': return tin.copy()
    if rule == 'tile':
        kh, kw = params
        return np.tile(tin, (kh, kw)) if tin.shape[0]*kh <= _MAX_DIM and tin.shape[1]*kw <= _MAX_DIM else None
    if rule == 'fixed':
        oh, ow = params
        if oh <= tin.shape[0] and ow <= tin.shape[1]: return tin[:oh, :ow]
        out = np.zeros((oh, ow), dtype=np.uint8)
        out[:min(oh, tin.shape[0]), :min(ow, tin.shape[1])] = tin[:min(oh, tin.shape[0]), :min(ow, tin.shape[1])]
        return out
    return None

def learn_color_map(pairs):
    lut = list(range(16))
    for b, a in pairs:
        if b.shape != a.shape: return None
        for bv, av in zip(b.ravel(), a.ravel()):
            if lut[int(bv)] != int(bv) and lut[int(bv)] != int(av): return None
            lut[int(bv)] = int(av)
    return tuple(lut)

def solve_symbolic(train_pairs, test_input):
    tin = to_grid(test_input)
    if tin is None: return None
    pairs = [(to_grid(b), to_grid(a)) for b, a in train_pairs if to_grid(b) is not None and to_grid(a) is not None]
    if not pairs: return None
    rule, params = learn_shape_rule(pairs)
    for d4 in D4_OPS:
        tpairs = []
        ok = True
        for b, a in pairs:
            s = apply_shape(rule, params, b)
            if s is None: ok = False; break
            g = apply_d4(s, d4)
            if g.shape != a.shape: ok = False; break
            tpairs.append((g, a))
        if ok and tpairs:
            cmap = learn_color_map(tpairs)
            if cmap is not None:
                st = apply_shape(rule, params, tin)
                if st is not None:
                    gt = apply_d4(st, d4)
                    lut = np.array(cmap, dtype=np.uint8)
                    return to_json(lut[gt])
    return None
