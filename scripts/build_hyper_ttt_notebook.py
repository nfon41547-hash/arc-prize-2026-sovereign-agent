import json
from pathlib import Path

# Load failed-in-aimo notebook cells
orig_nb = json.load(open(r'C:\Users\gemin\Downloads\failed-in-aimo.ipynb', encoding='utf-8'))

# Construct the unified Sovereign Hyper-TTT notebook
cells = []

# Cell 0: Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ARC Prize 2026 — Sovereign Hyper-TTT Solver (ARC-AGI-2)\n",
        "## Dual-Engine: Symbolic Morphism (0.01s) + Unsloth Test-Time Training (TTT) Turbo-DFS\n",
        "**Track:** ARC-AGI-2 ($700,000 USD Category) | **Hardware:** NVIDIA L4 / L4x4"
    ]
})

# Cell 1: Global timing
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import time\n",
        "global_end_time = time.time() + 12 * 3600 - 600\n",
        "print(f'[*] Global deadline: {global_end_time}')"
    ]
})

# Cell 2: Symbolic Morphism Engine (0.01s Fast Pass)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%%writefile symbolic_morphism.py\n",
        "import numpy as np\n",
        "from typing import Any, List, Optional, Set, Tuple\n",
        "\n",
        "_MAX_DIM = 64\n",
        "\n",
        "def to_grid(data: Any) -> Optional[np.ndarray]:\n",
        "    try:\n",
        "        if not isinstance(data, list) or not data or not isinstance(data[0], list):\n",
        "            return None\n",
        "        h, w = len(data), len(data[0])\n",
        "        if h == 0 or w == 0 or h > _MAX_DIM or w > _MAX_DIM:\n",
        "            return None\n",
        "        g = np.zeros((h, w), dtype=np.uint8)\n",
        "        for y, row in enumerate(data):\n",
        "            for x, v in enumerate(row):\n",
        "                g[y, x] = max(0, min(15, int(v)))\n",
        "        return g\n",
        "    except Exception:\n",
        "        return None\n",
        "\n",
        "def to_json(g: np.ndarray) -> List[List[int]]:\n",
        "    return [[int(v) for v in row] for row in g]\n",
        "\n",
        "D4_OPS = ['identity', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'transpose', 'anti_transpose']\n",
        "\n",
        "def apply_d4(g: np.ndarray, op: str) -> np.ndarray:\n",
        "    if op == 'rot90': return np.rot90(g, -1)\n",
        "    if op == 'rot180': return np.rot90(g, 2)\n",
        "    if op == 'rot270': return np.rot90(g, 1)\n",
        "    if op == 'flip_h': return np.fliplr(g)\n",
        "    if op == 'flip_v': return np.flipud(g)\n",
        "    if op == 'transpose': return g.T\n",
        "    if op == 'anti_transpose': return np.rot90(np.fliplr(g), 1)\n",
        "    return g\n",
        "\n",
        "def learn_shape_rule(pairs):\n",
        "    if not pairs: return 'unknown', None\n",
        "    if all(b.shape == a.shape for b, a in pairs): return 'same', None\n",
        "    ratios = {(a.shape[0] // b.shape[0], a.shape[1] // b.shape[1]) for b, a in pairs if a.shape[0] % b.shape[0] == 0 and a.shape[1] % b.shape[1] == 0}\n",
        "    if len(ratios) == 1: return 'tile', ratios.pop()\n",
        "    outs = {a.shape for _, a in pairs}\n",
        "    if len(outs) == 1: return 'fixed', outs.pop()\n",
        "    return 'unknown', None\n",
        "\n",
        "def apply_shape(rule, params, tin):\n",
        "    if rule == 'same': return tin.copy()\n",
        "    if rule == 'tile':\n",
        "        kh, kw = params\n",
        "        return np.tile(tin, (kh, kw)) if tin.shape[0]*kh <= _MAX_DIM and tin.shape[1]*kw <= _MAX_DIM else None\n",
        "    if rule == 'fixed':\n",
        "        oh, ow = params\n",
        "        if oh <= tin.shape[0] and ow <= tin.shape[1]: return tin[:oh, :ow]\n",
        "        out = np.zeros((oh, ow), dtype=np.uint8)\n",
        "        out[:min(oh, tin.shape[0]), :min(ow, tin.shape[1])] = tin[:min(oh, tin.shape[0]), :min(ow, tin.shape[1])]\n",
        "        return out\n",
        "    return None\n",
        "\n",
        "def learn_color_map(pairs):\n",
        "    lut = list(range(16))\n",
        "    for b, a in pairs:\n",
        "        if b.shape != a.shape: return None\n",
        "        for bv, av in zip(b.ravel(), a.ravel()):\n",
        "            if lut[int(bv)] != int(bv) and lut[int(bv)] != int(av): return None\n",
        "            lut[int(bv)] = int(av)\n",
        "    return tuple(lut)\n",
        "\n",
        "def solve_symbolic(train_pairs, test_input):\n",
        "    tin = to_grid(test_input)\n",
        "    if tin is None: return None\n",
        "    pairs = [(to_grid(b), to_grid(a)) for b, a in train_pairs if to_grid(b) is not None and to_grid(a) is not None]\n",
        "    if not pairs: return None\n",
        "    rule, params = learn_shape_rule(pairs)\n",
        "    for d4 in D4_OPS:\n",
        "        tpairs = []\n",
        "        ok = True\n",
        "        for b, a in pairs:\n",
        "            s = apply_shape(rule, params, b)\n",
        "            if s is None: ok = False; break\n",
        "            g = apply_d4(s, d4)\n",
        "            if g.shape != a.shape: ok = False; break\n",
        "            tpairs.append((g, a))\n",
        "        if ok and tpairs:\n",
        "            cmap = learn_color_map(tpairs)\n",
        "            if cmap is not None:\n",
        "                st = apply_shape(rule, params, tin)\n",
        "                if st is not None:\n",
        "                    gt = apply_d4(st, d4)\n",
        "                    lut = np.array(cmap, dtype=np.uint8)\n",
        "                    return to_json(lut[gt])\n",
        "    return None\n"
    ]
})

# Copy the remaining core TTT cells from failed-in-aimo
for c in orig_nb['cells']:
    src = ''.join(c['source'])
    if 'arc_loader.py' in src or 'arc_decoder.py' in src or 'arc_solver.py' in src or 'starter.py' in src or 'UNSLOTH_DISABLE_STATISTICS' in src or 'from arc_loader import' in src:
        cells.append(c)

unified_nb = {
    "cells": cells,
    "metadata": orig_nb.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5
}

target_path = r'starter_push_agi2\vibe-xi-omega-arc2.ipynb'
with open(target_path, 'w', encoding='utf-8') as f:
    json.dump(unified_nb, f, indent=1)

print('Unified Sovereign Hyper-TTT notebook written successfully!')
