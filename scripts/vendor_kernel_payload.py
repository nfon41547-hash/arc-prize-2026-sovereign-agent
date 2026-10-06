"""Vendor arc3sdk payload into the sovereign kernel notebook (lean rebuild).

Single source of truth: arc3sdk/*.py -> notebook vendor cell bytes ==
dataset patch bytes (verified byte-identical by test + release manifest).

Usage:
  py -3.12 scripts/vendor_kernel_payload.py          # regenerate notebook
  py -3.12 scripts/vendor_kernel_payload.py --check  # verify only
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IPYNB = ROOT / "starter_push" / "arc-agi-3-starter-kernel-v32-profile-3.ipynb"

MODULE_FILES = [
    "__init__.py",
    "v32_hardening.py",
    "offline_guard.py",
    "exploration_registry.py",
    "competition_contract.py",
    "egcd.py",
    "kaggle_native_adapter.py",
    "cass_xi/__init__.py",
    "cass_xi/active.py",
    "cass_xi/adapters.py",
    "cass_xi/intelligence.py",
    "cass_xi/llm_benchmark.py",
    "cass_xi/pareto.py",
    "cass_xi/profile.py",
    "cass_xi/surrogate.py",
    "cass_xi_wiring.py",
    "zero_waste.py",
    "cass_spx.py",
    "merge_cert.py",
    "gacl_select.py",
    "sovereign_duck_interceptor.py",
    "first_contact_learner.py",
    "long_horizon_controller.py",
    "rl_replay_buffer.py",
    "evidence_planner.py",
    "runtime_safety_contract.py",
    "realtime_abstract_cortex.py",
    "spectral_topological_reasoning.py",
    "causal_chain_reasoner.py",
    "algebraic_planning_engine.py",
    "fusion_supremacy.py",
    "taaf_stepenv_hook.py",
    "cass_phi.py",
    # v22 full decide chain (v27 lesson: working copy shadows dataset
    # mount for regular packages; anything outside the payload is dead).
    "unified_consensus_engine.py",
    "sovereign_memory_vram.py",
    "nvidia_dsl.py",
    "mcts_planner.py",
    "world_model_simulator.py",
    "human_cognitive_perception.py",
    "pure_abstract_axioms.py",
    "skills.py",
    "rune.py",
    "grid.py",
    "oracle.py",
    "principles.py",
    "sim.py",
    "category_engine.py",
    "skill_orchestrator.py",
    "skill_derived_profile.py",
    "object_segmentation.py",
    "hypothesis_ledger.py",
    "flux_search.py",
    "turn_memo.py",
    # v2-skill-orchestrator-2: full SOTA technique fusion.
    "algebraic_planning_engine.py",
    "causal_chain_reasoner.py",
    "fusion_supremacy.py",
    "spectral_topological_reasoning.py",
    "realtime_abstract_cortex.py",
    "world_model_simulator.py",
    "mcts_planner.py",
    # Original Algorithm: Transition Learner (learns from transitions directly).
    "transition_learner.py",
    # Meta-Evolution: Self-improving system coordinated with LLM feedback.
    "meta_evolution.py",
    # Φ-EVO: Phenomenological Self-Transcending Evolution Engine.
    "phi_evo.py",
    # Φ-EVO (evo_phi): MetricLiveness + HiddenDimensionMiner + OperatorGenome.
    "evo_phi.py",
    # v23-skills-1: ARC-specific skills wired into the decision chain.
    "abstraction_skill_registry.py",
    "arc_color_transform.py",
    "arc_object_tracker.py",
    "arc_pattern_completion.py",
    # V6 hierarchical controller & cognitive utility
    "cognitive_utility.py",
    "sobu_v6.py",
    "priority_scheduler_dprime.py",
    "duck_sobu_omega.py",
    # Fable layer (handbook harness core & patch)
    "fable_layer_core.py",
    "fable_layer_patch.py",
    "hindsight_pruner.py",
    "adsd_engine.py",
    "memadapter_engine.py",
    "agent_instruct_reasoner.py",
]

PAYLOAD_VERSION = "v28-fuse-1"
HOOK_VERSION = "taaf-stepenv-1"


def _pack(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data, 9)).decode("ascii")


def build_payloads() -> dict:
    payloads = {}
    for name in MODULE_FILES:
        data = (ROOT / "arc3sdk" / name).read_bytes()
        payloads[name] = _pack(data)
    return payloads


def _payload_digest(payloads: dict) -> str:
    # Version rides INSIDE the digest: a version bump alone forces refresh,
    # so the worker-visible label can never go stale on identical bytes.
    return hashlib.sha256(json.dumps(
        {"version": PAYLOAD_VERSION, "payloads": payloads},
        sort_keys=True).encode("ascii")).hexdigest()[:16]


def _cell_hash(src: str) -> str:
    import re as _re
    m = _re.search(r'_VENDOR_HASH = "([0-9a-f]{16})"', src)
    return m.group(1) if m else ""


def _hook_version(src: str) -> str:
    import re as _re
    m = _re.search(r"# hook v=([^\s]+)", src)
    return m.group(1) if m else ""


def _split_template(cell_src: str) -> tuple[str, str]:
    """Split existing vendor cell into (prolog, epilog) around _VENDOR dict."""
    marker = "_VENDOR = "
    i = cell_src.find(marker)
    if i < 0:
        raise ValueError("no _VENDOR dict in cell")
    prolog = cell_src[:i + len(marker)]
    # dict literal ends at the matching closing brace on its line-block:
    # parse from the dict start with ast to find the exact end.
    tail = cell_src[i + len(marker):]
    # the dict is one giant line; find the line end, then parse incrementally
    # (robust: scan for balanced braces).
    depth = 0
    instr = False
    esc = False
    for pos, ch in enumerate(tail):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return prolog, tail[pos + 1:]
    raise ValueError("unbalanced _VENDOR dict")


def make_vendor_cell(payloads: dict) -> dict:
    digest = _payload_digest(payloads)
    nb = json.loads(IPYNB.read_text(encoding="utf-8"))
    old_src = None
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        s = "".join(c.get("source", []))
        if "vendor-arc3sdk ok" in s:
            old_src = s
            break
    if old_src is None:
        raise ValueError("no vendor cell in notebook")
    prolog, epilog = _split_template(old_src)
    import re as _re
    epilog = _re.sub(r'_VENDOR_VERSION = "[^"]*"',
                     f'_VENDOR_VERSION = "{PAYLOAD_VERSION}"', epilog)
    epilog = _re.sub(r'_VENDOR_HASH = "[0-9a-f]*"',
                     f'_VENDOR_HASH = "{digest}"', epilog)
    src = prolog + json.dumps(payloads) + epilog
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def _find(cells: list, marker: str) -> int | None:
    return next((i for i, c in enumerate(cells)
                 if marker in "".join(c.get("source", []))), None)


def patch_notebook(write: bool = True) -> dict:
    nb = json.loads(IPYNB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    payloads = build_payloads()
    expected = _payload_digest(payloads)
    info = {"refreshed": False, "cells": len(cells),
            "version": PAYLOAD_VERSION}
    idx = _find(cells, "vendor-arc3sdk ok")
    if idx is None:
        raise ValueError("no vendor cell to refresh")
    cur = "".join(cells[idx].get("source", []))
    if _cell_hash(cur) != expected:
        old_id = cells[idx].get("id")
        cells[idx] = make_vendor_cell(payloads)
        if old_id is not None:
            cells[idx]["id"] = old_id
        info["refreshed"] = True
    if write and info["refreshed"]:
        IPYNB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        info["wrote"] = str(IPYNB)
    return info


def main() -> int:
    check = "--check" in sys.argv[1:]
    if check:
        nb = json.loads(IPYNB.read_text(encoding="utf-8"))
        payloads = build_payloads()
        expected = _payload_digest(payloads)
        for c in nb["cells"]:
            if c.get("cell_type") != "code":
                continue
            s = "".join(c.get("source", []))
            if "vendor-arc3sdk ok" in s:
                h = _cell_hash(s)
                ok = (h == expected)
                print(f"vendor check: {'OK' if ok else 'STALE'} "
                      f"cell={h} live={expected} version={PAYLOAD_VERSION}")
                return 0 if ok else 1
        print("vendor check: NO VENDOR CELL")
        return 1
    info = patch_notebook(write=True)
    print(f"vendor_kernel_payload {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
