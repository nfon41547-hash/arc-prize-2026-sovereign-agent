"""Dead code audit: find modules never imported by the decision chain.

Scans arc3sdk/*.py and checks which modules are never imported by:
- skill_orchestrator.py
- taaf_stepenv_hook.py
- unified_consensus_engine.py
- Any other module in the payload MODULE_FILES list

Output: list of dead modules (candidates for removal or wiring).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "arc3sdk"


def _imports_of(path: Path) -> set[str]:
    """Extract all imported module names from a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.level or 0) == 0:
                out.add(node.module.split(".")[0])
    return out


def main() -> int:
    # Modules in the payload (from vendor_kernel_payload.py)
    payload_modules = {
        "v32_hardening", "offline_guard", "exploration_registry",
        "competition_contract", "egcd", "kaggle_native_adapter",
        "cass_xi", "cass_xi_wiring", "zero_waste", "cass_spx",
        "merge_cert", "gacl_select", "sovereign_duck_interceptor",
        "first_contact_learner", "long_horizon_controller",
        "rl_replay_buffer", "evidence_planner", "runtime_safety_contract",
        "realtime_abstract_cortex", "spectral_topological_reasoning",
        "causal_chain_reasoner", "algebraic_planning_engine",
        "fusion_supremacy", "taaf_stepenv_hook", "cass_phi",
        "unified_consensus_engine", "sovereign_memory_vram",
        "nvidia_dsl", "mcts_planner", "world_model_simulator",
        "human_cognitive_perception", "pure_abstract_axioms",
        "skills", "rune", "grid", "oracle", "principles", "sim",
        "category_engine", "skill_orchestrator", "skill_derived_profile",
        "object_segmentation", "hypothesis_ledger", "flux_search",
        "turn_memo", "abstraction_skill_registry",
        "arc_color_transform", "arc_object_tracker", "arc_pattern_completion",
    }

    # Scan all imports in the decision chain
    chain_files = [
        SDK / "skill_orchestrator.py",
        SDK / "taaf_stepenv_hook.py",
        SDK / "unified_consensus_engine.py",
        SDK / "cass_phi.py",
        SDK / "zero_waste.py",
    ]
    imported: set[str] = set()
    for f in chain_files:
        if f.exists():
            imported.update(_imports_of(f))

    # Find dead modules (in payload but never imported)
    all_sdk = {p.stem for p in SDK.glob("*.py") if p.stem != "__init__"}
    dead = sorted(all_sdk - imported - payload_modules)

    print(f"Payload modules: {len(payload_modules)}")
    print(f"Imported by chain: {len(imported & all_sdk)}")
    print(f"Dead modules (not in payload, not imported): {len(dead)}")
    for d in dead:
        print(f"  - {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
