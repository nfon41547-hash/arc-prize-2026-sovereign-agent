"""Release manifest: byte-level source/vendor/dataset agreement."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "release_manifest.json"

MODULES = (
    "v32_hardening.py",
    "offline_guard.py",
    "exploration_registry.py",
    "competition_contract.py",
    "egcd.py",
    "kaggle_native_adapter.py",
    "realtime_abstract_cortex.py",
    "spectral_topological_reasoning.py",
    "causal_chain_reasoner.py",
    "algebraic_planning_engine.py",
    "fusion_supremacy.py",
    "taaf_stepenv_hook.py",
    "cass_phi.py",
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
    "object_segmentation.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _generator():
    path = ROOT / "scripts" / "vendor_kernel_payload.py"
    spec = importlib.util.spec_from_file_location("_vendor_payload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_manifest() -> dict:
    gen = _generator()
    payloads = gen.build_payloads()
    entries = []
    for name in MODULES:
        source = ROOT / "arc3sdk" / name
        if not source.is_file():
            raise FileNotFoundError(name)
        vendor_bytes = (zlib.decompress(base64.b64decode(payloads[name]))
                        if name in payloads else b"")
        if vendor_bytes != source.read_bytes():
            raise ValueError(f"{name}: vendor bytes != source bytes")
        entries.append({
            "source_module": f"arc3sdk/{name}",
            "sha256": _sha(source.read_bytes()),
        })
    nb = json.loads((ROOT / "starter_push" /
                     "arc-agi-3-starter-kernel-v32-profile-3.ipynb").read_text(
                         encoding="utf-8"))
    vendor_ok = False
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "vendor-arc3sdk ok" in src:
            vendor_ok = (gen._cell_hash(src) == gen._payload_digest(payloads)
                         and gen.PAYLOAD_VERSION in src)
    if not vendor_ok:
        raise ValueError("notebook vendor cell != live payload")
    return {
        "schema": "arc3.release-manifest.v1",
        "payload_version": gen.PAYLOAD_VERSION,
        "hook_version": gen.HOOK_VERSION,
        "modules": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    current = build_manifest()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"wrote {OUT}")
        return 0
    old = json.loads(OUT.read_text(encoding="utf-8"))
    if current != old:
        print("RELEASE_MANIFEST_MISMATCH: regenerate with --write and inspect drift")
        return 1
    print(f"manifest current: {len(current['modules'])} modules, "
          f"{current['payload_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
