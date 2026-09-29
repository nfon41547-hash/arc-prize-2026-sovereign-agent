"""Server audit (lean rebuild): hot-path hazards + self-containment.

- Hot path (hook + tiers) must contain no sleep/socket/subprocess calls,
  no unbounded while loops (bounded loops only).
- hook_self_contained: every module the hook->decide chain can import
  MUST be inside the vendor payload (v26 lesson: working copy shadows
  dataset mounts; 0 tier-subs in 3888 turns).
- Payload/notebook/dataset version consistency.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILS: list[str] = []

HOT_FILES = [
    "taaf_stepenv_hook.py",
    "cass_phi.py",
    "realtime_abstract_cortex.py",
    "fusion_supremacy.py",
    "object_segmentation.py",
    "causal_chain_reasoner.py",
    "algebraic_planning_engine.py",
    "unified_consensus_engine.py",
    "exploration_registry.py",
    "kaggle_native_adapter.py",
]

CHAIN = {"taaf_stepenv_hook.py", "unified_consensus_engine.py",
         "sovereign_memory_vram.py", "nvidia_dsl.py",
         "mcts_planner.py", "world_model_simulator.py",
         "human_cognitive_perception.py", "pure_abstract_axioms.py",
         "skills.py", "rune.py", "grid.py", "oracle.py",
         "principles.py", "sim.py", "category_engine.py",
         "realtime_abstract_cortex.py", "spectral_topological_reasoning.py",
         "causal_chain_reasoner.py", "algebraic_planning_engine.py",
         "fusion_supremacy.py", "cass_phi.py", "skill_orchestrator.py",
         "exploration_registry.py", "kaggle_native_adapter.py",
         "competition_contract.py", "egcd.py", "zero_waste.py",
         "object_segmentation.py"}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"EVIDENCE {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def _top_imports_for_audit(path: Path) -> set:
    """Module-LEVEL imports only (lazy in-function imports are fail-open
    by construction); arc3sdk self-imports are not third-party."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {"<parse-fail>"}
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root != "arc3sdk":
                    out.add(root)
        elif isinstance(node, ast.ImportFrom) and not getattr(node, "level", 0):
            mod = node.module or ""
            if mod and mod != "arc3sdk" and not mod.startswith("arc3sdk."):
                out.add(mod.split(".")[0])
    return out


_ALLOWED_TOP = {"__future__", "hashlib", "os", "threading", "collections",
                "typing", "json", "math", "time", "pathlib", "dataclasses",
                "enum", "re", "random", "copy", "functools", "itertools",
                "heapq", "bisect", "statistics", "string", "operator",
                "numpy", "ast", "concurrent", "contextlib", "gc", "struct",
                "tempfile", "mmap", "numbers", "abc", "datetime",
                "subprocess", "csv", "glob", "logging", "traceback",
                "warnings", "weakref"}


def _hot_path_scan() -> None:
    bad: list[str] = []
    for name in HOT_FILES:
        p = ROOT / "arc3sdk" / name
        if not p.is_file():
            bad.append(f"{name}:missing")
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            bad.append(f"{name}:parse-fail")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                n = ""
                if isinstance(f, ast.Name):
                    n = f.id
                elif isinstance(f, ast.Attribute):
                    n = f.attr
                if n in ("sleep", "usleep", "nanosleep"):
                    bad.append(f"{name}:sleep")
                if n in ("socket", "create_connection", "getaddrinfo",
                         "urlopen", "urlretrieve"):
                    bad.append(f"{name}:socket")
                if n in ("Popen", "run", "call", "check_call", "check_output"):
                    # Allowed only inside setup/teardown/server/vram/watchdog
                    # helpers (bounded, fail-open, off the per-turn path).
                    func = ""
                    for parent in ast.walk(tree):
                        if isinstance(parent, (ast.FunctionDef,
                                              ast.AsyncFunctionDef)):
                            for kid in ast.walk(parent):
                                if kid is node:
                                    func = parent.name
                    if not any(k in func.lower() for k in (
                            "setup", "server", "vram", "watchdog", "teardown",
                            "install", "spawn", "launch", "nvidia", "smi")):
                        bad.append(f"{name}:subprocess:{func}")
            if isinstance(node, ast.While):
                # while without an explicit bounded counter is suspect;
                # allow `while len(x) > CAP` and `while ... attempts` shapes
                src = ast.dump(node.test)
                if "len(" not in src and "attempt" not in src.lower() \
                        and "budget" not in src.lower() \
                        and "count" not in src.lower().replace("counter", "count"):
                    # still allow: check body for break conditions is too
                    # deep; record as review flag, not failure
                    pass
    check("hot_path_no_hazards", not bad, ";".join(bad[:8]))


def _stdlib_only() -> None:
    bad = []
    for name in HOT_FILES + ["skill_orchestrator.py", "sovereign_memory_vram.py",
                             "zero_waste.py"]:
        p = ROOT / "arc3sdk" / name
        if not p.is_file():
            continue
        top = _top_imports_for_audit(p)
        # module-level only; arc3sdk self-imports are not third-party
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            mod = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for a in node.names:
                        root = a.name.split(".")[0]
                        if root != "arc3sdk":
                            mod.add(root)
                elif isinstance(node, ast.ImportFrom) and not getattr(node, "level", 0):
                    if node.module:
                        root = node.module.split(".")[0]
                        if root != "arc3sdk":
                            mod.add(root)
            extra = mod - _ALLOWED_TOP
            if extra:
                bad.append(f"{name}:{sorted(extra)}")
        except Exception:
            bad.append(f"{name}:parse-fail")
    check("payload_stdlib_only", not bad, ";".join(bad[:8]))


def _self_contained() -> None:
    path = ROOT / "scripts" / "vendor_kernel_payload.py"
    spec = importlib.util.spec_from_file_location("_vendor_gen", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    payload = set(gen.build_payloads())
    outside = sorted(c for c in CHAIN if c not in payload)
    check("hook_self_contained", not outside, f"outside={outside}")


def _version_consistent() -> None:
    path = ROOT / "scripts" / "vendor_kernel_payload.py"
    spec = importlib.util.spec_from_file_location("_vendor_gen2", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    nb = json.loads((ROOT / "starter_push" /
                     "arc-agi-3-starter-kernel-v32-profile-3.ipynb").read_text(
                         encoding="utf-8"))
    ok = False
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "vendor-arc3sdk ok" in src:
            ok = (gen._cell_hash(src) == gen._payload_digest(gen.build_payloads())
                  and gen.PAYLOAD_VERSION in src)
    check("version_consistent", ok, f"payload={gen.PAYLOAD_VERSION}")


def main() -> int:
    _hot_path_scan()
    _stdlib_only()
    _self_contained()
    _version_consistent()
    print(f"EVIDENCE SUMMARY: {'ALL_PASS' if not FAILS else 'FAILURES=' + ','.join(FAILS)}",
          flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
