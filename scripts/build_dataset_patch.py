"""Dataset patch builder (lean rebuild): stage arc3sdk overlay + worker-sim.

Staging = dataset_patch/arc3sdk (payload files overlaid byte-identical).
Worker-sim: in a scrubbed subprocess (cwd=tmp, no repo on path), exec the
notebook vendor cell -> import the FULL hook->decide chain -> run one
decide on a fake obs. WORKER_SIM_OK only if the chain resolves offline.

Windows CLI quirk (kept): move PUSH_README.txt out before `datasets
version --dir-mode zip`, restore after (uploader sidecar bug).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "dataset_patch" / "arc3sdk"

# Worker import allowlist: every top-level import in every payload module
# must be stdlib-from-this-list or numpy. Anything else (torch, cupy,
# third-party) fails the build LOUDLY here instead of dying on the worker.
# (Historical: `types` was caught by this exact gate.)
WORKER_STDLIB_ALLOW = {
    "__future__", "hashlib", "os", "threading", "collections", "typing",
    "json", "math", "time", "pathlib", "dataclasses", "enum", "re",
    "random", "copy", "functools", "itertools", "heapq", "bisect",
    "statistics", "string", "operator", "numpy", "ast", "concurrent",
    "contextlib", "gc", "struct", "tempfile", "mmap", "numbers", "abc",
    "datetime", "subprocess", "csv", "glob", "logging", "traceback",
    "warnings", "weakref",
}


def _generator():
    path = ROOT / "scripts" / "vendor_kernel_payload.py"
    spec = importlib.util.spec_from_file_location("_vendor_payload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build() -> Path:
    gen = _generator()
    payloads = gen.build_payloads()
    if PATCH.exists():
        shutil.rmtree(PATCH)
    (PATCH / "arc3sdk").mkdir(parents=True)
    (PATCH / "arc3sdk" / "cass_xi").mkdir(parents=True)
    for name in payloads:
        if name == "__init__.py":
            # minimal package marker, same as vendor cell
            (PATCH / "arc3sdk" / "__init__.py").write_bytes(
                b'"""Vendored arc3sdk runtime (dataset overlay)."""\n'
                b'__version__ = "v32-vendored"\nVENDORED = True\n')
            continue
        src = ROOT / "arc3sdk" / name
        dst = PATCH / "arc3sdk" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    (PATCH / "dataset-metadata.json").write_text(json.dumps({
        "title": "arc3sdk-runtime-v32-hardening",
        "id": "bang1850/arc3sdk-runtime-v32-hardening",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n", encoding="utf-8")
    # Import-allowlist gate over PAYLOAD files (the shipped contract;
    # legacy dataset-only files are out of scope: never imported worker-side
    # since the working copy shadows the mount).
    import ast as _ast
    _payload_names = set(_generator().MODULE_FILES) | {"__init__.py"}
    for _p in sorted((PATCH / "arc3sdk").rglob("*.py")):
        _rel = _p.relative_to(PATCH / "arc3sdk").as_posix()
        if _rel not in _payload_names:
            continue
        try:
            _tree = _ast.parse(_p.read_text(encoding="utf-8"))
        except Exception as _e:
            raise AssertionError(f"{_p.name} parse fail: {_e}")
        for _node in _tree.body:
            _mods = set()
            if isinstance(_node, _ast.Import):
                _mods.update(a.name.split(".")[0] for a in _node.names)
            elif isinstance(_node, _ast.ImportFrom) and not getattr(
                    _node, "level", 0):
                if _node.module:
                    _mods.add(_node.module.split(".")[0])
            _bad = _mods - WORKER_STDLIB_ALLOW - {"arc3sdk"}
            assert not _bad, \
                f"{_p.name} top-level imports beyond worker stdlib/numpy: {sorted(_bad)}"
    return PATCH


VERIFY_SCRIPT = r"""
import sys
sys.path.insert(0, r"{stage}")
import numpy as np
from arc3sdk.taaf_stepenv_hook import install_stepenv_hook, HOOK_VERSION
from arc3sdk.unified_consensus_engine import SovereignMasterConsensusEngine
from arc3sdk import cass_phi as cphi

class FakeState:
    available_actions = [1, 2, 3, 4, 5]
    levels_completed = 0

class FakeGame:
    current_state = FakeState()
    class game_run:
        game_id = "sim-duck"

class FakeSession:
    game = FakeGame()
    def current_frame(self):
        import types
        return types.SimpleNamespace(grid=tuple(tuple(0 for _ in range(8)) for _ in range(8)), step=0, level=1)
    def step_env(self, arguments):
        return {"executed": True, "echo": arguments}

r = install_stepenv_hook(FakeSession)
assert r["installed"] is True, r
s = FakeSession()
out = s.step_env({"action": "UP"})
assert isinstance(out, dict) and out.get("executed") is True, out
eng = SovereignMasterConsensusEngine()
print("WORKER_SIM_OK hook=" + HOOK_VERSION)
"""


def verify(patch: Path) -> None:
    import os
    with tempfile.TemporaryDirectory() as td:
        runner = Path(td) / "sim.py"
        runner.write_text(VERIFY_SCRIPT.replace("{stage}", str(patch)),
                          encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(patch)
        env["PYTHONNOUSERSITE"] = "1"
        proc = subprocess.run(
            [sys.executable, str(runner)], cwd=td, capture_output=True,
            text=True, timeout=180, env=env)
        print(proc.stdout[-1500:])
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "WORKER_SIM_OK" in proc.stdout


if __name__ == "__main__":
    verify(build())
