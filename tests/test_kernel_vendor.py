"""Kernel vendor contract: payload sets, digest integrity, worker round-trip.

The round-trip runs the FULL hook->decide chain inside the vendored tree
alone (v26 lesson: working copy shadows dataset mounts; anything outside
the payload is dead code worker-side).
"""
import base64
import importlib.util
import os
import subprocess
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_generator():
    path = os.path.join(ROOT, "scripts", "vendor_kernel_payload.py")
    spec = importlib.util.spec_from_file_location("vendor_kernel_payload",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_payloads_decode_and_light_init():
    gen = _load_generator()
    payloads = gen.build_payloads()
    for name in ("unified_consensus_engine.py", "taaf_stepenv_hook.py",
                 "cass_phi.py", "skill_orchestrator.py",
                 "realtime_abstract_cortex.py", "fusion_supremacy.py",
                 "causal_chain_reasoner.py", "algebraic_planning_engine.py",
                 "v32_hardening.py", "exploration_registry.py",
                 "kaggle_native_adapter.py"):
        assert name in payloads, name
    for name, b64 in payloads.items():
        src = zlib.decompress(base64.b64decode(b64)).decode("utf-8")
        compile(src, name, "exec")
    init_src = zlib.decompress(
        base64.b64decode(payloads["__init__.py"])).decode("utf-8")
    for heavy in ("cupynumeric", "torch", "legate", "cupy"):
        assert heavy not in init_src, heavy


def test_digest_covers_version():
    gen = _load_generator()
    payloads = gen.build_payloads()
    d1 = gen._payload_digest(payloads)
    assert len(d1) == 16
    cell = gen.make_vendor_cell(payloads)
    src = "".join(cell["source"])
    assert gen._cell_hash(src) == d1
    assert gen.PAYLOAD_VERSION in src


def test_vendor_round_trip_subprocess(tmp_path):
    gen = _load_generator()
    cell = gen.make_vendor_cell(gen.build_payloads())
    cell_src = "".join(cell["source"])
    runner = tmp_path / "run_vendor.py"
    runner.write_text(
        "from pathlib import Path\n"
        "WORKING_DIR = Path(r'" + str(tmp_path) + "')\n"
        + cell_src + "\n"
        "from arc3sdk.v32_hardening import install_v32_hardening\n"
        "from arc3sdk import exploration_registry as er\n"
        "from arc3sdk.unified_consensus_engine import SovereignMasterConsensusEngine\n"
        "from arc3sdk import taaf_stepenv_hook as th\n"
        "from arc3sdk import cass_phi as cphi\n"
        "import numpy as np\n"
        "class FakeSolver:\n"
        "    def policy(self, obs): return 1\n"
        "assert install_v32_hardening(FakeSolver()) is True\n"
        "assert install_v32_hardening(None) is False\n"
        "g = np.zeros((10, 10), dtype=np.uint8); g[2:5, 2:5] = 3\n"
        "class FakeObs:\n"
        "    frame = g\n"
        "    grid = g\n"
        "    available_actions = [1, 2, 3, 4, 5]\n"
        "    game_id = 'rt-shadow'\n"
        "    score = 0.0\n"
        "    level = 0\n"
        "out = SovereignMasterConsensusEngine().decide(FakeObs())\n"
        "assert out is None or isinstance(out.get('action'), (int, dict))\n"
        "assert th.HOOK_VERSION == 'taaf-stepenv-1'\n"
        "assert isinstance(cphi.shared().metrics(), dict)\n"
        "print('ROUND_TRIP_OK')\n",
        encoding="utf-8",
    )
    proc = subprocess.run([sys.executable, str(runner)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=180)
    assert "vendor-arc3sdk ok" in proc.stdout, proc.stdout + proc.stderr
    assert "ROUND_TRIP_OK" in proc.stdout, proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr
