"""Notebook contract: vendor payload + hook marker + compilable cells."""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IPYNB = os.path.join(ROOT, "starter_push",
                     "arc-agi-3-starter-kernel-v32-profile-3.ipynb")


def _cells():
    nb = json.load(open(IPYNB, encoding="utf-8"))
    return nb["cells"]


def test_notebook_structure():
    cells = _cells()
    assert len(cells) == 19
    code = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code) == 9
    for i, c in enumerate(code):
        src = "".join(c.get("source", []))
        compile(src, f"cell{i}", "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_vendor_cell_current():
    import importlib.util
    path = os.path.join(ROOT, "scripts", "vendor_kernel_payload.py")
    spec = importlib.util.spec_from_file_location("vendor_kernel_payload",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    payloads = mod.build_payloads()
    for c in _cells():
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "vendor-arc3sdk ok" in src:
            assert mod._cell_hash(src) == mod._payload_digest(payloads)
            assert mod.PAYLOAD_VERSION in src
            return
    raise AssertionError("no vendor cell")


def test_hook_cell_marker():
    found = False
    for c in _cells():
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "# hook v=taaf-stepenv-1" in src:
            found = True
            assert "install_stepenv_hook" in src
    assert found, "hook cell marker missing"
    # every code cell we inject carries an id (no MissingIDFieldWarning)
    for c in _cells():
        tags = (c.get("metadata", {}) or {}).get("tags", [])
        if "vendor-arc3sdk" in tags or "sovereign-hook" in tags:
            assert c.get("id"), "injected cell missing id"
