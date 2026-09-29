"""Build the 27B A/B variant kernel (generated, never hand-edited).

Reads the main notebook (vendor cell passes through byte-identical, so
the v-current payload digest stays valid) and applies three surgical
replacements:
  cell 3 : vLLM profile -> dense27b (MTP_TOKENS 3 -> 0).
  cell 16: watchdog boot block -> AB27 boot-supervisor import + start;
           watchdog stop -> ab27 stop (bundle teardown_commands kept,
           best-effort).
Writes starter_push_ab/<variant>.ipynb + kernel-metadata.json (no id:
first push assigns one; dataset includes the boot dataset, model is the
official 27B FP8 mirror). Fails loudly when an anchor is missing.

Usage: py -3.12 scripts/make_ab_variant.py [--check]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "starter_push" / "arc-agi-3-starter-kernel-v32-profile-3.ipynb"
ABDIR = ROOT / "starter_push_ab"
CODE_FILE = "arc-agi-3-27b-ab.ipynb"

BOOT_DATASET = "bang1850/arc3-27b-boot"
BOOT_STAGING = ROOT / "dataset_27b"
BOOT_SOURCE = ROOT / "serving_27b"
MODEL_27B = "mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1"

PROFILE_OLD = "PUBLIC25_VLLM_PROFILE_NAME = 'kv5-bf16-mtp3-c8-cg32'"
PROFILE_NEW = "PUBLIC25_VLLM_PROFILE_NAME = 'dense27b-fp8-nomtp'"
MTP_OLD = '"TAAF_VLLM_MTP_TOKENS": "3"'
MTP_NEW = '"TAAF_VLLM_MTP_TOKENS": "0"'

BOOT_OLD = """if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
import vllm_server_watchdog as vllm_watchdog

vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')
vllm_watchdog.start_background(
    vllm_watchdog_setup,
    vllm_watchdog.WatchdogConfig(
        interval_seconds=15.0,
        request_timeout_seconds=5,
        failure_threshold=4,
        max_restart_attempts=2,
    ),
)"""

BOOT_NEW = """AB27_BOOT_DIR = next(
    (p.parent for p in Path("/kaggle/input").rglob("serving_setup_27b.py")
     if p.is_file()),
    None,
)
if AB27_BOOT_DIR is None:
    raise RuntimeError("AB27 boot dataset not mounted.")
if str(AB27_BOOT_DIR) not in sys.path:
    sys.path.insert(0, str(AB27_BOOT_DIR))
import serving_setup_27b as ab27_boot

ab27_handle = ab27_boot.start()
os.environ.update(ab27_handle["env"])"""

STOP_OLD = "vllm_watchdog.stop_background(timeout_seconds=15.0)"
STOP_NEW = "ab27_boot.stop(ab27_handle if 'ab27_handle' in dir() else None)"


def _replace_once(src: str, old: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f"anchor {what}: found {n}x, need exactly 1")
    return src.replace(old, BOOT_NEW if what == "boot" else (
        STOP_NEW if what == "stop" else (
            PROFILE_NEW if what == "profile" else MTP_NEW)))


def build(write: bool = True) -> dict:
    nb = json.loads(MAIN.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = [c for c in cells if c.get("cell_type") == "code"]
    assert len(cells) == 19 and len(code) == 9, "main notebook shape drifted"

    def src(i: int) -> str:
        return "".join(code[i].get("source", []))

    c3 = src(0)
    assert PROFILE_OLD in c3 and MTP_OLD in c3, "cell3 profile anchor missing"
    c3 = c3.replace(PROFILE_OLD, PROFILE_NEW).replace(MTP_OLD, MTP_NEW)

    c16 = src(7)
    assert BOOT_OLD in c16, "cell16 boot anchor missing"
    assert STOP_OLD in c16, "cell16 stop anchor missing"
    c16 = c16.replace(BOOT_OLD, BOOT_NEW).replace(STOP_OLD, STOP_NEW)

    code[0]["source"] = c3.splitlines(keepends=True)
    code[7]["source"] = c16.splitlines(keepends=True)
    import ast as _ast
    for c in (code[0], code[7]):
        compile("".join(c["source"]), "<variant>", "exec",
                flags=_ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    main_meta = json.loads((ROOT / "starter_push" / "kernel-metadata.json").read_text(encoding="utf-8"))
    datasets = list(main_meta.get("dataset_sources", []))
    if BOOT_DATASET not in datasets:
        datasets.append(BOOT_DATASET)
    meta = dict(main_meta)
    meta.update({
        "title": "arc-agi-3-27b-ab",
        "code_file": CODE_FILE,
        "dataset_sources": datasets,
        "model_sources": [MODEL_27B],
    })
    meta.pop("id", None)
    meta.pop("id_no", None)

    info = {"cells": len(cells), "code_file": CODE_FILE, "model": MODEL_27B}
    if write:
        ABDIR.mkdir(parents=True, exist_ok=True)
        (ABDIR / CODE_FILE).write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (ABDIR / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        BOOT_STAGING.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(BOOT_SOURCE.glob("*.py")):
            data = src_file.read_bytes()
            compile(data, src_file.name, "exec")
            (BOOT_STAGING / src_file.name).write_bytes(data)
        info["wrote"] = str(ABDIR)
        info["boot_files"] = sorted(p.name for p in BOOT_STAGING.glob("*.py"))
    return info


def main() -> int:
    check = "--check" in sys.argv[1:]
    info = build(write=not check)
    print(f"make_ab_variant {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
