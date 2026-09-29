"""A/B variant contract: generated notebook, boot staging, metadata."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gen():
    path = os.path.join(ROOT, "scripts", "make_ab_variant.py")
    spec = importlib.util.spec_from_file_location("make_ab_variant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_variant_notebook_profile_and_boot():
    gen = _load_gen()
    nb = json.load(open(os.path.join(
        ROOT, "starter_push_ab", gen.CODE_FILE), encoding="utf-8"))
    code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(nb["cells"]) == 19 and len(code) == 9
    c3 = "".join(code[0]["source"])
    assert "dense27b-fp8-nomtp" in c3
    assert '"TAAF_VLLM_MTP_TOKENS": "0"' in c3
    assert "kv5-bf16-mtp3" not in c3
    c16 = "".join(code[7]["source"])
    assert "serving_setup_27b" in c16 and "ab27_boot.start()" in c16
    assert "ab27_boot.stop(" in c16
    assert "vllm_server_watchdog" not in c16
    c9 = "".join(code[3]["source"])
    assert "ab27: Flash serving setup tolerated" in c9
    assert "serving_setup.py" in c9  # tolerate, not delete: keeps runtime side-effects


def test_variant_vendor_cell_matches_live_payload():
    import importlib.util as _ilu
    path = os.path.join(ROOT, "scripts", "vendor_kernel_payload.py")
    spec = _ilu.spec_from_file_location("vendor_kernel_payload", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen = _load_gen()
    nb = json.load(open(os.path.join(
        ROOT, "starter_push_ab", gen.CODE_FILE), encoding="utf-8"))
    payloads = mod.build_payloads()
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "vendor-arc3sdk ok" in src:
            assert mod._cell_hash(src) == mod._payload_digest(payloads)
            return
    raise AssertionError("no vendor cell in variant")


def test_variant_metadata_and_boot_staging():
    gen = _load_gen()
    meta = json.load(open(os.path.join(
        ROOT, "starter_push_ab", "kernel-metadata.json"), encoding="utf-8"))
    assert meta["code_file"] == gen.CODE_FILE
    assert meta["model_sources"] == [gen.MODEL_27B]
    assert gen.BOOT_DATASET in meta["dataset_sources"]
    assert meta.get("id") == "bang1850/arc-agi-3-27b-ab", \
        "variant must carry its own kernel id (never the main kernel id)"
    assert "arc-agi-3-starter-kernel" not in meta.get("id", "")
    assert "arc-prize-2026-arc-agi-3" in meta["competition_sources"]
    assert meta["machine_shape"] == "NvidiaRtxPro6000"
    live = open(os.path.join(ROOT, "serving_27b", "serving_setup_27b.py"),
                "rb").read()
    staged = open(os.path.join(ROOT, "dataset_27b", "serving_setup_27b.py"),
                  "rb").read()
    assert live == staged, "boot staging must equal source bytes"
    dm = json.load(open(os.path.join(ROOT, "dataset_27b",
                                     "dataset-metadata.json"), encoding="utf-8"))
    assert dm["id"] == gen.BOOT_DATASET
