"""Payload contract: exact module set, versions, digest integrity."""
import importlib.util
import zlib
import base64


def _load_generator():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "vendor_kernel_payload.py")
    spec = importlib.util.spec_from_file_location("vendor_kernel_payload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_payload_set_and_versions():
    gen = _load_generator()
    assert gen.PAYLOAD_VERSION == "v25-shadow-1"
    assert gen.HOOK_VERSION == "taaf-stepenv-1"
    payloads = gen.build_payloads()
    assert "unified_consensus_engine.py" in payloads
    assert "taaf_stepenv_hook.py" in payloads
    assert "cass_phi.py" in payloads
    assert "skill_orchestrator.py" in payloads
    for name, b64 in payloads.items():
        src = zlib.decompress(base64.b64decode(b64)).decode("utf-8")
        compile(src, name, "exec")
    # digest is content-sensitive (any byte drift forces cell refresh)
    altered = dict(payloads)
    first = next(iter(altered))
    altered[first] = altered[first][:-4] + ("AAAA" if not altered[first].endswith("AAAA") else "BBBB")
    assert gen._payload_digest(payloads) != gen._payload_digest(altered)
    assert len(gen._payload_digest(payloads)) == 16
    # cell hash extraction
    cell = gen.make_vendor_cell(payloads)
    src = "".join(cell["source"])
    assert gen._cell_hash(src) == gen._payload_digest(payloads)
    assert gen.PAYLOAD_VERSION in src
