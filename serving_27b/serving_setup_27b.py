"""27B dense-FP8 boot supervisor (AB arm). Own clean-room code: stdlib only.

Contrasts with the pinned Flash-Next stack (custom image, FP8-PLE patch,
MTP speculative decoding): this boots STOCK vLLM against an unmodified
Qwen3.8-27B-FP8 mirror (Apache-2.0) with MTP off and no model-specific
patches. Every step logs loudly: the first Kaggle boot is itself the
experiment (no 31GB model fits local dev VRAM, so no local boot proof).

Interface mirrors the watchdog shape the notebook expects:
  handle = start()  -> {"env": {...}, "log": path, "pid": int}
  stop(handle)      -> best-effort teardown, never raises.

Analyzer discovery contract (same as main stack):
  OPENAI_BASE_URL=http://127.0.0.1:1234/v1
  LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL = served name.

Tuning via env (AB27_*), all with documented defaults. Never raises
out of stop(); start() raises LOUDLY with the cause (boot failure must
be diagnosable from the worker log, never silent).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOST = "127.0.0.1"
PORT = 1234
DEFAULT_SERVED_NAME = "Qwen/Qwen3.8-27B-FP8"
MODEL_SLUG_HINTS = (
    "qwen3-8-27b-fp8-official",
    "qwen3-8-27b-fp8-repacked-v1",
)

_HANDLE: dict = {}


def _env(name: str, default: str) -> str:
    try:
        v = os.environ.get(name, default)
        return default if v is None or str(v) == "" else str(v)
    except Exception:
        return default


def find_model_dir() -> Path:
    """Locate the 27B model mount. Raises loudly when absent."""
    override = _env("AB27_MODEL_DIR", "")
    if override and Path(override).exists():
        return Path(override)
    try:
        paths = json.loads(os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "{}"))
        for ref, p in paths.items():
            low = str(ref).lower()
            if "27b" in low and p and Path(str(p)).exists():
                return Path(str(p))
    except Exception:
        pass
    roots = [Path("/kaggle/input")]
    try:
        roots += [p for p in Path("/kaggle/input/datasets").iterdir()]
    except Exception:
        pass
    for root in roots:
        try:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                low = child.name.lower()
                if any(h in low for h in MODEL_SLUG_HINTS) or "27b" in low:
                    return child
        except Exception:
            continue
    tried = [str(r) for r in roots]
    raise RuntimeError(
        "AB27 model mount not found (tried %s). Attach the 27B model "
        "as a model_source on the variant kernel." % tried)


def build_command(model_dir: Path) -> list[str]:
    """Stock-vLLM serve flags for dense FP8. No MTP, no PLE patch."""
    served = _env("AB27_SERVED_NAME", DEFAULT_SERVED_NAME)
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
        str(model_dir),
        "--served-model-name", served,
        "--host", HOST, "--port", str(PORT),
        "--load-format", "safetensors",
        "--dtype", "auto",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", _env("AB27_GPU_UTIL", "0.90"),
        "--max-model-len", _env("AB27_MAX_MODEL_LEN", "32768"),
        "--max-num-seqs", _env("AB27_MAX_SEQS", "8"),
        "--max-num-batched-tokens", _env("AB27_MAX_BATCHED", "8192"),
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
    ]


def _poll_ready(deadline_s: float) -> dict:
    url = "http://%s:%d/v1/models" % (HOST, PORT)
    start = time.monotonic()
    last = "not-started"
    while time.monotonic() - start < deadline_s:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                payload = json.loads(r.read().decode("utf-8", "replace"))
            ids = [str(d.get("id", "")) for d in payload.get("data", [])]
            return {"ready": True, "served_ids": ids,
                    "wait_s": round(time.monotonic() - start, 1)}
        except Exception as exc:
            last = "%s: %s" % (type(exc).__name__, str(exc)[:160])
            time.sleep(10.0)
    return {"ready": False, "last_error": last,
            "wait_s": round(time.monotonic() - start, 1)}


def start() -> dict:
    """Boot the server; returns handle. Raises loudly on failure."""
    global _HANDLE
    if _HANDLE.get("proc") is not None:
        return dict(_HANDLE)
    model_dir = find_model_dir()
    print("ab27: model_dir=%s" % model_dir, flush=True)
    try:
        import vllm  # noqa: F401
        print("ab27: vllm %s" % getattr(vllm, "__version__", "?"), flush=True)
    except Exception as exc:
        raise RuntimeError("ab27: stock vLLM not importable: %r. "
                           "Attach a runtime providing vLLM." % exc)
    cmd = build_command(model_dir)
    print("ab27: exec: %s" % " ".join(cmd[:6] + ["..."]), flush=True)
    log_path = Path(_env("AB27_LOG", "/tmp/ab27-vllm.log"))
    log_fh = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    deadline = float(_env("AB27_BOOT_TIMEOUT_S", "1800"))
    ready = _poll_ready(deadline)
    print("ab27: ready=%s" % json.dumps(ready)[:400], flush=True)
    if not ready.get("ready"):
        try:
            proc.terminate()
        except Exception:
            pass
        raise RuntimeError("ab27: server never became ready: %s (log %s)"
                           % (ready.get("last_error"), log_path))
    served = _env("AB27_SERVED_NAME", DEFAULT_SERVED_NAME)
    env = {"OPENAI_BASE_URL": "http://%s:%d/v1" % (HOST, PORT),
           "LOCAL_ANALYZER_MODEL_ID": served,
           "INFERENCE_ANALYZER_MODEL": served}
    _HANDLE = {"proc": proc, "pid": proc.pid, "env": env,
               "log": str(log_path), "model_dir": str(model_dir)}
    return {"pid": proc.pid, "env": dict(env), "log": str(log_path),
            "model_dir": str(model_dir)}


def stop(handle: dict | None = None) -> dict:
    """Best-effort teardown. Never raises."""
    try:
        proc = (_HANDLE.get("proc") if handle is None
                else (handle or {}).get("proc", _HANDLE.get("proc")))
        if proc is None:
            return {"stopped": False, "reason": "no-handle"}
        try:
            proc.terminate()
            proc.wait(timeout=60)
            outcome = "terminated"
        except Exception:
            try:
                proc.kill()
                outcome = "killed"
            except Exception as exc:
                return {"stopped": False, "reason": repr(exc)[:120]}
        _HANDLE.clear()
        return {"stopped": True, "outcome": outcome}
    except Exception as exc:
        return {"stopped": False, "reason": repr(exc)[:120]}
