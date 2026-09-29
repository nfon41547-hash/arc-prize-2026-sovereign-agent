"""CASS-Xi server operating profile — Kaggle worker (RTX PRO 6000) ONLY.

Pure data, no imports: safe inside the notebook vendor cell and the team
dataset patch. Local dev machines (e.g. RTX 3090) must NOT use this profile
to claim server numbers; server evidence comes from the worker log only.
"""

SERVER_PROFILE = {
    "target": "kaggle-worker-rtx-pro-6000",
    "operating_point": {
        "rank": 8,
        "active_layers": 4,
        "precision": "fp8",
        "fusion": "persistent",
    },
    "reason": "lowest-latency tier on the 96-row fp8/persistent sweep with "
              "mid capacity (rank 8 x 4 layers); bf16/unfused reserved for ablation only.",
    "vllm": {
        "kv_cache_memory_bytes": 7516192768,
        "canonical_kv_cache_memory_bytes": 5368709120,
        "canonical_profile": "kv5-bf16-mtp3-c8-cg32",
        "lossless_candidate": "kv5-bf16-mtp1-c8-cg32-pc1",
        "forbidden_env": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    },
    "benchmark": {
        "model": "HuggingFaceTB/SmolLM3-3B",
        "methods": ("frozen", "icl-memory", "online-lora", "cass-xi"),
        "seeds": (1, 2, 3),
        "rank": 8,
        "alpha": 16,
        "learning_rate": 2e-4,
        "train_steps": 8,
        "gate_threshold": 0.5,
        "risk_limit": 0.2,
        "memory_capacity": 32,
    },
}


def get():
    return dict(SERVER_PROFILE)


def operating_config():
    return dict(SERVER_PROFILE["operating_point"])
