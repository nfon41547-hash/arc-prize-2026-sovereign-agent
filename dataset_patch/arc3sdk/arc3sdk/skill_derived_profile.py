"""Evidence-gated runtime profile derived from the local NVIDIA skill catalog.

The profile is intentionally conservative: one RTX PRO GPU, one vLLM server,
fixed capture shapes, no speculative distributed dispatcher, and no allocator
mode that conflicts with PLE CPU-offload IPC.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SkillDerivedProfile:
    endpoint: str = "http://localhost:1234/v1"
    dispatcher: str = "alltoall"
    cuda_graphs: bool = True
    fixed_capture_size: int = 32
    mtp_tokens: int = 3
    max_seqs: int = 8
    max_batched_tokens: int = 12288
    expandable_segments: bool = False
    cpu_offload_ipc: bool = True

    def as_env(self) -> dict[str, str]:
        return {
            "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": str(self.fixed_capture_size),
            "TAAF_VLLM_MTP_TOKENS": str(self.mtp_tokens),
            "TAAF_VLLM_MAX_NUM_SEQS": str(self.max_seqs),
            "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": str(self.max_batched_tokens),
            "TAAF_MOE_DISPATCHER": self.dispatcher,
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:False",
            "VLLM_BASE_URL": self.endpoint,
            "LOCAL_ANALYZER_BASE_URL": self.endpoint,
            "OPENAI_BASE_URL": self.endpoint,
        }

    def apply(self) -> dict[str, str]:
        env = self.as_env()
        for key, value in env.items():
            os.environ[key] = value
        return env

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_skill_derived_profile() -> SkillDerivedProfile:
    profile = SkillDerivedProfile()
    profile.apply()
    return profile
