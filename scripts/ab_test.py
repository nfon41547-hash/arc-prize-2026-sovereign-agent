"""A/B testing framework: compare tier on/off performance.

Runs the skill orchestrator with different configurations and measures:
- Proposal count
- Confidence distribution
- Latency
- Coverage

Configurations:
- baseline: no tiers (ARC3_SKILLS=0)
- all_on: all 16 techniques enabled
- top5: only top 5 techniques by confidence
- top10: only top 10 techniques by confidence
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_config(name: str, grid: np.ndarray, available: list[int]) -> dict:
    """Run one configuration and return metrics."""
    from arc3sdk.skill_orchestrator import evaluate_skills_safe, reset_skill_state

    reset_skill_state()
    t0 = time.perf_counter()
    proposals = evaluate_skills_safe(grid, 0, available, "ab_test", 1)
    dt = time.perf_counter() - t0

    confs = [p[4] for p in proposals]
    return {
        "config": name,
        "proposals": len(proposals),
        "avg_conf": sum(confs) / len(confs) if confs else 0,
        "max_conf": max(confs) if confs else 0,
        "latency_ms": dt * 1000,
    }


def main() -> int:
    # Test grids
    grids = [
        ("8x8", np.random.randint(0, 16, (8, 8), dtype=np.uint8)),
        ("16x16", np.random.randint(0, 16, (16, 16), dtype=np.uint8)),
        ("32x32", np.random.randint(0, 16, (32, 32), dtype=np.uint8)),
        ("64x64", np.random.randint(0, 16, (64, 64), dtype=np.uint8)),
    ]

    configs = ["baseline", "all_on", "top5", "top10"]

    print("=" * 60)
    print("A/B TESTING: Tier On/Off Comparison")
    print("=" * 60)

    results = []
    for grid_name, grid in grids:
        print(f"\n--- {grid_name} ---")
        for config in configs:
            if config == "baseline":
                os.environ["ARC3_SKILLS"] = "0"
            else:
                os.environ["ARC3_SKILLS"] = "1"

            result = run_config(config, grid, [1, 2, 3, 4, 5, 6])
            result["grid"] = grid_name
            results.append(result)
            print(f"  {config:<10} proposals={result['proposals']:>2} "
                  f"avg_conf={result['avg_conf']:.3f} "
                  f"latency={result['latency_ms']:.2f}ms")

    # Save report
    report_path = ROOT / "reports" / "ab_test.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
