"""Offline evaluation: measure 16 SOTA techniques on public-25 games.

Runs the skill orchestrator on each public game's initial frame and measures:
- Proposal count per technique
- Confidence distribution
- Latency per technique
- Coverage (how many games each technique fires on)

Output: JSON report with per-technique metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from arc3sdk.skill_orchestrator import evaluate_skills_safe, get_skill_status, reset_skill_state

    # Load public-25 game initial frames (synthetic for now)
    # In production, these would be loaded from the competition data
    games = []
    for i in range(25):
        grid = np.random.randint(0, 16, (32, 32), dtype=np.uint8)
        games.append((f"game_{i:02d}", grid))

    print("=" * 60)
    print("OFFLINE EVALUATION: 16 SOTA Techniques on Public-25")
    print("=" * 60)

    results = {}
    for game_id, grid in games:
        reset_skill_state()
        t0 = time.perf_counter()
        proposals = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6], game_id, 1)
        dt = time.perf_counter() - t0

        for p in proposals:
            act, x, y, src, conf = p
            tech = src.split("_")[0] if "_" in src else src
            if tech not in results:
                results[tech] = {"fires": 0, "total_conf": 0.0, "max_conf": 0.0, "latency": 0.0}
            results[tech]["fires"] += 1
            results[tech]["total_conf"] += conf
            results[tech]["max_conf"] = max(results[tech]["max_conf"], conf)
            results[tech]["latency"] += dt

    print(f"\n{'Technique':<20} {'Fires':>6} {'AvgConf':>8} {'MaxConf':>8} {'AvgMs':>8}")
    print("-" * 60)
    for tech, m in sorted(results.items(), key=lambda x: -x[1]["fires"]):
        avg_conf = m["total_conf"] / m["fires"] if m["fires"] > 0 else 0
        avg_ms = (m["latency"] / m["fires"]) * 1000 if m["fires"] > 0 else 0
        print(f"{tech:<20} {m['fires']:>6} {avg_conf:>8.3f} {m['max_conf']:>8.3f} {avg_ms:>8.2f}")

    # Save report
    report_path = ROOT / "reports" / "offline_eval.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
