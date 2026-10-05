"""Full benchmark: measure per-technique performance on synthetic grids."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from arc3sdk.skill_orchestrator import evaluate_skills_safe, get_skill_status, reset_skill_state

    # Test grids of various sizes
    sizes = [8, 16, 32, 64]
    grids = []
    for size in sizes:
        grid = np.random.randint(0, 16, (size, size), dtype=np.uint8)
        grids.append((f"{size}x{size}", grid))

    # Add structured grids
    structured = np.zeros((32, 32), dtype=np.uint8)
    structured[5:10, 5:10] = 1
    structured[15:20, 15:20] = 2
    structured[25, 25] = 3
    grids.append(("structured", structured))

    print("=" * 60)
    print("FULL BENCHMARK: 16 SOTA Techniques")
    print("=" * 60)

    for name, grid in grids:
        reset_skill_state()
        t0 = time.perf_counter()
        proposals = evaluate_skills_safe(grid, 0, [1, 2, 3, 4, 5, 6], "bench", 1)
        dt = time.perf_counter() - t0

        print(f"\n{name}: {len(proposals)} proposals in {dt*1000:.2f}ms")
        for p in proposals[:5]:
            act, x, y, src, conf = p
            print(f"  act={act} x={x} y={y} src={src} conf={conf:.3f}")

    print("\n" + "=" * 60)
    print("STATUS")
    print("=" * 60)
    status = get_skill_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
