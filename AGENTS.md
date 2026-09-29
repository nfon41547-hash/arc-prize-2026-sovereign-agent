# AGENTS.md — ARC-AGI-3 Kaggle Starter (rebuilt 2026-09-29 after env wipe)

## HARD RULE: kernel push command (every single time, no exceptions)
```
python -m kaggle kernels push -p starter_push --accelerator NvidiaRtxPro6000
```
- Never push without `--accelerator NvidiaRtxPro6000`.
- `starter_push/kernel-metadata.json` code_file selects the notebook.
  Current code_file: `arc-agi-3-starter-kernel-v32-profile-3.ipynb` (19 cells,
  Tufa base + vendor cell + TAAF hook). Ask before changing it.
- Never auto-push after a user-cancelled run. Standing order (active):
  kernel COMPLETE -> submit immediately (1 submission/day quota).

## Dataset push (team runtime)
```
kaggle datasets version -p <staging> -m "<msg>" --dir-mode zip
```
- `--dir-mode zip` REQUIRED (preserves `arc3sdk/` subdir).
- Windows CLI bug: move `PUSH_README.txt` out of staging before pushing,
  restore after.

## Single source of truth
- `scripts/vendor_kernel_payload.py` generates the notebook vendor cell from
  `arc3sdk/*.py` (MODULE_FILES, PAYLOAD_VERSION). Notebook cell == dataset
  files, verified byte-identical. Regenerate, never hand-edit cells.
- Current payload: v23-strict-1. Hook: taaf-stepenv-1
  (`_HarnessGameSession.step_env` wrap — the TAAF solver has NO `.policy`).

## Score history (public/private leaderboard, verified numbers only)
- Best real: 3.68 (Duck-alone era). v24: 3.21, v25: 2.41 (both Duck-alone:
  tiers dead). v27: 0.37 (tiers live, harmful — see below).
- Top board (2026-09): ~19.45. Our tiers target efficiency, not reasoning.

## v28 root causes (proven from v27 worker log: 75 tier-subs, 0 from proof)
- v27 tiers substituted on uncalibrated statistical confidences
  (causal/cortex/skills/stagnation/fusion at 0.80-0.95), overriding Duck
  with goalless moves. Public 9.0->3.01, private 2.41->0.37.
- v26 root cause (earlier): unified_consensus_engine NOT in vendor payload;
  working copy shadows dataset mount -> lazy import failed silently every
  turn (0 tier-subs in 3888 turns). Fixed by vendoring the full decide
  hard-chain (v22) + hook_self_contained audit + one-time
  consensus-unavailable diagnostic.
- v28 fix: strict substitution allowlist (ARC3_SUB_ALLOW, default
  `ape,leap_photographic,leap_q,agno_offline_bfs_shortest_path`); everything
  else audited but never overrides. Denied reasons counted for autopsy.

## Standing orders (active unless lifted)
- No-memorization: no game-ID-keyed manuals/priors/playbacks in runtime.
  All learning within-run. (prior_world_memories.json + tool_priors.json
  deleted 2026-09-26.)
- Honest numbers only: verified log/test numbers, never fabricated claims.
- 1 competition submission/day. No git remote (local repo only).

## Environment facts
- Competition runs on Kaggle ONLY. Local machine is dev/test (RTX 3090).
- Heavy/data trees are git-ignored. reference_kernels/ holds third-party
  sources (Tufa duck-harness clone) + worker outputs; worker evidence that
  matters is distilled into reports/ (committed).
