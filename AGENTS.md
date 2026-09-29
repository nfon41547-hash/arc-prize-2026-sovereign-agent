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
kaggle datasets version -p dataset_patch -m "<msg>" --dir-mode zip
```
- Staging layout is `dataset_patch/payload/arc3sdk/*.py` (+
  `dataset-metadata.json` at root). The `payload/` wrapper is
  load-bearing: the CLI zips each top-level subdir FLAT (entries
  relative to the subdir), so only via the wrapper does `payload.zip`
  carry the `arc3sdk/` prefix (mount: `<slug>/arc3sdk/*.py`). Verified
  2026-09-29 against kaggle 2.2.4 source + remote zip inspection.
- Windows CLI quirk: move `PUSH_README.txt` OUTSIDE staging before
  pushing (a rename inside staging uploads the stray file), restore
  after. `scripts/build_dataset_patch.py` is the single source of truth
  for staging (byte-identical to the notebook vendor cell).

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
- Memorization lock LIFTED 2026-09-29 by user order ("ปลดล็อก"):
  trajectory/SFT training allowed ONLY on license-clean artifacts
  (verified SPDX/CC0/CC-BY/Apache-2.0/MIT per-artifact, recorded below)
  AND with game-level holdout (never train and evaluate on the same
  game IDs; public-25 gains from public-game training are contamination
  until proven on held-out games/private rerun). Runtime priors stay
  fail-open; honest-numbers and 1-submission/day unchanged.
- License audit log (verified via `kaggle datasets metadata` / repo files):
  - `jihangli1121/arc-agi-3-replays-v1`: CC BY 4.0 (clean w/ attribution;
    GT replays of 25 public games, scraped from three.arcprize.org).
  - `Tufalabs/duck-harness` (incl. `example-run`): NO license file,
    pyproject license None -> NOT clean, training/fork-mining banned
    until upstream adds terms (read-only reference OK, git-ignored).
  - `justforgags/arc3-duck-lora-sft`: CC0-1.0 (clean; Qwen3.6-targeted
    LoRA, incompatible with our Qwen3.8-Flash-Next runtime directly).
  - `thtennant/taaf-kaggle-source-share-fork`: CC0-1.0 (clean; source
    diff-mining only).
  - `travislambert/travis-lambert-memory-compression-v1`: CC-BY-SA-4.0
    (clean w/ attribution; share-alike binds derivatives).
  - `yousefturk/fluidmind-arc-agi-3`: license unknown -> banned.
- Honest numbers only: verified log/test numbers, never fabricated claims.
- 1 competition submission/day. No git remote (local repo only).

## Environment facts
- Competition runs on Kaggle ONLY. Local machine is dev/test (RTX 3090).
- Heavy/data trees are git-ignored. reference_kernels/ holds third-party
  sources (Tufa duck-harness clone) + worker outputs; worker evidence that
  matters is distilled into reports/ (committed).
