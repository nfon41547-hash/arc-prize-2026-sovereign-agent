"""Sovereign-agent notebook tune v1 — tested delta module.

Target: arc-prize-2026-sovereign-agent.ipynb (23 cells, Duck harness + patch,
Qwen3.8-Flash-Next INT4 + MTP draft + Pennyroyal SGLang). Ground truth from
cells 2/4/12/16/20; defaults verified against the patch source in cell 2.

This module changes NOTHING in starter_push/ (workspace single source of
truth untouched). It exposes copy-paste-ready deltas + a hardened
TRUE_SUBMISSION detector, with a self-test runnable via:
    python3 reports/sovereign_tune_v1.py
"""

TRUE_SUBMISSION_VALUES = {"1", "true", "yes", "on", "competition"}


def is_true_submission(env, gateway_probe=None):
    """3-signal detector. Old notebook checks signal (a) only.

    (a) KAGGLE_IS_COMPETITION_RERUN in TRUE_SUBMISSION_VALUES
    (b) KAGGLE_KERNEL_RUN_TYPE == "competition"
    (c) gateway http://gateway:8001/api/games answers (callable probe)
    """
    rerun = str(env.get("KAGGLE_IS_COMPETITION_RERUN", "")).strip().lower()
    if rerun in TRUE_SUBMISSION_VALUES:
        return True
    if str(env.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip().lower() == "competition":
        return True
    if gateway_probe is not None:
        try:
            return bool(gateway_probe())
        except Exception:
            return False
    return False


# setup_env (cell 4) deltas: key -> tuned value. Rationale inline.
TUNED_SETUP_ENV_DELTA = {
    # Shipped-but-dead guards: patch adds NoopRepeatGuard (512 states x 16
    # actions, overridable block-then-insist) yet defaults are off and cell 4
    # never enables them. Enabling saves goalless repeat actions (RHAE).
    "ARC3_DEATH_LEDGER": "1",        # was unset -> off; backs death verdicts
    "ARC3_NOOP_REPEAT_GUARD": "1",   # was unset -> off; activates guard store
    "ARC3_DEATH_REPEAT_GUARD": "1",  # was unset -> off; needs ledger above
    # Guards started at level 2; level 1 is cheapest (pace ref L1 ~= 27k vs
    # L6 ~= 67k tokens), so guard from the first level.
    "ARC3_GUARDS_FROM_LEVEL": 1,     # was 2
    # Determinism for RHAE: 0.7/0.95 wanders; 0.5/0.90 keeps tool calls stable
    # while preserving exploration (TOP_K stays 20).
    "LOCAL_ANALYZER_TEMPERATURE": "0.5",  # was 0.7
    "LOCAL_ANALYZER_TOP_P": "0.90",       # was 0.95
    # Keep earliest-turn grounding under drain (150 assistant turns, 58k drain
    # on a 128k window); cost is one floor slice, payoff is cross-level memory.
    "ARC3_CONTEXT_DRAIN_FLOOR": "1",  # was unset -> off
}

# Cell-16 sovereign hook replacement: idempotent, pins prior 7 + guard keys.
# MUST run before cell-10 imports (env is read at import time by tool_agent).
TUNED_SOVEREIGN_HOOK_ENV = {
    "ARC3_ACTION_ECHO": "1",
    "ARC3_PERSISTENT_FUNCTIONS": "1",
    "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_BATCH_NOOP_BLOCK": "1",
    "ARC3_STALE_STATE_BLOCK": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_PRIORITY_TAIL_LOOKUP": "remaining",
    "ARC3_DEATH_LEDGER": "1",
    "ARC3_NOOP_REPEAT_GUARD": "1",
    "ARC3_DEATH_REPEAT_GUARD": "1",
    "ARC3_GUARDS_FROM_LEVEL": "1",
    "EXPOSE_UNDO": "on",  # keep: UNDO cheaper than RESET; RESET stays hidden
}

# Serving CFG (cell 12) deltas. SPEC thresholds intentionally untouched:
# 1.0 = accept-all-draft (max speed); quality tradeoff needs a live A/B.
TUNED_SERVING_CFG_DELTA = {
    # 0.96 leaves no headroom for CUDA graphs + MTP draft + fp8 KV on the
    # RTX Pro 6000; one OOM kills the whole run (score 0). 0.90 is the proven
    # setup_commands band (0.9-0.92).
    "MEMFRAC": 0.90,  # was 0.96
}

# Explicit keeps (verified good, do NOT change):
#   PACE=0 (pace ref JSON is stale per its own limitations note; tail-value
#     scheduling without pace is correct until the ref is refreshed),
#   EXPOSE_RESET default off + ONLY_RESET_LEVELS=true, MEMORY_SECTIONS=off +
#   LEVEL_TRANSFER_GUIDANCE=1, MIDDLE_TRUNCATION=1, TOOL_STEPS=0.


def apply_setup_env(setup_env):
    out = dict(setup_env)
    out.update(TUNED_SETUP_ENV_DELTA)
    return out


def apply_cfg(cfg):
    out = dict(cfg)
    out.update(TUNED_SERVING_CFG_DELTA)
    return out


def _check(condition, label):
    print(("PASS " if condition else "FAIL ") + label)
    if not condition:
        raise SystemExit(1)


def main():
    _check(is_true_submission({"KAGGLE_IS_COMPETITION_RERUN": "1"}), "signal-a 1")
    _check(is_true_submission({"KAGGLE_IS_COMPETITION_RERUN": "YES"}), "signal-a yes")
    _check(is_true_submission({"KAGGLE_KERNEL_RUN_TYPE": "Competition"}), "signal-b")
    _check(is_true_submission({}, gateway_probe=lambda: True), "signal-c")
    _check(not is_true_submission({"KAGGLE_KERNEL_RUN_TYPE": "Batch"}), "push stays offline")
    _check(not is_true_submission({}), "empty env offline")
    base = {"ARC3_GUARDS_FROM_LEVEL": 2, "LOCAL_ANALYZER_TEMPERATURE": "0.7"}
    tuned = apply_setup_env(base)
    _check(tuned["ARC3_GUARDS_FROM_LEVEL"] == 1, "guards from level 1")
    _check(tuned["ARC3_NOOP_REPEAT_GUARD"] == "1", "noop guard on")
    _check(tuned["ARC3_DEATH_LEDGER"] == "1", "ledger on")
    _check(tuned["LOCAL_ANALYZER_TEMPERATURE"] == "0.5", "temp 0.5")
    _check(apply_cfg({"MEMFRAC": 0.96})["MEMFRAC"] == 0.90, "memfrac 0.90")
    _check(TUNED_SOVEREIGN_HOOK_ENV["EXPOSE_UNDO"] == "on", "undo pinned")
    print("tune-v1 self-test: ALL GREEN")


if __name__ == "__main__":
    main()
