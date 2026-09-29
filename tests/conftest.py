"""Suite isolation: hermetic tests (fail-open). Resets global singletons,
RNGs, env and sys.path between tests so pass/fail never depends on order."""
import os
import sys

import pytest


def _snapshot_env():
    return dict(os.environ), list(sys.path)


def _restore_env(snap):
    env, path = snap
    try:
        os.environ.clear()
        os.environ.update(env)
    except Exception:
        pass
    try:
        sys.path[:] = path
    except Exception:
        pass


def _reset_all():
    try:
        import numpy as _np
        _np.random.seed(0)
    except Exception:
        pass
    try:
        import random as _py_random
        _py_random.seed(0)
    except Exception:
        pass
    try:
        from arc3sdk.sovereign_memory_vram import GlobalMemoryBank
        GlobalMemoryBank().reset_episodic()
    except Exception:
        pass
    try:
        from arc3sdk import exploration_registry as er
        er.reset_all()
    except Exception:
        pass
    try:
        from arc3sdk import causal_chain_reasoner as ccr
        ccr.reset()
    except Exception:
        pass
    try:
        from arc3sdk import fusion_supremacy as fsx
        fsx.reset_shared()
    except Exception:
        pass
    try:
        from arc3sdk import cass_phi as cphi
        cphi.reset_shared()
    except Exception:
        pass
    try:
        from arc3sdk import hypothesis_ledger as _hl
        _hl.reset_shared()
    except Exception:
        pass
    try:
        from arc3sdk import realtime_abstract_cortex as _ctx
        try:
            _ctx._LAST_OP.clear()
        except Exception:
            pass
        _ctx.reset_episodic(None)
    except Exception:
        pass
    try:
        from arc3sdk.unified_consensus_engine import SovereignMasterConsensusEngine
        kernel = getattr(SovereignMasterConsensusEngine(), "kernel", None)
        if hasattr(kernel, "reset"):
            kernel.reset()
        for attr in ("fatal_state_actions", "fatal_clicks"):
            try:
                getattr(kernel, attr).clear()
            except Exception:
                pass
        try:
            kernel._mcts_planner = None
        except Exception:
            pass
        try:
            kernel._ape_miss_key = None
        except Exception:
            pass
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_globals():
    _snap = _snapshot_env()
    _reset_all()
    yield
    _reset_all()
    _restore_env(_snap)
