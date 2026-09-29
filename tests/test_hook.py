"""Hook seam locks: install, substitute, strict allowlist, kill-switches.

Uses FakeSession mirroring the real TAAF contract (verified against
reference source): step_env(self, arguments), current_frame().grid,
game.current_state.available_actions/levels_completed,
game.game_run.game_id.
"""
import os

import numpy as np
import pytest

from arc3sdk import taaf_stepenv_hook as th
from arc3sdk import cass_phi as cphi


class _FakeState:
    def __init__(self, grid, avail=(1, 2, 3, 4, 5, 6), levels_completed=0):
        self._grid = grid
        self.available_actions = list(avail)
        self.levels_completed = levels_completed
        self.won = False


class _FakeGame:
    def __init__(self, grid, gid="zz99-test", avail=(1, 2, 3, 4, 5, 6)):
        self.current_state = _FakeState(grid, avail)
        self.game_run = type("R", (), {"game_id": gid})()


class _FakeSession:
    step_env = None

    def __init__(self, grid, gid="zz99-test", avail=(1, 2, 3, 4, 5, 6)):
        self.game = _FakeGame(grid, gid, avail)

    def current_frame(self):
        g = self.game.current_state._grid
        tup = tuple(tuple(int(c) for c in row) for row in g)
        return type("F", (), {"grid": tup, "step": 1, "level": 1})()


def _grid():
    g = np.zeros((10, 10), dtype=np.uint8)
    g[2:5, 2:5] = 3
    return g


@pytest.fixture(autouse=True)
def _clean():
    th.reset_telemetry()
    cphi.reset_shared()
    try:
        _FakeSession.step_env = None
    except Exception:
        pass
    try:
        delattr(_FakeSession, "_sovereign_wrapped")
    except Exception:
        pass
    old = dict(os.environ)
    for k in ("ARC3_TIER_GATE", "ARC3_TAAF_TIERS", "ARC3_PHI",
              "ARC3_SUB_ALLOW"):
        os.environ.pop(k, None)
    yield
    th.reset_telemetry()
    cphi.reset_shared()
    try:
        _FakeSession.step_env = None
    except Exception:
        pass
    try:
        delattr(_FakeSession, "_sovereign_wrapped")
    except Exception:
        pass
    os.environ.clear()
    os.environ.update(old)


def _install(decide_out):
    import arc3sdk.unified_consensus_engine as uce

    class _FakeEngine:
        def decide(self, obs):
            return decide_out

    import unittest.mock as _mock
    _patcher = _mock.patch.object(uce, "SovereignMasterConsensusEngine",
                                  lambda *a, **k: _FakeEngine())
    _patcher.start()
    rec = {}

    def _orig(self, arguments):
        rec["args"] = arguments
        return {"executed": True}

    _FakeSession.step_env = _orig
    out = th.install_stepenv_hook(_FakeSession)
    assert out["installed"] is True
    return rec, _patcher


def test_install_idempotent_and_bad_input():
    assert th.install_stepenv_hook(None)["installed"] is False
    assert th.install_stepenv_hook(object())["installed"] is False

    class _C:
        pass

    assert th.install_stepenv_hook(_C)["installed"] is False

    class _D:
        def step_env(self, arguments):
            return {"ok": True}

    assert th.install_stepenv_hook(_D)["installed"] is True
    assert th.install_stepenv_hook(_D)["reason"] == "already"


def test_substitute_proof_reason():
    rec, p = _install({"action": 1, "confidence": 0.95, "reason": "ape"})
    try:
        s = _FakeSession(_grid())
        s.step_env({"action": "UP"})
        assert rec["args"] == {"action": "ACTION1"}
    finally:
        p.stop()


def test_deny_statistical_reason():
    """v27 lesson: causal/cortex/skills clicks never override (audited)."""
    rec, p = _install({"action": 1, "confidence": 0.95,
                       "reason": "causal_chain_argmax"})
    try:
        s = _FakeSession(_grid())
        s.step_env({"action": "UP"})
        assert rec["args"] == {"action": "UP"}
    finally:
        p.stop()


@pytest.mark.parametrize("reason", [
    "causal_chain_argmax", "cortex_empirical_rank", "fusion_click",
    "cortex_centroid_click", "cogniarc_stagnation_escape",
    "skill_raycast_open_corridor", "terminal_horizon_mcts_geodesic",
    "grandmaster_batched_plan", "sovereign_eikonal_geodesic",
])
def test_deny_each_v27_harm_reason(reason, monkeypatch):
    import arc3sdk.unified_consensus_engine as uce

    class _FakeEngine:
        def decide(self, obs):
            return {"action": 1, "confidence": 0.99, "reason": reason}

    monkeypatch.setattr(uce, "SovereignMasterConsensusEngine",
                        lambda *a, **k: _FakeEngine())
    rec = {}

    def _orig(self, arguments):
        rec["args"] = arguments
        return {"executed": True}

    monkeypatch.setattr(_FakeSession, "step_env", _orig, raising=False)
    assert th.install_stepenv_hook(_FakeSession)["installed"] is True
    _FakeSession(_grid()).step_env({"action": "UP"})
    assert rec["args"] == {"action": "UP"}


@pytest.mark.parametrize("reason", [
    "ape", "leap_photographic:Q0.85v3", "leap_q:Q0.50",
    "agno_offline_bfs_shortest_path",
])
def test_allow_proof_reasons(reason, monkeypatch):
    import arc3sdk.unified_consensus_engine as uce

    class _FakeEngine:
        def decide(self, obs):
            return {"action": 1, "confidence": 0.95, "reason": reason}

    monkeypatch.setattr(uce, "SovereignMasterConsensusEngine",
                        lambda *a, **k: _FakeEngine())
    rec = {}

    def _orig(self, arguments):
        rec["args"] = arguments
        return {"executed": True}

    monkeypatch.setattr(_FakeSession, "step_env", _orig, raising=False)
    assert th.install_stepenv_hook(_FakeSession)["installed"] is True
    _FakeSession(_grid()).step_env({"action": "UP"})
    assert rec["args"] == {"action": "ACTION1"}


def test_env_override_opens_and_closes(monkeypatch):
    import arc3sdk.unified_consensus_engine as uce

    class _FakeEngine:
        def decide(self, obs):
            return {"action": 1, "confidence": 0.95,
                    "reason": "causal_chain_argmax"}

    monkeypatch.setattr(uce, "SovereignMasterConsensusEngine",
                        lambda *a, **k: _FakeEngine())
    rec = {}

    def _orig(self, arguments):
        rec["args"] = arguments
        return {"executed": True}

    monkeypatch.setattr(_FakeSession, "step_env", _orig, raising=False)
    assert th.install_stepenv_hook(_FakeSession)["installed"] is True
    # operator override opens it
    monkeypatch.setenv("ARC3_SUB_ALLOW", "causal_chain_argmax")
    _FakeSession(_grid()).step_env({"action": "UP"})
    assert rec["args"] == {"action": "ACTION1"}
    # empty allowlist closes everything
    monkeypatch.setenv("ARC3_SUB_ALLOW", "")
    _FakeSession(_grid()).step_env({"action": "UP"})
    assert rec["args"] == {"action": "UP"}


def test_kill_switches(monkeypatch):
    import arc3sdk.unified_consensus_engine as uce

    class _FakeEngine:
        def decide(self, obs):
            return {"action": 1, "confidence": 0.99, "reason": "ape"}

    monkeypatch.setattr(uce, "SovereignMasterConsensusEngine",
                        lambda *a, **k: _FakeEngine())
    rec = {}

    def _orig(self, arguments):
        rec["args"] = arguments
        return {"executed": True}

    monkeypatch.setattr(_FakeSession, "step_env", _orig, raising=False)
    assert th.install_stepenv_hook(_FakeSession)["installed"] is True
    monkeypatch.setenv("ARC3_TAAF_TIERS", "0")
    _FakeSession(_grid()).step_env({"action": "UP"})
    assert rec["args"] == {"action": "UP"}


def test_consensus_import_failure_diagnosed_once(monkeypatch, capsys):
    import sys as _sys
    from arc3sdk import taaf_stepenv_hook as thmod
    monkeypatch.setattr(thmod, "_CONSENSUS_DIAG_DONE", False)
    rec, p = _install({"action": 1, "confidence": 0.99, "reason": "ape"})
    try:
        monkeypatch.setitem(_sys.modules,
                            "arc3sdk.unified_consensus_engine", None)
        s = _FakeSession(_grid())
        s.step_env({"action": "UP"})
        assert rec["args"] == {"action": "UP"}
        assert "consensus-unavailable" in capsys.readouterr().out
        s.step_env({"action": "UP"})
        assert "consensus-unavailable" not in capsys.readouterr().out
    finally:
        p.stop()


def test_hook_ledger_retrodiction_two_turns():
    """Executed tier proposal resolves on next turn's realized outcome."""
    from arc3sdk import hypothesis_ledger as _hl
    _hl.reset_shared()
    rec, p = _install({"action": 1, "confidence": 0.95, "reason": "ape"})
    try:
        _FakeSession(_grid()).step_env({"action": "UP"})
        g2 = _grid()
        g2[4, 4] = 7
        _FakeSession(g2).step_env({"action": "UP"})
        st = _hl.shared_ledger().stats().get("tier:ape", {})
        assert st.get("resolved", 0) == 1 and st.get("confirmed", 0) == 1
    finally:
        p.stop()


def test_hook_shadow_proposal_never_resolves():
    """Denied tier proposal expires unscored (volume only)."""
    from arc3sdk import hypothesis_ledger as _hl
    _hl.reset_shared()
    rec, p = _install({"action": 1, "confidence": 0.9,
                       "reason": "causal_chain_argmax"})
    try:
        _FakeSession(_grid()).step_env({"action": "UP"})
        g2 = _grid()
        g2[4, 4] = 7
        _FakeSession(g2).step_env({"action": "UP"})
        st = _hl.shared_ledger().stats().get(
            "shadow:causal_chain_argmax", {})
        assert st.get("proposed", 0) >= 1 and st.get("resolved", 0) == 0
    finally:
        p.stop()


def test_shadow_denied_records_would_be():
    """Shadow mode: denied tier logs count + mean-conf + would-be action.

    Behavior-neutral (analyzer still acts); the phi artifact carries the
    calibration evidence a future allowlist decision must cite.
    """
    rec, p = _install({"action": 2, "confidence": 0.9,
                       "reason": "causal_chain_argmax"})
    try:
        s = _FakeSession(_grid())
        s.step_env({"action": "UP"})
        assert rec["args"] == {"action": "UP"}
        ent = th._DENIED.get("causal_chain_argmax")
        assert isinstance(ent, list) and ent[0] >= 1
        assert abs(ent[1] / ent[0] - 0.9) < 1e-9
        pend = cphi.shared()._pending.get("zz99-test")
        assert pend is not None
        dec = str(pend.get("decision", ""))
        assert "denied:causal_chain_argmax:would2@0.90" in dec
    finally:
        p.stop()


def test_never_raises_fuzz(monkeypatch):
    import arc3sdk.unified_consensus_engine as uce

    class _Boom:
        def decide(self, obs):
            raise RuntimeError("boom")

    monkeypatch.setattr(uce, "SovereignMasterConsensusEngine",
                        lambda *a, **k: _Boom())

    class _D:
        def step_env(self, arguments):
            return {"ok": True}

    th.install_stepenv_hook(_D)
    d = _D()
    assert d.step_env(None) == {"ok": True}
    assert d.step_env({"action": "UP"}) == {"ok": True}
    assert d.step_env("garbage") == {"ok": True}
    assert th._build_obs(object()) is None
    assert th._translate("garbage", set()) is None
