"""Hypothesis ledger locks: retrodiction math, shadow discipline, bounds."""
import pytest

from arc3sdk import hypothesis_ledger as hl


@pytest.fixture(autouse=True)
def _clean():
    hl.reset_shared()
    yield
    hl.reset_shared()


def test_propose_observe_confirm_refute():
    led = hl.HypothesisLedger()
    led.propose("g", 0, 1, 0.9, tag="tier:ape")
    assert led.observe("g", 0, True) is True
    assert led.weight("tier:ape") == pytest.approx((1 + 1) / (1 + 2))
    led.propose("g", 0, 2, 0.8, tag="tier:ape")
    assert led.observe("g", 0, False) is False
    assert led.weight("tier:ape") == pytest.approx((1 + 1) / (2 + 2))
    st = led.stats()["tier:ape"]
    assert st["proposed"] == 2 and st["resolved"] == 2
    assert st["confirmed"] == 1 and st["mean_conf"] == pytest.approx(0.85)


def test_weight_neutral_without_data():
    assert hl.HypothesisLedger().weight("tier:nope") == 0.5


def test_shadow_never_resolved():
    led = hl.HypothesisLedger()
    led.propose("g", 0, 1, 0.95, tag="shadow:fusion_click")
    assert led.observe("g", 0, True) is None
    st = led.stats()["shadow:fusion_click"]
    assert st["proposed"] == 1 and st["resolved"] == 0
    assert led.weight("shadow:fusion_click") == 0.5


def test_level_move_expires_pending():
    led = hl.HypothesisLedger()
    led.propose("g", 0, 1, 0.9, tag="tier:ape")
    assert led.observe("g", 1, True) is None
    assert led.stats()["tier:ape"]["resolved"] == 0


def test_newer_proposal_supersedes():
    led = hl.HypothesisLedger()
    led.propose("g", 0, 1, 0.9, tag="tier:ape")
    led.propose("g", 0, 2, 0.7, tag="tier:ape")
    assert led.observe("g", 0, True) is True
    st = led.stats()["tier:ape"]
    assert st["proposed"] == 2 and st["resolved"] == 1


def test_bounds_and_garbage():
    led = hl.HypothesisLedger()
    led.propose(None, None, None, None, None)
    led.propose("g", 0, 1, 99.0, tag="tier:ape")  # conf out of range
    led.propose("g", 0, 1, 0.9, tag="tier:ape")
    assert led.observe("g", 0, True) is True
    assert led.observe("nogame", 0, True) is None
    assert led.observe(None, None, None) is None
    assert led.weight(None) == 0.5
    assert isinstance(led.stats(), dict)
    for i in range(200):
        led.propose(f"g{i}", 0, 1, 0.5, tag=f"t{i}")
    assert len(led._pending) <= 64 and len(led._tags) <= 64
    led.reset()
    assert led.stats() == {} and led.weight("tier:ape") == 0.5


def test_singleton_shared():
    hl.reset_shared()
    assert hl.shared_ledger() is hl.shared_ledger()
    hl.shared_ledger().propose("g", 0, 1, 0.5, tag="t")
    assert hl.shared_ledger().stats()["t"]["proposed"] == 1
