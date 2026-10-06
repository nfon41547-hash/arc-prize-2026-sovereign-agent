import numpy as np
from arc3sdk.fable_layer_core import Ledger, key_of, detect_budget, budget_alert, small_far_changes

def test_fable_ledger_and_keys():
    led = Ledger(1)
    led.feed("RIGHT", {"executed_count": 1, "gameplay_changed": True}, np.zeros((10, 10)), np.ones((10, 10)))
    assert led.n_actions == 1
    assert key_of("RIGHT") == "RIGHT"
    assert key_of("MOUSE(row=10, col=20)") == "MOUSE"

def test_fable_small_far_changes():
    before = np.zeros((64, 64), dtype=np.uint8)
    after = np.zeros((64, 64), dtype=np.uint8)
    # Main activity near center
    after[30:35, 30:35] = 1
    # Small far indicator in bottom corner
    after[60:62, 60:62] = 2
    patches = small_far_changes(before, after)
    assert isinstance(patches, list)
