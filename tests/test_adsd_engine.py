"""Test suite for Auto-Diagnosis and Skill Discovery (ADSD)."""
import numpy as np
import pytest
from arc3sdk.adsd_engine import AutoDiagnosisEngine, DiscoveredSkillLibrary, get_adsd_library

def test_auto_diagnosis_wall_collision():
    grid = np.zeros((10, 10), dtype=np.uint8)
    diag = AutoDiagnosisEngine.diagnose_failure(
        before_grid=grid,
        after_grid=grid, # Identical grid = no-op
        action="UP",
        result={"board_changed": False},
        recent_history=[]
    )
    assert diag["root_cause"] == "DIRECTIONAL_OBSTACLE_COLLISION"
    assert diag["suggested_skill_family"] == "OrthogonalBypassSkill"
    assert diag["confidence"] > 0.90

def test_skill_discovery_execution():
    lib = get_adsd_library()
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[4:6, 4:6] = 2 # Add foreground block
    
    # Test Orthogonal Bypass execution
    bypass_prop = lib.execute_skill("OrthogonalBypassSkill", grid, [2, 4], {"blocked_action": 1})
    assert bypass_prop is not None
    assert bypass_prop[0] in [2, 4]
    
    # Test Centroid Mouse execution
    mouse_prop = lib.execute_skill("SalientCentroidFocusSkill", grid, [6], {"bg": 0})
    assert mouse_prop is not None
    assert mouse_prop[0] == 6 # Action 6 = MOUSE
    assert mouse_prop[1] == 4 or mouse_prop[1] == 5 # cx
