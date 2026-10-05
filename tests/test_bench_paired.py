import numpy as np
from scripts.bench_duck_sobu_paired import (
    GameRunMetrics,
    compute_bayesian_paired_posterior,
    evaluate_paired_runs,
)

def test_paired_evaluation_promoted():
    baseline = [
        GameRunMetrics("g1", levels_completed=2, total_actions=40, dead_work_actions=10, final_score=2.0),
        GameRunMetrics("g2", levels_completed=1, total_actions=30, dead_work_actions=12, final_score=1.0),
    ]
    omega = [
        GameRunMetrics("g1", levels_completed=3, total_actions=35, dead_work_actions=4, final_score=3.0),
        GameRunMetrics("g2", levels_completed=2, total_actions=25, dead_work_actions=3, final_score=2.0),
    ]
    res = evaluate_paired_runs(baseline, omega)
    assert res.promoted is True
    assert res.score_omega > res.score_baseline
    assert res.apl_omega < res.apl_baseline
    assert res.dead_work_omega < res.dead_work_baseline
    assert res.prob_delta_positive >= 0.90

def test_bayesian_posterior_zero_deltas():
    deltas = np.array([0.0, 0.0, 0.0])
    p = compute_bayesian_paired_posterior(deltas)
    assert p == 0.0
