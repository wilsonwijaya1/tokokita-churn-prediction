import numpy as np
import pytest

from tokokita.decision import (RetentionEconomics, best_threshold, expected_value_policy, net_benefit,
                               retention_value, sweep_thresholds)
from tokokita.metrics import (bootstrap_metrics, cluster_resamples, evaluate, expected_calibration_error, lift_at_k,
                              precision_at_k, recall_at_k)


def test_precision_recall_lift_at_k_known_values():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert precision_at_k(y, score, 0.2) == 1.0               # top 2 are both churners
    assert precision_at_k(y, score, 0.4) == pytest.approx(0.5)
    assert recall_at_k(y, score, 0.2) == pytest.approx(2 / 3)
    assert lift_at_k(y, score, 0.2) == pytest.approx(1.0 / 0.3)


def test_ece_is_zero_for_perfectly_calibrated_and_large_for_overconfident():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert expected_calibration_error(y, np.full(10, 0.5)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(10, 0.95)) == pytest.approx(0.45)


def test_evaluate_returns_expected_keys_and_perfect_ranking_scores_one():
    y = np.array([0, 0, 0, 1, 1, 1] * 5)
    out = evaluate(y, y * 0.8 + 0.1)
    assert out["AUC"] == pytest.approx(1.0) and out["PR_AUC"] == pytest.approx(1.0)
    assert {"AUC", "PR_AUC", "Precision@10%", "Recall@10%", "Lift@10%", "Brier", "ECE"} <= set(out)


def test_cluster_bootstrap_is_reproducible_and_resamples_whole_customers():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    r1 = [idx.tolist() for idx in cluster_resamples(groups, 5, seed=3)]
    r2 = [idx.tolist() for idx in cluster_resamples(groups, 5, seed=3)]
    assert r1 == r2
    for idx in r1:
        counts = {g: sum(groups[i] == g for i in idx) for g in "abc"}
        # a customer is drawn k times -> all of its rows appear k times (2k, 1k, 3k rows)
        assert counts["a"] % 2 == 0 and counts["c"] % 3 == 0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    n = 600
    groups = np.repeat(np.arange(n // 2), 2)
    y = rng.integers(0, 2, n)
    p = np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1)
    reps = bootstrap_metrics(y, {"m": p}, groups, n_boot=200, seed=1)["m"]["AUC"]
    point = evaluate(y, p)["AUC"]
    assert reps.quantile(0.025) <= point <= reps.quantile(0.975)


def test_net_benefit_arithmetic():
    econ = RetentionEconomics(contact_cost=10_000, save_rate=0.5, margin=1.0, retained_months=3.0)
    value = retention_value([300_000, 300_000, 300_000], econ)       # 1.0 * 3 * 300k/3 = 300k each
    assert value.tolist() == [300_000, 300_000, 300_000]
    churned = np.array([1, 0, 1])
    # contact customers 0 and 1: gain = 1*0.5*300k = 150k, cost = 2*10k
    assert net_benefit([True, True, False], churned, value, econ) == pytest.approx(150_000 - 20_000)
    assert net_benefit([False, False, False], churned, value, econ) == 0.0


def test_threshold_sweep_is_monotone_in_contacts_and_finds_the_optimum():
    econ = RetentionEconomics(contact_cost=10_000, save_rate=0.5, margin=1.0, retained_months=3.0)
    prob = np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1])
    churned = np.array([1, 1, 1, 0, 0, 0])
    value = np.full(6, 300_000.0)
    sw = sweep_thresholds(prob, churned, value, econ)
    assert (sw["n_contacted"].diff().dropna() <= 0).all()            # higher threshold -> fewer contacts
    best = best_threshold(sw)
    assert best["n_contacted"] == 3 and best["net_benefit"] == pytest.approx(3 * 150_000 - 3 * 10_000)
    # the expected-value rule contacts exactly those whose expected saving beats the cost
    assert expected_value_policy(prob, value, econ).sum() == 6      # 0.1*0.5*300k = 15k > 10k
    assert expected_value_policy(prob, value, econ.with_(contact_cost=50_000)).tolist() == [True, True, True, True, False, False]
