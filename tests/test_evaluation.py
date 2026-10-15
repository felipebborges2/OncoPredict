import numpy as np
import pytest

from oncopredict.evaluation import (bootstrap_ci, mcnemar_test, net_benefit, sensitivity_threshold,
                                    threshold_metrics, youden_threshold)

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.6, 0.3, 0.4, 0.2, 0.1, 0.05])


def test_threshold_metrics_counts_missed_cancers_as_false_negatives():
    m = threshold_metrics(Y, P, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 0, 4)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(1.0)


def test_sensitivity_threshold_reaches_target_with_highest_cutoff():
    t = sensitivity_threshold(Y, P, target=1.0)
    assert t == pytest.approx(0.3)
    assert threshold_metrics(Y, P, t)["sensitivity"] == 1.0


def test_youden_threshold_maximizes_sensitivity_plus_specificity():
    t = youden_threshold(Y, P)
    m = threshold_metrics(Y, P, t)
    best = max(threshold_metrics(Y, P, c)["sensitivity"] + threshold_metrics(Y, P, c)["specificity"] for c in P)
    assert m["sensitivity"] + m["specificity"] == pytest.approx(best)


def test_mcnemar_identical_predictions_is_not_significant():
    pred = (P >= 0.5).astype(int)
    assert mcnemar_test(Y, pred, pred) == (0, 0, 1.0)


def test_mcnemar_counts_discordant_pairs():
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # all correct
    b = np.array([0, 0, 1, 1, 0, 0, 0, 0])  # two misses
    n_ab, n_ba, p = mcnemar_test(Y, a, b)
    assert (n_ab, n_ba) == (2, 0)
    assert p == pytest.approx(0.5)


def test_net_benefit_matches_formula():
    model, treat_all = net_benefit(Y, P, [0.5])
    # At pt = 0.5: TP = 3, FP = 0, n = 8, weight = 1
    assert model[0] == pytest.approx(3 / 8)
    assert treat_all[0] == pytest.approx(0.5 - 0.5)


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    first = bootstrap_ci(y, p, 0.5, n_boot=300, seed=1)
    second = bootstrap_ci(y, p, 0.5, n_boot=300, seed=1)
    assert first.equals(second)
    assert ((first.ci_low <= first.estimate) & (first.estimate <= first.ci_high)).all()
