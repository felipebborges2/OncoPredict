"""Metrics, operating thresholds, confidence intervals and statistical tests.

Convention: label 1 = malignant (positive class); `proba` is P(malignant).
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict

from oncopredict.config import CONFIDENCE_LEVEL, N_BOOTSTRAP, RANDOM_STATE


def threshold_metrics(y_true, proba, threshold):
    """Confusion-matrix counts and clinical rates when predicting malignant if proba >= threshold."""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "accuracy": (tp + tn) / len(y_pred),
    }


def out_of_fold_proba(estimator, X, y, cv):
    """P(malignant) for each training sample, predicted by a model that did not see it.

    These predictions let us choose a decision threshold without using the test set.
    """
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def youden_threshold(y_true, proba):
    """Threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr - fpr)])


def sensitivity_threshold(y_true, proba, target):
    """Highest threshold whose sensitivity is at least `target`.

    Among all thresholds that detect enough cancers, the highest one gives
    the fewest false alarms (best specificity).
    """
    _, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr >= target)])


def bootstrap_ci(y_true, proba, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Point estimates and percentile bootstrap confidence intervals.

    The test set is resampled with replacement `n_boot` times; the spread of
    each metric across resamples shows how much it would vary with a
    different sample of patients of the same size.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)

    def compute(y, p):
        m = threshold_metrics(y, p, threshold)
        return {
            "auc": roc_auc_score(y, p),
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "accuracy": m["accuracy"],
            "brier": brier_score_loss(y, p),
        }

    point = compute(y_true, proba)
    samples = []
    while len(samples) < n_boot:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # a resample with a single class has no AUC
        samples.append(compute(y_true[idx], proba[idx]))
    samples = pd.DataFrame(samples)

    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return pd.DataFrame({
        "estimate": pd.Series(point),
        "ci_low": samples.quantile(alpha),
        "ci_high": samples.quantile(1 - alpha),
    })


def mcnemar_test(y_true, pred_a, pred_b):
    """Exact McNemar test: do two classifiers make different errors on the same patients?

    Only discordant cases matter: b = A right and B wrong, c = A wrong and B right.
    Under the null hypothesis (equal error rates) b ~ Binomial(b + c, 0.5).
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p_value = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p_value


def net_benefit(y_true, proba, thresholds):
    """Decision curve analysis: net benefit of acting on the model at each threshold probability.

    NB(pt) = TP/n - FP/n * pt / (1 - pt). The weight pt / (1 - pt) is the
    exchange rate between harms and benefits implied by the threshold: a
    clinician who would biopsy at pt = 10% accepts 9 unnecessary biopsies
    per cancer found.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    prevalence = y_true.mean()
    model, treat_all = [], []
    for pt in thresholds:
        weight = pt / (1 - pt)
        pred = proba >= pt
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        model.append(tp / n - fp / n * weight)
        treat_all.append(prevalence - (1 - prevalence) * weight)
    return np.array(model), np.array(treat_all)
