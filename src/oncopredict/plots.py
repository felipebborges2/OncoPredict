"""Figures for model evaluation. Every function saves one PNG and closes it."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve

from oncopredict.evaluation import net_benefit, threshold_metrics
from oncopredict.models import MODEL_LABELS

MODEL_COLORS = {"logistic_regression": "#4C72B0", "random_forest": "#55A868", "svm": "#C44E52"}
DPI = 150


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_test, test_probas, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2,
                label=f"{MODEL_LABELS[name]} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (AUC = 0.500)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves on the held-out test set\n(positive class = malignant)")
    ax.legend(loc="lower right")
    _save(fig, path)


def _draw_confusion_matrix(ax, metrics, title):
    cm = np.array([[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred: malignant", "Pred: benign"])
    ax.set_yticks([0, 1], ["True: malignant", "True: benign"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=15,
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
    ax.set_title(f"{title}\nFN = {metrics['fn']} (missed cancers) | FP = {metrics['fp']}", fontsize=10)


def plot_threshold_analysis(y_train, oof_proba, y_test, test_proba, thresholds, operating, model_name, path):
    """Left: sensitivity/specificity vs. threshold on out-of-fold training predictions
    (where thresholds are chosen). Right: test confusion matrices at 0.5 and at the
    operating threshold (where they are evaluated)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.4, 1, 1]})

    grid = np.linspace(0, 1, 501)
    rates = [threshold_metrics(y_train, oof_proba, t) for t in grid]
    axes[0].plot(grid, [m["sensitivity"] for m in rates], color="#C44E52", linewidth=2, label="Sensitivity")
    axes[0].plot(grid, [m["specificity"] for m in rates], color="#55A868", linewidth=2, label="Specificity")
    styles = {"Default": ":", "Youden": "--"}
    for label, t in thresholds.items():
        style = styles.get(label.split()[0], "-")
        axes[0].axvline(t, color="#333333", linestyle=style, linewidth=1.3, label=f"{label}: {t:.3f}")
    axes[0].set_xlabel("Decision threshold on P(malignant)")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Threshold selection (out-of-fold, training set)")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(loc="lower center", fontsize=9)

    default = threshold_metrics(y_test, test_proba, 0.5)
    chosen = threshold_metrics(y_test, test_proba, thresholds[operating])
    _draw_confusion_matrix(axes[1], default, "Test set, threshold 0.50")
    _draw_confusion_matrix(axes[2], chosen, f"Test set, {operating} ({thresholds[operating]:.3f})")

    fig.suptitle(f"Decision threshold analysis: {MODEL_LABELS[model_name]}", fontsize=13)
    _save(fig, path)


def plot_calibration(y_test, test_probas, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in test_probas.items():
        CalibrationDisplay.from_predictions(
            y_test, proba, n_bins=5, strategy="quantile", ax=ax, color=MODEL_COLORS[name],
            name=f"{MODEL_LABELS[name]} (Brier = {brier_score_loss(y_test, proba):.3f})",
        )
    ax.set_xlabel("Mean predicted P(malignant)")
    ax.set_ylabel("Observed fraction malignant")
    ax.set_title("Calibration on the test set (5 quantile bins)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, path)


def plot_decision_curve(y_test, test_probas, path):
    grid = np.linspace(0.01, 0.80, 80)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    treat_all = None
    for name, proba in test_probas.items():
        nb, treat_all = net_benefit(y_test, proba, grid)
        ax.plot(grid, nb, color=MODEL_COLORS[name], linewidth=2, label=MODEL_LABELS[name])
    ax.plot(grid, treat_all, color="gray", linewidth=1.3, label="Biopsy everyone")
    ax.axhline(0, color="black", linewidth=1, label="Biopsy no one")
    prevalence = np.mean(y_test)
    ax.set_ylim(-0.05, prevalence + 0.05)
    ax.set_xlabel("Threshold probability (risk at which a biopsy would be ordered)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis on the test set")
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_learning_curve(estimator, X_train, y_train, cv, scoring, model_name, path):
    """Learning curve computed on the training set only; the test set stays untouched."""
    sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for scores, color, label in [(train_scores, "#4C72B0", "Training"),
                                 (val_scores, "#C44E52", "Cross-validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", color=color, label=label)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Training samples")
    ax.set_ylabel(scoring.replace("_", " ").upper())
    ax.set_title(f"Learning curve: {MODEL_LABELS[model_name]}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, path)
