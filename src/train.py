"""
train.py — Train, select, evaluate and save the OncoPredict model.

Usage (from any directory):
    python src/train.py

Pipeline:
    1. Stratified train/test split (the test set is used once, at the end).
    2. Grid search with stratified CV for each candidate model (training set only).
    3. Model selection by mean CV score.
    4. Decision threshold selection on out-of-fold training predictions.
    5. Test evaluation with bootstrap confidence intervals, McNemar tests,
       calibration and decision curve analysis.
    6. SHAP explanation of the deployed model and serialization.
"""

import itertools
import logging

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone

from oncopredict.config import (FIGURES_DIR, MODEL_PATH, MODELS_DIR, REPORTS_DIR, SCORING,
                                TARGET_SENSITIVITY)
from oncopredict.data import CLASS_NAMES, load_data, split_data
from oncopredict.evaluation import (bootstrap_ci, mcnemar_test, out_of_fold_proba,
                                    sensitivity_threshold, threshold_metrics, youden_threshold)
from oncopredict.explain import explain_model, plot_shap
from oncopredict.models import MODEL_LABELS, cv_summary, make_cv, select_best, tune_models
from oncopredict.plots import (plot_calibration, plot_decision_curve, plot_learning_curve,
                               plot_roc_curves, plot_threshold_analysis)
from oncopredict.report import write_report

log = logging.getLogger("oncopredict")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    log.info("Train: %d samples | Test: %d samples | Malignant prevalence: %.1f%%",
             len(X_train), len(X_test), 100 * y.mean())

    # 1. Tuning and selection — training set only
    searches = tune_models(X_train, y_train)
    best_name = select_best(searches)
    best_pipeline = searches[best_name].best_estimator_
    for name, search in searches.items():
        mean, std = cv_summary(search)
        log.info("%-20s CV %s = %.4f ± %.4f  %s", MODEL_LABELS[name], SCORING, mean, std, search.best_params_)
    log.info("Selected by CV: %s", MODEL_LABELS[best_name])

    # 2. Threshold selection — out-of-fold predictions on the training set
    oof_proba = out_of_fold_proba(clone(best_pipeline), X_train, y_train, make_cv())
    operating = f"Sensitivity ≥ {TARGET_SENSITIVITY:.0%}"
    thresholds = {
        "Default": 0.5,
        "Youden": youden_threshold(y_train, oof_proba),
        operating: sensitivity_threshold(y_train, oof_proba, TARGET_SENSITIVITY),
    }

    # 3. Test evaluation — first and only use of the test set
    test_probas = {name: s.best_estimator_.predict_proba(X_test)[:, 1] for name, s in searches.items()}
    best_proba = test_probas[best_name]

    comparison = []
    for name, search in searches.items():
        mean, std = cv_summary(search)
        ci = bootstrap_ci(y_test, test_probas[name], 0.5)
        comparison.append({
            "model": name, "cv_mean": mean, "cv_std": std,
            "test_auc": ci.loc["auc", "estimate"],
            "test_auc_ci_low": ci.loc["auc", "ci_low"], "test_auc_ci_high": ci.loc["auc", "ci_high"],
            "test_brier": ci.loc["brier", "estimate"],
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        })
    comparison = pd.DataFrame(comparison).sort_values("cv_mean", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    test_preds = {name: (p >= 0.5).astype(int) for name, p in test_probas.items()}
    mcnemar = {(a, b): mcnemar_test(y_test, test_preds[a], test_preds[b])
               for a, b in itertools.combinations(searches, 2)}

    train_rates = {label: threshold_metrics(y_train, oof_proba, t) for label, t in thresholds.items()}
    test_rates = {label: threshold_metrics(y_test, best_proba, t) for label, t in thresholds.items()}
    final_cis = {
        "Threshold 0.50": bootstrap_ci(y_test, best_proba, 0.5),
        f"{operating} ({thresholds[operating]:.3f})": bootstrap_ci(y_test, best_proba, thresholds[operating]),
    }
    for label, rates in test_rates.items():
        log.info("Test @ %-18s threshold=%.3f  sensitivity=%.3f  specificity=%.3f  FN=%d  FP=%d",
                 label, thresholds[label], rates["sensitivity"], rates["specificity"], rates["fn"], rates["fp"])

    # 4. Figures
    plot_roc_curves(y_test, test_probas, FIGURES_DIR / "roc_curves.png")
    plot_calibration(y_test, test_probas, FIGURES_DIR / "calibration.png")
    plot_decision_curve(y_test, test_probas, FIGURES_DIR / "decision_curve.png")
    plot_threshold_analysis(y_train, oof_proba, y_test, best_proba, thresholds, operating,
                            best_name, FIGURES_DIR / "threshold_analysis.png")
    plot_learning_curve(clone(best_pipeline), X_train, y_train, make_cv(), SCORING, best_name,
                        FIGURES_DIR / "learning_curve.png")
    explanation = explain_model(best_pipeline, X_train, X_test)
    shap_case = plot_shap(explanation, y_test, best_name,
                          FIGURES_DIR / "shap_beeswarm.png", FIGURES_DIR / "shap_waterfall.png")

    write_report(REPORTS_DIR / "final_report.md", best_name=best_name, comparison=comparison,
                 thresholds=thresholds, operating=operating, train_rates=train_rates,
                 test_rates=test_rates, final_cis=final_cis, mcnemar=mcnemar,
                 n_train=len(X_train), n_test=len(X_test), shap_case=shap_case)

    # 5. Artifact: the pipeline plus everything inference needs to use it correctly
    joblib.dump({
        "pipeline": best_pipeline,
        "model_name": best_name,
        "threshold": thresholds[operating],
        "threshold_rule": operating,
        "feature_names": list(X.columns),
        "class_names": CLASS_NAMES,
        "feature_ranges": X_train.agg(["min", "max"]).T,
        "sklearn_version": sklearn.__version__,
    }, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH.relative_to(MODEL_PATH.parents[1]))
    log.info("Report saved to reports/final_report.md")


if __name__ == "__main__":
    main()
