"""Markdown report with the final results, regenerated on every training run."""

from oncopredict.config import CONFIDENCE_LEVEL, CV_FOLDS, N_BOOTSTRAP, SCORING, TARGET_SENSITIVITY
from oncopredict.models import MODEL_LABELS


def _ci(row, pct=True):
    fmt = "{:.1%}" if pct else "{:.3f}"
    return f"{fmt.format(row.estimate)} ({fmt.format(row.ci_low)} – {fmt.format(row.ci_high)})"


def _metrics_table(cis):
    rows = [("AUC", "auc", False), ("Sensitivity", "sensitivity", True),
            ("Specificity", "specificity", True), ("Accuracy", "accuracy", True),
            ("Brier score", "brier", False)]
    header = "| Metric | " + " | ".join(cis) + " |\n|---|" + "---|" * len(cis) + "\n"
    body = "".join(
        f"| {label} | " + " | ".join(_ci(ci.loc[key], pct) for ci in cis.values()) + " |\n"
        for label, key, pct in rows
    )
    return header + body


def write_report(path, *, best_name, comparison, thresholds, operating, train_rates,
                 test_rates, final_cis, mcnemar, n_train, n_test, shap_case):
    level = f"{CONFIDENCE_LEVEL:.0%}"
    lines = [
        "# OncoPredict-ML: final model report",
        "",
        "*Generated automatically by `src/train.py`. Positive class = malignant.*",
        "",
        "## Protocol",
        "",
        f"- Stratified hold-out split: {n_train} training / {n_test} test samples.",
        f"- Hyperparameters tuned with {CV_FOLDS}-fold stratified CV on the training set, scoring = `{SCORING}`.",
        "- Model selected by **mean CV score** (the test set is not used for any decision).",
        "- Decision thresholds selected on **out-of-fold** training predictions.",
        f"- Test metrics reported once, with {level} percentile bootstrap CIs ({N_BOOTSTRAP} resamples).",
        "",
        "## Model comparison",
        "",
        f"| Model | CV {SCORING.upper().replace('_', ' ')} (mean ± sd) | Test AUC ({level} CI) | Test Brier | Best parameters |",
        "|---|---|---|---|---|",
    ]
    for r in comparison.itertuples():
        name = f"**{MODEL_LABELS[r.model]}** (selected)" if r.model == best_name else MODEL_LABELS[r.model]
        lines.append(
            f"| {name} | {r.cv_mean:.4f} ± {r.cv_std:.4f} | {r.test_auc:.3f} ({r.test_auc_ci_low:.3f} – "
            f"{r.test_auc_ci_high:.3f}) | {r.test_brier:.3f} | `{r.best_params}` |"
        )

    lines += [
        "",
        "Overlapping confidence intervals mean the test set (n = "
        f"{n_test}) cannot separate the candidates; the ranking comes from cross-validation.",
        "",
        "## Pairwise McNemar tests (test set, threshold 0.50)",
        "",
        "| Comparison | A right, B wrong | A wrong, B right | p-value |",
        "|---|---|---|---|",
    ]
    for (a, b), (n_ab, n_ba, p) in mcnemar.items():
        lines.append(f"| {MODEL_LABELS[a]} vs {MODEL_LABELS[b]} | {n_ab} | {n_ba} | {p:.3f} |")

    lines += [
        "",
        f"## Decision threshold ({MODEL_LABELS[best_name]})",
        "",
        "| Rule | Threshold | Out-of-fold sensitivity | Out-of-fold specificity | Test FN (missed cancers) | Test FP |",
        "|---|---|---|---|---|---|",
    ]
    for label, t in thresholds.items():
        tr, te = train_rates[label], test_rates[label]
        marker = " (deployed)" if label == operating else ""
        lines.append(
            f"| {label}{marker} | {t:.3f} | {tr['sensitivity']:.1%} | {tr['specificity']:.1%} | {te['fn']} | {te['fp']} |"
        )

    lines += [
        "",
        f"The deployed rule is the highest threshold that reached at least {TARGET_SENSITIVITY:.0%} "
        "sensitivity on out-of-fold predictions: missing a cancer is treated as costlier than a false alarm.",
        "",
        f"## Final test performance ({level} bootstrap CI)",
        "",
        _metrics_table(final_cis),
        f"SHAP waterfall example: malignant test case #{shap_case} (the borderline one, prediction closest to 50%).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
