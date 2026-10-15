# OncoPredict-ML: final model report

*Generated automatically by `src/train.py`. Positive class = malignant.*

## Protocol

- Stratified hold-out split: 455 training / 114 test samples.
- Hyperparameters tuned with 5-fold stratified CV on the training set, scoring = `roc_auc`.
- Model selected by **mean CV score** (the test set is not used for any decision).
- Decision thresholds selected on **out-of-fold** training predictions.
- Test metrics reported once, with 95% percentile bootstrap CIs (2000 resamples).

## Model comparison

| Model | CV ROC AUC (mean ± sd) | Test AUC (95% CI) | Test Brier | Best parameters |
|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.9959 ± 0.0046 | 0.996 (0.986 – 1.000) | 0.021 | `{'C': 1, 'solver': 'liblinear'}` |
| SVM | 0.9949 ± 0.0050 | 0.995 (0.983 – 1.000) | 0.022 | `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}` |
| Random Forest | 0.9889 ± 0.0067 | 0.995 (0.986 – 1.000) | 0.032 | `{'max_depth': 5, 'min_samples_leaf': 1, 'n_estimators': 200}` |

Overlapping confidence intervals mean the test set (n = 114) cannot separate the candidates; the ranking comes from cross-validation.

## Pairwise McNemar tests (test set, threshold 0.50)

| Comparison | A right, B wrong | A wrong, B right | p-value |
|---|---|---|---|
| Logistic Regression vs Random Forest | 3 | 2 | 1.000 |
| Logistic Regression vs SVM | 3 | 2 | 1.000 |
| Random Forest vs SVM | 3 | 3 | 1.000 |

## Decision threshold (Logistic Regression)

| Rule | Threshold | Out-of-fold sensitivity | Out-of-fold specificity | Test FN (missed cancers) | Test FP |
|---|---|---|---|---|---|
| Default | 0.500 | 95.3% | 98.6% | 2 | 1 |
| Youden | 0.307 | 97.6% | 97.2% | 1 | 1 |
| Sensitivity ≥ 98% (deployed) | 0.216 | 98.2% | 95.4% | 1 | 3 |

The deployed rule is the highest threshold that reached at least 98% sensitivity on out-of-fold predictions: missing a cancer is treated as costlier than a false alarm.

## Final test performance (95% bootstrap CI)

| Metric | Threshold 0.50 | Sensitivity ≥ 98% (0.216) |
|---|---|---|
| AUC | 0.996 (0.986 – 1.000) | 0.996 (0.986 – 1.000) |
| Sensitivity | 95.2% (87.5% – 100.0%) | 97.6% (92.1% – 100.0%) |
| Specificity | 98.6% (95.2% – 100.0%) | 95.8% (90.8% – 100.0%) |
| Accuracy | 97.4% (93.9% – 100.0%) | 96.5% (93.0% – 99.1%) |
| Brier score | 0.021 (0.006 – 0.041) | 0.021 (0.006 – 0.041) |

SHAP waterfall example: malignant test case #112 (the borderline one, prediction closest to 50%).
