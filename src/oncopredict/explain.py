"""SHAP explanations for the deployed model."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

from oncopredict.config import RANDOM_STATE
from oncopredict.models import MODEL_LABELS


def explain_model(pipeline, X_train, X_test, max_background=100):
    """SHAP values of P(malignant) for the full pipeline, in original feature units.

    The permutation explainer is model-agnostic, so it explains exactly the
    model that is saved and used for inference (scaler included), whatever
    algorithm won the selection. Background data come from the training set.
    """
    background = shap.maskers.Independent(
        X_train.sample(min(max_background, len(X_train)), random_state=RANDOM_STATE)
    )
    explainer = shap.explainers.Permutation(
        lambda X: pipeline.predict_proba(X)[:, 1],
        background,
        feature_names=list(X_test.columns),
        seed=RANDOM_STATE,
    )
    return explainer(X_test, silent=True)


def plot_shap(explanation, y_test, model_name, beeswarm_path, waterfall_path):
    plt.figure()
    shap.plots.beeswarm(explanation, show=False, max_display=15)
    plt.title(f"SHAP values for P(malignant): {MODEL_LABELS[model_name]}", pad=15)
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Explain the malignant test case whose prediction is closest to 50%:
    # borderline cases are the ones where an explanation is most useful.
    malignant = np.flatnonzero(np.asarray(y_test) == 1)
    predicted = explanation.base_values + explanation.values.sum(axis=1)
    case = malignant[np.argmin(np.abs(predicted[malignant] - 0.5))]

    plt.figure()
    shap.plots.waterfall(explanation[case], show=False, max_display=12)
    plt.title(f"Individual explanation: malignant test case #{case}", pad=15)
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    return int(case)
