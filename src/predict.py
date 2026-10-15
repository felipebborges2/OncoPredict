"""
predict.py — Inference with the trained OncoPredict model.

Usage (from any directory):
    python src/predict.py                                  # built-in synthetic sample
    python src/predict.py --input data.csv                 # external CSV
    python src/predict.py --input data.csv --output out.csv

The CSV must have one row per sample and one column per feature, using the
feature names of the Breast Cancer Wisconsin dataset (e.g. "mean radius").
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from oncopredict.config import MODEL_PATH
from oncopredict.models import MODEL_LABELS

# Hypothetical malignant tumor, close to the malignant-class means of the
# dataset. It does not correspond to any real patient.
SYNTHETIC_SAMPLE = [
    17.5, 21.0, 115.0, 950.0, 0.110,
    0.160, 0.180, 0.095, 0.210, 0.065,
    0.80, 1.20, 5.50, 75.0, 0.007,
    0.025, 0.030, 0.012, 0.018, 0.004,
    22.0, 28.0, 145.0, 1500.0, 0.145,
    0.380, 0.450, 0.175, 0.350, 0.110,
]


def load_artifact(path=MODEL_PATH):
    if not Path(path).exists():
        sys.exit(f"Model not found at {path}. Run `python src/train.py` first.")
    return joblib.load(path)


def load_input(csv_path, feature_names):
    """Read a CSV and return its feature columns in training order."""
    df = pd.read_csv(csv_path)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    extra = [c for c in df.columns if c not in feature_names]
    if extra:
        print(f"Warning: ignoring columns that are not model features: {extra}")
    return df[feature_names]


def out_of_range(X, feature_ranges):
    """Features outside the range seen in training, per row.

    The model was never shown such values, so its output there is an extrapolation.
    """
    low, high = feature_ranges["min"], feature_ranges["max"]
    outside = (X < low) | (X > high)
    return {i: list(X.columns[row]) for i, row in zip(X.index, outside.to_numpy()) if row.any()}


def predict(artifact, X):
    """P(malignant) and predicted class at the deployed threshold."""
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    labels = [artifact["class_names"][int(p >= artifact["threshold"])] for p in proba]
    return pd.DataFrame({"p_malignant": proba, "prediction": labels}, index=X.index)


def print_results(results, artifact, source, warnings):
    print("=" * 60)
    print("  OncoPredict — inference")
    print("=" * 60)
    print(f"  Source    : {source}")
    print(f"  Model     : {MODEL_LABELS[artifact['model_name']]}")
    print(f"  Threshold : {artifact['threshold']:.3f} ({artifact['threshold_rule']})")
    for i, row in results.iterrows():
        bar = "█" * int(row.p_malignant * 30)
        print(f"\n  Sample {i}: {row.prediction.upper():<9} P(malignant) = {row.p_malignant:6.1%}  {bar}")
        if i in warnings:
            print(f"    Warning: outside training range -> {warnings[i]}")
    print("\n" + "=" * 60)
    print("  Educational project. Not a medical device and not a diagnosis.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="OncoPredict — inference on new samples.")
    parser.add_argument("--input", "-i", help="CSV with tumor features. Defaults to a synthetic sample.")
    parser.add_argument("--output", "-o", help="Optional path to save predictions as CSV.")
    args = parser.parse_args()

    artifact = load_artifact()
    features = artifact["feature_names"]

    if args.input:
        if not Path(args.input).exists():
            sys.exit(f"File not found: {args.input}")
        try:
            X = load_input(args.input, features)
        except ValueError as error:
            sys.exit(str(error))
        source = args.input
    else:
        X = pd.DataFrame([SYNTHETIC_SAMPLE], columns=features)
        source = "synthetic sample (default)"

    results = predict(artifact, X)
    print_results(results, artifact, source, out_of_range(X, artifact["feature_ranges"]))

    if args.output:
        results.to_csv(args.output, index_label="sample")
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
