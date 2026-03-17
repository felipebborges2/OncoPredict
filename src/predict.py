from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer


def build_synthetic_sample():
    """
    Cria uma amostra sintética de tumor com valores realistas.

    Os valores são baseados nas médias do dataset Breast Cancer Wisconsin,
    com pequenas variações para simular um caso clínico hipotético.
    Este é um paciente fictício — não representa nenhum caso real.
    """
    feature_names = load_breast_cancer().feature_names

    synthetic_values = [
        17.5,   # mean radius
        21.0,   # mean texture
        115.0,  # mean perimeter
        950.0,  # mean area
        0.110,  # mean smoothness
        0.160,  # mean compactness
        0.180,  # mean concavity
        0.095,  # mean concave points
        0.210,  # mean symmetry
        0.065,  # mean fractal dimension
        0.80,   # radius error
        1.20,   # texture error
        5.50,   # perimeter error
        75.0,   # area error
        0.007,  # smoothness error
        0.025,  # compactness error
        0.030,  # concavity error
        0.012,  # concave points error
        0.018,  # symmetry error
        0.004,  # fractal dimension error
        22.0,   # worst radius
        28.0,   # worst texture
        145.0,  # worst perimeter
        1500.0, # worst area
        0.145,  # worst smoothness
        0.380,  # worst compactness
        0.450,  # worst concavity
        0.175,  # worst concave points
        0.350,  # worst symmetry
        0.110,  # worst fractal dimension
    ]

    return pd.DataFrame([synthetic_values], columns=feature_names)


def main():
    model_path = Path("models/best_model.pkl")

    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode src/train.py primeiro.")

    model = joblib.load(model_path)

    sample = build_synthetic_sample()

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    class_names = load_breast_cancer().target_names
    predicted_class = class_names[prediction]

    print("=" * 50)
    print("  OncoPredict — Inferência em Nova Amostra")
    print("=" * 50)

    print("\nCaracterísticas do tumor (amostra sintética):")
    for feature, value in zip(sample.columns, sample.values[0]):
        print(f"  {feature:<35} {value:.4f}")

    print("\nResultado da predição:")
    print(f"  Classe prevista : {predicted_class.upper()}")

    print("\nProbabilidades por classe:")
    for class_name, prob in zip(class_names, probabilities):
        bar = "█" * int(prob * 30)
        print(f"  {class_name:<12} {prob:.1%}  {bar}")

    print("=" * 50)

    if predicted_class == "malignant":
        print("  ⚠ ATENÇÃO: predição indica tumor MALIGNO.")
    else:
        print("  ✓ Predição indica tumor BENIGNO.")

    print("\n  AVISO: Este sistema é educacional.")
    print("  Não substitui diagnóstico médico profissional.")
    print("=" * 50)


if __name__ == "__main__":
    main()
