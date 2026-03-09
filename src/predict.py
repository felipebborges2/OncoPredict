from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer


def main():
    model_path = Path("models/best_model.pkl")

    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode src/train.py primeiro.")

    model = joblib.load(model_path)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    sample = X.iloc[[0]]
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    class_names = data.target_names

    print("Amostra usada para teste:")
    print(sample)

    print("\nPredição:")
    print(class_names[prediction])

    print("\nProbabilidades:")
    for class_name, prob in zip(class_names, probability):
        print(f"{class_name}: {prob:.4f}")


if __name__ == "__main__":
    main()