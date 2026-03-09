from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def build_models():
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True))
        ])
    }
    return models


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = build_models()

    results = []
    best_model = None
    best_score = -1
    best_name = None

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    for name, pipeline in models.items():
        print(f"\nTreinando modelo: {name}")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {acc:.4f}")
        print("Matriz de confusão:")
        print(confusion_matrix(y_test, y_pred))
        print("Relatório de classificação:")
        print(classification_report(y_test, y_pred))

        results.append({
            "model": name,
            "accuracy": acc
        })

        report_path = reports_dir / f"{name}_classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))

        if acc > best_score:
            best_score = acc
            best_model = pipeline
            best_name = name

    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    results_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    if best_model is not None:
        model_path = models_dir / "best_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"\nMelhor modelo salvo em: {model_path}")
        print(f"Melhor modelo: {best_name} | Acurácia: {best_score:.4f}")

    print("\nResumo dos modelos:")
    print(results_df)


if __name__ == "__main__":
    main()