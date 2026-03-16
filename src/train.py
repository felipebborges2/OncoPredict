from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc_curves(trained_models, X_test, y_test, figures_dir):
    figures_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))

    for name, pipeline in trained_models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Aleatório (AUC = 0.500)")
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    plt.ylabel("Sensibilidade (Taxa de Verdadeiros Positivos)")
    plt.title("Curvas ROC — Comparação de Modelos")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()
    print("\nCurvas ROC salvas em figures/roc_curves.png")


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
    trained_models = {}
    best_model = None
    best_score = -1
    best_name = None

    models_dir = Path("models")
    reports_dir = Path("reports")
    figures_dir = Path("figures")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    for name, pipeline in models.items():
        print(f"\nTreinando modelo: {name}")

        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"Cross-validation (5 folds) — Média: {cv_mean:.4f} | Desvio padrão: {cv_std:.4f}")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        print(f"Acurácia no conjunto de teste: {acc:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print("Matriz de confusão:")
        print(confusion_matrix(y_test, y_pred))
        print("Relatório de classificação:")
        print(classification_report(y_test, y_pred))

        results.append({
            "model": name,
            "accuracy": acc,
            "auc": roc_auc,
            "cv_mean": cv_mean,
            "cv_std": cv_std
        })

        trained_models[name] = pipeline

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

    plot_roc_curves(trained_models, X_test, y_test, figures_dir)

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