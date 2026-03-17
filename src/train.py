from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt


def plot_threshold_analysis(best_model, best_name, X_test, y_test, reports_dir, figures_dir):
    """
    Analisa o impacto de diferentes limiares de decisão no melhor modelo.

    Por padrão, modelos classificam como positivo (maligno) quando a
    probabilidade >= 0.5. Mas em oncologia, pode ser preferível usar um
    limiar menor para maximizar a detecção de casos malignos, aceitando
    mais falsos alarmes em troca de não perder nenhum câncer real.

    O índice de Youden = sensibilidade + especificidade - 1
    aponta o limiar que maximiza ambas simultaneamente.
    """
    import numpy as np

    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    specificity = 1 - fpr
    youden_index = tpr + specificity - 1
    best_idx = youden_index.argmax()
    best_threshold = thresholds[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = specificity[best_idx]

    # Avalia o modelo com o limiar padrão (0.5) e o limiar ótimo
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= best_threshold).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Sensibilidade e especificidade por limiar
    axes[0].plot(thresholds, tpr[:-1] if len(thresholds) < len(tpr) else tpr,
                 label="Sensibilidade", color="#e74c3c", linewidth=2)
    axes[0].plot(thresholds, specificity[:-1] if len(thresholds) < len(specificity) else specificity,
                 label="Especificidade", color="#2ecc71", linewidth=2)
    axes[0].axvline(best_threshold, color="#3498db", linestyle="--", linewidth=1.5,
                    label=f"Limiar ótimo ({best_threshold:.2f})")
    axes[0].axvline(0.5, color="gray", linestyle=":", linewidth=1.5,
                    label="Limiar padrão (0.50)")
    axes[0].set_xlabel("Limiar de decisão")
    axes[0].set_ylabel("Taxa")
    axes[0].set_title("Sensibilidade vs. Especificidade por limiar")
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])

    # Gráfico 2: Matrizes de confusão — padrão vs. ótimo
    for ax, y_pred, title in [
        (None, None, None),  # placeholder para alinhar layout
    ]:
        pass

    cm_default = confusion_matrix(y_test, y_pred_default)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)

    import numpy as np
    x = [0, 1]
    labels = ["Maligno", "Benigno"]

    for i, (cm, title, color) in enumerate([
        (cm_default, f"Limiar padrão (0.50)\nFN={cm_default[1,0]} | FP={cm_default[0,1]}", "#95a5a6"),
        (cm_optimal, f"Limiar ótimo ({best_threshold:.2f})\nFN={cm_optimal[1,0]} | FP={cm_optimal[0,1]}", "#3498db"),
    ]):
        ax = axes[1] if i == 0 else None
        if i == 1:
            break
        im = axes[1].imshow(cm, cmap="Blues")
        axes[1].set_title(title, fontsize=10)
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Prev: Maligno", "Prev: Benigno"])
        axes[1].set_yticklabels(["Real: Maligno", "Real: Benigno"])
        for r in range(2):
            for c in range(2):
                axes[1].text(c, r, cm[r, c], ha="center", va="center",
                             fontsize=14, color="white" if cm[r, c] > cm.max() / 2 else "black")

    plt.suptitle(f"Análise de limiar de decisão — {best_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_analysis.png", dpi=150)
    plt.close()

    # Salva relatório textual
    report_path = reports_dir / "threshold_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Análise de Limiar de Decisão — {best_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Limiar padrão (0.50):\n")
        f.write(f"  Falsos Negativos (cânceres não detectados): {cm_default[1,0]}\n")
        f.write(f"  Falsos Positivos (alarmes falsos):          {cm_default[0,1]}\n")
        f.write(f"  Sensibilidade: {(cm_default[1,1] / (cm_default[1,0] + cm_default[1,1])):.4f}\n")
        f.write(f"  Especificidade: {(cm_default[0,0] / (cm_default[0,0] + cm_default[0,1])):.4f}\n\n")
        f.write(f"Limiar ótimo pelo índice de Youden ({best_threshold:.4f}):\n")
        f.write(f"  Falsos Negativos (cânceres não detectados): {cm_optimal[1,0]}\n")
        f.write(f"  Falsos Positivos (alarmes falsos):          {cm_optimal[0,1]}\n")
        f.write(f"  Sensibilidade: {best_sensitivity:.4f}\n")
        f.write(f"  Especificidade: {best_specificity:.4f}\n")

    print(f"\nAnálise de limiar salva em reports/threshold_analysis.txt")
    print(f"Limiar ótimo (Youden): {best_threshold:.4f} "
          f"| Sensibilidade: {best_sensitivity:.4f} "
          f"| Especificidade: {best_specificity:.4f}")


def plot_learning_curve(best_model, best_name, X, y, figures_dir):
    """
    Gera a curva de aprendizado do melhor modelo.

    Treina o modelo repetidamente com subconjuntos crescentes dos dados
    (de 10% a 100%) e plota a acurácia de treino e de validação cruzada
    para cada tamanho. Isso responde: "mais dados melhorariam o modelo?"

    - Se a curva de validação ainda está subindo no final → mais dados ajudariam
    - Se ambas as curvas convergem em valor alto → o modelo está saturado (bom sinal)
    - Se há grande gap entre treino e validação → overfitting
    """
    import numpy as np

    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X, y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#3498db", label="Acurácia de treino")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color="#3498db")
    plt.plot(train_sizes, val_mean, "o-", color="#e74c3c", label="Acurácia de validação (CV)")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color="#e74c3c")

    plt.xlabel("Tamanho do conjunto de treinamento (amostras)")
    plt.ylabel("Acurácia")
    plt.title(f"Curva de Aprendizado — {best_name}")
    plt.legend(loc="lower right")
    plt.ylim(0.85, 1.01)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "learning_curve.png", dpi=150)
    plt.close()
    print("\nCurva de aprendizado salva em figures/learning_curve.png")


def plot_shap_values(trained_models, X_train, X_test, feature_names, figures_dir):
    """
    Gera explicações SHAP para o Random Forest.

    SHAP (SHapley Additive exPlanations) calcula a contribuição de cada
    feature para cada predição individual — não apenas a importância global.

    Usamos o Random Forest com TreeExplainer (exato e rápido) em vez do
    SVM, pois modelos lineares e baseados em kernel requerem KernelExplainer,
    que é uma aproximação lenta. Em projetos reais, é prática comum usar
    um modelo de árvore especificamente para análise de interpretabilidade.
    """
    import shap

    rf_pipeline = trained_models["random_forest"]
    rf_model = rf_pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(rf_model)

    # shap_values retorna um objeto Explanation com shape (n_samples, n_features, n_classes)
    # Usamos classe 0 (maligno) pois é o de maior interesse clínico
    shap_explanation = explainer(X_test)
    shap_malignant = shap_explanation[:, :, 0]

    # Beeswarm plot — visão global: impacto de cada feature em todas as amostras
    plt.figure()
    shap.plots.beeswarm(shap_malignant, show=False, max_display=20)
    plt.title("SHAP — Impacto das features na classificação de tumor maligno", pad=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfall plot — visão individual: explicação de uma única amostra
    plt.figure()
    shap.plots.waterfall(shap_malignant[0], show=False, max_display=15)
    plt.title("SHAP — Explicação individual (amostra #1)", pad=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nGráficos SHAP salvos em figures/")


def plot_feature_importance(trained_models, feature_names, figures_dir):
    figures_dir.mkdir(exist_ok=True)

    # Random Forest — importância baseada em quanto cada feature reduz a impureza nas árvores
    rf_pipeline = trained_models["random_forest"]
    rf_importances = pd.Series(
        rf_pipeline.named_steps["model"].feature_importances_,
        index=feature_names
    ).sort_values(ascending=True).tail(15)

    plt.figure(figsize=(8, 7))
    rf_importances.plot(kind="barh", color="#3498db")
    plt.title("Random Forest — Top 15 Features mais importantes")
    plt.xlabel("Importância (redução média de impureza)")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_rf.png", dpi=150)
    plt.close()

    # Regressão Logística — coeficientes indicam o peso de cada feature na decisão
    lr_pipeline = trained_models["logistic_regression"]
    lr_coefs = pd.Series(
        lr_pipeline.named_steps["model"].coef_[0],
        index=feature_names
    ).sort_values(ascending=True)
    top_lr = pd.concat([lr_coefs.head(10), lr_coefs.tail(10)]).sort_values()

    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in top_lr.values]
    plt.figure(figsize=(8, 7))
    top_lr.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Regressão Logística — Top features por coeficiente\n(verde = empurra para benigno | vermelho = empurra para maligno)")
    plt.xlabel("Coeficiente (após normalização)")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_lr.png", dpi=150)
    plt.close()

    print("\nGráficos de feature importance salvos em figures/")


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
            ("model", RandomForestClassifier(random_state=42))
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True))
        ])
    }
    return models


def build_param_grids():
    """
    Define as grades de hiperparâmetros para cada modelo.

    No Pipeline do sklearn, os parâmetros são nomeados como
    '<nome_do_passo>__<nome_do_parâmetro>'. Por isso usamos
    'model__C', 'model__n_estimators', etc.
    """
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "svm": {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    }


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = build_models()
    param_grids = build_param_grids()

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
        print(f"\n{'='*50}")
        print(f"Modelo: {name}")
        print(f"{'='*50}")

        # GridSearchCV testa todas as combinações de hiperparâmetros via
        # validação cruzada (cv=5) e retorna o melhor modelo automaticamente.
        # n_jobs=-1 usa todos os núcleos disponíveis para paralelizar.
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        cv_mean = grid_search.best_score_
        cv_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

        print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
        print(f"CV Acurácia (5 folds) — Média: {cv_mean:.4f} | Desvio padrão: {cv_std:.4f}")

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
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
            "cv_std": cv_std,
            "best_params": str(grid_search.best_params_)
        })

        trained_models[name] = best_pipeline

        report_path = reports_dir / f"{name}_classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(f"Best Hyperparameters:\n{grid_search.best_params_}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))

        if acc > best_score:
            best_score = acc
            best_model = best_pipeline
            best_name = name

    plot_roc_curves(trained_models, X_test, y_test, figures_dir)
    plot_learning_curve(best_model, best_name, X, y, figures_dir)
    plot_shap_values(trained_models, X_train, X_test, X.columns.tolist(), figures_dir)
    plot_feature_importance(trained_models, X.columns.tolist(), figures_dir)
    plot_threshold_analysis(best_model, best_name, X_test, y_test, reports_dir, figures_dir)

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