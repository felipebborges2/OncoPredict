# OncoPredict-ML

Pipeline completo de Machine Learning para classificação de tumores de mama, aplicado ao dataset Breast Cancer Wisconsin.

> **Projeto educacional e de portfólio.** Não tem finalidade clínica e não substitui diagnóstico médico profissional.

Desenvolvido por **Felipe Borges** — Estudante de Física Médica @ UFCSPA, na intersecção entre IA, Ciência de Dados e Tecnologia em Saúde.

---

## Sobre o projeto

O objetivo é demonstrar um pipeline de ML supervisionado end-to-end aplicado a um problema de classificação clínica, cobrindo todas as etapas fundamentais:

- Análise exploratória de dados
- Pré-processamento e normalização de features
- Treinamento e comparação de múltiplos algoritmos
- Avaliação com métricas robustas (acurácia, AUC, validação cruzada, matriz de confusão)
- Análise de feature importance
- Serialização do melhor modelo e inferência em novos dados

**Dataset:** [Breast Cancer Wisconsin](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) — 569 amostras, 30 features morfológicas de tumores, classificação binária (maligno/benigno).

---

## Resultados

| Modelo | Acurácia | AUC | CV Médio | CV Std |
|---|---|---|---|---|
| **Logistic Regression** | **98.2%** | **0.9954** | **98.1%** | **0.0065** |
| SVM | 98.2% | 0.9950 | 97.4% | 0.0147 |
| Random Forest | 95.6% | 0.9931 | 95.8% | 0.0238 |

O melhor modelo foi a **Regressão Logística** — mesma acurácia que o SVM, mas com menor variação entre rodadas de validação cruzada, indicando maior estabilidade e melhor generalização.

### Curvas ROC
![ROC Curves](figures/roc_curves.png)

### Feature Importance
![Feature Importance RF](figures/feature_importance_rf.png)
![Feature Importance LR](figures/feature_importance_lr.png)

### Separabilidade das classes
![Class Separability](figures/class_separability.png)

---

## Estrutura do projeto

```
OncoPredict-ML/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # Análise exploratória dos dados
│   └── 02_model_training.ipynb         # Análise visual dos resultados
├── src/
│   ├── train.py                        # Pipeline de treinamento e avaliação
│   └── predict.py                      # Inferência em nova amostra
├── models/
│   └── best_model.pkl                  # Melhor modelo serializado
├── reports/
│   ├── model_comparison.csv            # Comparação de métricas
│   └── *_classification_report.txt    # Relatórios por modelo
├── figures/                            # Gráficos gerados
├── test_env.py                         # Verificação do ambiente
├── requirements.txt
└── README.md
```

---

## Como executar

### 1. Clone o repositório e configure o ambiente

```bash
git clone https://github.com/felipebborges2/OncoPredict.git
cd OncoPredict
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Treine os modelos

```bash
python src/train.py
```

Isso irá:
- Treinar os 3 modelos com validação cruzada
- Gerar relatórios em `reports/`
- Gerar gráficos (ROC, feature importance) em `figures/`
- Salvar o melhor modelo em `models/best_model.pkl`

### 3. Execute uma inferência

```bash
python src/predict.py
```

Roda o modelo treinado em uma amostra sintética com valores realistas, simulando a chegada de um novo caso.

### 4. Explore os notebooks

```bash
jupyter notebook
```

- `01_exploratory_analysis.ipynb` — análise do dataset antes do treinamento
- `02_model_training.ipynb` — visualização e interpretação dos resultados

---

## Stack

- Python 3.9+
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- joblib
- Jupyter Notebook

---

## Autor

**Felipe Borges**
Estudante de Física Médica @ UFCSPA
Interesses: IA aplicada à saúde, radiômica, aprendizado de máquina em oncologia
