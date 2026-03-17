"""
predict.py — Inferência com o modelo treinado do OncoPredict.

Uso:
    python src/predict.py                        # usa amostra sintética
    python src/predict.py --input dados.csv      # usa arquivo CSV externo

O CSV deve conter uma linha por amostra e uma coluna por feature,
com os mesmos nomes de colunas usados no treinamento.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer


EXPECTED_FEATURES = load_breast_cancer().feature_names.tolist()


def load_input(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"O arquivo CSV está faltando as seguintes colunas:\n{missing}"
        )

    extra = [c for c in df.columns if c not in EXPECTED_FEATURES]
    if extra:
        print(f"  Aviso: colunas ignoradas (não fazem parte do modelo): {extra}")

    return df[EXPECTED_FEATURES]


def build_synthetic_sample() -> pd.DataFrame:
    """
    Cria uma amostra sintética de tumor com valores realistas.

    Os valores simulam um tumor maligno hipotético — baseados nas médias
    do dataset Breast Cancer Wisconsin para a classe maligna.
    Não representa nenhum caso real.
    """
    synthetic_values = [
        17.5, 21.0, 115.0, 950.0, 0.110,
        0.160, 0.180, 0.095, 0.210, 0.065,
        0.80,  1.20,   5.50, 75.0, 0.007,
        0.025, 0.030, 0.012, 0.018, 0.004,
        22.0, 28.0, 145.0, 1500.0, 0.145,
        0.380, 0.450, 0.175, 0.350, 0.110,
    ]
    return pd.DataFrame([synthetic_values], columns=EXPECTED_FEATURES)


def print_result(sample: pd.DataFrame, model, source: str):
    class_names = load_breast_cancer().target_names

    print("\n" + "=" * 52)
    print("  OncoPredict — Resultado da Inferência")
    print("=" * 52)
    print(f"  Fonte dos dados : {source}")
    print(f"  Amostras        : {len(sample)}")

    for i, row in sample.iterrows():
        prediction = model.predict(row.to_frame().T)[0]
        probabilities = model.predict_proba(row.to_frame().T)[0]
        predicted_class = class_names[prediction]

        print(f"\n  --- Amostra #{i + 1} ---")
        print(f"  Classe prevista : {predicted_class.upper()}")
        print("  Probabilidades  :")
        for class_name, prob in zip(class_names, probabilities):
            bar = "█" * int(prob * 28)
            print(f"    {class_name:<12} {prob:.1%}  {bar}")

        if predicted_class == "malignant":
            print("  ⚠  Predição indica tumor MALIGNO.")
        else:
            print("  ✓  Predição indica tumor BENIGNO.")

    print("\n" + "=" * 52)
    print("  AVISO: Este sistema é educacional.")
    print("  Não substitui diagnóstico médico profissional.")
    print("=" * 52)


def main():
    parser = argparse.ArgumentParser(
        description="OncoPredict — inferência em novos dados clínicos."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Caminho para um arquivo CSV com as features do tumor. "
             "Se não informado, usa uma amostra sintética.",
    )
    args = parser.parse_args()

    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode src/train.py primeiro.")

    model = joblib.load(model_path)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
        sample = load_input(args.input)
        source = str(input_path)
    else:
        sample = build_synthetic_sample()
        source = "amostra sintética (padrão)"

    print_result(sample, model, source)


if __name__ == "__main__":
    main()
