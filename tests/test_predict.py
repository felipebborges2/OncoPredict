import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oncopredict.data import CLASS_NAMES, FEATURE_NAMES, load_data
from predict import SYNTHETIC_SAMPLE, load_input, out_of_range, predict


@pytest.fixture(scope="module")
def artifact():
    X, y = load_data()
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, y)
    return {
        "pipeline": pipeline,
        "threshold": 0.3,
        "class_names": CLASS_NAMES,
        "feature_ranges": X.agg(["min", "max"]).T,
    }


def test_load_input_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"mean radius": [10.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_input(path, FEATURE_NAMES)


def test_load_input_drops_extra_columns_and_orders_features(tmp_path):
    X, _ = load_data()
    df = X.head(3)[FEATURE_NAMES[::-1]].assign(patient_id=[1, 2, 3])
    path = tmp_path / "ok.csv"
    df.to_csv(path, index=False)
    loaded = load_input(path, FEATURE_NAMES)
    assert list(loaded.columns) == FEATURE_NAMES


def test_predict_uses_stored_threshold(artifact):
    X, _ = load_data()
    results = predict(artifact, X.head(20))
    expected = results.p_malignant.ge(artifact["threshold"]).map({True: "malignant", False: "benign"})
    assert (results.prediction == expected).all()


def test_synthetic_sample_is_classified_malignant(artifact):
    X = pd.DataFrame([SYNTHETIC_SAMPLE], columns=FEATURE_NAMES)
    assert predict(artifact, X).prediction.iloc[0] == "malignant"


def test_out_of_range_flags_extrapolation(artifact):
    X, _ = load_data()
    sample = X.head(1).copy()
    sample["mean radius"] = 1000.0
    assert out_of_range(sample, artifact["feature_ranges"]) == {0: ["mean radius"]}
