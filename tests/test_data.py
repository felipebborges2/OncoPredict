from sklearn.datasets import load_breast_cancer

from oncopredict.data import CLASS_NAMES, load_data, split_data


def test_positive_class_is_malignant():
    X, y = load_data()
    raw = load_breast_cancer()
    malignant_raw = raw.target == list(raw.target_names).index("malignant")
    assert (y.to_numpy() == malignant_raw.astype(int)).all()
    assert y.sum() == 212  # malignant cases in the Wisconsin dataset
    assert CLASS_NAMES[1] == "malignant"


def test_split_is_stratified_and_reproducible():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 114
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert (split_data(X, y)[1].index == X_test.index).all()
