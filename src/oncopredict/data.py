"""Dataset loading and train/test split."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from oncopredict.config import RANDOM_STATE, TEST_SIZE

# Index = label. Label 1 (malignant) is the positive class throughout the project.
CLASS_NAMES = ("benign", "malignant")
FEATURE_NAMES = [str(name) for name in load_breast_cancer().feature_names]


def load_data():
    """Return features X and target y, where y = 1 means malignant.

    scikit-learn ships this dataset with 0 = malignant and 1 = benign. Using
    that encoding directly makes predict_proba(X)[:, 1] the probability of
    a *benign* tumor, which silently inverts sensitivity, false negatives and
    threshold reasoning. The labels are flipped here, once, so that every
    downstream metric refers to detecting cancer.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int).rename("malignant")
    return X, y


def split_data(X, y):
    """Stratified hold-out split. The test set is used once, for final reporting."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
