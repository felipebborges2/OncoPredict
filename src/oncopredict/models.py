"""Candidate models, hyperparameter grids and model selection."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from oncopredict.config import CV_FOLDS, RANDOM_STATE, SCORING

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
}


def make_cv():
    """Shuffled, stratified folds with a fixed seed, so every run sees the same splits."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_models():
    # The scaler lives inside each pipeline so it is fitted only on the
    # training folds during cross-validation (no information leaks from
    # the validation fold into the normalization statistics).
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            # probability=True fits an internal Platt scaling with its own
            # random CV; without random_state results change on every run.
            ("model", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def build_param_grids():
    """Grids are named '<pipeline step>__<parameter>', as required by Pipeline."""
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3],
        },
        "svm": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale"],
        },
    }


def tune_models(X_train, y_train):
    """Run a grid search per model on the training set only.

    Returns {name: fitted GridSearchCV}. The test set is never touched here.
    """
    models = build_models()
    grids = build_param_grids()
    searches = {}
    for name, pipeline in models.items():
        search = GridSearchCV(pipeline, grids[name], cv=make_cv(), scoring=SCORING, n_jobs=-1)
        search.fit(X_train, y_train)
        searches[name] = search
    return searches


def cv_summary(search):
    """Mean and standard deviation of the CV score for the best configuration."""
    i = search.best_index_
    return search.cv_results_["mean_test_score"][i], search.cv_results_["std_test_score"][i]


def select_best(searches):
    """Pick the model with the highest mean cross-validated score.

    Selecting on the test set would turn it into part of the training process
    and make the reported test performance optimistically biased.
    """
    return max(searches, key=lambda name: cv_summary(searches[name])[0])
