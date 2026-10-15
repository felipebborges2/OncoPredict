"""Project-wide settings. Every path is resolved from the repository root, so
scripts work no matter which directory they are launched from."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ROC AUC is threshold-independent: it measures how well the model ranks
# malignant cases above benign ones. The operating threshold is chosen in a
# separate step, so tuning should not depend on the arbitrary 0.5 cut-off.
SCORING = "roc_auc"

# Clinical operating point: the highest threshold whose out-of-fold
# sensitivity (share of cancers detected) is at least this value.
TARGET_SENSITIVITY = 0.98

N_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95
