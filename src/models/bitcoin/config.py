from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BITCOIN_DIR = Path(__file__).resolve().parent
SRC_DIR = BITCOIN_DIR.parent.parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models" / "bitcoin"
REPORTS_DIR = PROJECT_ROOT / "reports" / "bitcoin"

# Input data file
BITCOIN_DATA_PATH = PROCESSED_DATA_DIR / "bitcoin_features.csv"

# Output files
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_bitcoin.pkl"
TREE_MODEL_PATH = MODELS_DIR / "decision_tree_bitcoin.pkl"

LOGISTIC_METRICS_PATH = REPORTS_DIR / "logistic_metrics.json"
TREE_METRICS_PATH = REPORTS_DIR / "tree_metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# -----------------------------------------------------------------------------
# Data columns
# -----------------------------------------------------------------------------

DATE_COL = "date"
TARGET_COL = "price_direction"

# Feature set recommended by your EDA for t-1 -> t prediction
FEATURE_COLS = [
    "log_return",
    "log_close_open_ratio",
    "log_high_low_ratio",
    "volatility_7",
    "log_volume_change",
]

# Optional columns that may exist but are explicitly excluded
EXCLUDED_COLS = [
    "log_close",
    "ma_7",
    "ma_30",
    "log_market_cap",
    "log_volume",
]

# -----------------------------------------------------------------------------
# Time-series setup
# -----------------------------------------------------------------------------

# We predict y(t) from X(t-1), so features will be shifted by 1 day.
LAG_STEPS = 1

# -----------------------------------------------------------------------------
# Time-series split (chronological)
# -----------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-8

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Logistic Regression settings
# -----------------------------------------------------------------------------

LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

# -----------------------------------------------------------------------------
# Decision Tree settings
# -----------------------------------------------------------------------------

TREE_PARAMS = {
    "max_depth": 4,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
}

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0