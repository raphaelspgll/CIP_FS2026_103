from pathlib import Path

# -----------------------------------------------------------------------------
# Base paths
# -----------------------------------------------------------------------------

BITCOIN_VIZ_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = BITCOIN_VIZ_DIR.parent
SRC_DIR = VISUALIZATION_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# -----------------------------------------------------------------------------
# Input artifacts from modelling pipeline
# -----------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models" / "bitcoin"
REPORTS_DIR = PROJECT_ROOT / "reports" / "bitcoin"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_bitcoin.pkl"
TREE_MODEL_PATH = MODELS_DIR / "decision_tree_bitcoin.pkl"

LOGISTIC_METRICS_PATH = REPORTS_DIR / "logistic_metrics.json"
TREE_METRICS_PATH = REPORTS_DIR / "tree_metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# -----------------------------------------------------------------------------
# Output directory for plots
# -----------------------------------------------------------------------------

PLOTS_DIR = REPORTS_DIR / "plots"

CONFUSION_MATRIX_LOGISTIC_PATH = PLOTS_DIR / "confusion_matrix_logistic.png"
CONFUSION_MATRIX_TREE_PATH = PLOTS_DIR / "confusion_matrix_tree.png"
LOGISTIC_COEFFICIENTS_PATH = PLOTS_DIR / "logistic_coefficients.png"
TREE_FEATURE_IMPORTANCE_PATH = PLOTS_DIR / "tree_feature_importance.png"
ACCURACY_COMPARISON_PATH = PLOTS_DIR / "accuracy_comparison.png"

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

FIGSIZE_STANDARD = (8, 5)
FIGSIZE_WIDE = (10, 6)
DPI = 300