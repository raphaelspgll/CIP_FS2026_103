import matplotlib.pyplot as plt
import pandas as pd

from src.models.bitcoin.config import FEATURE_COLS, LAG_STEPS
from src.visualization.bitcoin.config import (
    DPI,
    FIGSIZE_STANDARD,
    LOGISTIC_COEFFICIENTS_PATH,
    PLOTS_DIR,
    TREE_FEATURE_IMPORTANCE_PATH,
)


def ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_lagged_feature_names() -> list[str]:
    return [f"{col}_lag{LAG_STEPS}" for col in FEATURE_COLS]


def plot_logistic_coefficients(logistic_model) -> None:
    """
    Plot and save logistic regression coefficients.
    Assumes a sklearn Pipeline with final step named 'model'.
    """
    ensure_plots_dir()

    feature_names = get_lagged_feature_names()
    coefficients = logistic_model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    ).sort_values("coefficient")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.barh(coef_df["feature"], coef_df["coefficient"])
    ax.set_title("Bitcoin Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(LOGISTIC_COEFFICIENTS_PATH, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_tree_feature_importance(tree_model) -> None:
    """
    Plot and save decision tree feature importances.
    """
    ensure_plots_dir()

    feature_names = get_lagged_feature_names()
    importances = tree_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance")

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title("Bitcoin Decision Tree Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(TREE_FEATURE_IMPORTANCE_PATH, dpi=DPI, bbox_inches="tight")
    plt.close()