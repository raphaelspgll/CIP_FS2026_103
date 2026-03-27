import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.visualization.bitcoin.config import (
    CONFUSION_MATRIX_LOGISTIC_PATH,
    CONFUSION_MATRIX_TREE_PATH,
    DPI,
    FIGSIZE_STANDARD,
    PLOTS_DIR,
)


def ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    confusion_matrix: list | np.ndarray,
    title: str,
    output_path,
) -> None:
    """
    Plot and save a confusion matrix.
    """
    ensure_plots_dir()

    cm = np.array(confusion_matrix)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_logistic_confusion_matrix(logistic_metrics: dict) -> None:
    """
    Plot confusion matrix for logistic regression using test metrics.
    """
    cm = logistic_metrics["test"]["confusion_matrix"]
    plot_confusion_matrix(
        confusion_matrix=cm,
        title="Bitcoin Logistic Regression Confusion Matrix (Test Set)",
        output_path=CONFUSION_MATRIX_LOGISTIC_PATH,
    )


def plot_tree_confusion_matrix(tree_metrics: dict) -> None:
    """
    Plot confusion matrix for decision tree using test metrics.
    """
    cm = tree_metrics["test"]["confusion_matrix"]
    plot_confusion_matrix(
        confusion_matrix=cm,
        title="Bitcoin Decision Tree Confusion Matrix (Test Set)",
        output_path=CONFUSION_MATRIX_TREE_PATH,
    )