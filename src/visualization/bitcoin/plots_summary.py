import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.bitcoin.config import (
    ACCURACY_COMPARISON_PATH,
    DPI,
    FIGSIZE_WIDE,
    PLOTS_DIR,
)


def ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_accuracy_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Plot and save accuracy comparison for both models across splits.
    """
    ensure_plots_dir()

    # reshape for plotting
    df_melted = comparison_df.melt(
        id_vars="model",
        value_vars=["train_accuracy", "validation_accuracy", "test_accuracy"],
        var_name="split",
        value_name="accuracy",
    )

    # clean split names
    df_melted["split"] = df_melted["split"].str.replace("_accuracy", "")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for model in df_melted["model"].unique():
        subset = df_melted[df_melted["model"] == model]
        ax.plot(subset["split"], subset["accuracy"], marker="o", label=model)

    ax.set_title("Bitcoin Model Accuracy Comparison")
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Accuracy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(ACCURACY_COMPARISON_PATH, dpi=DPI, bbox_inches="tight")
    plt.close()