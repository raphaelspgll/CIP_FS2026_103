from src.visualization.bitcoin.load import (
    load_logistic_metrics,
    load_logistic_model,
    load_model_comparison,
    load_tree_metrics,
    load_tree_model,
)
from src.visualization.bitcoin.plots_classification import (
    plot_logistic_confusion_matrix,
    plot_tree_confusion_matrix,
)
from src.visualization.bitcoin.plots_model import (
    plot_logistic_coefficients,
    plot_tree_feature_importance,
)
from src.visualization.bitcoin.plots_summary import (
    plot_accuracy_comparison,
)


def main() -> None:
    print("Starting Bitcoin visualization pipeline...")

    # -------------------------------------------------------------------------
    # Load artifacts
    # -------------------------------------------------------------------------

    logistic_model = load_logistic_model()
    tree_model = load_tree_model()

    logistic_metrics = load_logistic_metrics()
    tree_metrics = load_tree_metrics()

    comparison_df = load_model_comparison()

    # -------------------------------------------------------------------------
    # Classification plots
    # -------------------------------------------------------------------------

    print("\n[1/3] Plotting confusion matrices...")
    plot_logistic_confusion_matrix(logistic_metrics)
    plot_tree_confusion_matrix(tree_metrics)

    # -------------------------------------------------------------------------
    # Model interpretation plots
    # -------------------------------------------------------------------------

    print("\n[2/3] Plotting model insights...")
    plot_logistic_coefficients(logistic_model)
    plot_tree_feature_importance(tree_model)

    # -------------------------------------------------------------------------
    # Summary plots
    # -------------------------------------------------------------------------

    print("\n[3/3] Plotting summary comparison...")
    plot_accuracy_comparison(comparison_df)

    print("\nBitcoin visualization pipeline finished successfully.")


if __name__ == "__main__":
    main()