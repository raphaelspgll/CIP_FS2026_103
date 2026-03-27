from __future__ import annotations

from .evaluate import main as evaluate_main
from .train_logistic import save_model as save_logistic_model
from .train_logistic import train_logistic_model
from .train_tree import save_model as save_tree_model
from .train_tree import train_tree_model


def main() -> None:
    """
    Run the full Bitcoin modelling pipeline:
    1. Train logistic regression
    2. Save logistic regression model
    3. Train decision tree
    4. Save decision tree model
    5. Evaluate both models
    """
    print("Starting Bitcoin model pipeline...")

    print("\n[1/3] Training Logistic Regression...")
    logistic_model = train_logistic_model()
    save_logistic_model(logistic_model)
    print("Logistic Regression model saved.")

    print("\n[2/3] Training Decision Tree...")
    tree_model = train_tree_model()
    save_tree_model(tree_model)
    print("Decision Tree model saved.")

    print("\n[3/3] Evaluating models...")
    evaluate_main()
    print("Evaluation complete.")

    print("\nBitcoin pipeline finished successfully.")


if __name__ == "__main__":
    main()