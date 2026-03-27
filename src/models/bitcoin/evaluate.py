from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.bitcoin.config import (
    LOGISTIC_METRICS_PATH,
    LOGISTIC_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    TREE_METRICS_PATH,
    TREE_MODEL_PATH,
)
from src.models.bitcoin.data import load_and_split_data


def load_model(model_path: Path):
    """
    Load a saved model from disk.
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_metrics(model, X, y, split_name: str) -> dict:
    """
    Compute evaluation metrics for one dataset split.
    """
    y_pred = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """
    Save metrics dictionary as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def evaluate_logistic() -> dict:
    """
    Evaluate the saved logistic regression model on train/val/test.
    """
    dataset = load_and_split_data()
    model = load_model(LOGISTIC_MODEL_PATH)

    results = {
        "model_name": "logistic_regression",
        "train": compute_metrics(model, dataset.X_train, dataset.y_train, "train"),
        "validation": compute_metrics(model, dataset.X_val, dataset.y_val, "validation"),
        "test": compute_metrics(model, dataset.X_test, dataset.y_test, "test"),
    }

    save_metrics(results, LOGISTIC_METRICS_PATH)
    return results


def evaluate_tree() -> dict:
    """
    Evaluate the saved decision tree model on train/val/test.
    """
    dataset = load_and_split_data()
    model = load_model(TREE_MODEL_PATH)

    results = {
        "model_name": "decision_tree",
        "train": compute_metrics(model, dataset.X_train, dataset.y_train, "train"),
        "validation": compute_metrics(model, dataset.X_val, dataset.y_val, "validation"),
        "test": compute_metrics(model, dataset.X_test, dataset.y_test, "test"),
    }

    save_metrics(results, TREE_METRICS_PATH)
    return results


def build_comparison_table(logistic_results: dict, tree_results: dict) -> pd.DataFrame:
    """
    Build a compact comparison table for accuracy across splits.
    """
    comparison_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "train_accuracy": logistic_results["train"]["accuracy"],
                "validation_accuracy": logistic_results["validation"]["accuracy"],
                "test_accuracy": logistic_results["test"]["accuracy"],
            },
            {
                "model": "decision_tree",
                "train_accuracy": tree_results["train"]["accuracy"],
                "validation_accuracy": tree_results["validation"]["accuracy"],
                "test_accuracy": tree_results["test"]["accuracy"],
            },
        ]
    )

    return comparison_df


def save_comparison_table(comparison_df: pd.DataFrame) -> None:
    """
    Save the model comparison table as CSV.
    """
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)


def main() -> None:
    """
    Evaluate both saved Bitcoin models and save results.
    """
    logistic_results = evaluate_logistic()
    tree_results = evaluate_tree()

    comparison_df = build_comparison_table(logistic_results, tree_results)
    save_comparison_table(comparison_df)

    print("Saved evaluation files:")
    print(f"- {LOGISTIC_METRICS_PATH}")
    print(f"- {TREE_METRICS_PATH}")
    print(f"- {MODEL_COMPARISON_PATH}")
    print("\nModel comparison:")
    print(comparison_df)


if __name__ == "__main__":
    main()