from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from src.models.bitcoin.config import TREE_MODEL_PATH, TREE_PARAMS
from src.models.bitcoin.data import load_and_split_data


def build_tree_model() -> DecisionTreeClassifier:
    """
    Build a Decision Tree classifier using parameters from config.py.
    """
    model = DecisionTreeClassifier(**TREE_PARAMS)
    return model


def train_tree_model() -> DecisionTreeClassifier:
    """
    Load Bitcoin data, train the Decision Tree model,
    and return the fitted estimator.
    """
    dataset = load_and_split_data()

    model = build_tree_model()
    model.fit(dataset.X_train, dataset.y_train)

    return model


def save_model(
    model: DecisionTreeClassifier,
    output_path: Path = TREE_MODEL_PATH,
) -> None:
    """
    Save the trained Decision Tree model to disk as a pickle file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def main() -> None:
    """
    Train and save the Bitcoin Decision Tree model.
    """
    model = train_tree_model()
    save_model(model)
    print(f"Saved decision tree model to: {TREE_MODEL_PATH}")


if __name__ == "__main__":
    main()