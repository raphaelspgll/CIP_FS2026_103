from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.bitcoin.config import LOGISTIC_MODEL_PATH, LOGISTIC_PARAMS
from src.models.bitcoin.data import load_and_split_data


def build_logistic_pipeline() -> Pipeline:
    """
    Build a logistic regression pipeline with feature scaling.

    StandardScaler is recommended for Logistic Regression so that
    features are on comparable scales.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )
    return pipeline


def train_logistic_model() -> Pipeline:
    """
    Load Bitcoin data, train the logistic regression pipeline,
    and save the fitted model to disk.
    """
    dataset = load_and_split_data()

    model = build_logistic_pipeline()
    model.fit(dataset.X_train, dataset.y_train)

    return model


def save_model(model: Pipeline, output_path: Path = LOGISTIC_MODEL_PATH) -> None:
    """
    Save the trained model to disk as a pickle file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def main() -> None:
    """
    Train and save the Bitcoin logistic regression model.
    """
    model = train_logistic_model()
    save_model(model)
    print(f"Saved logistic regression model to: {LOGISTIC_MODEL_PATH}")


if __name__ == "__main__":
    main()