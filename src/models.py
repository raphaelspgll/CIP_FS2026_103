import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import FEATURE_COLS  # single source of truth for feature column names

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TARGET_COL = "price_direction"


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split per coin by date with no shuffle; recombine for return."""
    trains, tests = [], []
    for _, group in df.groupby("coin_id"):
        group = group.sort_values("date").reset_index(drop=True)
        split_idx = int(len(group) * (1 - test_size))
        trains.append(group.iloc[:split_idx])
        tests.append(group.iloc[split_idx:])
    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)


def evaluate_model(model, train, test, feature_cols, target_col):
    X_train, y_train = train[feature_cols], train[target_col]
    X_test,  y_test  = test[feature_cols],  test[target_col]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if isinstance(model, LinearRegression):
        preds = (preds >= 0.5).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
    }


def main():
    dfs = []
    for path in glob.glob(os.path.join(PROCESSED_DIR, "*_features.csv")):
        dfs.append(pd.read_csv(path, parse_dates=["date"]))
    if not dfs:
        print("No feature files found. Run features.py first.")
        return

    df = pd.concat(dfs, ignore_index=True).dropna(subset=FEATURE_COLS + [TARGET_COL])
    train, test = chronological_split(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree":       DecisionTreeClassifier(),
        "LinearRegression":   LinearRegression(),
    }

    for coin_id in sorted(df["coin_id"].unique()):
        print(f"\n=== {coin_id} ===")
        coin_train = train[train["coin_id"] == coin_id]
        coin_test  = test[test["coin_id"]  == coin_id]
        for name, model in models.items():
            m = evaluate_model(model, coin_train, coin_test, FEATURE_COLS, TARGET_COL)
            print(
                f"  {name:25s}  acc={m['accuracy']:.4f}  "
                f"prec={m['precision']:.4f}  "
                f"rec={m['recall']:.4f}  "
                f"f1={m['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
