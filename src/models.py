import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import FEATURE_COLS  # single source of truth for feature column names

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
IMAGES_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
TARGET_COL    = "price_direction"

METRICS = ["accuracy", "precision", "recall", "f1"]


def chronological_split(df: pd.DataFrame, train_size: float = 0.70, val_size: float = 0.15):
    """Split each coin's data chronologically into train, validation, and test sets.

    Data is sorted by date per coin and split without shuffling to preserve
    temporal order and avoid data leakage. The remaining proportion after
    train_size and val_size is used as the test set.

    Args:
        df:         DataFrame containing all coins. Must have a 'coin' and 'date' column.
        train_size: Fraction of rows assigned to the training set. Default is 0.70.
        val_size:   Fraction of rows assigned to the validation set. Default is 0.15.
                    The test set receives the remaining 1 - train_size - val_size = 0.15.

    Returns:
        A tuple of three DataFrames: (train, val, test), each containing rows
        from all coins concatenated in chronological order.
    """
    trains, vals, tests = [], [], []
    for _, group in df.groupby("coin"):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        idx_train = int(n * train_size)
        idx_val   = int(n * (train_size + val_size))
        trains.append(group.iloc[:idx_train])
        vals.append(group.iloc[idx_train:idx_val])
        tests.append(group.iloc[idx_val:])
    return (
        pd.concat(trains, ignore_index=True),
        pd.concat(vals,   ignore_index=True),
        pd.concat(tests,  ignore_index=True),
    )


def evaluate_model(model, train: pd.DataFrame, eval_set: pd.DataFrame,
                   feature_cols: list, target_col: str) -> dict:
    """Fit a model on the training set and evaluate it on the given evaluation set.

    Args:
        model:        A scikit-learn estimator with fit() and predict() methods.
        train:        DataFrame used for fitting the model. Must contain feature_cols and target_col.
        eval_set:     DataFrame used for evaluation (validation or test set).
                      Must contain feature_cols and target_col.
        feature_cols: List of column names used as input features.
        target_col:   Name of the binary target column (0 or 1).

    Returns:
        A dict with four rounded metrics:
        {
            "accuracy":  float,
            "precision": float,
            "recall":    float,
            "f1":        float,
        }
    """
    X_train, y_train = train[feature_cols], train[target_col]
    X_eval,  y_eval  = eval_set[feature_cols], eval_set[target_col]
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    return {
        "accuracy":  round(accuracy_score(y_eval, preds), 4),
        "precision": round(precision_score(y_eval, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_eval, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_eval, preds, zero_division=0), 4),
    }


def plot_results(results: dict, save_dir: str) -> None:
    """Create and save a grouped bar chart of model evaluation results per coin.

    Produces one subplot per coin. Each subplot shows accuracy, precision, recall,
    and F1 for both models side by side on the test set. A red dashed line marks
    the 0.50 random baseline and a grey dashed line marks the 0.58 realistic
    accuracy ceiling derived from the EDA autocorrelation analysis.

    Args:
        results:  Nested dict of the form
                  {coin: {model_name: {"val": metrics_dict, "test": metrics_dict}}}
                  where metrics_dict contains keys accuracy, precision, recall, f1.
        save_dir: Directory where the output PNG will be saved.

    Returns:
        None. Saves the figure to save_dir/model_results.png and prints the path.
    """
    coins       = sorted(results.keys())
    model_names = list(next(iter(results.values())).keys())
    n_coins     = len(coins)
    x           = np.arange(len(METRICS))
    width       = 0.35
    colors      = ["#4C72B0", "#DD8452"]  # blue = LogisticRegression, orange = DecisionTree

    fig, axes = plt.subplots(1, n_coins, figsize=(5 * n_coins, 5), sharey=True)
    if n_coins == 1:
        axes = [axes]

    for ax, coin in zip(axes, coins):
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            test_metrics = results[coin][model_name]["test"]
            values = [test_metrics[m] for m in METRICS]
            offset = (i - (len(model_names) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, label=model_name,
                          color=color, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

        # Reference lines from EDA findings
        ax.axhline(0.50, color="red",  linestyle="--", linewidth=1.2, label="Random baseline (0.50)")
        ax.axhline(0.58, color="grey", linestyle=":",  linewidth=1.2, label="Realistic ceiling (0.58)")

        ax.set_title(coin, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in METRICS])
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Score")
        ax.legend(fontsize=7, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Model Evaluation on Test Set  |  70 / 15 / 15 chronological split",
        fontweight="bold", fontsize=12, y=1.02,
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "model_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")


def main():
    """Load feature CSVs, split data, train each model per coin, and print and plot results.

    Reads bitcoin_features.csv, xrp_features.csv, and icp_features.csv from the
    processed data directory, applies a chronological 70/15/15 train/validation/test
    split per coin, then trains and evaluates LogisticRegression and DecisionTree on
    both the validation and test sets. Results are printed to stdout and saved as a
    grouped bar chart to images/model_results.png.
    """
    target_files = ["bitcoin_features.csv", "xrp_features.csv", "icp_features.csv"]
    dfs = []
    for filename in target_files:
        path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        dfs.append(pd.read_csv(path, parse_dates=["date"]))
    if not dfs:
        print("No feature files found. Run the EDA pipeline first.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Lag all feature columns by 1 day per coin so that models predict
    # price_direction(t) from indicators at t-1, not same-day values.
    # Using same-day features would be data leakage: e.g. log_return(t) > 0
    # if and only if price_direction(t) == 1.
    lagged_groups = []
    for _, group in df.groupby("coin"):
        group = group.sort_values("date").copy()
        group[FEATURE_COLS] = group[FEATURE_COLS].shift(1)
        lagged_groups.append(group)
    df = pd.concat(lagged_groups, ignore_index=True)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    train, val, test = chronological_split(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree":       DecisionTreeClassifier(),
    }

    results = {}
    for coin in sorted(df["coin"].unique()):
        results[coin] = {}
        coin_train = train[train["coin"] == coin]
        coin_val   = val[val["coin"]     == coin]
        coin_test  = test[test["coin"]   == coin]
        for name, model in models.items():
            val_m  = evaluate_model(model, coin_train, coin_val,  FEATURE_COLS, TARGET_COL)
            test_m = evaluate_model(model, coin_train, coin_test, FEATURE_COLS, TARGET_COL)
            results[coin][name] = {"val": val_m, "test": test_m}

    for coin, coin_results in results.items():
        print(f"\n=== {coin} ===")
        for name, splits in coin_results.items():
            print(
                f"  {name:25s}"
                f"  val  acc={splits['val']['accuracy']:.4f}  prec={splits['val']['precision']:.4f}"
                f"  rec={splits['val']['recall']:.4f}  f1={splits['val']['f1']:.4f}"
            )
            print(
                f"  {'':25s}"
                f"  test acc={splits['test']['accuracy']:.4f}  prec={splits['test']['precision']:.4f}"
                f"  rec={splits['test']['recall']:.4f}  f1={splits['test']['f1']:.4f}"
            )

    plot_results(results, IMAGES_DIR)


if __name__ == "__main__":
    main()
