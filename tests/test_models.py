import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import chronological_split, evaluate_model, FEATURE_COLS


def make_df(n=100, coin_id="bitcoin"):
    np.random.seed(42)
    return pd.DataFrame({
        "date":            pd.date_range("2022-01-01", periods=n),
        "coin_id":         [coin_id] * n,
        "daily_return":    np.random.randn(n) * 0.02,
        "ma_7":            np.random.uniform(30000, 50000, n),
        "ma_30":           np.random.uniform(30000, 50000, n),
        "volatility_7":    np.random.uniform(0.01, 0.05, n),
        "vol_change":      np.random.randn(n) * 0.1,
        "price_direction": np.random.randint(0, 2, n).astype(float),
    })


def test_chronological_split_no_data_leakage():
    df = make_df(n=100)
    train, test = chronological_split(df, test_size=0.2)
    assert train["date"].max() < test["date"].min()


def test_chronological_split_sizes():
    df = make_df(n=100)
    train, test = chronological_split(df, test_size=0.2)
    assert len(test) == 20
    assert len(train) == 80


def test_split_is_per_coin():
    df_btc = make_df(n=100, coin_id="bitcoin")
    df_xrp = make_df(n=100, coin_id="ripple")
    df = pd.concat([df_btc, df_xrp], ignore_index=True)
    train, test = chronological_split(df, test_size=0.2)
    assert len(test[test["coin_id"] == "bitcoin"]) == 20
    assert len(test[test["coin_id"] == "ripple"]) == 20


def test_evaluate_model_returns_all_metrics():
    from sklearn.linear_model import LogisticRegression
    df = make_df(n=200)
    train, test = chronological_split(df, test_size=0.2)
    metrics = evaluate_model(
        LogisticRegression(max_iter=1000), train, test, FEATURE_COLS, "price_direction"
    )
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_feature_cols_constant():
    assert FEATURE_COLS == ["daily_return", "ma_7", "ma_30", "volatility_7", "vol_change"]
