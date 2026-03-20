import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import engineer


def base_df(n=40, coin_id="bitcoin", start_price=30000.0, step=100.0):
    """n rows with linearly increasing price."""
    return pd.DataFrame({
        "date":           pd.date_range("2023-01-01", periods=n),
        "coin_id":        [coin_id] * n,
        "price_usd":      [start_price + i * step for i in range(n)],
        "market_cap_usd": [6e11] * n,
        "volume_24h_usd": [2e10] * n,
    })


def test_price_direction_is_1_for_increasing_prices():
    df = base_df()
    result = engineer(df)
    # All valid (non-NaN) price_direction values should be 1 (prices always increase)
    valid = result["price_direction"].dropna()
    assert (valid == 1).all()


def test_price_direction_is_not_lagged():
    df = base_df(n=10)
    result = engineer(df)
    # price_direction for row index 1 should be non-NaN (direction of day 1 vs day 0)
    # If it were lagged, row 1 would be NaN
    assert not pd.isna(result["price_direction"].iloc[1])


def test_feature_columns_are_lagged_by_one():
    df = base_df(n=10)
    result = engineer(df)
    # Row index 1 should have NaN for all lagged feature columns
    # (because shift(1) moves row 0's values to row 1, but row 0 itself was NaN for daily_return)
    assert pd.isna(result["daily_return"].iloc[1])
    assert pd.isna(result["ma_7"].iloc[1])


def test_ma7_window():
    df = base_df(n=40)
    result = engineer(df)
    prices = [30000.0 + i * 100 for i in range(40)]
    # Unlagged ma_7 at row index 7 = mean(prices[1..7])
    # After lag, this appears at row index 8
    expected = sum(prices[1:8]) / 7
    assert abs(result["ma_7"].iloc[8] - expected) < 1.0


def test_no_cross_coin_leakage():
    btc = base_df(n=20, coin_id="bitcoin", start_price=30000.0, step=100.0)
    xrp = base_df(n=20, coin_id="ripple",  start_price=1.0,     step=0.01)
    combined = pd.concat([btc, xrp], ignore_index=True)
    result = engineer(combined)
    btc_returns = result[result["coin_id"] == "bitcoin"]["daily_return"].dropna()
    xrp_returns = result[result["coin_id"] == "ripple"]["daily_return"].dropna()
    # BTC daily return ≈ 100/30000; XRP daily return ≈ 0.01/1.0
    assert btc_returns.mean() < 0.01
    assert xrp_returns.mean() > 0.005


def test_output_contains_all_feature_columns():
    df = base_df(n=40)
    result = engineer(df)
    for col in ["daily_return", "ma_7", "ma_30", "volatility_7", "vol_change", "price_direction"]:
        assert col in result.columns, f"Missing column: {col}"
