import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import clean


def base_df():
    return pd.DataFrame({
        "date":             ["2024-01-01", "2024-01-02", "2024-01-03"],
        "coin_id":          ["bitcoin"] * 3,
        "coin_name":        ["Bitcoin"] * 3,
        "symbol":           ["BTC"] * 3,
        "price_usd":        [40000.0, 41000.0, 42000.0],
        "market_cap_usd":   [8e11, 8.2e11, 8.4e11],
        "volume_24h_usd":   [2e10, 2.1e10, 2.2e10],
        "price_change_pct": [1.0, 2.5, 2.44],
        "scraped_at":       ["2024-01-01T10:00:00Z"] * 3,
    })


def test_drops_row_when_price_usd_missing():
    df = base_df()
    df.loc[1, "price_usd"] = np.nan
    result = clean(df)
    assert len(result) == 2
    assert "2024-01-02" not in result["date"].astype(str).values


def test_dedup_keeps_latest_scraped_at():
    df = base_df()
    dup = df.iloc[0].copy()
    dup["scraped_at"] = "2024-01-01T12:00:00Z"
    dup["price_usd"] = 99999.0
    df = pd.concat([df, pd.DataFrame([dup])], ignore_index=True)
    result = clean(df)
    mask = result["date"].astype(str) == "2024-01-01"
    assert result[mask]["price_usd"].iloc[0] == 99999.0


def test_forward_fills_market_cap_and_sets_imputed_flag():
    df = base_df()
    df.loc[1, "market_cap_usd"] = np.nan
    result = clean(df)
    mask = result["date"].astype(str) == "2024-01-02"
    assert result[mask]["market_cap_usd"].iloc[0] == 8e11
    assert result[mask]["market_cap_usd_imputed"].iloc[0] == True


def test_prefers_scraped_price_change_pct_when_present():
    df = base_df()
    result = clean(df)
    # Original scraped value for row 1 is 2.5; computed would be ~2.44
    mask = result["date"].astype(str) == "2024-01-02"
    assert result[mask]["price_change_pct"].iloc[0] == 2.5


def test_recomputes_price_change_pct_when_missing():
    df = base_df()
    df.loc[2, "price_change_pct"] = np.nan
    result = clean(df)
    mask = result["date"].astype(str) == "2024-01-03"
    val = result[mask]["price_change_pct"].iloc[0]
    expected = (42000.0 / 41000.0 - 1) * 100
    assert abs(val - expected) < 0.001


def test_flags_outliers_above_50_pct():
    df = base_df()
    df.loc[1, "price_change_pct"] = 60.0
    result = clean(df)
    mask = result["date"].astype(str) == "2024-01-02"
    assert result[mask]["is_outlier"].iloc[0] == True
    mask2 = result["date"].astype(str) == "2024-01-01"
    assert result[mask2]["is_outlier"].iloc[0] == False


def test_forward_fills_volume_and_sets_imputed_flag():
    df = base_df()
    df.loc[1, "volume_24h_usd"] = np.nan
    result = clean(df)
    mask = result["date"].astype(str) == "2024-01-02"
    assert result[mask]["volume_24h_usd"].iloc[0] == 2e10
    assert result[mask]["volume_24h_usd_imputed"].iloc[0] == True


def test_sorted_by_coin_id_and_date():
    df = base_df().iloc[::-1].reset_index(drop=True)
    result = clean(df)
    dates = result["date"].astype(str).tolist()
    assert dates == sorted(dates)
