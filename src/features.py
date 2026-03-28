"""
features.py
Defines canonical feature columns and engineers log-transformed predictors
from raw OHLCV data for the cryptocurrency price prediction pipeline.

FEATURE_COLS is the single source of truth for the five stationary predictors
used in model training. models.py imports this list for training and evaluation.
coinmarket_eda.py imports it as LAG_FEATURE_COLS for the lag-1 correlation analysis.

The engineer() function computes all features for a single coin DataFrame.
The engineer_all() function processes all coins and saves feature CSVs to
data/processed/. Run this script directly to regenerate the feature CSVs.
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

try:
    # When run as a script, __file__ gives the script's location
    _BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive consoles where __file__ is not defined
    _BASE = os.path.join(os.getcwd(), "src")

RAW_DIR       = os.path.join(_BASE, "../data/raw")
PROCESSED_DIR = os.path.join(_BASE, "../data/processed")

COINS = {
    "Bitcoin": "bitcoin_historical.csv",
    "XRP":     "xrp_historical.csv",
    "ICP":     "icp_historical.csv",
}

# Moving average window lengths (days)
MA_SHORT = 7
MA_LONG  = 30

# Stationary predictors used by the ML models (single source of truth).
# Non-stationary level features (log_close, ma_7, ma_30, log_volume,
# log_market_cap) are intentionally excluded: they trend with price levels
# and produce spurious correlations with the binary target.
FEATURE_COLS = [
    "log_return",
    "log_close_open_ratio",
    "log_high_low_ratio",
    "volatility_7",
    "log_volume_change",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-transformed features and the binary price-direction target.

    All features are computed on same-day OHLCV values without lagging.
    The calling code in models.py shifts FEATURE_COLS by one day per coin
    to produce t-1 predictors for the target at t, avoiding data leakage.

    New columns added:
        log_close            : log(close) — log price level
        log_volume           : log(volume) — log volume
        log_market_cap       : log(market_cap) — log market cap
        log_return           : log(close_t / close_{t-1}) — daily log return
        log_close_open_ratio : log(close / open) — positive on bullish days
        log_high_low_ratio   : log(high / low) — intraday range, always >= 0
        ma_7                 : MA_SHORT-day rolling mean of log_close
        ma_30                : MA_LONG-day rolling mean of log_close
        volatility_7         : MA_SHORT-day rolling std of log_return
        log_volume_change    : log(volume_t / volume_{t-1}) — log volume change
        price_direction      : 1 if close > open (bullish), else 0 — ML target

    Args:
        df: DataFrame with columns date, open, high, low, close, volume,
            market_cap, and coin, sorted ascending by date.

    Returns:
        Copy of df with all feature and target columns added. NaN rows from
        rolling windows are kept so time-series plots span the full date range.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Log-transformed price and market features (non-stationary level features)
    df["log_close"]            = np.log(df["close"])
    df["log_volume"]           = np.log(df["volume"])
    df["log_market_cap"]       = np.log(df["market_cap"])

    # Log return: log(P_t / P_{t-1}) — stationary, preferred over pct_change
    df["log_return"]           = np.log(df["close"] / df["close"].shift(1))

    # Intraday log ratios (stationary and scale-invariant)
    df["log_close_open_ratio"] = np.log(df["close"] / df["open"])
    df["log_high_low_ratio"]   = np.log(df["high"] / df["low"])

    # Rolling indicators on log-transformed series
    df["ma_7"]                 = df["log_close"].rolling(window=MA_SHORT).mean()
    df["ma_30"]                = df["log_close"].rolling(window=MA_LONG).mean()
    df["volatility_7"]         = df["log_return"].rolling(window=MA_SHORT).std()
    df["log_volume_change"]    = np.log(df["volume"] / df["volume"].shift(1))

    # Binary ML target: 1 = bullish day (close > open), 0 = bearish/flat
    df["price_direction"]      = (df["close"] > df["open"]).astype(int)

    return df


def engineer_all(datasets: dict, output_dir: str) -> dict:
    """Engineer features for all coins and save one CSV per coin.

    Calls engineer() on each coin DataFrame and writes the result to
    output_dir/<coin_name>_features.csv (e.g. bitcoin_features.csv).

    Args:
        datasets:   Dictionary mapping coin name to its raw DataFrame.
        output_dir: Directory where the feature CSVs will be saved.
                    Created automatically if it does not exist.

    Returns:
        Dictionary mapping coin name to the enriched DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    enriched = {}
    for coin, df in datasets.items():
        enriched_df = engineer(df)
        out_name = coin.lower().replace(" ", "_") + "_features.csv"
        out_path = os.path.join(output_dir, out_name)
        enriched_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        enriched[coin] = enriched_df
    return enriched


def main() -> None:
    """Load raw OHLCV data, engineer features for all coins, and save CSVs.

    Reads bitcoin_historical.csv, xrp_historical.csv, and icp_historical.csv
    from data/raw/, computes all features via engineer(), and saves the
    enriched DataFrames to data/processed/.
    """
    datasets = {}
    for coin, filename in COINS.items():
        path = os.path.join(RAW_DIR, filename)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["coin"] = coin
        datasets[coin] = df

    engineer_all(datasets, PROCESSED_DIR)
    print(f"\nFeature CSVs saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
