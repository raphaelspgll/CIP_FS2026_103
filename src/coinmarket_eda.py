"""
coinmarket_eda.py
Exploratory Data Analysis for cryptocurrency price prediction.

Research questions:
1. Which market indicators have the strongest relationship to price changes?
2. How accurately can t-1 indicators predict next-day price direction (up/down)?
3. Which ML model achieves the highest accuracy?

This script:
- Loads historical OHLCV data for Bitcoin, XRP, and ICP
- Runs a data quality check
- Engineers log-transformed features for correlation analysis and ML
- Checks stationarity and uses only stationary features for lag-1 analysis
- Produces distribution, correlation, autocorrelation, cross-asset, and
  volatility plots saved to images/
- Saves enriched DataFrames to data/data_processed/*_features.csv
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

try:
    # When run as a script, __file__ gives the script's location
    _BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive consoles (e.g. PyCharm) where __file__ is not defined
    _BASE = os.path.join(os.getcwd(), "src")

RAW_DIR       = os.path.join(_BASE, "../data/data_raw")       # raw CSVs from scraper
PROCESSED_DIR = os.path.join(_BASE, "../data/data_processed") # engineered feature CSVs
IMAGES_DIR    = os.path.join(_BASE, "../images")              # all plot outputs

COINS = {
    "Bitcoin": "bitcoin_historical.csv",
    "XRP":     "xrp_historical.csv",
    "ICP":     "icp_historical.csv",
}

# Consistent brand colors used across all multi-coin plots
COLORS = {"Bitcoin": "#F7931A", "XRP": "#346AA9", "ICP": "#29ABE2"}

# Moving average window lengths (days)
MA_SHORT = 7
MA_LONG  = 30

# All log-transformed features — used for the general correlation heatmap.
# Includes non-stationary level features (log_close, ma_7, ma_30, etc.)
# which are informative for pairwise feature relationships.
FEATURE_COLS = [
    "log_close", "log_volume", "log_market_cap",
    "log_return", "log_close_open_ratio", "log_high_low_ratio",
    "ma_7", "ma_30", "volatility_7", "log_volume_change",
]

# Stationary-only features — used for the lag-1 analysis.
# Non-stationary features (log_close, ma_7, ma_30, log_volume, log_market_cap)
# trend with price levels and produce spurious correlations with price_direction.
LAG_FEATURE_COLS = [
    "log_return", "log_close_open_ratio", "log_high_low_ratio",
    "volatility_7", "log_volume_change",
]

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_figure(filename: str, images_dir: str) -> None:
    """Apply tight layout, save figure to images_dir, and close.

    Args:
        filename:   PNG filename (no path prefix).
        images_dir: Directory where the file will be saved.
    """
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {os.path.join(images_dir, filename)}")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(raw_dir: str) -> dict:
    """Load all three cryptocurrency CSV files and parse dates.

    Args:
        raw_dir: Path to the directory containing the raw CSV files.

    Returns:
        Dictionary mapping coin name to its DataFrame, sorted ascending by date.
        Each DataFrame has an additional 'coin' column for stacking/faceting.
    """
    datasets = {}
    for coin, filename in COINS.items():
        path = os.path.join(raw_dir, filename)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["coin"] = coin  # used in combined plots
        datasets[coin] = df
    return datasets


# ---------------------------------------------------------------------------
# 2. Data overview and quality check
# ---------------------------------------------------------------------------

def _print_basic_stats(coin: str, df: pd.DataFrame) -> None:
    """Print shape, dtypes, sample rows, descriptive stats, missing values, duplicates.

    Args:
        coin: Coin name for the section header.
        df:   Raw DataFrame for this coin.
    """
    print(f"\n{'=' * 60}")
    print(f"  {coin}")
    print(f"{'=' * 60}")
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- First 3 Rows ---")
    print(df.head(3).to_string())

    print("\n--- Descriptive Statistics ---")
    print(df.describe().to_string())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values.")

    print("\n--- Duplicate Rows ---")
    print(f"{df.duplicated().sum()} duplicate rows found.")


def _check_ohlcv_violations(coin: str, df: pd.DataFrame) -> None:
    """Check and report OHLCV logic violations (e.g. high < low, volume < 0).

    Args:
        coin: Coin name for the section header.
        df:   Raw DataFrame for this coin.
    """
    violations = {
        "high < open":    (df["high"] < df["open"]).sum(),
        "high < close":   (df["high"] < df["close"]).sum(),
        "low > open":     (df["low"] > df["open"]).sum(),
        "low > close":    (df["low"] > df["close"]).sum(),
        "high < low":     (df["high"] < df["low"]).sum(),
        "volume < 0":     (df["volume"] < 0).sum(),
        "market_cap < 0": (df["market_cap"] < 0).sum(),
    }
    print("\n--- OHLCV Logic Violations ---")
    any_violation = False
    for check, count in violations.items():
        if count > 0:
            print(f"  {check}: {count} rows")
            any_violation = True
    if not any_violation:
        print("  No violations found.")


def data_overview(datasets: dict) -> None:
    """Print a structured data quality report to stdout for each coin.

    Args:
        datasets: Dictionary mapping coin name to its raw DataFrame.
    """
    for coin, df in datasets.items():
        _print_basic_stats(coin, df)
        _check_ohlcv_violations(coin, df)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(datasets: dict) -> dict:
    """Compute log-transformed indicators and the ML target variable for each coin.

    All derived features use logarithmic scaling, which is methodologically
    appropriate for crypto data: prices follow log-normal distributions, log
    returns are symmetric and additive over time, and log-transforming volume/
    market cap removes scale dominance in cross-coin correlation analysis.

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
        price_direction      : 1 if close > open (bullish), else 0 -- ML target

    Raw OHLCV columns are kept for the quality check and for price_direction.
    NaN rows from rolling windows are kept so time-series plots span the full range.

    Args:
        datasets: Dictionary mapping coin name to its raw DataFrame.

    Returns:
        Dictionary mapping coin name to enriched DataFrames (copies).
        Feature CSVs are also saved to data_processed/.
    """
    enriched = {}
    for coin, df in datasets.items():
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

        # Save enriched data for downstream ML script
        out_name = coin.lower().replace(" ", "_") + "_features.csv"
        out_path = os.path.join(PROCESSED_DIR, out_name)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        enriched[coin] = df
    return enriched


# ---------------------------------------------------------------------------
# 4. Stationarity check
# ---------------------------------------------------------------------------

def analyze_stationarity(datasets: dict) -> None:
    """Check stationarity of key features by comparing rolling means over time.

    Stationarity is required for meaningful correlation analysis. Non-stationary
    features (those whose mean drifts over time) produce spurious correlations
    with the binary target price_direction.

    log_close, ma_7, ma_30, log_volume, log_market_cap are NON-STATIONARY:
    they trend with price levels over the 3-year window.

    log_return, log_close_open_ratio, log_high_low_ratio, volatility_7, and
    log_volume_change are STATIONARY: their mean stays stable over time.

    This is why the lag-1 analysis uses LAG_FEATURE_COLS (stationary only)
    rather than all FEATURE_COLS.

    Args:
        datasets: Dictionary mapping coin name to enriched DataFrame.
    """
    WINDOW = 60  # 2-month rolling window for mean comparison

    check_cols = {
        "log_close":  "NON-STATIONARY (trends with price level)",
        "log_return": "STATIONARY     (mean-reverting around zero)",
    }

    for coin, df in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"  Stationarity Check: {coin}")
        print(f"{'=' * 60}")
        print(f"  {'Feature':<25} {'Mean (first 60d)':<22} {'Mean (last 60d)':<22} Verdict")
        print(f"  {'-' * 85}")

        for col, verdict in check_cols.items():
            series = df[col].dropna()
            mean_first = series.iloc[:WINDOW].mean()
            mean_last  = series.iloc[-WINDOW:].mean()
            print(f"  {col:<25} {mean_first:<22.4f} {mean_last:<22.4f} {verdict}")

        print(f"\n  Lag-1 analysis uses only stationary features:")
        print(f"  {LAG_FEATURE_COLS}")


# ---------------------------------------------------------------------------
# 5. Distribution plots
# ---------------------------------------------------------------------------

def plot_return_histograms(datasets: dict, images_dir: str) -> None:
    """Plot log return distribution histogram for each coin (1x3 subplots).

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where the PNG will be saved.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (coin, df) in zip(axes, datasets.items()):
        sns.histplot(df["log_return"].dropna(), bins=60, kde=True, ax=ax, color=COLORS[coin])
        ax.set_title(f"{coin} - Log Return Distribution")
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Count")
    _save_figure("hist_log_return.png", images_dir)


def plot_return_boxplot(datasets: dict, images_dir: str) -> None:
    """Boxplot comparing log return spread and outliers across all coins.

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where the PNG will be saved.
    """
    combined = pd.concat(
        [df[["coin", "log_return"]].dropna() for df in datasets.values()],
        ignore_index=True,
    )
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="coin", y="log_return", hue="coin", data=combined,
                palette=COLORS, legend=False)
    plt.title("Log Return Distribution by Coin")
    plt.xlabel("Coin")
    plt.ylabel("Log Return")
    _save_figure("boxplot_log_return.png", images_dir)


def plot_price_trend(datasets: dict, images_dir: str) -> None:
    """Plot cumulative log return over time (all coins start at 0).

    Cumulative log return = log(P_t / P_0), making relative performance
    directly comparable across coins regardless of absolute price level.

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where the PNG will be saved.
    """
    plt.figure(figsize=(12, 4))
    for coin, df in datasets.items():
        cum_log_return = df["log_return"].cumsum()
        plt.plot(df["date"], cum_log_return, label=coin, color=COLORS[coin], alpha=0.9)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.title("Cumulative Log Return Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    _save_figure("price_trend.png", images_dir)


def plot_log_trends(datasets: dict, images_dir: str) -> None:
    """Plot log volume and log market cap over time for all coins.

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where the PNGs will be saved.
    """
    trend_cols = {
        "volume_trend.png":     ("log_volume",    "Log Volume"),
        "market_cap_trend.png": ("log_market_cap", "Log Market Cap (USD)"),
    }
    for filename, (col, ylabel) in trend_cols.items():
        plt.figure(figsize=(12, 4))
        for coin, df in datasets.items():
            plt.plot(df["date"], df[col], label=coin, color=COLORS[coin], alpha=0.9)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(f"Cryptocurrency {ylabel} Over Time")
        plt.legend()
        plt.xticks(rotation=45)
        _save_figure(filename, images_dir)


def plot_distributions(datasets: dict, images_dir: str) -> None:
    """Orchestrate all distribution and trend plots.

    Produces 5 PNG files:
        hist_log_return.png      - log return histograms per coin
        boxplot_log_return.png   - cross-coin log return boxplot
        price_trend.png          - cumulative log return over time
        volume_trend.png         - log volume over time
        market_cap_trend.png     - log market cap over time

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where PNG files will be saved.
    """
    plot_return_histograms(datasets, images_dir)
    plot_return_boxplot(datasets, images_dir)
    plot_price_trend(datasets, images_dir)
    plot_log_trends(datasets, images_dir)


# ---------------------------------------------------------------------------
# 6. Correlation analysis
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(coin: str, df: pd.DataFrame, images_dir: str) -> None:
    """Plot Pearson correlation heatmap for all FEATURE_COLS.

    Includes all log-transformed features (stationary and non-stationary) to
    show pairwise feature relationships. Not used for predictive analysis.

    Args:
        coin:       Coin name for the plot title and filename.
        df:         Enriched DataFrame for this coin.
        images_dir: Directory where the PNG will be saved.
    """
    corr_matrix = df[FEATURE_COLS].dropna().corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.4)
    plt.title(f"{coin} - Feature Correlation Matrix")
    slug = coin.lower().replace(" ", "_")
    _save_figure(f"corr_heatmap_{slug}.png", images_dir)


def plot_lag_correlation(coin: str, df: pd.DataFrame, images_dir: str) -> pd.Series:
    """Plot correlation of stationary t-1 features with next-day price direction.

    Uses LAG_FEATURE_COLS (stationary features only) to avoid spurious correlations
    from non-stationary level features (log_close, ma_7, ma_30, etc.).

    Correlation method: Pearson, which is mathematically equivalent to
    point-biserial correlation when the target variable is binary (0/1).
    P-values are computed via scipy.stats.pearsonr (two-tailed t-test,
    t = r * sqrt(n-2) / sqrt(1-r^2)) and printed to stdout as a
    significance table (* p<0.05, ** p<0.01, ns = not significant).

    Args:
        coin:       Coin name for the plot title and filename.
        df:         Enriched DataFrame for this coin.
        images_dir: Directory where the PNG will be saved.

    Returns:
        lag_corr: Correlation Series sorted by absolute value descending.
                  Passed to plot_top_indicator_scatter for indicator selection.
    """
    lag_df = df[LAG_FEATURE_COLS].copy()
    # Shift price_direction back by 1: today's features predict tomorrow's direction
    lag_df["target"] = df["price_direction"].shift(-1)
    lag_df = lag_df.dropna()

    # Compute Pearson r and two-tailed p-value for each feature.
    # Pearson correlation is mathematically equivalent to point-biserial
    # correlation when the target is binary, so no special test is needed.
    r_vals = {}
    p_vals = {}
    for feat in LAG_FEATURE_COLS:
        r, p = stats.pearsonr(lag_df[feat], lag_df["target"])
        r_vals[feat] = r
        p_vals[feat] = p

    # Sort by absolute correlation descending (same ordering as before)
    lag_corr = pd.Series(r_vals).sort_values(key=abs, ascending=False)

    # Print significance table to stdout
    print(f"\n{coin} — Lag-1 Pearson Correlation with Next-Day Price Direction")
    print(f"  n = {len(lag_df)},  95% significance threshold: p < 0.05")
    print(f"  {'Feature':<28} {'r':>8} {'p-value':>10}  sig")
    print("  " + "-" * 55)
    for feat in lag_corr.index:
        r = r_vals[feat]
        p = p_vals[feat]
        if p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  {feat:<28} {r:>8.4f} {p:>10.4f}  {sig}")

    bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in lag_corr.values]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=lag_corr.values, y=lag_corr.index, hue=lag_corr.index,
                palette=bar_colors, orient="h", legend=False)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title(f"{coin} - Indicator Correlation with Next-Day Price Direction (Lag-1)")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Indicator")
    slug = coin.lower().replace(" ", "_")
    _save_figure(f"lag1_correlation_{slug}.png", images_dir)
    return lag_corr


def plot_top_indicator_scatter(coin: str, df: pd.DataFrame,
                                lag_corr: pd.Series, images_dir: str) -> None:
    """Scatter plots of the top-3 lag-1 indicators vs next-day log_return.

    X-axis: indicator value at day t-1 (the predictor).
    Y-axis: log_return at day t (the outcome being predicted).

    Indicators are ranked by absolute Pearson correlation with next-day
    price direction. Each subplot includes a linear regression line.

    Note: Uses matplotlib directly (ax.scatter + np.polyfit) rather than
    seaborn due to a known seaborn/pandas 3.0 incompatibility with the
    PlotData internal data-lookup path.

    Args:
        coin:       Coin name for the plot titles and filename.
        df:         Enriched DataFrame for this coin.
        lag_corr:   Sorted correlation Series from plot_lag_correlation.
        images_dir: Directory where the PNG will be saved.
    """
    top3 = lag_corr.abs().nlargest(3).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, indicator in zip(axes, top3):
        col = str(indicator)
        # Use the indicator from t-1 and next-day log_return as the outcome
        plot_df = pd.DataFrame({
            col: df[col].values[:-1],
            "next_log_return": df["log_return"].values[1:],
        }).dropna().reset_index(drop=True)
        x_vals = plot_df[col].to_numpy(dtype=float).ravel()
        y_vals = plot_df["next_log_return"].to_numpy(dtype=float).ravel()

        ax.scatter(x_vals, y_vals, alpha=0.3, color=COLORS[coin], s=10)

        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="black", linewidth=1.5)

        ax.set_title(f"{coin}: {col} (t-1) vs Next-Day Log Return")
        ax.set_xlabel(f"{col} at t-1")
        ax.set_ylabel("Log Return at t")
        ax.tick_params(axis="x", rotation=30)

    slug = coin.lower().replace(" ", "_")
    _save_figure(f"scatter_top_indicators_{slug}.png", images_dir)


def plot_correlations(datasets: dict, images_dir: str) -> None:
    """Orchestrate all correlation plots for each coin.

    Produces 9 PNG files (3 chart types x 3 coins):
        corr_heatmap_{coin}.png           - full feature correlation matrix
        lag1_correlation_{coin}.png       - stationary indicator lag-1 bar chart
        scatter_top_indicators_{coin}.png - top-3 indicator scatter plots

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where PNG files will be saved.
    """
    for coin, df in datasets.items():
        plot_correlation_heatmap(coin, df, images_dir)
        lag_corr = plot_lag_correlation(coin, df, images_dir)
        plot_top_indicator_scatter(coin, df, lag_corr, images_dir)


# ---------------------------------------------------------------------------
# 7. New analyses
# ---------------------------------------------------------------------------

def analyze_autocorrelation(datasets: dict, images_dir: str) -> None:
    """Plot autocorrelation of log_return for lags 1-20 with 95% confidence bands.

    Bars outside ±1.96/√n (the 95% confidence band) indicate statistically
    significant autocorrelation — i.e., past returns carry predictive signal.
    Significant lag-1 autocorrelation directly supports research question 2.

    Produces one PNG per coin (3 total):
        autocorr_{coin}.png

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where PNG files will be saved.
    """
    MAX_LAG = 20

    for coin, df in datasets.items():
        returns = df["log_return"].dropna()
        n = len(returns)
        conf_band = 1.96 / np.sqrt(n)

        acf_vals = [returns.autocorr(lag=k) for k in range(1, MAX_LAG + 1)]
        lags = list(range(1, MAX_LAG + 1))

        # Print lag-1 ACF result with significance verdict
        acf_lag1 = acf_vals[0]
        sig_lag1 = "significant" if abs(acf_lag1) > conf_band else "not significant"
        print(f"\n{coin} — Autocorrelation of Log Return")
        print(f"  n = {n},  95% confidence band: ±{conf_band:.4f}")
        print(f"  Lag-1 ACF: {acf_lag1:.4f}  ({sig_lag1} at 95% level)")
        sig_count = sum(1 for v in acf_vals if abs(v) > conf_band)
        print(f"  Lags 1–{MAX_LAG} with |ACF| > band: {sig_count} of {MAX_LAG}")

        # Highlight bars that exceed the confidence band in red
        bar_colors = ["#e74c3c" if abs(v) > conf_band else "#95a5a6" for v in acf_vals]

        plt.figure(figsize=(10, 4))
        plt.bar(lags, acf_vals, color=bar_colors, alpha=0.8)
        plt.axhline(conf_band,  color="blue", linestyle="--", linewidth=1,
                    label=f"95% confidence band (±{conf_band:.3f})")
        plt.axhline(-conf_band, color="blue", linestyle="--", linewidth=1)
        plt.axhline(0, color="black", linewidth=0.6)
        plt.title(f"{coin} - Autocorrelation of Log Return (Lags 1–{MAX_LAG})")
        plt.xlabel("Lag (days)")
        plt.ylabel("Autocorrelation")
        plt.legend()
        slug = coin.lower().replace(" ", "_")
        _save_figure(f"autocorr_{slug}.png", images_dir)


def analyze_cross_asset_correlation(datasets: dict, images_dir: str) -> None:
    """Plot log_return Pearson correlation matrix across all three coins.

    Reveals the degree of co-movement between BTC, XRP, and ICP.
    High correlation implies shared market-wide signals (BTC dominance);
    low correlation implies coin-specific factors are more important.

    Produces one PNG:
        cross_asset_correlation.png

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where the PNG will be saved.
    """
    returns_df = pd.DataFrame({
        coin: df.set_index("date")["log_return"]
        for coin, df in datasets.items()
    }).dropna()

    corr = returns_df.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Cross-Asset Log Return Correlation (BTC / XRP / ICP)")
    _save_figure("cross_asset_correlation.png", images_dir)


def analyze_volatility(datasets: dict, images_dir: str) -> None:
    """Plot 30-day rolling volatility of log_return over time for all coins.

    Reveals volatility clustering (GARCH-like behavior) and market regimes:
    - Peaks correspond to bear markets and crash events
    - Calm periods correspond to trending bull markets

    Dashed horizontal lines show each coin's overall mean volatility for reference.

    Produces one PNG:
        rolling_volatility.png

    Args:
        datasets:   Dictionary mapping coin name to enriched DataFrame.
        images_dir: Directory where the PNG will be saved.
    """
    plt.figure(figsize=(12, 4))
    for coin, df in datasets.items():
        rolling_vol = df["log_return"].rolling(window=MA_LONG).std()
        mean_vol = df["log_return"].std()
        plt.plot(df["date"], rolling_vol, label=coin, color=COLORS[coin], alpha=0.9)
        plt.axhline(mean_vol, color=COLORS[coin], linestyle="--",
                    linewidth=0.8, alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Rolling 30-Day Volatility (Std of Log Return)")
    plt.title("Rolling Volatility Over Time — Dashed Lines Show Overall Mean")
    plt.legend()
    plt.xticks(rotation=45)
    _save_figure("rolling_volatility.png", images_dir)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full EDA pipeline in sequence.

    Steps:
        1.  Create output directories
        2.  Load raw data
        3.  Print data quality report
        4.  Engineer log-transformed features
        5.  Check stationarity
        6.  Plot distributions
        7.  Plot correlations (with stationary lag-1 analysis)
        8.  Analyze autocorrelation
        9.  Analyze cross-asset correlation
        10. Analyze rolling volatility
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("Loading data...")
    raw = load_data(RAW_DIR)

    print("\nRunning data overview...")
    data_overview(raw)

    print("\nEngineering features...")
    enriched = engineer_features(raw)

    print("\nChecking stationarity...")
    analyze_stationarity(enriched)

    print("\nPlotting distributions...")
    plot_distributions(enriched, IMAGES_DIR)

    print("\nPlotting correlations...")
    plot_correlations(enriched, IMAGES_DIR)

    print("\nAnalyzing autocorrelation...")
    analyze_autocorrelation(enriched, IMAGES_DIR)

    print("\nAnalyzing cross-asset correlation...")
    analyze_cross_asset_correlation(enriched, IMAGES_DIR)

    print("\nAnalyzing rolling volatility...")
    analyze_volatility(enriched, IMAGES_DIR)

    print("\n" + "=" * 60)
    print("EDA complete. Output files:")
    print(f"  Feature CSVs : {PROCESSED_DIR}/")
    print(f"  Plot PNGs    : {IMAGES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
