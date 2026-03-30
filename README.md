# Cryptocurrency Price Direction Prediction
### CIP – Data Collection, Integration and Preprocessing (HSLU)

## Overview
This project is part of the course *Data Collection, Integration and Preprocessing (CIP02)*.

The central research question is: **can publicly available market indicators from day t-1 predict the direction of price movement on day t?**

A binary classification pipeline was built to predict the daily price movement direction (up / down) of six major cryptocurrencies using market indicators from the preceding day. Historical data spanning three years was collected from CoinGecko, cleaned, and enriched with technical features.

## Research Questions
1. Which market indicators have the strongest relationship with daily cryptocurrency price changes?
2. How accurately can historical indicators from day t-1 predict price direction on day t?
3. Which machine learning model achieves the highest accuracy under these conditions?

## Coins Covered
Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB), Solana (SOL), Ripple (XRP), Internet Computer (ICP)

## Data Sources
All data is collected from **CoinGecko** historical data pages, which provide daily OHLCV figures going back several years.

Raw data schema per coin (`data/raw/{coin_id}.csv`):
| Field | Type | Description |
|---|---|---|
| `date` | string | ISO 8601 (YYYY-MM-DD) |
| `coin_id` | string | CoinGecko slug |
| `coin_name` | string | Human-readable name |
| `symbol` | string | Ticker (e.g. BTC) |
| `price_usd` | float | Daily closing price in USD |
| `market_cap_usd` | float | Market capitalisation in USD |
| `volume_24h_usd` | float | 24-hour trading volume in USD |
| `price_change_pct` | float | Day-over-day % change (recomputed) |
| `scraped_at` | string (UTC) | Timestamp of scrape |


## Data Acquisition
The scraper is built on **Scrapy** with the `scrapy-playwright` extension, which drives a headless Chromium browser to handle JavaScript-rendered tables.

Key configuration:
- `DOWNLOAD_DELAY`: 2.5 s (polite crawl rate)
- `AUTOTHROTTLE_ENABLED`: True (adaptive rate limiting)
- `PLAYWRIGHT_BROWSER`: chromium (headless JS rendering)
- `ROBOTSTXT_OBEY`: True

The spider applies a three-year date filter and repeatedly clicks "Show More" to load the full history. For each row it reads: date, market cap, 24h volume, and closing price.


## Data Processing
Raw CSVs are processed through a five-step cleaning pipeline (`src/preprocessing.py`). Output is written to `data/processed/{coin_id}_cleaned.csv`.

1. **Type casting** — date parsed to `datetime64[ns]`; numeric fields cast to `float64`
2. **Deduplication** — duplicates on `(coin_id, date)` dropped, retaining the most recent scrape
3. **Missing value handling**:
   - `price_usd`: row dropped (core signal, cannot be fabricated)
   - `market_cap_usd`, `volume_24h_usd`: forward-filled within coin group, flagged with imputation indicator
   - `price_change_pct`: recomputed as $p_t / p_{t-1} - 1$
4. **Outlier flagging** — rows where `|price_change_pct| > 50%` are flagged `is_outlier = True` but retained
5. **Sorting** — output sorted by `(coin_id, date)` ascending


## Feature Engineering
Features are computed per coin (to prevent cross-coin leakage) and lagged by one day (`src/features.py`):

| Feature | Formula | Lagged? |
|---|---|---|
| `daily_return` | $p_t / p_{t-1} - 1$ | Yes |
| `ma_7` | 7-day rolling mean of `price_usd` | Yes |
| `ma_30` | 30-day rolling mean of `price_usd` | Yes |
| `volatility_7` | 7-day rolling std of `daily_return` | Yes |
| `vol_change` | $v_t / v_{t-1} - 1$ | Yes |
| `price_direction` | 1 if `daily_return > 0`, else 0 | No (target) |

Features from day t-1 → predict `price_direction` at day t.


## Models
Three classifiers are evaluated under a strict **chronological 80/20 train/test split** (no shuffling, to prevent look-ahead bias):

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline (`max_iter=1000`) |
| Decision Tree | Default depth, no pruning |
| Linear Regression | Continuous output thresholded at 0.5 |

**Evaluation metrics:** accuracy (primary), precision, recall, F1-score.


## Key Design Decisions
- **No shuffling of time-series data** — chronological splits strictly prevent future data leakage
- **Per-coin feature grouping** — rolling windows and lags computed independently per coin
- **Outliers flagged, not dropped** — extreme price events carry real market information


## Limitations
- Feature set limited to price-derived technical indicators; sentiment, on-chain metrics, and macro variables are not included
- CoinGecko does not expose intra-day open prices, so `price_change_pct` is always a day-over-day return
- Decision Tree is not tuned; hyperparameter search could reduce overfitting


## Project Structure
```
CIP_FS2026_103/
│── data/           # raw and processed datasets
│── docs/           # reports, feasibility study, documentation
│── latex/          # project report (LaTeX source)
│── notebooks/      # exploratory analysis (EDA)
│── scraper/        # web scraping scripts (Scrapy, Playwright)
│── scripts/        # data processing and utility scripts
│── src/            # core logic (preprocessing, features, models)
│── tests/          # testing code
│── .venv/          # virtual environment (not tracked)
```


## Technologies
- Python 3.10+
- Scrapy 2.13, scrapy-playwright 0.0.43
- Pandas 3.0.1, NumPy 2.4.3
- scikit-learn 1.8.0
- Matplotlib 3.10.8, Seaborn 0.13.2

## Authors
Lemma Emanuel
Spagolla Raphaël
Krishnathasan Tharrmeehan
