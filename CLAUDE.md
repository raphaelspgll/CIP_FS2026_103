# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HSLU graduate data science project (CIP — Data Collection, Integration and Preprocessing, FS 2026, Group 103). Goal: predict the **direction** of daily cryptocurrency price movement (up/down) on day `t` using publicly available market indicators from day `t-1`. Binary classification using Logistic Regression and Decision Tree models trained on 3 years of historical data.

Coins in scope: Bitcoin, Ethereum, Binance Coin, Solana, Ripple (XRP), Internet Computer (ICP).

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape historical data (Playwright-based)
python src/coinmarket_scrape.py          # CoinMarketCap scraper (feature/CoinMarket branch)
# python src/coingecko_scrape.py         # CoinGecko scraper (feature/CoinGecko branch, in progress)

# Clean and engineer features
python src/preprocessing.py              # Output: data/processed/{coin_id}_cleaned.csv
python src/features.py

# Explore in notebooks (run in order)
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

## Branch Strategy

- `feature/CoinMarket` — working CoinMarketCap scraper (`src/coinmarket_scrape.py`) + 3-year historical CSVs for BTC, XRP, ICP in `data/`
- `feature/CoinGecko` — CoinGecko scraper implementation (in progress, no `.py` files yet)

## Data Fields

All raw scraped output must use these field names (no currency symbols or commas in numeric values):

| Field | Type | Description |
|---|---|---|
| `date` | string | ISO 8601 (`YYYY-MM-DD`) |
| `coin_id` | string | Slug (e.g. `bitcoin`) |
| `coin_name` | string | Human-readable name |
| `symbol` | string | Ticker (e.g. `BTC`) |
| `price_usd` | float | Closing price in USD |
| `market_cap_usd` | float | Market capitalisation in USD |
| `volume_24h_usd` | float | 24-hour trading volume in USD |
| `price_change_pct` | float | % price change vs. previous day |
| `scraped_at` | string | UTC timestamp of scrape |

## Scraper Pattern

Current implementation (`coinmarket_scrape.py`) uses Playwright:
- Navigates to CoinMarketCap historical data pages
- Clicks "Load More" until 3-year threshold is reached
- Parses HTML table rows, strips `$` and `,` from numeric fields
- Outputs one CSV per coin sorted by date into `data/`

CoinGecko target URL pattern: `https://www.coingecko.com/en/coins/{coin_id}/historical_data`

Use `scrapy-playwright` or direct Playwright for JS-rendered tables. Respect crawl rate (`DOWNLOAD_DELAY = 2.5`, `AUTOTHROTTLE_ENABLED = True`). Log start date, end date, and item count per coin on completion.

## Data Processing (`src/preprocessing.py`)

Steps in order, operating on Pandas DataFrames:

1. Parse `date` as `datetime64`, cast numeric fields to `float64`
2. Deduplicate `(coin_id, date)` — keep latest `scraped_at`
3. Missing values: drop row if `price_usd` missing; forward-fill `market_cap_usd` / `volume_24h_usd` within coin group and flag with `_imputed` boolean; recompute `price_change_pct` from `price_usd.shift(1)` if missing
4. Flag `price_change_pct > ±50%` as `is_outlier = True` — do not drop
5. Sort by `(coin_id, date)` ascending → save to `data/processed/{coin_id}_cleaned.csv`

Raw data in `data/raw/` is read-only after scraping; all mutations go to `data/processed/`.

## Feature Engineering (`src/features.py`)

Applied per coin after sorting by date:

| Feature | Formula |
|---|---|
| `daily_return` | `(price_t / price_{t-1}) - 1` |
| `ma_7` | 7-day simple moving average of `price_usd` |
| `ma_30` | 30-day simple moving average of `price_usd` |
| `volatility_7` | Rolling 7-day std of `daily_return` |
| `vol_change` | `volume_24h_usd / volume_24h_usd.shift(1) - 1` |
| `price_direction` | **Target**: `1` if `daily_return > 0` else `0` |

## Modeling (`src/models.py`)

- **Task:** binary classification — predict `price_direction`
- **Split:** chronological (no shuffle) — last 20% of dates as test set
- **Models:** `LogisticRegression` (baseline), `DecisionTreeClassifier`, `LinearRegression` (for comparison)
- **Primary metric:** accuracy; secondary: precision, recall, F1
- **Input features:** all features from day `t-1` (lagged), excluding `price_direction`

## Critical Rules

- **Never shuffle time-series data** — chronological train/test split only
- **Feature engineering must be grouped by `coin_id`** to avoid cross-coin leakage
- All features must be lagged by 1 day (`shift(1)`) before modeling to prevent look-ahead bias
- Target variable is `price_direction` (binary) — do not regress on raw price
