# Crypto Price Prediction Pipeline — Design Spec
**Date:** 2026-03-20
**Branch:** feature/CoinGecko
**Scope:** Full pipeline — scraper, preprocessing, feature engineering, modeling

---

## Overview

Implement a full data pipeline to predict the daily direction (up/down) of cryptocurrency prices using market indicators from the previous day. The pipeline follows Option A: a Scrapy project for data collection and standalone scripts for analysis.

---

## Repository Structure

```
CIP_FS2026_103/
├── scraper/
│   ├── scrapy.cfg
│   └── coingecko/
│       ├── __init__.py
│       ├── settings.py
│       ├── items.py
│       ├── pipelines.py
│       └── spiders/
│           ├── __init__.py
│           └── coingecko_spider.py
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── models.py
└── data/
    ├── raw/
    └── processed/
```

**Run order:**
```bash
cd scraper && scrapy crawl coingecko          # → data/raw/{coin_id}.csv
python src/preprocessing.py                   # → data/processed/{coin_id}_cleaned.csv
python src/features.py                        # → data/processed/{coin_id}_features.csv
python src/models.py                          # → prints metrics per model per coin
```

**Dependencies to add to `requirements.txt`:** `scrapy`, `scrapy-playwright`

---

## Section 1: Scraper (Scrapy + scrapy-playwright)

**Spider:** `coingecko_spider.py`

- Target URL: `https://www.coingecko.com/en/coins/{coin_id}/historical_data`
- Coins: `bitcoin`, `ripple`, `internet-computer` (CoinGecko slugs for BTC, XRP, ICP). Ethereum, Binance Coin, and Solana are deferred to a future branch.
- Date range: 3 years back from spider start date (computed dynamically)
- Uses `scrapy-playwright` middleware to handle JS-rendered data tables
- Crawl rate: `DOWNLOAD_DELAY = 2.5`, `AUTOTHROTTLE_ENABLED = True`, `ROBOTSTXT_OBEY = True`
- Logs start date, end date, and row count per coin on spider close

**Item fields** (`items.py`):

| Field | Type | Description |
|---|---|---|
| `date` | string | ISO 8601 (`YYYY-MM-DD`) |
| `coin_id` | string | CoinGecko slug |
| `coin_name` | string | Human-readable name |
| `symbol` | string | Ticker (e.g. `BTC`) |
| `price_usd` | float | Closing price in USD |
| `market_cap_usd` | float | Market cap in USD |
| `volume_24h_usd` | float | 24h trading volume in USD |
| `price_change_pct` | float | % price change vs previous day (scraped value) |
| `scraped_at` | string | UTC timestamp of scrape |

**Pipeline** (`pipelines.py`):
- Writes to `data/raw/{coin_id}.csv`, append mode
- Header written only on file creation
- Deduplicates on `(coin_id, date)` at spider close — keeps latest `scraped_at`

---

## Section 2: Preprocessing (`src/preprocessing.py`)

Reads `data/raw/{coin_id}.csv` per coin, applies steps in order, writes to `data/processed/{coin_id}_cleaned.csv`:

1. Parse `date` as `datetime64`; cast numeric columns to `float64`
2. Deduplicate `(coin_id, date)` — keep row with latest `scraped_at`
3. Missing values:
   - Drop row if `price_usd` is missing
   - Forward-fill `market_cap_usd` / `volume_24h_usd` within coin group; add `market_cap_usd_imputed` and `volume_24h_usd_imputed` boolean flags
   - Use scraped `price_change_pct` when present; recompute from `price_usd / price_usd.shift(1) - 1` only when missing
4. Flag `abs(price_change_pct) > 50` as `is_outlier = True` (no automatic drop; for human inspection only)
5. Sort by `(coin_id, date)` ascending before saving

---

## Section 3: Feature Engineering (`src/features.py`)

Reads each `data/processed/{coin_id}_cleaned.csv`, computes features per coin (sorted by date, grouped by `coin_id`), writes to `data/processed/{coin_id}_features.csv`:

**Step 1 — Compute raw features:**

| Feature | Formula |
|---|---|
| `daily_return` | `price_usd / price_usd.shift(1) - 1` |
| `ma_7` | 7-day rolling mean of `price_usd` |
| `ma_30` | 30-day rolling mean of `price_usd` |
| `volatility_7` | 7-day rolling std of `daily_return` |
| `vol_change` | `volume_24h_usd / volume_24h_usd.shift(1) - 1` |
| `price_direction` | `1` if `daily_return > 0` else `0` — **target variable, not lagged** |

**Step 2 — Lag features only (not the target):**

Shift `daily_return`, `ma_7`, `ma_30`, `volatility_7`, `vol_change` back by 1 day (`shift(1)`). This means day `t`'s row contains day `t-1` indicator values. `price_direction` is **not** lagged — it represents the direction on day `t` that the model predicts.

---

## Section 4: Modeling (`src/models.py`)

- Loads all `data/processed/{coin_id}_features.csv` files, concatenates
- Drops rows with NaN in feature or target columns (from rolling/lag windows)
- Feature columns: `daily_return`, `ma_7`, `ma_30`, `volatility_7`, `vol_change` (all lagged to t-1)
- Target: `price_direction` (day `t`)
- **Train/test split:** computed per coin group — last 20% of dates per coin as test, no shuffle. Per-coin test sets are then recombined for evaluation.
- **Models:**
  - `LogisticRegression` (baseline)
  - `DecisionTreeClassifier`
  - `LinearRegression` (thresholded at 0.5 for binary output — included as a regression baseline for academic comparison, not a production model)
- **Output:** accuracy, precision, recall, F1 printed per model per coin

---

## Constraints

- Never shuffle time-series data — chronological split only
- Only feature columns (`daily_return`, `ma_7`, `ma_30`, `volatility_7`, `vol_change`) are lagged; `price_direction` is never lagged
- Feature engineering must be grouped by `coin_id` to avoid cross-coin leakage
- Train/test split must also be computed per `coin_id` group
- Target variable is `price_direction` (binary) — do not regress on raw price
- Raw data in `data/raw/` is read-only after scraping
