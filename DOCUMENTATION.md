# CIP_FS2026_103 — Project Documentation

**Course:** Data Collection, Integration and Preprocessing (CIP02), FS 2026
**Group:** 103
**Institution:** HSLU — Hochschule Luzern

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Data Pipeline](#4-data-pipeline)
5. [Scraper (`scraper/coingecko/`)](#5-scraper-scrapercoingecko)
6. [Preprocessing (`src/preprocessing.py`)](#6-preprocessing-srcpreprocessingpy)
7. [Feature Engineering (`src/features.py`)](#7-feature-engineering-srcfeaturespy)
8. [Modeling (`src/models.py`)](#8-modeling-srcmodelspy)
9. [Data Schema](#9-data-schema)
10. [Tests](#10-tests)
11. [Branch Strategy](#11-branch-strategy)
12. [Design Decisions & Constraints](#12-design-decisions--constraints)

---

## 1. Project Overview

This project builds a binary classification pipeline to predict the **direction** of daily cryptocurrency price movement (up/down) on day `t` using publicly available market indicators from day `t-1`.

**Coins in scope:**

| Coin | Symbol | CoinGecko Slug |
|---|---|---|
| Bitcoin | BTC | `bitcoin` |
| Ethereum | ETH | `ethereum` |
| Binance Coin | BNB | `binancecoin` |
| Solana | SOL | `solana` |
| Ripple | XRP | `ripple` |
| Internet Computer | ICP | `internet-computer` |

**Data source:** CoinGecko historical data pages (3 years of daily OHLCV data)
**Models:** Logistic Regression (baseline), Decision Tree, Linear Regression (for comparison)
**Primary metric:** Accuracy; secondary: Precision, Recall, F1

---

## 2. Repository Structure

```
CIP_FS2026_103/
├── src/
│   ├── preprocessing.py         # Data cleaning: type casting, dedup, imputation, outlier flagging
│   ├── features.py              # Feature engineering: technical indicators, lag-by-1
│   └── models.py                # Modeling: chronological split, evaluation, 3 classifiers
├── scraper/
│   └── coingecko/
│       ├── settings.py          # Scrapy + scrapy-playwright configuration
│       ├── items.py             # CryptoItem definition (9 fields)
│       ├── pipelines.py         # CSV export pipeline with deduplication
│       └── spiders/
│           └── coingecko_spider.py  # Playwright-based async spider
├── tests/
│   ├── test_preprocessing.py    # 8 tests — data cleaning
│   ├── test_features.py         # 6 tests — feature engineering
│   ├── test_models.py           # 5 tests — model training & evaluation
│   ├── test_spider_parsing.py   # 4 tests — spider helper functions
│   ├── test_items.py            # 1 test  — CryptoItem field presence
│   └── test_pipeline.py         # 5 tests — CSV export pipeline
├── data/
│   ├── raw/                     # Read-only after scraping (one CSV per coin)
│   └── processed/               # Output of preprocessing and feature engineering
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── docs/
│   └── feasibility_study_103.md
├── requirements.txt
├── CLAUDE.md                    # AI assistant instructions (project spec)
└── DOCUMENTATION.md             # This file
```

---

## 3. Setup & Installation

**Requirements:** Python 3.10+, pip

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Playwright browser (required for scraping)
playwright install chromium
```

**Key dependencies:**

| Package | Version | Purpose |
|---|---|---|
| pandas | 3.0.1 | Data manipulation |
| numpy | 2.4.3 | Numerical operations |
| scikit-learn | 1.8.0 | ML models and metrics |
| scrapy | 2.13.0 | Web scraping framework |
| scrapy-playwright | 0.0.43 | JS-rendered page support |
| matplotlib / seaborn | 3.10.8 / 0.13.2 | Visualisation |

---

## 4. Data Pipeline

The pipeline runs in four sequential stages:

```
[1. Scrape]  →  data/raw/{coin_id}.csv
      ↓
[2. Preprocess]  →  data/processed/{coin_id}_cleaned.csv
      ↓
[3. Feature Engineering]  →  data/processed/{coin_id}_features.csv
      ↓
[4. Modeling]  →  printed evaluation table (accuracy, precision, recall, F1)
```

**Run order:**

```bash
# Stage 1: Scrape (from repo root)
cd scraper
scrapy crawl coingecko

# Stage 2: Preprocess
python src/preprocessing.py

# Stage 3: Feature engineering
python src/features.py

# Stage 4: Modeling
python src/models.py

# Optional: explore in notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

---

## 5. Scraper (`scraper/coingecko/`)

### Overview

A [Scrapy](https://scrapy.org/) spider that uses [scrapy-playwright](https://github.com/scrapy-plugins/scrapy-playwright) to scrape JS-rendered historical data tables from CoinGecko.

**Target URL pattern:**
`https://www.coingecko.com/en/coins/{coin_id}/historical_data`

### Configuration (`settings.py`)

| Setting | Value | Purpose |
|---|---|---|
| `COIN_LIST` | 6 coins (slug + name + symbol) | Coins to scrape |
| `DOWNLOAD_DELAY` | 2.5 sec | Polite crawl rate |
| `AUTOTHROTTLE_ENABLED` | True | Adaptive rate limiting |
| `PLAYWRIGHT_BROWSER_TYPE` | chromium | Headless browser |
| `ROBOTSTXT_OBEY` | True | Respects robots.txt |

### Spider (`coingecko_spider.py`)

**Start requests:** One Playwright-enabled request per coin. Waits for `table tbody tr` to appear (30 s timeout) before parsing.

**Parsing logic:**

For each table row, the spider reads cells in this column order:

| Index | Field | Notes |
|---|---|---|
| 0 | `date` | Already in ISO `YYYY-MM-DD` format from CoinGecko |
| 1 | `market_cap_usd` | `$` and `,` stripped |
| 2 | `volume_24h_usd` | `$` and `,` stripped |
| 3 | `price_usd` | Daily price |

**Date filter:** Only rows with `date >= today - 3 years` are yielded.

**`price_change_pct` note:** The CoinGecko historical data table does not expose an open price column, so `price_change_pct` is always yielded as `None`. The preprocessing step always recomputes it as day-over-day change `price_usd / price_usd.shift(1) - 1`.

**"Show More" pagination:** The spider navigates to the date-ranged URL (`?start=...&end=...`), then repeatedly clicks "Show More" (up to 80 times) until the oldest visible row reaches the 3-year cutoff.

**Logging:** On spider close, logs `start_date`, `end_date`, and item count per coin.

### Pipeline (`pipelines.py`)

- Writes one CSV file per `coin_id` to `data/raw/`
- Appends to existing files; writes header only once
- On spider close: deduplicates by `(coin_id, date)`, keeping the row with the latest `scraped_at`

### Helper functions

| Function | Signature | Description |
|---|---|---|
| `parse_price` | `(text: str) -> float \| None` | Strips `$`, `,`, whitespace; returns float or None for empty/invalid input |
| `_oldest_table_date` | `(page) -> str \| None` | Returns the oldest ISO date string visible in the table, or None |
| `_click_button_by_text` | `(page, text: str) -> bool` | Clicks the first visible button matching `text`; returns True if found |

---

## 6. Preprocessing (`src/preprocessing.py`)

### Entry point

```python
clean(df: pd.DataFrame) -> pd.DataFrame
```

Reads raw CSVs from `data/raw/`, applies the pipeline below, and writes cleaned output to `data/processed/{coin_id}_cleaned.csv`.

### Processing steps (in order)

**1. Type casting**
- `date` → `datetime64[ns]`
- `price_usd`, `market_cap_usd`, `volume_24h_usd`, `price_change_pct` → `float64`

**2. Deduplication**
- Sort by `scraped_at` descending, then drop duplicates on `(coin_id, date)` keeping the first (latest scrape).

**3. Missing value handling**

| Field | Strategy |
|---|---|
| `price_usd` | Drop row if missing — price is required |
| `market_cap_usd` | Forward-fill within coin group; set `market_cap_usd_imputed = True` for filled rows |
| `volume_24h_usd` | Forward-fill within coin group; set `volume_24h_usd_imputed = True` for filled rows |
| `price_change_pct` | Recompute as `price_usd / price_usd.shift(1) - 1` if missing; prefer scraped value when present |

**4. Outlier flagging**
- `is_outlier = True` where `|price_change_pct| > 0.50` (50%)
- Outlier rows are retained — not dropped

**5. Sorting**
- Output sorted by `(coin_id, date)` ascending

### Output columns

All input columns plus: `market_cap_usd_imputed`, `volume_24h_usd_imputed`, `is_outlier`

---

## 7. Feature Engineering (`src/features.py`)

### Entry point

Reads `data/processed/{coin_id}_cleaned.csv`, applies features per coin, and writes to `data/processed/{coin_id}_features.csv`.

### Features computed

All features are computed **per coin** (grouped by `coin_id`) to prevent cross-coin leakage, then **lagged by 1 day** (`shift(1)`) so that the model sees only day `t-1` information when predicting day `t`.

| Feature | Formula | Lagged? |
|---|---|---|
| `daily_return` | `price_usd / price_usd.shift(1) - 1` | Yes |
| `ma_7` | 7-day rolling mean of `price_usd` | Yes |
| `ma_30` | 30-day rolling mean of `price_usd` | Yes |
| `volatility_7` | 7-day rolling std of `daily_return` | Yes |
| `vol_change` | `volume_24h_usd / volume_24h_usd.shift(1) - 1` | Yes |
| `price_direction` | `1` if `daily_return > 0` else `0` | **No** — this is the target |

### Lag semantics

```
Day t-1 features  →  Day t target
──────────────────────────────────
daily_return[t-1]
ma_7[t-1]           price_direction[t]
ma_30[t-1]
volatility_7[t-1]
vol_change[t-1]
```

The first row per coin will have `NaN` for all lagged features and is dropped during modeling.

### Exported constant

```python
FEATURE_COLS = ["daily_return", "ma_7", "ma_30", "volatility_7", "vol_change"]
```

This constant is imported by `models.py` to avoid duplication.

---

## 8. Modeling (`src/models.py`)

### Task

Binary classification: predict `price_direction` (0 = down, 1 = up) for day `t` using features from day `t-1`.

### Train/test split

**Function:** `chronological_split(df, test_size=0.2)`

- Split is performed **per coin** independently
- The last 20% of dates (by chronological order) form the test set
- **No shuffling** — time-series integrity is preserved
- Prevents forward-looking bias: all training data precedes all test data within each coin

### Models evaluated

| Model key | Class | Notes |
|---|---|---|
| `LogisticRegression` | `LogisticRegression(max_iter=1000)` | Baseline |
| `DecisionTree` | `DecisionTreeClassifier()` | Default depth (no pruning) |
| `LinearRegression` | `LinearRegression()` | Continuous output thresholded at 0.5 for binary classification |

### Evaluation

**Function:** `evaluate_model(model, train, test, feature_cols, target_col)`

Returns a dict with four metrics (rounded to 4 decimal places):

| Metric | Description |
|---|---|
| `accuracy` | Fraction of correct predictions |
| `precision` | Positive predictive value |
| `recall` | Sensitivity / true positive rate |
| `f1` | Harmonic mean of precision and recall |

### Output format

Results are printed as a table grouped by coin:

```
=== bitcoin ===
  LogisticRegression         acc=0.5342  prec=0.5201  rec=0.6100  f1=0.5615
  DecisionTree               acc=0.5123  prec=0.5034  rec=0.5800  f1=0.5392
  LinearRegression           acc=0.5198  prec=0.5100  rec=0.5950  f1=0.5492
```

---

## 9. Data Schema

### Raw (`data/raw/{coin_id}.csv`)

| Field | Type | Description |
|---|---|---|
| `date` | string (`YYYY-MM-DD`) | ISO 8601 date |
| `coin_id` | string | CoinGecko slug (e.g. `bitcoin`) |
| `coin_name` | string | Human-readable name |
| `symbol` | string | Ticker (e.g. `BTC`) |
| `price_usd` | float | Daily closing price in USD |
| `market_cap_usd` | float | Market capitalisation in USD |
| `volume_24h_usd` | float | 24-hour trading volume in USD |
| `price_change_pct` | float | Intra-day % price change (scraped); day-over-day (recomputed) |
| `scraped_at` | string (UTC ISO timestamp) | Time of scrape |

No currency symbols (`$`) or thousands separators (`,`) in numeric fields.

### Cleaned (`data/processed/{coin_id}_cleaned.csv`)

All raw fields plus:

| Field | Type | Description |
|---|---|---|
| `market_cap_usd_imputed` | bool | True if `market_cap_usd` was forward-filled |
| `volume_24h_usd_imputed` | bool | True if `volume_24h_usd` was forward-filled |
| `is_outlier` | bool | True if `|price_change_pct| > 50%` |

### Features (`data/processed/{coin_id}_features.csv`)

All cleaned fields plus:

| Field | Type | Description |
|---|---|---|
| `daily_return` | float | Day-over-day return (lagged) |
| `ma_7` | float | 7-day moving average of price (lagged) |
| `ma_30` | float | 30-day moving average of price (lagged) |
| `volatility_7` | float | 7-day rolling std of daily_return (lagged) |
| `vol_change` | float | Day-over-day volume change ratio (lagged) |
| `price_direction` | int (0/1) | Target — 1 if price went up, 0 otherwise (not lagged) |

---

## 10. Tests

All tests are in `tests/` and run with pytest:

```bash
pytest tests/
```

### Test coverage summary

| File | Tests | What is verified |
|---|---|---|
| `test_preprocessing.py` | 8 | Type casting, dedup (keeps latest scraped_at), forward-fill with imputed flags, price_change_pct preference, outlier flagging, sort order |
| `test_features.py` | 6 | Target is not lagged, features are lagged by exactly 1 (row 1 is NaN for `daily_return` and `vol_change`), ma_7 window, no cross-coin leakage, all feature columns present |
| `test_models.py` | 5 | No data leakage in chronological split, 80/20 split sizes, per-coin split, all four metrics returned, FEATURE_COLS constant |
| `test_spider_parsing.py` | 4 | `parse_price` strips `$` and `,`, handles plain floats, returns None for empty; `parse_date_str` produces ISO format |
| `test_items.py` | 1 | All 9 required fields present on CryptoItem |
| `test_pipeline.py` | 5 | CSV creation, dedup keeps latest scraped_at, separate file per coin, header written once |

---

## 11. Branch Strategy

| Branch | Status | Contents |
|---|---|---|
| `main` | Stable | Base project structure |
| `feature/CoinMarket` | Complete | CoinMarketCap Playwright scraper (`src/coinmarket_scrape.py`) + 3-year historical CSVs for BTC, XRP, ICP |
| `feature/CoinGecko` | In progress | CoinGecko Scrapy spider + full preprocessing, feature engineering, and modeling pipeline |

The `feature/CoinGecko` branch contains the complete data science pipeline (preprocessing → features → modeling) and is the primary development branch.

---

## 12. Design Decisions & Constraints

### No shuffling of time-series data

Train/test splits are always chronological. Shuffling would allow the model to implicitly learn from future data, producing optimistic evaluation metrics that do not reflect real-world performance.

### Per-coin feature grouping

All rolling windows, lags, and target computation are applied within each coin's time series independently. Computing features across coins (e.g., a rolling mean that spans from BTC rows to ETH rows) would be meaningless and introduce data leakage.

### Lag-by-1 for all features, not for the target

The prediction task is: given what we observe on day `t-1`, predict the direction on day `t`. Features are shifted by one period; the target is not. The first row per coin always has `NaN` features after lagging and is dropped before modeling.

### Forward-fill for volume and market cap, drop for price

Price is the core signal — a missing price cannot be imputed without introducing fabricated information. Volume and market cap can be reasonably approximated by the last known value over short gaps (e.g., weekends, holidays), so forward-fill is used with an `_imputed` flag for transparency.

### Outliers are flagged, not dropped

Price swings above ±50% are rare but real events (e.g., exchange listings, market crashes). Dropping them would silently remove historically significant data. The `is_outlier` flag allows downstream inspection and selective filtering.

### `price_change_pct` semantics

The CoinGecko historical data table does not expose an open price, so the spider always yields `price_change_pct = None`. The preprocessing step always recomputes it as day-over-day change `price_usd / price_usd.shift(1) - 1`. Model features use `daily_return` (also day-over-day), so `price_change_pct` is informational only.
