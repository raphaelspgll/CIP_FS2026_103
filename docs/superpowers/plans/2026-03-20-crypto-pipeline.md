# Crypto Price Prediction Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full pipeline that scrapes 3 years of BTC/XRP/ICP historical data from CoinGecko, cleans it, engineers features, and trains three ML models to predict daily price direction.

**Architecture:** Scrapy project with scrapy-playwright for JS-rendered CoinGecko pages writes raw CSVs to `data/raw/`; standalone Python scripts handle preprocessing, feature engineering, and modeling in sequence.

**Tech Stack:** Python 3, Scrapy 2.13, scrapy-playwright, Playwright (chromium), Pandas, NumPy, scikit-learn, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `scraper/scrapy.cfg` | Scrapy project config |
| `scraper/coingecko/__init__.py` | Package marker |
| `scraper/coingecko/settings.py` | COIN_LIST, crawl rate, playwright handler config |
| `scraper/coingecko/items.py` | CryptoItem (9 fields) |
| `scraper/coingecko/pipelines.py` | Append CSV per coin + dedup on spider close |
| `scraper/coingecko/spiders/__init__.py` | Package marker |
| `scraper/coingecko/spiders/coingecko_spider.py` | Navigate CoinGecko, wait for JS table, parse 3 years of rows |
| `src/preprocessing.py` | Clean raw CSVs → `data/processed/{coin_id}_cleaned.csv` |
| `src/features.py` | Compute indicators, lag features → `data/processed/{coin_id}_features.csv` |
| `src/models.py` | Chronological split per coin, train 3 models, print metrics |
| `tests/__init__.py` | Package marker |
| `tests/test_items.py` | Verify CryptoItem has all required fields |
| `tests/test_pipeline.py` | CSV write, append mode, dedup logic |
| `tests/test_spider_parsing.py` | Price string parsing, date string parsing |
| `tests/test_preprocessing.py` | Each cleaning step independently |
| `tests/test_features.py` | Feature formulas, lag behaviour, no cross-coin leakage |
| `tests/test_models.py` | Chronological split, per-coin grouping, metric output |

**Run order after implementation:**
```bash
pip install -r requirements.txt
playwright install chromium
cd scraper && scrapy crawl coingecko    # → data/raw/{coin_id}.csv
cd .. && python src/preprocessing.py    # → data/processed/{coin_id}_cleaned.csv
python src/features.py                  # → data/processed/{coin_id}_features.csv
python src/models.py                    # prints metrics per model per coin
```

---

### Task 1: Project Scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `scraper/scrapy.cfg`
- Create: `scraper/coingecko/__init__.py`
- Create: `scraper/coingecko/spiders/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add scrapy and scrapy-playwright to requirements.txt**

Append to `requirements.txt`:
```
scrapy==2.13.0
scrapy-playwright==0.0.43
```

- [ ] **Step 2: Create Scrapy project config**

`scraper/scrapy.cfg`:
```ini
[settings]
default = coingecko.settings

[deploy]
project = coingecko
```

- [ ] **Step 3: Create package markers and data directories**

Create these as empty files:
- `scraper/coingecko/__init__.py`
- `scraper/coingecko/spiders/__init__.py`
- `tests/__init__.py`
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt scraper/ data/raw/.gitkeep data/processed/.gitkeep tests/__init__.py
git commit -m "feat: scaffold Scrapy project structure and test directory"
```

---

### Task 2: CryptoItem

**Files:**
- Create: `scraper/coingecko/items.py`
- Create: `tests/test_items.py`

- [ ] **Step 1: Write the failing test**

`tests/test_items.py`:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scraper'))

from coingecko.items import CryptoItem


def test_crypto_item_has_required_fields():
    item = CryptoItem()
    expected = {
        "date", "coin_id", "coin_name", "symbol",
        "price_usd", "market_cap_usd", "volume_24h_usd",
        "price_change_pct", "scraped_at",
    }
    assert expected.issubset(set(item.fields.keys()))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_items.py -v
```
Expected: `ModuleNotFoundError` — items.py does not exist yet.

- [ ] **Step 3: Write minimal implementation**

`scraper/coingecko/items.py`:
```python
import scrapy


class CryptoItem(scrapy.Item):
    date             = scrapy.Field()
    coin_id          = scrapy.Field()
    coin_name        = scrapy.Field()
    symbol           = scrapy.Field()
    price_usd        = scrapy.Field()
    market_cap_usd   = scrapy.Field()
    volume_24h_usd   = scrapy.Field()
    price_change_pct = scrapy.Field()
    scraped_at       = scrapy.Field()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_items.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/coingecko/items.py tests/test_items.py
git commit -m "feat: add CryptoItem with 9 required fields"
```

---

### Task 3: Scrapy Settings

**Files:**
- Create: `scraper/coingecko/settings.py`

- [ ] **Step 1: Create settings.py**

`scraper/coingecko/settings.py`:
```python
BOT_NAME = "coingecko"

SPIDER_MODULES = ["coingecko.spiders"]

COIN_LIST = [
    {"coin_id": "bitcoin",           "name": "Bitcoin",           "symbol": "BTC"},
    {"coin_id": "ripple",            "name": "XRP",               "symbol": "XRP"},
    {"coin_id": "internet-computer", "name": "Internet Computer", "symbol": "ICP"},
]

# scrapy-playwright configuration
DOWNLOAD_HANDLERS = {
    "http":  "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": True}
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

DOWNLOAD_DELAY = 2.5
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
ROBOTSTXT_OBEY = True

DEFAULT_REQUEST_HEADERS = {
    "Accept-Language": "en",
    "User-Agent": "Mozilla/5.0 (compatible; CIPResearchBot/1.0; academic project)",
}

ITEM_PIPELINES = {
    "coingecko.pipelines.CsvExportPipeline": 300,
}

RAW_DATA_DIR = "../data/raw"
```

- [ ] **Step 2: Commit**

```bash
git add scraper/coingecko/settings.py
git commit -m "feat: add Scrapy settings with scrapy-playwright config and COIN_LIST"
```

---

### Task 4: CSV Export Pipeline

**Files:**
- Create: `scraper/coingecko/pipelines.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline.py`:
```python
import csv
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scraper'))

from coingecko.pipelines import CsvExportPipeline


def make_item(coin_id, date, scraped_at, price=50000.0):
    return {
        "coin_id":          coin_id,
        "date":             date,
        "coin_name":        "Bitcoin",
        "symbol":           "BTC",
        "price_usd":        price,
        "market_cap_usd":   1e12,
        "volume_24h_usd":   5e10,
        "price_change_pct": 1.5,
        "scraped_at":       scraped_at,
    }


class FakeSpider:
    name = "coingecko"

    class logger:
        @staticmethod
        def info(msg):
            pass


@pytest.fixture
def pipeline(tmp_path):
    p = CsvExportPipeline()
    p.raw_dir = str(tmp_path)
    return p, tmp_path


def test_writes_csv_on_first_item(pipeline):
    p, tmp_path = pipeline
    p.process_item(make_item("bitcoin", "2024-01-01", "2024-01-01T10:00:00Z"), FakeSpider())
    p.close_spider(FakeSpider())
    out = tmp_path / "bitcoin.csv"
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["date"] == "2024-01-01"


def test_dedup_keeps_latest_scraped_at(pipeline):
    p, tmp_path = pipeline
    p.process_item(make_item("bitcoin", "2024-01-01", "2024-01-01T08:00:00Z", price=49000.0), FakeSpider())
    p.process_item(make_item("bitcoin", "2024-01-01", "2024-01-01T12:00:00Z", price=50000.0), FakeSpider())
    p.close_spider(FakeSpider())
    rows = list(csv.DictReader((tmp_path / "bitcoin.csv").open()))
    assert len(rows) == 1
    assert float(rows[0]["price_usd"]) == 50000.0


def test_separate_csv_per_coin(pipeline):
    p, tmp_path = pipeline
    p.process_item(make_item("bitcoin", "2024-01-01", "2024-01-01T10:00:00Z"), FakeSpider())
    p.process_item(make_item("ripple",  "2024-01-01", "2024-01-01T10:00:00Z"), FakeSpider())
    p.close_spider(FakeSpider())
    assert (tmp_path / "bitcoin.csv").exists()
    assert (tmp_path / "ripple.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```
Expected: `ModuleNotFoundError` — pipelines.py does not exist.

- [ ] **Step 3: Write minimal implementation**

`scraper/coingecko/pipelines.py`:
```python
import csv
import os
import pandas as pd

FIELDS = [
    "date", "coin_id", "coin_name", "symbol",
    "price_usd", "market_cap_usd", "volume_24h_usd",
    "price_change_pct", "scraped_at",
]


class CsvExportPipeline:
    def __init__(self):
        self.raw_dir = "../data/raw"
        self.file_handles = {}
        self.writers = {}
        self.rows_per_coin = {}

    def open_spider(self, spider):
        self.raw_dir = spider.settings.get("RAW_DATA_DIR", "../data/raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def process_item(self, item, spider):
        coin_id = item["coin_id"]
        if coin_id not in self.file_handles:
            path = os.path.join(self.raw_dir, f"{coin_id}.csv")
            file_exists = os.path.isfile(path)
            fh = open(path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if not file_exists:
                writer.writeheader()
            self.file_handles[coin_id] = fh
            self.writers[coin_id] = writer
            self.rows_per_coin[coin_id] = 0
        self.writers[coin_id].writerow({k: item.get(k, "") for k in FIELDS})
        self.rows_per_coin[coin_id] += 1
        return item

    def close_spider(self, spider):
        for coin_id, fh in self.file_handles.items():
            fh.close()
            path = os.path.join(self.raw_dir, f"{coin_id}.csv")
            self._dedup(path)
            spider.logger.info(
                f"[{coin_id}] Wrote {self.rows_per_coin[coin_id]} rows (after dedup)"
            )

    def _dedup(self, path):
        df = pd.read_csv(path)
        df = (
            df.sort_values("scraped_at")
            .drop_duplicates(subset=["coin_id", "date"], keep="last")
        )
        df.to_csv(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```
Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/coingecko/pipelines.py tests/test_pipeline.py
git commit -m "feat: add CSV export pipeline with per-coin files and dedup on close"
```

---

### Task 5: CoinGecko Spider

**Files:**
- Create: `scraper/coingecko/spiders/coingecko_spider.py`
- Create: `tests/test_spider_parsing.py`

Note: The spider targets CoinGecko's historical data table which is JS-rendered. The table columns assumed are: Date | Market Cap | Volume | Open | Close. `price_change_pct` is computed as `(close - open) / open * 100`. If the page structure differs, update the `css` selectors in `parse()`.

- [ ] **Step 1: Write failing tests for parsing helpers**

`tests/test_spider_parsing.py`:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scraper'))

from coingecko.spiders.coingecko_spider import parse_price, parse_date_str


def test_parse_price_strips_currency_and_commas():
    assert parse_price("$1,234,567.89") == 1234567.89


def test_parse_price_handles_plain_float():
    assert parse_price("50000.0") == 50000.0


def test_parse_price_returns_none_for_empty():
    assert parse_price("") is None
    assert parse_price("N/A") is None


def test_parse_date_str_converts_to_iso():
    assert parse_date_str("Jan 01, 2024") == "2024-01-01"
    assert parse_date_str("Dec 31, 2021") == "2021-12-31"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_spider_parsing.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the spider**

`scraper/coingecko/spiders/coingecko_spider.py`:
```python
import scrapy
from datetime import datetime, timezone, timedelta
from coingecko.items import CryptoItem


def parse_price(text: str):
    """Strip $, commas, whitespace; return float or None."""
    cleaned = str(text).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date_str(text: str) -> str:
    """Convert 'Jan 01, 2024' to '2024-01-01'."""
    return datetime.strptime(text.strip(), "%b %d, %Y").strftime("%Y-%m-%d")


class CoinGeckoSpider(scrapy.Spider):
    name = "coingecko"

    def start_requests(self):
        coin_list = self.settings.get("COIN_LIST", [])
        self.cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=3 * 365)
        ).strftime("%Y-%m-%d")
        self.logger.info(f"Scraping from {self.cutoff_date} to today")

        for coin in coin_list:
            url = (
                f"https://www.coingecko.com/en/coins"
                f"/{coin['coin_id']}/historical_data"
            )
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_coroutines": [
                        {
                            "method": "wait_for_selector",
                            "args": ["table tbody tr"],
                            "kwargs": {"timeout": 30000},
                        },
                    ],
                    "coin": coin,
                },
                callback=self.parse,
            )

    async def parse(self, response, **kwargs):
        coin = response.meta["coin"]
        page = response.meta.get("playwright_page")
        if page:
            await page.close()

        rows = response.css("table tbody tr")
        self.logger.info(f"[{coin['coin_id']}] Found {len(rows)} candidate rows")

        start_date = end_date = None
        count = 0

        for row in rows:
            cells = row.css("td")
            if len(cells) < 5:
                continue
            try:
                date_str = parse_date_str(cells[0].css("::text").get(""))
            except ValueError:
                continue

            if date_str < self.cutoff_date:
                continue

            price_usd      = parse_price(cells[4].css("::text").get(""))  # Close
            market_cap_usd = parse_price(cells[1].css("::text").get(""))
            volume_24h_usd = parse_price(cells[2].css("::text").get(""))
            open_price     = parse_price(cells[3].css("::text").get(""))

            if price_usd and open_price:
                price_change_pct = round((price_usd - open_price) / open_price * 100, 4)
            else:
                price_change_pct = None

            yield CryptoItem(
                date             = date_str,
                coin_id          = coin["coin_id"],
                coin_name        = coin["name"],
                symbol           = coin["symbol"],
                price_usd        = price_usd,
                market_cap_usd   = market_cap_usd,
                volume_24h_usd   = volume_24h_usd,
                price_change_pct = price_change_pct,
                scraped_at       = datetime.now(timezone.utc).isoformat(),
            )

            if start_date is None or date_str < start_date:
                start_date = date_str
            if end_date is None or date_str > end_date:
                end_date = date_str
            count += 1

        self.logger.info(
            f"[{coin['coin_id']}] Scraped {count} rows | {start_date} → {end_date}"
        )
```

- [ ] **Step 4: Run parsing tests to verify they pass**

```bash
pytest tests/test_spider_parsing.py -v
```
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/coingecko/spiders/coingecko_spider.py tests/test_spider_parsing.py
git commit -m "feat: add CoinGecko spider with playwright and 3-year date filter"
```

---

### Task 6: Preprocessing

**Files:**
- Create: `src/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

`tests/test_preprocessing.py`:
```python
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


def test_sorted_by_coin_id_and_date():
    df = base_df().iloc[::-1].reset_index(drop=True)
    result = clean(df)
    dates = result["date"].astype(str).tolist()
    assert dates == sorted(dates)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_preprocessing.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/preprocessing.py`:
```python
import glob
import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
NUMERIC_COLS = ["price_usd", "market_cap_usd", "volume_24h_usd", "price_change_pct"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Type casting
    df["date"] = pd.to_datetime(df["date"])
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Dedup: keep latest scraped_at per (coin_id, date)
    df = (
        df.sort_values("scraped_at")
        .drop_duplicates(subset=["coin_id", "date"], keep="last")
        .reset_index(drop=True)
    )

    # 3a. Drop rows with missing price_usd
    df = df.dropna(subset=["price_usd"]).reset_index(drop=True)

    # 3b. Forward-fill market_cap_usd and volume_24h_usd per coin; flag imputed rows
    for col in ["market_cap_usd", "volume_24h_usd"]:
        imputed_col = f"{col}_imputed"
        missing_before = df[col].isna()
        df[col] = df.groupby("coin_id")[col].transform(lambda s: s.ffill())
        df[imputed_col] = missing_before & df[col].notna()

    # 3c. Recompute price_change_pct where missing (prefer scraped value)
    computed = (
        df.groupby("coin_id")["price_usd"]
        .transform(lambda s: (s / s.shift(1) - 1) * 100)
    )
    df["price_change_pct"] = df["price_change_pct"].where(
        df["price_change_pct"].notna(), computed
    )

    # 4. Flag outliers (for human inspection; do not drop)
    df["is_outlier"] = df["price_change_pct"].abs() > 50

    # 5. Sort
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        coin_id = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        cleaned = clean(df)
        out_path = os.path.join(PROCESSED_DIR, f"{coin_id}_cleaned.csv")
        cleaned.to_csv(out_path, index=False)
        print(f"[{coin_id}] {len(cleaned)} rows → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocessing.py -v
```
Expected: all 7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add preprocessing module with type casting, dedup, imputation, and outlier flagging"
```

---

### Task 7: Feature Engineering

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

`tests/test_features.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/features.py`:
```python
import glob
import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

FEATURE_COLS = ["daily_return", "ma_7", "ma_30", "volatility_7", "vol_change"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)

    def _compute(group):
        g = group.copy()

        # Step 1: compute raw features and target
        g["daily_return"]    = g["price_usd"] / g["price_usd"].shift(1) - 1
        g["ma_7"]            = g["price_usd"].rolling(7).mean()
        g["ma_30"]           = g["price_usd"].rolling(30).mean()
        g["volatility_7"]    = g["daily_return"].rolling(7).std()
        g["vol_change"]      = g["volume_24h_usd"] / g["volume_24h_usd"].shift(1) - 1

        # Target: direction of price on day t (not lagged)
        g["price_direction"] = (g["daily_return"] > 0).astype(float)
        g["price_direction"] = g["price_direction"].where(g["daily_return"].notna())

        # Step 2: lag only the feature columns, never the target
        g[FEATURE_COLS] = g[FEATURE_COLS].shift(1)

        return g

    return (
        df.groupby("coin_id", group_keys=False)
        .apply(_compute)
        .reset_index(drop=True)
    )


def main():
    for path in glob.glob(os.path.join(PROCESSED_DIR, "*_cleaned.csv")):
        coin_id = os.path.basename(path).replace("_cleaned.csv", "")
        df = pd.read_csv(path, parse_dates=["date"])
        result = engineer(df)
        out_path = os.path.join(PROCESSED_DIR, f"{coin_id}_features.csv")
        result.to_csv(out_path, index=False)
        print(f"[{coin_id}] {len(result)} rows → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```
Expected: all 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add feature engineering with lag-by-1 and per-coin grouping"
```

---

### Task 8: Modeling

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`src/models.py`:
```python
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEATURE_COLS = ["daily_return", "ma_7", "ma_30", "volatility_7", "vol_change"]
TARGET_COL = "price_direction"


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split per coin by date with no shuffle; recombine for return."""
    trains, tests = [], []
    for _, group in df.groupby("coin_id"):
        group = group.sort_values("date").reset_index(drop=True)
        split_idx = int(len(group) * (1 - test_size))
        trains.append(group.iloc[:split_idx])
        tests.append(group.iloc[split_idx:])
    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)


def evaluate_model(model, train, test, feature_cols, target_col):
    X_train, y_train = train[feature_cols], train[target_col]
    X_test,  y_test  = test[feature_cols],  test[target_col]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if isinstance(model, LinearRegression):
        preds = (preds >= 0.5).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
    }


def main():
    dfs = []
    for path in glob.glob(os.path.join(PROCESSED_DIR, "*_features.csv")):
        dfs.append(pd.read_csv(path, parse_dates=["date"]))
    if not dfs:
        print("No feature files found. Run features.py first.")
        return

    df = pd.concat(dfs, ignore_index=True).dropna(subset=FEATURE_COLS + [TARGET_COL])
    train, test = chronological_split(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree":       DecisionTreeClassifier(),
        "LinearRegression":   LinearRegression(),
    }

    for coin_id in sorted(df["coin_id"].unique()):
        print(f"\n=== {coin_id} ===")
        coin_train = train[train["coin_id"] == coin_id]
        coin_test  = test[test["coin_id"]  == coin_id]
        for name, model in models.items():
            m = evaluate_model(model, coin_train, coin_test, FEATURE_COLS, TARGET_COL)
            print(
                f"  {name:25s}  acc={m['accuracy']:.4f}  "
                f"prec={m['precision']:.4f}  "
                f"rec={m['recall']:.4f}  "
                f"f1={m['f1']:.4f}"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```
Expected: all 5 PASS

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```
Expected: all tests across all modules PASS

- [ ] **Step 6: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add modeling module with per-coin chronological split and three classifiers"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Install all dependencies**

```bash
pip install -r requirements.txt
playwright install chromium
```

- [ ] **Step 2: Run complete test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: all tests PASS with no warnings about missing modules.

- [ ] **Step 3: Smoke-test the scraper (optional — requires network)**

```bash
cd scraper
scrapy crawl coingecko --logfile=../scrapy.log -L INFO
```
Check `scrapy.log` for lines like:
```
[bitcoin] Scraped 1095 rows | 2023-03-20 → 2026-03-19
[ripple] Scraped 1095 rows | ...
[internet-computer] Scraped 1095 rows | ...
```

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete crypto price prediction pipeline"
```
