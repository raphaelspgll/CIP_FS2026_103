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
