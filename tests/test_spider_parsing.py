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
