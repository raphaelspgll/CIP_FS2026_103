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
