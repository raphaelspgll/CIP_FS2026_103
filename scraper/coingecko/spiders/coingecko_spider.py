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

            # Note: this is open-to-close (same day). preprocessing.py fills missing values
            # using close-to-close (previous day). The field is not used in modeling.
            if price_usd is not None and open_price is not None and open_price != 0.0:
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
