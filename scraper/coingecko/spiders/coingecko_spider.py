import scrapy
from scrapy import Selector
from datetime import datetime, timezone, timedelta
from coingecko.items import CryptoItem


def parse_price(text: str):
    """Strip $, commas, whitespace; return float or None."""
    cleaned = str(text).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


class CoinGeckoSpider(scrapy.Spider):
    name = "coingecko"

    def start_requests(self):
        coin_list = self.settings.get("COIN_LIST", [])
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.cutoff_date = (
            datetime.now(timezone.utc) - timedelta(days=3 * 365)
        ).strftime("%Y-%m-%d")
        self.logger.info(f"Scraping from {self.cutoff_date} to {self.today}")

        for coin in coin_list:
            # Initial Scrapy request uses the base URL (robots.txt compliant).
            # Playwright will then navigate to the date-ranged URL inside the browser.
            base_url = (
                f"https://www.coingecko.com/en/coins/{coin['coin_id']}/historical_data"
            )
            yield scrapy.Request(
                base_url,
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

    async def _oldest_table_date(self, page) -> str | None:
        """Return the oldest date string visible in the table, or None."""
        rows = await page.query_selector_all("table tbody tr")
        oldest = None
        for row in rows:
            cells = await row.query_selector_all("td")
            if not cells:
                continue
            text = (await cells[0].inner_text()).strip()
            if len(text) == 10 and (oldest is None or text < oldest):
                oldest = text
        return oldest

    async def _click_button_by_text(self, page, text: str) -> bool:
        """Click the first visible button whose text matches. Returns True if found."""
        buttons = await page.query_selector_all("button")
        for btn in buttons:
            t = (await btn.inner_text()).strip()
            if t == text:
                await btn.click()
                return True
        return False

    async def parse(self, response, **kwargs):
        coin = response.meta["coin"]
        page = response.meta.get("playwright_page")

        if page:
            # Navigate to 3-year date range via browser (bypasses Scrapy robots.txt check)
            date_url = (
                f"https://www.coingecko.com/en/coins/{coin['coin_id']}/historical_data"
                f"?start={self.cutoff_date}&end={self.today}"
            )
            await page.goto(date_url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_selector("table tbody tr", timeout=30000)

            # Loop "Show More" until oldest row is at or before cutoff
            max_clicks = 80  # safety cap: 80 * ~20 rows >> 1095 rows needed
            for i in range(max_clicks):
                oldest = await self._oldest_table_date(page)
                if oldest is None or oldest <= self.cutoff_date:
                    self.logger.info(
                        f"[{coin['coin_id']}] Oldest date {oldest!r} reached cutoff "
                        f"after {i} 'Show More' clicks"
                    )
                    break
                clicked = await self._click_button_by_text(page, "Show More")
                if not clicked:
                    self.logger.warning(
                        f"[{coin['coin_id']}] 'Show More' not found after {i} clicks; "
                        f"oldest date: {oldest}"
                    )
                    break
                await page.wait_for_timeout(1500)
            else:
                oldest = await self._oldest_table_date(page)
                self.logger.warning(
                    f"[{coin['coin_id']}] Hit max_clicks={max_clicks}; oldest date: {oldest}"
                )

            html = await page.content()
            await page.close()
            sel = Selector(text=html)
        else:
            sel = response

        # Table columns: Date | Market Cap | Volume | Price
        rows = sel.css("table tbody tr")
        self.logger.info(f"[{coin['coin_id']}] Found {len(rows)} candidate rows")

        start_date = end_date = None
        count = 0

        for row in rows:
            cells = row.css("td")
            if len(cells) < 4:
                continue

            date_str = cells[0].css("::text").get("").strip()
            if not date_str or len(date_str) != 10:
                continue

            if date_str < self.cutoff_date:
                continue

            market_cap_usd = parse_price(cells[1].css("::text").get(""))
            volume_24h_usd = parse_price(cells[2].css("::text").get(""))
            price_usd      = parse_price(cells[3].css("::text").get(""))

            # price_change_pct unavailable from this table;
            # preprocessing.py recomputes it from price_usd.shift(1).
            yield CryptoItem(
                date             = date_str,
                coin_id          = coin["coin_id"],
                coin_name        = coin["name"],
                symbol           = coin["symbol"],
                price_usd        = price_usd,
                market_cap_usd   = market_cap_usd,
                volume_24h_usd   = volume_24h_usd,
                price_change_pct = None,
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
