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
