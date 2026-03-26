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
