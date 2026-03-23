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
            try:
                count = self._dedup(path)
                spider.logger.info(f"[{coin_id}] Wrote {count} rows (after dedup)")
            except Exception as e:
                spider.logger.info(f"[{coin_id}] Dedup failed: {e}")

    def _dedup(self, path):
        df = pd.read_csv(path)
        df = df.sort_values("scraped_at").drop_duplicates(
            subset=["coin_id", "date"], keep="last"
        )
        df[FIELDS].to_csv(path, index=False)
        return len(df)
