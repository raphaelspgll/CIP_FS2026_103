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
        df.sort_values(["coin_id", "date", "scraped_at"])
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
    df["is_outlier"] = df["price_change_pct"].abs().gt(50).fillna(False)

    # 5. Sort
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        try:
            df = pd.read_csv(path)
            cleaned = clean(df)
        except Exception as e:
            print(f"[{path}] ERROR: {e} — skipping")
            continue
        for coin_id, group in cleaned.groupby("coin_id"):
            out_path = os.path.join(PROCESSED_DIR, f"{coin_id}_cleaned.csv")
            group.to_csv(out_path, index=False)
            print(f"[{coin_id}] {len(group)} rows → {out_path}")


if __name__ == "__main__":
    main()
