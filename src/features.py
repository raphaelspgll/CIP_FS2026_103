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
