import pandas as pd


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def klines_to_dataframe(raw_klines: list) -> pd.DataFrame:
    df = pd.DataFrame(raw_klines)

    # --- Validate structure ---
    if df.shape[1] != 12:
        raise ValueError(f"Expected 12 columns, got {df.shape[1]}")

    df.columns = KLINE_COLUMNS

    # --- Convert timestamps ---
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # --- Convert numeric columns ---
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # --- Convert trades to int ---
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    # --- Drop useless column ---
    df = df.drop(columns=["ignore"])

    # --- Sort and reset index ---
    df = df.sort_values("open_time")
    df = df.reset_index(drop=True)

    return df