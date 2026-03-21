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

    if df.shape[1] != 12:
        raise ValueError(f"Expected 12 columns, got {df.shape[1]}")

    df.columns = KLINE_COLUMNS
    return df