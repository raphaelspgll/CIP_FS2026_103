from src.acquisition.binance.endpoints import (
    get_historical_klines,
    get_yesterday_date_string,
)
from src.acquisition.binance.parser import klines_to_dataframe
from src.acquisition.binance.save_raw import save_dataframe_to_csv


SYMBOLS = ["BTCUSDT", "XRPUSDT", "ICPUSDT"]
INTERVAL = "1d"
START_DATE = "2023-01-01"


if __name__ == "__main__":
    end_date = get_yesterday_date_string()

    for symbol in SYMBOLS:
        print(f"\n--- Processing {symbol} ---")

        raw_data = get_historical_klines(
            symbol=symbol,
            interval=INTERVAL,
            start_date=START_DATE,
            end_date=end_date,
            limit=1000,
        )

        df = klines_to_dataframe(raw_data)
        df["symbol"] = symbol

        print("DataFrame shape:", df.shape)
        print("First date:", df["open_time"].min())
        print("Last date:", df["open_time"].max())

        output_path = f"data/raw/binance/{symbol.lower()}_{INTERVAL}.csv"
        save_dataframe_to_csv(df, output_path)