from src.acquisition.binance.endpoints import (
    get_historical_klines,
    get_yesterday_date_string,
)
from src.acquisition.binance.parser import klines_to_dataframe


if __name__ == "__main__":
    end_date = get_yesterday_date_string()

    raw_data = get_historical_klines(
        symbol="BTCUSDT",
        interval="1d",
        start_date="2023-01-01",
        end_date=end_date,
        limit=1000,
    )

    df = klines_to_dataframe(raw_data)

    print("\nDataFrame shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nHead:")
    print(df.head())