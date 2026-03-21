from src.acquisition.binance.endpoints import get_klines


if __name__ == "__main__":
    data = get_klines(
        symbol="BTCUSDT",
        interval="1d",
        start_date="2023-01-01",
        end_date="2023-01-10",
        limit=10,
    )

    print("Number of rows:", len(data))
    print("\nFirst row:")
    print(data[0])

    print("\nLast row:")
    print(data[-1])