from src.acquisition.binance.endpoints import get_klines


if __name__ == "__main__":
    data = get_klines(symbol="BTCUSDT", interval="1d", limit=5)

    print("Number of rows:", len(data))
    print("\nFirst row:")
    print(data[0])