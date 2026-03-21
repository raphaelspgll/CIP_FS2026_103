from .client import send_request


def get_klines(symbol: str, interval: str, limit: int = 10) -> list:
    endpoint = "/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    return send_request(endpoint, params)