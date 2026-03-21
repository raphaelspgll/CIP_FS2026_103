from datetime import datetime

from .client import send_request


def date_to_milliseconds(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def get_klines(
    symbol: str,
    interval: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 1000,
) -> list:
    endpoint = "/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    if start_date is not None:
        params["startTime"] = date_to_milliseconds(start_date)

    if end_date is not None:
        params["endTime"] = date_to_milliseconds(end_date)

    return send_request(endpoint, params)