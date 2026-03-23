from datetime import datetime, timedelta

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


def get_yesterday_date_string() -> str:
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def get_historical_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    limit: int = 1000,
) -> list:
    all_data = []
    current_start_ms = date_to_milliseconds(start_date)
    end_ms = date_to_milliseconds(end_date)

    while current_start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_ms,
            "endTime": end_ms,
            "limit": limit,
        }

        batch = send_request("/api/v3/klines", params)

        if not batch:
            break

        all_data.extend(batch)

        last_open_time = batch[-1][0]
        next_start_ms = last_open_time + 1

        if next_start_ms <= current_start_ms:
            break

        current_start_ms = next_start_ms

        print(f"Downloaded batch: {len(batch)} rows, total so far: {len(all_data)}")

        if len(batch) < limit:
            break

    return all_data