import requests


BASE_URL = "https://api.binance.com"


def send_request(endpoint: str, params: dict = None) -> list:
    url = BASE_URL + endpoint

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    return response.json()