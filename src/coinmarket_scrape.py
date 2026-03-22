from playwright.sync_api import sync_playwright
import pandas as pd
import os
from datetime import datetime, timedelta

output_dir = '../data/raw/'
os.makedirs(output_dir, exist_ok=True)

coins = [
    ("bitcoin", "bitcoin_historical.csv"),
    ("xrp", "xrp_historical.csv"),
    ("internet-computer", "icp_historical.csv"),
]

target_date = datetime.now() - timedelta(days=3 * 365)

def scrape_coin(page, coin_slug):
    page.goto(f"https://coinmarketcap.com/currencies/{coin_slug}/historical-data/")
    page.wait_for_timeout(5000)

    while True:
        rows = page.query_selector_all("tbody tr")
        last_row = rows[-1].query_selector_all("td")
        last_date = datetime.strptime(last_row[0].inner_text(), "%b %d, %Y")

        print(f"[{coin_slug}] Oldest date: {last_date.strftime('%Y-%m-%d')}")

        if last_date <= target_date:
            print(f"[{coin_slug}] 3 Years reached!")
            break

        try:
            page.locator("text=Load More").click()
            page.wait_for_timeout(2000)
        except:
            print(f"[{coin_slug}] The “Load More” button is no longer found")
            break

    rows = page.query_selector_all("tbody tr")
    data = []
    for row in rows:
        cells = row.query_selector_all("td")
        if cells:
            data.append([cell.inner_text() for cell in cells])

    return data

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    for coin_slug, file_name in coins:
        print(f"\n=== Scraping {coin_slug} ===")
        data = scrape_coin(page, coin_slug)

        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "market_cap"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "market_cap"]:
            df[col] = df[col].str.replace("$", "", regex=False)\
                             .str.replace(",", "", regex=False)\
                             .astype(float)
        df = df.sort_values("date").reset_index(drop=True)

        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(df)} lines)")

    browser.close()