from src.acquisition.binance.scraper import open_binance_homepage


if __name__ == "__main__":
    html = open_binance_homepage()
    print("Downloaded HTML length:", len(html))