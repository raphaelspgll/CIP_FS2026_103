from src.acquisition.binance.scraper import open_kline_docs_page


if __name__ == "__main__":
    html = open_kline_docs_page()
    print("Downloaded HTML length:", len(html))