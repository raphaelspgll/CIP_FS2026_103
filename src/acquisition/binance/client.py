import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


DOCS_URL = "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints"


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    return webdriver.Chrome(options=options)


def handle_cookie_popup(driver: webdriver.Chrome) -> None:
    reject_xpaths = [
        "//button[contains(., 'Reject Additional Cookies')]",
        "//button[contains(., 'Reject All')]",
        "//button[contains(., 'Decline')]",
        "//button[contains(., 'Only necessary')]",
    ]

    accept_xpaths = [
        "//button[contains(., 'Accept Cookies & Continue')]",
        "//button[contains(., 'Accept Cookies')]",
        "//button[contains(., 'Accept')]",
        "//button[contains(., 'Allow all')]",
    ]

    possible_xpaths = reject_xpaths + accept_xpaths

    for xpath in possible_xpaths:
        try:
            button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )

            try:
                button.click()
            except Exception:
                driver.execute_script("arguments[0].click();", button)

            print(f"Clicked cookie button: {button.text}")
            time.sleep(1)
            return

        except TimeoutException:
            continue

    print("No cookie popup handled.")


def open_kline_docs_page() -> str:
    driver = build_driver()

    try:
        driver.get(DOCS_URL)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        handle_cookie_popup(driver)
        time.sleep(2)

        body = driver.find_element(By.TAG_NAME, "body").text

        print("Page title:", driver.title)
        print("Current URL:", driver.current_url)
        print("Body text length:", len(body))

        if "Kline/Candlestick data" in body:
            print("\nSUCCESS: Found 'Kline/Candlestick data' on the page.\n")
        else:
            print("\nWARNING: 'Kline/Candlestick data' not found.\n")

        print("----- PAGE PREVIEW -----")
        print(body[:1500])

        return driver.page_source

    finally:
        driver.quit()


if __name__ == "__main__":
    html = open_kline_docs_page()
    print("\nHTML length:", len(html))