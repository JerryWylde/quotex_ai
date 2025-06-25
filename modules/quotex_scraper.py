# executor/quotex_scraper.py
import os
import time
import logging
import pickle
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

class QuotexScraper:
    def __init__(self, headless=False):
        try:
            logging.info("🔍 Initializing QuotexScraper with cookies...")

            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            if headless:
                options.add_argument("--headless=new")

            self.driver = uc.Chrome(options=options, headless=headless, version_main=137)
            self.driver.set_page_load_timeout(60)
            self.driver.get("https://quotex.com/en")
            time.sleep(5)

            cookies_path = "cookies/quotex_cookies.pkl"
            if not os.path.exists(cookies_path):
                raise FileNotFoundError("❌ Cookie file not found. Run browser_login.py first.")

            with open(cookies_path, "rb") as f:
                cookies = pickle.load(f)

            for cookie in cookies:
                cookie.pop("sameSite", None)
                self.driver.add_cookie(cookie)

            self.driver.refresh()
            time.sleep(5)

            if "login" in self.driver.current_url:
                logging.warning("🔒 Cookie login failed. Run browser_login.py again.")
                raise Exception("Login via cookies failed.")

            logging.info("✅ Logged in using cookies.")

        except Exception as e:
            logging.exception(f"❌ Failed to init scraper: {e}")
            raise

    def get_latest_trade_result(self) -> str:
        try:
            self.driver.get("https://quotex.com/en/trade/history/")
            time.sleep(4)

            result_xpath = '(//div[contains(@class, "history-item__result")])[1]'
            result_element = self.driver.find_element(By.XPATH, result_xpath)
            result_text = result_element.text.strip().upper()

            if "WON" in result_text:
                return "WON"
            elif "LOST" in result_text:
                return "LOST"
            else:
                return "UNKNOWN"

        except Exception as e:
            logging.error(f"❌ Failed to scrape trade result: {e}")
            return "SCRAPE_FAILED"

    def close(self):
        try:
            self.driver.quit()
            logging.info("🛑 Scraper browser closed.")
        except Exception as e:
            logging.error(f"❌ Failed to close browser: {e}")
