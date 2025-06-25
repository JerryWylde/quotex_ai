import os
import time
import logging
import pickle
import undetected_chromedriver as uc

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class QuotexExecutor:
    def __init__(self, headless=False):
        try:
            logging.info("🚀 Initializing QuotexExecutor with persistent login via cookies...")

            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument("--remote-debugging-port=9222")
            if headless:
                options.add_argument("--headless=new")

            self.driver = uc.Chrome(options=options, headless=headless, version_main=137)
            self.driver.set_page_load_timeout(60)
            self.driver.get("https://quotex.com/en")
            time.sleep(5)

            # 🔁 Load cookies
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
                logging.warning("🔒 Login failed after cookie injection. Run browser_login.py again.")
                raise Exception("❌ Cookie login failed.")

            logging.info("✅ QuotexExecutor successfully logged in using cookies.")

        except Exception as e:
            logging.exception(f"❌ Initialization failed: {e}")
            raise


    def place_trade(self, direction):
        try:
            logging.info(f"🖱️ Attempting to place a {direction.upper()} trade...")

            self.driver.get("https://quotex.com/en/trade")
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "section-deal__button"))
            )

            if direction.lower() in ["call", "up"]:
                selector = "button.call-btn.section-deal__button"
            elif direction.lower() in ["put", "down"]:
                selector = "button.put-btn.section-deal__button"
            else:
                logging.error("❌ Invalid trade direction specified.")
                return "FAIL"

            button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            self.driver.execute_script("arguments[0].click();", button)

            logging.info(f"✅ {direction.upper()} trade clicked successfully.")
            return "EXECUTED"

        except Exception as e:
            logging.error(f"❌ Trade execution failed: {e}")
            return "FAIL"


    def ensure_demo_mode(self):
        try:
            self.driver.implicitly_wait(10)

            demo_xpath = "//div[contains(text(), 'Demo account')]"
            if "Demo account" in self.driver.page_source:
                demo_button = self.driver.find_element(By.XPATH, demo_xpath)
                self.driver.execute_script("arguments[0].click();", demo_button)
                logging.info("✅ Switched to Demo account.")
                time.sleep(2)
            else:
                logging.info("✅ Already in demo or live account view.")

        except Exception as e:
            logging.warning(f"⚠️ Could not ensure Demo mode: {e}")


    def close(self):
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                logging.info("🛑 Browser session closed.")
        except Exception as e:
            logging.error(f"❌ Error closing browser: {e}")
