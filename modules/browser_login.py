# browser_login.py
import time
import pickle
import os
import undetected_chromedriver as uc

# Use a dedicated Chrome user data directory for persistent sessions
CHROME_USER_DATA_DIR = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\AutomationProfile")
COOKIE_FILE = "cookies/quotex_cookies.pkl"

def manual_login_and_save_cookies():
    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={CHROME_USER_DATA_DIR}")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    print("Launching Chrome (undetected) for manual login...")
    driver = uc.Chrome(options=options)

    driver.get("https://quotex.com/en/sign-in")

    print("🔐 Please login manually and complete captcha within 90 seconds...")
    time.sleep(90)  # Adjust if needed

    cookies = driver.get_cookies()
    os.makedirs(os.path.dirname(COOKIE_FILE), exist_ok=True)
    with open(COOKIE_FILE, "wb") as f:
        pickle.dump(cookies, f)

    print("✅ Cookies saved successfully.")
    driver.quit()

if __name__ == "__main__":
    manual_login_and_save_cookies()
