# store_quotex_cookies.py

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pickle
import os
import time

chrome_options = Options()
chrome_options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get("https://quotex.com/en/sign-in")

print("🔐 Please login manually. You have 60 seconds...")
time.sleep(60)

# Save cookies
os.makedirs("cookies", exist_ok=True)
with open("cookies/quotex_cookies.pkl", "wb") as f:
    pickle.dump(driver.get_cookies(), f)

print("✅ Cookies saved to cookies/quotex_cookies.pkl")
driver.quit()
