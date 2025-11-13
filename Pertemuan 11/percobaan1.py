from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time


# --- Opsi (headless optional) ---
opts = Options()
# opts.add_argument("--headless=new")   # aktifkan jika ingin tanpa UI
opts.add_argument("--start-maximized")


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

try:
    driver.get("https://lms.unm.ac.id/")

    print(driver.title)

    time.sleep(5)

finally:
    driver.quit()