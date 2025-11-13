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
    time.sleep(5)

    deskripsi = driver.find_element(By.XPATH, '//*[@id="inst22"]/div/div/div/div[1]/div/div/p[1]')
    print(deskripsi.text)

finally:
    driver.quit()