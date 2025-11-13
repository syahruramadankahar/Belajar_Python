from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


# --- Opsi (headless optional) ---
opts = Options()
# opts.add_argument("--headless=new")   # aktifkan jika ingin tanpa UI
opts.add_argument("--start-maximized")


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

try:
    driver.get("https://lms.unm.ac.id/")

    wait = WebDriverWait(driver, 10)
    login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[1]/header/div/nav/ul[2]/li[1]/a')))
    print(login_button.click())

    time.sleep(5)

finally:
    driver.quit()