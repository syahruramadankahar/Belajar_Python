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
    login_button.click()

    time.sleep(1)

    #login
    username = wait.until(EC.element_to_be_clickable((By.ID, 'login_username')))
    password = wait.until(EC.element_to_be_clickable((By.ID, 'login_password')))

    username.send_keys("240210501012")
    password.send_keys("ADam211004")

    submit_login = driver.find_element(By.XPATH, "/html/body/div[1]/div[1]/div[3]/div/div/div[2]/div/div/form/button")
    submit_login.click()

    #cari mata kuliahnya
    cari_matkul = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[1]/div[3]/div/div/div/div[1]/div/div[3]/div[2]/aside[2]/div[3]/div/div/div/div[2]/div/div/div[1]/div/div/div[21]")))
    cari_matkul.click()

    #klik tombol view mata kuliahnya
    klik_view = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[1]/div[3]/div/div/div/div[1]/div/div[3]/div[2]/aside[2]/div[3]/div/div/div/div[2]/div/div/div[1]/div/div/div[21]/div[1]/div/ul/li/a")))
    klik_view.click()
    
    #klik content course
    content_course = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[1]/div[5]/div[3]/div[2]/div/div/div/div/div/div/div/div[2]/div[2]/li[1]/div/div/div[1]/h4/a')))
    content_course.click()

    #klik course
    klik_course = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[1]/div[5]/div[3]/div[2]/div/div/div/div/div/div/div/div[2]/div[2]/li[1]/div/div/div[2]/div/div[3]/ul/li[3]/div/div/div[2]/div[1]/a/span')))
    klik_course.click()

    #klik ajukan kehadiran
    ajukan_kehadiran = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[1]/div[5]/div[3]/div[2]/div/div/div/div/div/table[1]/tbody/tr[6]/td[3]/a')))
    ajukan_kehadiran.click()

    #klik jenis kehadiran seperti (hadir,sakit,alpa dan izin)
    jenis_kehadiran = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[1]/div[5]/div[3]/div[2]/div/div/div/div/div/form/fieldset/div/div/div[2]/fieldset/div/label[1]/span')))
    jenis_kehadiran.click()

    #klik simpan perubahan / submit kehadiran
    simpan = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[1]/div[5]/div[3]/div[2]/div/div/div/div/div/form/div[2]/div[2]/fieldset/div/div[1]/span[2]')))
    simpan.click()

    time.sleep(10)

finally:
    driver.quit()