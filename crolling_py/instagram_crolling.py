from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import json
import time
import random

# WebDriver 설정 (Chrome 예제)
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
di = {}
errror_di = {}

try:
    with open("./instagram_posts.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    for j in range(1,73):
        # 타겟 URL 설정
        url = f"https://nmixx.net/photos?page={j}"  # 여기에 크롤링할 URL을 입력하세요.
        driver.get(url)

        # 페이지 스크롤 끝까지 로드
        last_height = driver.execute_script("return document.body.scrollHeight")  # 초기 문서 높이
        while True:
            # 페이지 스크롤 다운
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            # 새로운 높이 가져오기
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:  # 더 이상 스크롤할 높이가 없으면 중단
                break
            last_height = new_height

        # 크롤링 시작
        for i in range(1, 101):
            try:
                xpath1 = f"/html/body/div[2]/div[7]/div[{i}]/b/a[2]"
                xpath2 = f"/html/body/div[2]/div[7]/div[{i}]/b/a[1]"

                # 이미지 URL
                star = driver.find_element(By.XPATH, xpath1).get_attribute("href")

                # 인스타 URL
                img = driver.find_element(By.XPATH, xpath2).get_attribute("href")
                print(f"{j} - {i}")
                di[img[34:]] = star
            except:
                errror_di[j] = i
                print(f"errror {j} - {i}")

finally:
    # WebDriver 종료
    driver.quit()

data.append(di)
with open("./nmixx_official_posts.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("\n\n\n\n")
print(errror_di)