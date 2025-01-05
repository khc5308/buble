import time, json
from selenium import webdriver
from selenium.webdriver.common.by import By # 요소 찾는거 도와주는거   ex) By.id, By.CSS_SELECTOR
from selenium.webdriver.common.keys import Keys

# Chrome 드라이버 초기화
driver = webdriver.Chrome()

# Twitter 미디어 페이지 열기
driver.get('https://x.com/NMIXX_official')
time.sleep(5)  # 페이지 로드 대기

post_cnt_str = driver.find_elements(By.XPATH, '/html/body/div[1]/div/div/div[2]/main/div/div/div/div/div/div[1]/div[1]/div/div/div/div/div/div[2]/div/div')
post_cnt = 8073

# 더 많은 미디어를 로드하기 위해 스크롤하기
def scroll_down():
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END) # 페이지 끝까지 내리기
    time.sleep(0.8)  # 속도 조절

# 데이터 수집
media_data = []

while True:
    # 모든 미디어 게시물 찾기
    articles = driver.find_elements(By.TAG_NAME, 'article')
    for article in articles:
        try:
            
            # 트윗의 날짜 가져오기
            text = article.find_element(By.XPATH, './/*[@class="css-146c3p1 r-8akbws r-krxsd3 r-dnmrzs r-1udh08x r-bcqeeo r-1ttztb7 r-qvutc0 r-37j5jr r-a023e6 r-rjixqe r-16dba41 r-bnwqim"]')

            # 트윗 링크 가져오기 
            link_element = article.find_element(By.XPATH, './/a[contains(@href, "/status/")]')
            post_link = link_element.get_attribute('href')

            # 미디어 URL 가져오기
            media_elements = article.find_elements(By.XPATH, './/img[contains(@src, "twimg.com/media/")]')
            for media_element in media_elements:
                media_url = media_element.get_attribute('src')
                media_data.append({
                    'text': text,
                    'media_url': media_url,
                    'post_link': post_link
                })
        except :
            print(f"게시물 처리 중 오류 발생, 속도를 줄여보는거 추천")

    # 더 많은 콘텐츠를 로드하기 위해 스크롤 다운
    scroll_down()

    # 종료 조건 확인 (자신의 조건에 맞게 설정 가능)
    if len(media_data) >= post_cnt:  # 일단 이거 게시물 
        break

# 브라우저 닫기
driver.quit()

# 수집한 데이터를 JSON 파일로 저장
with open('twitter_media.json', 'w') as f:
    json.dump(media_data, f, indent=4)

print(f"{len(media_data)}개의 미디어 항목을 twitter_media.json 파일로 저장했습니다.")
