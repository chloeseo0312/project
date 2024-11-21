from pykrx import stock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from transformers import pipeline

### 주식 ###

## 어제 날짜와 형식화된 어제 날짜 계산

# 평일
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
formatted_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

# # 주말 지나고 월요일
# yesterday = (datetime.today() - timedelta(days=3)).strftime('%Y%m%d')
# formatted_yesterday = (datetime.today() - timedelta(days=3)).strftime('%Y%m%d')

# 어제 코스닥 시장의 시세 데이터 가져오기
kosdaq_data = stock.get_market_ohlcv_by_ticker(yesterday, market="KOSDAQ")

# 등락률이 상한가 범위 (29.7% 이상 30% 이하)인 종목 필터링 후 복사본 생성
filtered_data = kosdaq_data[(kosdaq_data['등락률'] >= 29.7) & (kosdaq_data['등락률'] <= 30)].copy()

# 데이터 타입을 float64로 변환하여 오버플로우 방지 -> 이거 안하면 int32끼리 곱했을 때 값 너무 커져서 거래대금 (-)로 출력됨
filtered_data['종가'] = filtered_data['종가'].astype('float64')
filtered_data['거래량'] = filtered_data['거래량'].astype('float64')

# PER, 시가총액 추가
fundamental_data = stock.get_market_fundamental_by_ticker(yesterday, market="KOSDAQ")
cap_data = stock.get_market_cap_by_ticker(yesterday, market="KOSDAQ")[['시가총액']]

# 종목명 추가
filtered_data['종목명'] = [stock.get_market_ticker_name(ticker) for ticker in filtered_data.index]


# 필요한 정보 선택
result_df = filtered_data[['종목명', '거래량', '등락률']].copy()
result_df['거래대금'] = filtered_data['종가'] * filtered_data['거래량']
result_df = result_df.join(fundamental_data[['PER']], how='left')
result_df = result_df.join(cap_data, how='left')


# 날짜 열 추가
result_df['날짜'] = formatted_yesterday

# print(result_df)

# 200 영업일 전 날짜 계산
two_hundred_days_ago = stock.get_nearest_business_day_in_a_week((datetime.strptime(yesterday, '%Y%m%d') - timedelta(days=280)).strftime('%Y%m%d'))

# 각 종목의 200 영업일치 데이터를 저장할 딕셔너리와 평균 데이터를 저장할 리스트 생성
historical_data_dict = {}

# 각 종목의 200일치 데이터 평균 계산
averages = []
for ticker in filtered_data.index:
    # 각 종목의 200일치 데이터 가져오기
    historical_data = stock.get_market_ohlcv_by_date(two_hundred_days_ago, yesterday, ticker)
    
    # 각 종목의 200일치 데이터를 딕셔너리에 저장
    historical_data = historical_data.reset_index()  # 날짜를 컬럼으로 추가

    historical_data['PER'] = fundamental_data.loc[ticker, 'PER'] if ticker in fundamental_data.index else np.nan
    historical_data['시가총액'] = cap_data.loc[ticker, '시가총액'] if ticker in cap_data.index else np.nan
    historical_data['거래대금'] = historical_data['종가'] * historical_data['거래량']
    historical_data['종목명'] = stock.get_market_ticker_name(ticker)
    historical_data_dict[ticker] = historical_data.copy()
    
    mean_volume = historical_data['거래량'].mean()
    mean_change_rate = historical_data['등락률'].mean()
    mean_trade_value = (historical_data['종가'] * historical_data['거래량']).mean()
    # mean_trade_value = (historical_data['거래대금']).mean()
    mean_per = fundamental_data.loc[ticker, 'PER'] if ticker in fundamental_data.index else np.nan
    mean_market_cap = cap_data.loc[ticker, '시가총액'].mean() if ticker in cap_data.index else np.nan

    averages.append({
        '종목명': stock.get_market_ticker_name(ticker),
        'mean_거래량': mean_volume,
        'mean_등락률_200d': mean_change_rate,
        'mean_거래대금': mean_trade_value,
        'mean_PER': mean_per,
        'mean_시가총액': mean_market_cap
    })
print()

# 저장할 디렉토리 지정
output_dir = "C:/Users/chloeseo/final_project/today/today_stocks/200days"

# 저장할 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 각 종목별로 200일치 데이터를 CSV 파일로 저장
for ticker, data in historical_data_dict.items():
    stock_name = stock.get_market_ticker_name(ticker)
    filename = os.path.join(output_dir, f"{stock_name}_{yesterday}_200days.csv")
    data.to_csv(filename, encoding='cp949', index=False)
    print(f"{filename} 저장 완료")
print()

# 평균 데이터 DataFrame 생성
average_df = pd.DataFrame(averages)

# 합치기
merged_df = pd.merge(result_df, average_df, on='종목명')

### mean_등락률_7d 위함
# 7 영업일 전 날짜 계산
seven_days_ago = stock.get_nearest_business_day_in_a_week((datetime.strptime(yesterday, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d'))

# 각 종목의 7일 평균 등락률 계산
seven_day_averages = []
for ticker in filtered_data.index:
    historical_data = stock.get_market_ohlcv_by_date(seven_days_ago, yesterday, ticker)
    mean_change_rate_7d = historical_data['등락률'].mean()

    seven_day_averages.append({
        '종목명': stock.get_market_ticker_name(ticker),
        'mean_등락률_7d': mean_change_rate_7d
    })

# 7일 평균 데이터 DataFrame 생성
seven_day_avg_df = pd.DataFrame(seven_day_averages)

# 기존 DataFrame과 합치기
final_df = pd.merge(merged_df, seven_day_avg_df, on='종목명')

column_order = ['날짜', '종목명', '거래량', '등락률', '거래대금', 'PER', '시가총액', 'mean_거래량', 'mean_등락률_7d', 'mean_등락률_200d', 'mean_거래대금', 'mean_PER', 'mean_시가총액']
final_df = final_df[column_order]

### 200d-7d 추가!!!
final_df['mean_등락률_7d'] = final_df['mean_등락률_7d'].abs()
final_df['mean_등락률_200d'] = final_df['mean_등락률_200d'].abs()
final_df['mean_등락률_200d-7d'] = final_df['mean_등락률_200d'] - final_df['mean_등락률_7d']

# 최종 DataFrame에 컬럼 순서를 추가 조정
column_order = [
    '날짜', '종목명', '거래량', '등락률', '거래대금', 'PER', '시가총액',
    'mean_거래량', 'mean_등락률_7d', 'mean_등락률_200d', 'mean_등락률_200d-7d',
    'mean_거래대금', 'mean_PER', 'mean_시가총액'
]
final_df = final_df[column_order]


# 거래대금/시가총액 컬럼 추가
final_df['거래대금/시가총액'] = final_df['거래대금'] / final_df['시가총액']

# 거래대금 - mean_거래대금 컬럼 추가
final_df['dif_거래대금'] = final_df['거래대금'] - final_df['mean_거래대금']

# 최종 DataFrame에 컬럼 순서를 추가 조정
column_order = [
    '날짜', '종목명', '거래량', '등락률', '거래대금', 'PER', '시가총액', 'mean_거래량',
    'mean_등락률_7d', 'mean_등락률_200d', 'mean_등락률_200d-7d', 'mean_거래대금',
    'mean_PER', 'mean_시가총액', '거래대금/시가총액', 'dif_거래대금'
]
df = final_df[column_order]

df.to_csv('C:/Users/chloeseo/final_project/today/today_stocks/ceiling_today.csv', encoding='cp949', index=False)

print('오늘 지표 점수까지 출력 완료')
print()

#########################################################################################################################

### 매집봉 ###
# historical_data_dict 생성됐는지 확인
print(f"filtered_data 종목 수: {len(filtered_data)}")  # 필터된 종목 수 확인
print()
print(f"historical_data_dict 생성된 종목 수: {len(historical_data_dict)}")  # 딕셔너리에 저장된 종목 수 확인
print()

# 딕셔너리에 데이터가 없다면 경고 메시지 출력
if not historical_data_dict:
    print("historical_data_dict에 데이터가 없습니다.")

# 매집봉 분석을 위해 각 종목의 200일 데이터를 결합하여 분석
combined_data = []
for ticker, stock_data in historical_data_dict.items():
    # 10일 이동 평균 거래량 계산
    stock_data['10일_평균_거래량'] = stock_data['거래량'].rolling(window=10).mean()
    
    # 10일 평균 대비 거래량 증가율 계산
    stock_data['거래량_증가율_10일'] = (stock_data['거래량'] / stock_data['10일_평균_거래량'] - 1) * 100
    
    # 기준 설정: 10일 평균 대비 500% 이상 거래량 증가, 등락률이 3% 이하
    volume_increase_threshold = 500
    price_change_threshold = 3
    
    # 10일 평균 대비 거래량 증가일 필터링
    potential_abnormal_days = stock_data[
        (stock_data['거래량_증가율_10일'] >= volume_increase_threshold) & 
        (stock_data['등락률'].abs() <= price_change_threshold)
    ].copy()
    
    # '이후 60일 동안' 등락률 절댓값이 15% 이상인 날이 있는지 확인하는 함수
    def has_large_price_change_within_60_days(index):
        end_index = min(index + 60, len(stock_data) - 1)
        next_60_days = stock_data['등락률'].iloc[index + 1:end_index + 1]
        return (next_60_days.abs() >= 15).any()
    
    # 각 행에 대해 조건을 만족하는지 확인하고 필터링
    potential_abnormal_days['이후_급변_여부'] = potential_abnormal_days.index.map(has_large_price_change_within_60_days).astype(bool)
    
    # '조회종목'과 '조회날짜' 컬럼 추가
    stock_name = stock.get_market_ticker_name(ticker)
    potential_abnormal_days['조회종목'] = stock_name
    potential_abnormal_days['조회날짜'] = yesterday
    
    # 리스트에 추가
    combined_data.append(potential_abnormal_days)

# 리스트를 데이터프레임으로 결합
final_df = pd.concat(combined_data, ignore_index=True)

# 최종 데이터프레임 출력
accumulation_today = final_df[['날짜', '거래량', '등락률', '거래량_증가율_10일', '이후_급변_여부', '조회종목', '조회날짜']]

accumulation_today['날짜'] = pd.to_datetime(accumulation_today['날짜'], errors='coerce').dt.strftime('%Y-%m-%d')

### 3년치 매집봉 파일 불러와 아래 붙여서 저장

df_3y = pd.read_csv("C:/Users/chloeseo/final_project/today/today_stocks/매집봉/accumulation.csv", encoding='cp949')
df_3y['날짜'] = pd.to_datetime(df_3y['날짜'], errors='coerce').dt.strftime('%Y-%m-%d')

# accumulation 데이터프레임과 df_3y 데이터프레임 결합
combined_accumulation = pd.concat([df_3y, accumulation_today], ignore_index=True)

# 결합된 데이터프레임을 파일로 저장
accumulation_path = "C:/Users/chloeseo/final_project/accumulation_update.csv"
combined_accumulation.to_csv(accumulation_path, encoding='cp949', index=False)
print(f"결합된 데이터프레임이 {accumulation_path}에 저장되었습니다.")


#########################################################################################################################

### 크롤링 ###
# 날짜 컬럼을 datetime 형식으로 변환
df['날짜'] = pd.to_datetime(df['날짜'])

# 일주일 전 날짜를 계산
df['일주일전'] = df['날짜'] - pd.DateOffset(days=7)

# 종목명과 날짜 추출
df_name = df['종목명'].tolist()
df_start = df['일주일전'].dt.strftime('%Y-%m-%d').tolist()  # 일주일 전 날짜 리스트
df_end = df['날짜'].dt.strftime('%Y-%m-%d').tolist()  # 날짜 리스트

# chrome 브라우저를 적용시킬 때 적용할 옵션 생성
options = Options()
options.add_argument('--start-maximized')
options.add_experimental_option('detach', True)
options.add_experimental_option("excludeSwitches", ["enable-logging!"])

# 다운로드 경로 설정
folder_path = "C:/Users/chloeseo/final_project/today/today_news/news"
download_folder = "C:/Users/chloeseo/Downloads"  # Chrome의 기본 다운로드 폴더
save_folder = folder_path

# news 폴더 초기화 (다운로드 전에 수행)
for file_name in os.listdir(save_folder):
    file_path = os.path.join(save_folder, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)  # 기존 파일 삭제
# print("news 폴더의 기존 파일이 모두 삭제되었습니다.")

# chrome 드라이버 객체 생성
driver = webdriver.Chrome(options=options)

# 검색할 url 생성 (bigkinds로 한번에 이동)
url = 'https://www.bigkinds.or.kr/'
driver.get(url)

# 로그인 버튼 클릭
login_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="header"]/div[1]/div/div[1]/button[1]')))
login_button.click()

# 로그인 정보 입력
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="login-user-id"]'))).send_keys("whyun1199@naver.com")
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="login-user-password"]'))).send_keys("qwerasdf@1234")
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="login-btn"]'))).click()

# 로그인 후 페이지 로딩 대기
time.sleep(3)

# 검색 및 기간 설정
for search_name, start_date_value, end_date_value in zip(df_name, df_start, df_end):
    # 검색창 요소를 찾기 위한 재시도 로직
    attempt = 0
    while attempt < 3:
        try:
            # 검색창 찾고 텍스트 입력
            search_window = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="total-search-key"]')))
            search_window.clear()  # 기존 값 지우기
            search_window.send_keys(search_name)
            search_window.send_keys(Keys.ENTER)
            break  # 성공적으로 실행되면 반복문 탈출
        except Exception as e:
            attempt += 1
            print(f"검색창 찾기 중 오류 발생: {e}")
            time.sleep(2)

    # 기간 설정
    try:
        # 기간 설정 섹션을 닫고 다시 여는 과정 추가
        driver.find_element(By.XPATH, '//*[@id="collapse-step-1"]').click()
        time.sleep(1)  # 잠시 대기하여 로딩 완료를 보장

        # 기간 선택 버튼 클릭
        driver.find_element(By.XPATH, '//*[@id="collapse-step-1-body"]/div[3]/div/div[1]/div[1]/a').click()
        time.sleep(1)

        # 종료일 설정
        end_date = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="search-end-date"]')))
        driver.execute_script("arguments[0].value = '';", end_date)  # JavaScript로 값 지우기
        time.sleep(0.5)  # 잠시 대기
        driver.execute_script(f"arguments[0].value = '{end_date_value}';", end_date)  # JavaScript로 날짜 값 설정

        # 시작일 설정
        start_date = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="search-begin-date"]')))
        driver.execute_script("arguments[0].value = '';", start_date)  # JavaScript로 값 지우기
        time.sleep(0.5)  # 잠시 대기
        driver.execute_script(f"arguments[0].value = '{start_date_value}';", start_date)  # JavaScript로 날짜 값 설정

        # 검색 버튼 클릭
        search_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="search-foot-div"]/div[2]/button[2]')))
        driver.execute_script("arguments[0].click();", search_button)  # JavaScript로 버튼 클릭

        # 건수 100건씩 보기 및 다운로드 처리
        select_element = driver.find_element(By.XPATH, '//*[@id="select2"]')
        select = Select(select_element)
        select.select_by_value("100")

        # step3 버튼 클릭 후 엑셀 다운로드 버튼이 보일 때까지 대기 후 클릭
        driver.find_element(By.XPATH, '//*[@id="collapse-step-3"]').click()
        time.sleep(5)
        
        # 다운로드 버튼을 찾고 클릭
        try:
            download_button = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.XPATH, '//*[@id="analytics-data-download"]/div[3]/button'))
            )
            driver.execute_script("arguments[0].click();", download_button)  # JavaScript로 버튼 클릭
            time.sleep(5)

            # 파일 이름 설정
            new_file_name = f"{search_name}_{end_date_value}.xlsx"
            new_file_path = os.path.join(save_folder, new_file_name)

            # 기본 다운로드 경로에서 새 파일 이름으로 복사
            latest_file = max([f for f in os.listdir(download_folder) if f.endswith(".xlsx")], key=lambda f: os.path.getctime(os.path.join(download_folder, f)))
            shutil.move(os.path.join(download_folder, latest_file), new_file_path)
            print(f"파일 저장 완료: {new_file_path}")

        except Exception as e:
            print(f"엑셀 다운로드 버튼 클릭 중 오류 발생 또는 버튼이 보이지 않음: {e}")

    except Exception as e:
        print(f"기간 설정 중 오류 발생: {e}")

    # 검색 완료 후 검색창 초기화
    try:
        # 새 검색을 위한 리셋 과정
        reset_button = driver.find_element(By.XPATH, '//*[@id="header"]/div[1]/div/h1/a/img')
        reset_button.click()
        time.sleep(2)  # 리셋 후 대기 시간 추가
    except Exception as e:
        print(f"검색 조건 초기화 중 오류 발생: {e}")

# 드라이버 종료
driver.quit()

#########################################################################################################################

### 감성분석 ###
# 폴더 경로 설정 (엑셀 파일들이 저장된 폴더 경로)
folder_path = "C:/Users/chloeseo/final_project/today/today_news/news"
# 병합된 결과를 저장할 경로
combined_result_file = "C:/Users/chloeseo/final_project/today/today_news/classify/종합_감성분석_결과_today.xlsx"

# 감성 분석 모델 생성
model_name = 'snunlp/KR-FinBert-SC'
ko_finbert = pipeline(task='text-classification', model=model_name)

# 감성 분석 함수 정의
def classify(data):
    neg_count = pos_count = neutral_count = 0
    for text in data:
        res = ko_finbert(text)
        if res[0]['label'] == 'negative':
            neg_count += 1
        elif res[0]['label'] == 'positive':
            pos_count += 1
        else:
            neutral_count += 1
    total = len(data)
    return (pos_count / total) * 100, (neg_count / total) * 100, (neutral_count / total) * 100

# 모든 감성 분석 결과를 담을 리스트
all_results = []

# 폴더 내 모든 엑셀 파일을 대상으로 감성 분석 수행
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        
        # 파일명에서 종목명과 날짜 추출
        stock_name, date_str = file_name.split('_')[0], file_name.split('_')[1].replace('.xlsx', '')
        print(f"\n파일: {file_name} 분석 중... (종목명: {stock_name}, 날짜: {date_str})")

        try:
            # 엑셀 파일 읽기
            df = pd.read_excel(file_path)
            
            # 파일에 '제목'과 '본문' 열이 있는지 확인
            if '제목' in df.columns and '본문' in df.columns:
                # '제목' 열에서 종목명이 포함된 행의 인덱스 찾기
                title = df['제목'].to_list()
                index = [idx for idx, data in enumerate(title) if stock_name in data]

                # 종목명을 포함한 제목 개수 확인
                num_titles_with_stock = len(index)
                print(f"종목명을 포함한 제목 개수: {num_titles_with_stock}")

                if num_titles_with_stock > 0:
                    # '제목'과 '본문' 열을 결합하여 감성 분석할 데이터 생성
                    df_news = df.loc[index, ['제목', '본문']]
                    df_news['total'] = df_news['제목'] + ' ' + df_news['본문']
                    texts = [text.strip() for text in df_news['total']]

                    # 감성 분석 수행
                    pos_ratio, neg_ratio, neutral_ratio = classify(texts)
                    print(f"긍정 비율: {pos_ratio}%, 부정 비율: {neg_ratio}%, 중립 비율: {neutral_ratio}%")
                else:
                    print(f"파일 '{file_name}'에 '{stock_name}' 키워드가 포함된 제목이 없습니다.")
                    pos_ratio = neg_ratio = neutral_ratio = 0.0  # 감성 분석 결과가 없으므로 0으로 설정
                
                # 결과를 리스트에 추가
                result_data = {
                    '종목명': stock_name,
                    '날짜': date_str,
                    '종목명 포함 제목 개수': num_titles_with_stock,
                    '긍정 비율(%)': pos_ratio,
                    '부정 비율(%)': neg_ratio,
                    '중립 비율(%)': neutral_ratio
                }
                all_results.append(result_data)
            else:
                print(f"파일 '{file_name}'에 '제목' 또는 '본문' 열이 없습니다. 열 이름을 확인하세요.")

        except Exception as e:
            print(f"파일 '{file_name}' 처리 중 오류 발생: {e}")

# 모든 데이터를 하나의 데이터프레임으로 병합
if all_results:
    combined_df = pd.DataFrame(all_results)
    # 병합된 데이터프레임을 엑셀 파일로 저장
    combined_df.to_excel(combined_result_file, index=False)
    print(f"\n모든 파일이 병합되어 '{combined_result_file}'에 저장되었습니다.")
else:
    print("병합할 데이터가 없습니다.")

print()

### 당일 상한가 주식 + 뉴스 감성분석 합치기 ### 

# Load the uploaded files
file_path_excel = "C:/Users/chloeseo/final_project/today/today_news/classify/종합_감성분석_결과_today.xlsx"
file_path_csv = "C:/Users/chloeseo/final_project/today/today_stocks/ceiling_today.csv"

# Read the Excel and CSV files
df1 = pd.read_excel(file_path_excel)
df2 = pd.read_csv(file_path_csv, encoding='cp949')

# 날짜 컬럼을 문자열로 변환하여 'YYYY-MM-DD' 형식으로 변환
df2['날짜'] = pd.to_datetime(df2['날짜'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

# Merge the two dataframes on '종목명' and '날짜' columns, using an outer join to include all rows and handle missing values with NaN
today_final_df = pd.merge(df1, df2, on=['종목명', '날짜'], how='outer')

today_final_df.to_csv("C:/Users/chloeseo/final_project/today/today_final.csv", encoding='cp949', index=False)

### 위에서 사용한 변수 그대로 불러와 합치는 방법
# ## 맨 처음에 3년치 데이터랑 합칠때만 아래 파일 사용
# df = pd.read_csv("C:/Users/chloeseo/final_project/stock2/final_merged_indicators_df_re.csv", encoding='cp949')
## 위의 파일이랑 한 번 합친 후부터는 keep_update.csv를 계속 업데이트 할 것
df = pd.read_csv("C:/Users/chloeseo/final_project/keep_update.csv", encoding='cp949')

keep_update = pd.concat([df, today_final_df], axis=0)

keep_update.to_csv("C:/Users/chloeseo/final_project/keep_update.csv", encoding='cp949', index=False)

#########################################################################################################################

### 지표별 점수내기 ###

# 문자열에서 쉼표와 공백을 제거하고 숫자로 변환
file_path='C:/Users/chloeseo/final_project/keep_update.csv'
data = pd.read_csv(file_path,encoding='cp949')

# 분석에 사용할 컬럼 지정
input_cols = ['거래대금/시가총액', 'mean_등락률_7d']

# IQR 계산
Q1 = data[input_cols].quantile(0.25)
Q3 = data[input_cols].quantile(0.75)
IQR = Q3 - Q1

# 이상치 경계값 계산 (컬럼별)
min_values = Q3 + 1.5 * IQR
max_values = data[input_cols].max()

# 거리 및 구간 설정 (컬럼별 구간)
segments_dict = {}
for col in input_cols:
    distance = abs(max_values[col] - min_values[col])
    segment_size = distance / 100
    segments_dict[col] = [min_values[col] + i * segment_size for i in range(101)]

# 점수 계산 함수
def get_score(value, col):
    """각 값에 대해 구간별 점수를 계산하는 함수"""
    segments = segments_dict[col]
    for i in range(100):
        if segments[i] <= value < segments[i + 1]:
            return i + 1  # 해당 구간의 점수 반환
    if value >= segments[-1]:  # 마지막 구간보다 큰 경우
        return 101
    return 0  # 구간에 속하지 않는 경우

# 점수 계산 및 새로운 열 추가 (컬럼별 점수 계산)
for col in input_cols:
    data[f'{col} 점수'] = data[col].apply(lambda x: get_score(x, col))

############################################################################

# 시가총액 계산 따로 실시
P1=data['시가총액'].min()
P3=data['시가총액'].quantile(0.25)

# 20등분으로 나눌 구간 크기 계산
segment_size = (P3 - P1) / 100 # P1, P3 사용

# 각 구간의 경계 설정
segments = [P1 + i * segment_size for i in range(101)]  # 0~30까지 포함

# 각 구간별 점수 매기기
scores = {f"{segments[i]:.2f} ~ {segments[i + 1]:.2f}": i + 1 for i in range(100)}

# 구간 점수 계산 함수
def get_score(value):
    for i in range(100):
        if segments[i] <= value < segments[i + 1]:
            return i + 1  # 해당 구간의 점수 반환
    if value <= segments[-1]:  # 최초 구간보다 작은 경우
        return 101
    return 0  # 구간에 속하지 않는 경우 (필요 시 처리)

data['시가총액 점수'] = data['시가총액'].apply(get_score)

#############################################################################

# 분석에 사용할 컬럼 지정
input_cols = ['PER', 'mean_PER']

# IQR 계산 (0을 제외한 값들로)
Q1 = data[input_cols].quantile(0.25)
Q3 = data[input_cols].quantile(0.75)
IQR = Q3 - Q1

# 이상치 경계값 계산 (컬럼별)
min_values = Q3 - 1.5 * IQR
max_values = Q3 + 1.5 * IQR  # IQR 기준 상한을 계산 (outlier 제외)

# 거리 및 구간 설정 (컬럼별 구간)
segments_dict = {}
for col in input_cols:
    # 각 컬럼에 대한 최댓값과 최솟값을 구하기
    min_val = min_values[col] if min_values[col] > 0 else 0  # 최소값은 0 이상
    max_val = data[col].max()  # 실제 컬럼에서의 최대값 사용

    distance = max_val - min_val
    segment_size = distance / 100  # 100개의 구간으로 나누기
    segments_dict[col] = [min_val + i * segment_size for i in range(101)]  # 100개의 구간

# 점수 계산 함수
def get_score(value, col):
    """각 값에 대해 구간별 점수를 계산하는 함수"""
    segments = segments_dict[col]
    for i in range(100):
        if segments[i] < value <= segments[i + 1]:
            return i + 1  # 해당 구간의 점수 반환
    if value == 0:  # 0인경우
        return 50
    return 0  # 구간에 속하지 않는 경우

# 점수 계산 및 새로운 열 추가 (컬럼별 점수 계산)
for col in input_cols:
    data[f'{col} 점수'] = data[col].apply(lambda x: get_score(x, col))


############################################################################
# 변동성 갭
# 분석에 사용할 단일 컬럼 지정
col = 'mean_등락률_200d-7d'

# IQR 계산
Q1 = data[col].quantile(0.25)
Q3 = data[col].quantile(0.75)
IQR = Q3 - Q1

# 이상치 경계값 계산 (하한과 상한)
min_value = data[col].min()   # 하한
max_value = Q1 - 1.5 * IQR  # 상한

# 거리 및 구간 설정 (단일 컬럼)
distance = abs(min_value - max_value)
segment_size = distance / 100
segments = [min_value + i * segment_size for i in range(101)]  # 100개의 구간

# 점수 계산 함수
def get_score(value):
    """각 값에 대해 구간별 점수를 계산하는 함수"""
    for i in range(100):
        if segments[i] <= value < segments[i + 1]:
            return 100 - (i + 1)  # 해당 구간의 점수 반환
    return 0  # 구간에 속하지 않는 경우

# 점수 계산 및 새로운 열 추가 (단일 컬럼 점수 계산)
data[f'{col} 점수'] = data[col].apply(get_score)


## 변동성 갭 (2)

# Q1, Q3, IQR 계산
Q1 = data[col].quantile(0.25)
Q3 = data[col].quantile(0.75)
IQR = Q3 - Q1

# 이상치 경계값 계산 (하한과 상한)
min_value = 0  # 하한 (0 이상)
max_value = data[col].max()  # 상한 (컬럼의 최댓값)

# 거리 및 구간 설정 (단일 컬럼)
distance = abs(min_value - max_value)
segment_size = distance / 100
segments = [min_value + i * segment_size for i in range(101)]  # 100개의 구간

# 점수 계산 함수
def get_score(value):
    """각 값에 대해 구간별 점수를 계산하는 함수"""

    # 0 이상의 값에 대해서만 구간 점수 계산
    if value >= 0:
        for i in range(100):
            if segments[i] <= value < segments[i + 1]:
                return i + 1  # 해당 구간의 점수 반환
    return None  # 0 미만 값에 대해서는 변경하지 않음 (기존 점수 유지)

# 기존 점수 열을 바꿀 때, 0 이상인 값에 대해서만 새로 계산된 점수 적용
data[f'{col} 점수'] = data.apply(lambda row: get_score(row[col]) if row[col] >= 0 else row[f'{col} 점수'], axis=1)

###############################################################################
# 각 지표 점수 더한 최종 점수
data['최종 점수'] = data[['거래대금/시가총액 점수', 'mean_등락률_7d 점수', '시가총액 점수', 'PER 점수', 'mean_PER 점수', 'mean_등락률_200d-7d 점수']].sum(axis=1)

data.to_csv("C:/Users/chloeseo/final_project/keep_update.csv", encoding='cp949', index=False)

print('지표별 점수 계산 완료')

print()

#########################################################################################################################

### 종목별 평균 점수 ### 
score_mean = data[data['최종 점수'] != 0].groupby('종목명')['최종 점수'].mean().reset_index()
score_mean.to_csv("C:/Users/chloeseo/final_project/score_mean.csv", encoding='cp949', index=False)

print('keep_update와 score_mean 생성까지 모든 과정이 성공적으로 완료되었습니다.')