import pandas as pd
import os

INPUT_FILE = '../../shared-data/merged-data.csv'          # 상대 경로 (시간순 정렬된 데이터)
OUTPUT_DIR = '../data/time-based-split/'                  # 현재 디렉토리 기준 출력 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(INPUT_FILE)

# 총 행 수 및 분할 단위 계산
total_rows = len(df)
chunk_size = total_rows // 10

# 데이터 분할 및 저장
for i in range(10):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 9 else total_rows  # 마지막은 나머지 포함
    split_df = df.iloc[start:end]

    if i < 8:
        split_type = 'train'
    elif i == 8:
        split_type = 'val'
    else:
        split_type = 'test'

    output_filename = f'{i+1}_{split_type}.csv'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    split_df.to_csv(output_path, index=False)

print(f"[Complete] {total_rows} rows split into 10 files under '{OUTPUT_DIR}'")