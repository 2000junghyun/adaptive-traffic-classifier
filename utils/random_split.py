import pandas as pd
import os
from sklearn.utils import shuffle

# 입력 및 출력 경로
INPUT_FILE = '../../shared-data/merged-data.csv'
OUTPUT_DIR = '../data/random-split/not-preprocessed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV 파일 로드 및 셔플
df = pd.read_csv(INPUT_FILE)
df = shuffle(df, random_state=42).reset_index(drop=True)  # 재현성을 위한 seed 고정

# 총 행 수 및 분할 크기
total_rows = len(df)
chunk_size = total_rows // 10

# 분할 및 저장
for i in range(10):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 9 else total_rows
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

print(f"[Complete] {total_rows} rows randomly split into 10 files under '{OUTPUT_DIR}'")
