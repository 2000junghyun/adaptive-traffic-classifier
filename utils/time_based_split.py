import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "preprocessed-merged-traffic.csv"
OUTPUT_DIR = DATA_DIR / "time-based-split"
TRAIN_DIR = OUTPUT_DIR / "train"
os.makedirs(TRAIN_DIR, exist_ok=True)

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
        output_path = TRAIN_DIR / f"{i+1}_train.csv"
    elif i == 8:
        output_path = OUTPUT_DIR / "9_val.csv"
    else:
        output_path = OUTPUT_DIR / "10_test.csv"

    split_df.to_csv(output_path, index=False)

print(f"[Complete] {total_rows} rows split into 10 files under '{OUTPUT_DIR}'")