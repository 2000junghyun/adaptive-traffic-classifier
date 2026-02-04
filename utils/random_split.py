import os
from pathlib import Path

import pandas as pd
from sklearn.utils import shuffle


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "preprocessed-merged-traffic.csv"
OUTPUT_DIR = DATA_DIR / "random-split"
TRAIN_DIR = OUTPUT_DIR / "train"
os.makedirs(TRAIN_DIR, exist_ok=True)

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
        output_path = TRAIN_DIR / f"{i+1}_train.csv"
    elif i == 8:
        output_path = OUTPUT_DIR / "9_val.csv"
    else:
        output_path = OUTPUT_DIR / "10_test.csv"

    split_df.to_csv(output_path, index=False)

print(f"[Complete] {total_rows} rows randomly split into 10 files under '{OUTPUT_DIR}'")