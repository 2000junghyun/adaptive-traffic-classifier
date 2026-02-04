import os
from pathlib import Path

import pandas as pd

# 입력 및 출력 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "preprocessed-merged-traffic.csv"
OUTPUT_DIR = DATA_DIR / "hybrid-split"
TRAIN_DIR = OUTPUT_DIR / "train"
os.makedirs(TRAIN_DIR, exist_ok=True)

try:
    # CSV 파일 로드 (시간순 정렬되어 있다고 가정)
    df = pd.read_csv(INPUT_FILE)
    total_rows = len(df)

    if total_rows < 10:
        raise ValueError("Dataset too small to split into 10 parts.")

    # validation/test는 랜덤 샘플링 (10%씩)
    val_df = df.sample(frac=0.1, random_state=42)
    remaining_df = df.drop(val_df.index)

    test_df = remaining_df.sample(frac=0.1111, random_state=24)  # 0.1111 ≒ 10% of original
    train_df = remaining_df.drop(test_df.index).reset_index(drop=True)

    # 학습 데이터 (80%)를 시간 순서대로 8등분
    train_total = len(train_df)
    chunk_size = train_total // 8

    for i in range(8):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 7 else train_total
        chunk = train_df.iloc[start:end]
        output_path = TRAIN_DIR / f"{i+1}_train.csv"
        chunk.to_csv(output_path, index=False)

    # validation/test 저장
    val_df.to_csv(OUTPUT_DIR / "9_val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "10_test.csv", index=False)

    print(f"[✓] Hybrid split completed: {len(train_df)} train rows, {len(val_df)} val rows, {len(test_df)} test rows")
    print(f"[✓] Output directory: {OUTPUT_DIR}")

except FileNotFoundError:
    print(f"[ERROR] Input file not found: {INPUT_FILE}")
except pd.errors.EmptyDataError:
    print(f"[ERROR] Input file is empty or corrupt: {INPUT_FILE}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")