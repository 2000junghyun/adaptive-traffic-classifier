import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT_FILES = '../data/random-split/not-preprocessed/*.csv'
OUTPUT_DIR = '../data/random-split/preprocessed'

KEEP_COLS = [
    'Destination Port',
    'Flow Duration',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Min',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Min',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Fwd Header Length',
    'Bwd Packets/s',
    'Min Packet Length',
    'Packet Length Std',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'Down/Up Ratio',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Idle Mean',
    'Idle Std',
    'Idle Min',
    'Label_binary'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=INPUT_FILES, help='Input file pattern (supports wildcard)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    input_pattern = args.input
    output_dir = args.output

    files = glob.glob(input_pattern)
    print(f"[*] {len(files)} files found for pattern: {input_pattern}")
    for file in files:
        preprocess_file(file, output_dir)


def preprocess_file(input_path, output_dir):
    filename = os.path.basename(input_path)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[!] Fail to read {filename}: {e}")
        return

    try:
        df = preprocess_pipeline(df)
    except Exception as e:
        print(f"[!] Fail to preprocess {filename}: {e}")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        df.to_csv(save_path, index=False)
        print(f"[+] {filename} => {save_path}")
        # 저장 성공 시 원본 삭제
        try:
            os.remove(input_path)
            print(f"[-] Deleted original file: {input_path}")
        except Exception as del_err:
            print(f"[!] Fail to delete {input_path}: {del_err}")
    except Exception as e:
        print(f"[!] Fail to save {filename}: {e}")


def preprocess_pipeline(df):
    df = strip_column_whitespace(df)
    df = binary_labeling(df)
    df = select_keep_columns(df)
    df = fill_missing_values(df)
    df = clip_outliers_iqr(df)
    df = minmax_scale(df)
    return df


def strip_column_whitespace(df):
    df.columns = df.columns.str.strip()
    return df

def binary_labeling(df, label_col='Label', new_col='Label_binary'):
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataframe")
    df[new_col] = df[label_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
    return df

def select_keep_columns(df, keep_cols=KEEP_COLS):
    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
    return df[keep_cols].copy()

def fill_missing_values(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df

def clip_outliers_iqr(df, exclude_cols=['Label', 'Label_binary']):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    for col in numerical_cols:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df

def minmax_scale(df, exclude_cols=['Label', 'Label_binary']):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


if __name__ == "__main__":
    main()