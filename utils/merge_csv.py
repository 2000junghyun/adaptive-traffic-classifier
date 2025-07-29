import os
import csv

INPUT_DIR = '../../shared-data/traffic-samples'
OUTPUT_DIR = '../../shared-data'
OUTPUT_FILE_NAME = 'merged-traffic.csv'

def main():
    try:
        file_paths = get_csv_file_paths(INPUT_DIR)
        print(f"Found {len(file_paths)} CSV files: {file_paths}")

        if not file_paths:
            print(f"[ERROR] No CSV files in '{INPUT_DIR}' directory.")
            return

        matched, fieldnames = check_fieldnames_match(file_paths)
        if not matched:
            print("[ERROR] Field names do not match across files.")
            return

        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
        merge_csv_files(file_paths, output_path, fieldnames)

    except Exception as e:
        print(f"[ERROR] Unexpected error in main(): {e}")


# Directory 내 CSV 파일 목록 확인
def get_csv_file_paths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]


# 필드명 일치 확인
def check_fieldnames_match(file_paths):
    fieldnames = None
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            else:
                if reader.fieldnames != fieldnames:
                    print(f"[ERROR] Field name mismatch in: {path}")
                    return False, None
    return True, fieldnames


# CSV 파일 통합
def merge_csv_files(file_paths, output_file, fieldnames):
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            for path in file_paths:
                with open(path, 'r', encoding='utf-8') as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        writer.writerow(row)
        print(f"[✓] CSV merge complete: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to merge CSV files: {e}")


if __name__ == "__main__":
    main()