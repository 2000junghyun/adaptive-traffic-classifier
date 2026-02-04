## Overview

- **네트워크 트래픽 데이터**를 기반으로 악성(1) / 정상(0) 여부를 분류하는 **머신러닝 기반 이진 분류기**
- 단순히 한 번 학습한 모델이 아니라, **데이터가 점진적으로 추가되는 환경에서도 적응적으로 성능을 유지**할 수 있도록 설계

<br>

실제 보안 운영 환경을 반영하기 위해 다음 3가지 데이터 분할 방식으로 모델을 학습·평가:
- **Random Split**: 무작위로 데이터셋을 학습/검증/테스트로 분리
- **Hybrid Split**: 시간 순서를 고려하되 일부 무작위성을 가미
- **Time-based Split**: 시간 순으로 데이터를 순차 분리하여 학습 (실제 실시간 데이터 수집 환경 유사)

## Tech Stack

- **언어**: Python 3.9
- **데이터 처리**: pandas, NumPy, scikit-learn
- **시각화**: matplotlib
- **환경**: Jupyter Notebook 기반 분석 및 실험

## Dataset

[IDS 2017 | Datasets | Research | Canadian Institute for Cybersecurity | UNB](https://www.unb.ca/cic/datasets/ids-2017.html)

## Directory Structure

```bash
.
├── data
│   ├── merged-traffic.csv               # (Optional) Raw 병합본
│   ├── preprocessed-merged-traffic.csv  # 전처리 완료 데이터
│   ├── hybrid-split/                    # Hybrid 방식 분할 데이터
│   │   ├── train/                       # 1~8 train chunk
│   │   ├── 9_val.csv
│   │   ├── 10_test.csv
│   │   └── pred-results/                # 학습 반복별 test 예측 결과
│   ├── random-split/                    # Random 방식 분할 데이터
│   └── time-based-split/                # Time-based 방식 분할 데이터
│
├── notebooks
│   ├── 00_intro.ipynb                       # 프로젝트 개요
│   ├── 01_eda.ipynb                         # 탐색적 데이터 분석(EDA)
│   ├── 02_data_preprocessing.ipynb          # 데이터 전처리
│   ├── 03_model_training/                   # 분할 방식별 모델 학습
│   │   ├── model_training_hybrid.ipynb
│   │   ├── model_training_random.ipynb
│   │   └── model_training_time_based.ipynb
│   └── 04_eval_visualization/               # 학습 결과 시각화
│       ├── eval_visualization_hybrid.ipynb
│       ├── eval_visualization_random.ipynb
│       └── eval_visualization_time_based.ipynb
│
├── utils
│   ├── hybrid_split.py               # Hybrid 데이터 분할 스크립트
│   ├── merge_csv.py                  # CSV 병합 스크립트
│   ├── random_split.py               # Random 데이터 분할 스크립트
│   └── time_based_split.py           # Time-based 데이터 분할 스크립트
│
├── .gitignore
└── README.md
```

## How It Works

### 0. (Optional) Raw CSV 병합

- 여러 일자/구간으로 나뉜 원본 CSV를 하나의 파일로 통합
- `utils/merge_csv.py`는 입력 디렉토리의 CSV들에 대해 **스키마(컬럼명) 일치 여부를 검사**한 뒤 병합

### 1. 데이터 로드 및 전처리 (notebooks/02_data_preprocessing.ipynb)

- 원본 `merged-traffic.csv`를 로드한 뒤, 보안 분류 문제에 맞게 **데이터 품질/누수(leakage)/중복 특징**을 정리
- 주요 처리 단계
    - **극소수 클래스 제거**: 전체 데이터의 0.1% 미만인 라벨은 제거(학습 안정성/노이즈 감소)
    - **이진 라벨링**: `Label_binary = 0(BENIGN) / 1(그 외)`
    - **불필요한 컬럼 제거**
        - 상수 컬럼(고유값 1개)
        - 결측 비율 90% 초과 컬럼
        - 누수 컬럼: `Label` 제거
        - **상관계수 0.95 이상** 고상관 피처 제거(파생명/결측치/고유값 기준 휴리스틱으로 드롭 대상 선택)
    - **결측/무한대 처리**: `inf → NaN`, 수치형은 평균/범주형은 최빈값으로 대치
    - **이상치 처리**: 수치형 피처에 대해 IQR 기반 클리핑
    - **스케일링**: Min-Max scaling(0~1)
- 산출물
    - `data/preprocessed-merged-traffic.csv`

### 2. 데이터 분할 (utils/*.py)

전처리된 단일 CSV를 10개 파일로 분할해 **점진적 학습 시나리오**를 만들고, 검증/테스트는 고정합니다.

- **Random Split** (`utils/random_split.py`)
    - 전체 데이터를 셔플(`random_state=42`) 후 10등분
    - 1~8: train, 9: val, 10: test
- **Time-based Split** (`utils/time_based_split.py`)
    - 데이터 순서를 유지한 채 10등분(시간축을 가정)
    - 1~8: train, 9: val, 10: test
- **Hybrid Split** (`utils/hybrid_split.py`)
    - val(10%) / test(≈10%)는 랜덤 샘플링으로 고정
    - 나머지 train(≈80%)는 **시간 순서대로 8등분**

각 split의 산출물은 `data/<split>/` 하위에 다음과 같이 저장:

- `data/<split>/train/1_train.csv` ~ `8_train.csv`
- `data/<split>/9_val.csv`, `data/<split>/10_test.csv`

### 3. 적응형(점진적) 학습 실험: Full retrain 반복 (notebooks/03_model_training/*)

- 핵심 아이디어: “데이터가 순차적으로 유입될 때, 누적 학습 데이터가 늘어남에 따라 성능이 어떻게 변화하는가”를 관찰
- 분할 방식별 노트북이 동일한 프로토콜로 학습
    - `notebooks/03_model_training/model_training_random.ipynb`
    - `notebooks/03_model_training/model_training_time_based.ipynb`
    - `notebooks/03_model_training/model_training_hybrid.ipynb`

학습 루프(공통):
- `n = 1..8`에 대해, `train/1..n` 파일을 concat하여 **Full retrain**
- 모델: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`
- `val`과 `test`는 고정해두고 매 반복마다 평가
    - 지표: Accuracy, ROC-AUC
- 반복별 test 예측 결과를 저장(실험 재현/시각화용)
    - 경로: `data/<split>/pred-results/test_result_{n}_files.csv`
    - 컬럼: `Label_binary`(정답), `y_pred`(예측), `y_prob`(확률)

### 4. 평가 및 시각화 (notebooks/04_eval_visualization/*)

- 저장된 `pred-results`를 불러와, 학습 데이터가 적을 때(예: 1개 파일)와 충분히 누적됐을 때(예: 8개 파일)의 성능을 비교
- 산출물(예시: random split 노트북)
    - `classification_report`(precision/recall/F1)
    - Confusion Matrix heatmap
    - ROC Curve / PR Curve 및 AUC/AP 비교

## How to Run Locally

### 0. (Optional) Raw CSV 병합

```bash
# NOTE: utils/merge_csv.py는 기본 경로가 외부(shared-data)로 설정돼 있습니다.
# 필요하면 스크립트 상수(INPUT_DIR/OUTPUT_DIR)를 프로젝트 환경에 맞게 수정하세요.
python utils/merge_csv.py
```

### 1. 데이터 전처리

- `notebooks/02_data_preprocessing.ipynb` 실행
- 산출물: `data/preprocessed-merged-traffic.csv`

### 2. 데이터 분할

```bash
# 예: time-based split 데이터 생성
python utils/time_based_split.py

# 또는
python utils/random_split.py
python utils/hybrid_split.py
```

### 3. 모델 학습

- Jupyter Notebook 실행 후 `notebooks/03_model_training/` 내 각 분할 방식별 파일 실행

### 4. 결과 시각화

- `notebooks/04_eval_visualization/` 실행하여 그래프 및 지표 확인

## Features / Main Logic

- **Leakage-aware Preprocessing (실험 신뢰성 강화)**
    - `Label` 제거, 상수/결측 과다/고상관 피처 제거를 통해 “테스트 성능이 과대평가되는 상황”을 줄이고, 실제 운영 환경에 가까운 검증을 지향

- **Operational Split 전략 3종 비교**
    - Random / Time-based / Hybrid split을 동일한 프로토콜로 돌려, 시간 순서(데이터 드리프트) 반영 여부에 따른 성능 차이를 비교 가능

- **점진적 데이터 누적 시나리오를 Full retrain으로 재현**
    - `train` 파일을 1개→8개로 늘리며 매번 전체를 다시 학습(full retrain)하여, “데이터가 쌓일수록 성능이 어떻게 개선/변동하는지”를 정량화

- **Artifact-first Experiment Tracking**
    - 각 반복(n)마다 `test_result_{n}_files.csv`를 저장해, 모델을 다시 학습하지 않아도 동일 조건에서 평가/시각화를 반복 가능

- **Reproducibility Controls**
    - random split/모델 학습에서 `random_state`를 고정해 재현 가능한 실험을 제공

- **Extensible Baseline**
    - 동일한 데이터/아티팩트 구조를 유지한 채 다른 모델(예: Logistic Regression, GBM 계열)로 교체해 비교 실험 가능
    - 현재는 RandomForest 기반

## Future

- 데이터 불균형 대응(오버샘플링/언더샘플링)
- Feature selection 자동화
- 실시간 추론 API 구현

## Motivation / Impact

- **보안 운영 환경 재현**: 실제 네트워크 로그 수집 상황을 반영한 데이터 흐름 설계
- **효율적인 위협 탐지 실험**: 다양한 데이터 분할 방식에 따른 모델 성능 변화 분석
- **운영 적용 가능성 검증**: 향후 실시간 탐지 시스템과 연동 가능성 확인