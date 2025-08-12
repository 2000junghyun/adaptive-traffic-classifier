# Adaptive Traffic Classifier

## Overview

This project is a **machine learning–based binary classifier** that determines whether network traffic is **malicious (1)** or **benign (0)**. It is designed not as a one-time trained model but as an **adaptive system** that can maintain performance even when new data is continuously added.

<br>

To reflect real-world security operations, the model was trained and evaluated using three different data-splitting strategies:

- **Random Split**: Randomly divides the dataset into training, validation, and test sets
- **Hybrid Split**: Primarily time-based ordering with some randomization
- **Time-based Split**: Sequentially splits data by time, simulating real-time data collection environments


## Tech Stack

- **Language**: Python 3.9
- **Data Processing**: pandas, NumPy, scikit-learn
- **Visualization**: matplotlib
- **Environment**: Jupyter Notebook for analysis and experiments


## Dataset

[IDS 2017 | Datasets | Research | Canadian Institute for Cybersecurity | UNB](https://www.unb.ca/cic/datasets/ids-2017.html)


## Directory Structure

```
.
├── data
│   ├── hybrid-split/                    # Hybrid split data
│   ├── random-split/                    # Random split data
│   ├── time-based-split/                # Time-based split data
│   └── preprocessed-merged-traffic.csv  # Preprocessed merged dataset
│
├── notebooks
│   ├── 00_intro.ipynb                       # Project introduction
│   ├── 01_eda.ipynb                         # Exploratory Data Analysis (EDA)
│   ├── 02_data_preprocessing.ipynb          # Data preprocessing
│   ├── 03_model_training/                   # Model training for each split method
│   │   ├── model_training_hybrid.ipynb
│   │   ├── model_training_random.ipynb
│   │   └── model_training_time_based.ipynb
│   └── 04_eval_visualization/               # Evaluation and visualization
│       ├── eval_visualization_hybrid.ipynb
│       ├── eval_visualization_random.ipynb
│       └── eval_visualization_time_based.ipynb
│
├── pred-results
│   ├── hybrid-split/                 # Prediction results (Hybrid)
│   ├── random-split/                 # Prediction results (Random)
│   └── time-based-split/             # Prediction results (Time-based)
│
├── utils
│   ├── hybrid_split.py               # Hybrid split script
│   ├── merge_csv.py                  # CSV merge script
│   ├── random_split.py               # Random split script
│   └── time_based_split.py           # Time-based split script
│
├── .gitignore
└── README.md
```

## How It Works

### 1. Data Loading & Preprocessing

- Load raw network traffic CSV files
- Perform **outlier handling**, **missing value removal**, and **binary labeling (`Label_binary`)**
- Drop constant-value columns and columns with more than 90% missing values
- Apply Min-Max scaling

### 2. Data Splitting

- Split datasets using **Random**, **Hybrid**, and **Time-based** methods
- Create training, validation, and test sets

### 3. Model Training

- Run training notebooks in `notebooks/03_model_training/` for each split method
- Test various models such as Logistic Regression and Random Forest
- Observe **performance changes as more data is added incrementally**

### 4. Evaluation & Visualization

- Calculate Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Visualize learning curves, ROC curves, and PR curves
- Save prediction results in `pred-results/` for comparison

## How to Run Locally

### 1. Data Preprocessing

```bash
# Example: Generate time-based split data
python utils/time_based_split.py
```

### 2. Model Training

- Launch Jupyter Notebook and run each split-specific training notebook in `notebooks/03_model_training/`

### 3. Result Visualization

- Run notebooks in `notebooks/04_eval_visualization/` to view graphs and metrics

## Features / Main Logic

- **Adaptive Learning Simulation**
    
    Designed to simulate real-world environments where data accumulates daily, enabling incremental learning experiments.
    
- **Multiple Data Split Strategies**
    
    Compare the performance impact of Random, Hybrid, and Time-based splits.
    
- **Automated Data Handling**
    
    `utils/` scripts automate dataset splitting and merging processes.
    
- **Quantitative Performance Analysis**
    
    Use visualizations to track model performance over time.
    

## Future Work

- Implement online (incremental) learning algorithms
- Address data imbalance using oversampling/undersampling
- Automate feature selection
- Develop a real-time inference API

## Motivation / Impact

- **Realistic Security Environment Simulation**: Mimics actual network log collection workflows
- **Efficient Threat Detection Research**: Compares how different splitting strategies impact model accuracy
- **Operational Readiness Validation**: Prepares the foundation for integrating with real-time detection systems
>>>>>>> c90c0ca (Update README file)
