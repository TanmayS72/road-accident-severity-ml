# Road Traffic Accident Severity Prediction & Risk Analysis
### 2nd-Year Engineering Machine Learning Lab Project

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)](https://plotly.com/)

A machine learning web application that analyzes real-world road traffic collision records, forecasts accident injury severity, benchmarks standard syllabus-aligned classifiers, and visualizes accident risk patterns.

---

## 📌 Academic Scope & Objectives

Designed for an undergraduate engineering curriculum, this project focuses on **standard, syllabus-friendly machine learning algorithms** that are straightforward to explain and defend in a viva examination:

1. **K-Nearest Neighbors (KNN)** (Distance-based instance learning with scaled numerical features)
2. **Logistic Regression** (Regularized multinomial linear classification with class balancing)
3. **Decision Tree** (Non-linear recursive partitioning with depth regularization)
4. **Random Forest** (Ensemble bagging of de-correlated decision trees with class balancing)

> [!NOTE]
> Advanced boosting libraries (XGBoost, HistGradientBoosting, AdaBoost) are excluded to ensure strict alignment with the 2nd-year engineering ML curriculum.

---

## 🗂️ Project Architecture

```text
ML-Project/
├── .streamlit/
│   └── config.toml                  # Dark navy, slate & cyan styling tokens
├── data/
│   ├── RTA Dataset.csv              # Primary collision registry (12,316 rows × 32 columns)
│   └── cleaned.csv                  # Imputed verification dataset (12,316 rows × 15 columns)
├── models/
│   ├── k_nearest_neighbors.joblib   # Serialized KNN pipeline (with StandardScaler)
│   ├── logistic_regression.joblib   # Serialized Logistic Regression pipeline (with StandardScaler)
│   ├── decision_tree.joblib         # Serialized Decision Tree pipeline
│   ├── random_forest.joblib         # Serialized Random Forest pipeline
│   ├── best_model.joblib            # Serialized top model pipeline (Random Forest)
│   ├── metrics_cache.joblib         # Precomputed multi-class benchmark scorecard
│   ├── test_data.joblib             # Stratified 20% test partition (2,464 records)
│   └── metadata.joblib              # Pipeline metadata, class labels & feature lists
├── src/
│   ├── __init__.py                  # Package marker
│   ├── data_loader.py               # Dataset loading, temporal extraction & mapping validation
│   ├── preprocessing.py             # ColumnTransformer & Scikit-learn Pipeline builders
│   ├── train_models.py              # Model training, stratified splits & artifact serialization
│   ├── evaluation.py                # Multi-class scoring, confusion matrices & per-class metrics
│   └── visualizations.py            # Responsive Plotly charts & styled metric cards
├── pages/
│   ├── 1_Dashboard.py               # Operational KPIs, diurnal distribution & top causes
│   ├── 2_Exploratory_Data_Analysis.py # Univariate, bivariate & correlation discovery
│   ├── 3_Severity_Prediction.py     # Real-time incident severity forecasting engine
│   ├── 4_Model_Comparison.py       # 4-algorithm benchmark, confusion matrices & feature importance
│   └── 5_Risk_Analysis.py           # Multi-factor hazard matrices & safety policy directives
├── app.py                           # Executive welcome portal & ML architecture flow
├── requirements.txt                 # Cleaned dependencies (no XGBoost)
└── README.md                        # Project documentation
```

---

## 📊 Dataset Audit & Target Verification

### 1. Primary Dataset: `data/RTA Dataset.csv`
- **Dimensions**: 12,316 rows × 32 columns
- **Duplicate Records**: 0 duplicate rows
- **Missing Values**: 16 columns contain nulls (e.g., `Defect_of_vehicle`: 4,427, `Service_year_of_vehicle`: 3,928)
- **Target Column**: `Accident_severity` (String labels)
  - `Slight Injury`: 10,415 records (84.57%)
  - `Serious Injury`: 1,743 records (14.15%)
  - `Fatal injury`: 158 records (1.28%)

### 2. Verification Dataset: `data/cleaned.csv`
- **Dimensions**: 12,316 rows × 15 columns (14 categorical features + integer target)
- **Duplicate Records**: 390 duplicate rows (due to projection into 15 dimensions)
- **Missing Values**: 0 nulls
- **Verified Numeric Target Mapping**:
  - `0` &rarr; **Fatal injury** (158 records)
  - `1` &rarr; **Serious Injury** (1,743 records)
  - `2` &rarr; **Slight Injury** (10,415 records)

---

## 🛡️ Preprocessing & Preventing Data Leakage

To guarantee academic rigor and prevent data snooping:
1. **Stratified Split**: 80% train (9,852 records) and 20% test (2,464 records), maintaining class ratios.
2. **Leak-Free Transformation**: The `ColumnTransformer` is fitted **strictly on the training partition (`X_train`)**. The test partition (`X_test`) is transformed only during evaluation.
3. **Self-Contained Pipelines**: Preprocessing is attached directly inside each model's `Pipeline([('preprocessor', ...), ('classifier', ...)])`.
4. **Feature Exclusions**:
   - `Accident_severity`: Target variable.
   - `Casualty_severity`: Direct post-incident casualty injury level; leaks target severity.
   - `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Work_of_casuality`, `Fitness_of_casuality`: Post-accident victim documentation (contains 36% 'na' values).
   - `Time`: Replaced by numerical `Hour` (0-23).

---

## ⚖️ Class Imbalance & Model Performance

### The "Accuracy Paradox"
Because **84.57%** of recorded collisions are minor (*Slight Injury*), raw accuracy is fundamentally misleading:
- **K-Nearest Neighbors (KNN)** achieved **84.70% raw accuracy**, but because standard KNN does not support `class_weight='balanced'`, it assigned 99% of samples to the majority class, achieving **0.00% recall on Fatal accidents**.
- **Random Forest** and **Decision Tree** use cost-sensitive balanced weighting (`class_weight='balanced'`), heavily penalizing errors on rare fatal/serious cases.

### Official Benchmark Scorecard (Held-Out Test Set: 2,464 Cases)

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Fatal Recall | Serious Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ | **78.25%** | **46.26%** | **0.4704** | **0.7848** | **0.4832** | **0.4626** | **19.35%** | **33.24%** |
| **Decision Tree** | 62.87% | 51.40% | 0.3829 | 0.6893 | 0.3932 | 0.5140 | 48.39% | 38.68% |
| **K-Nearest Neighbors** | 84.70% | 34.89% | 0.3385 | 0.7891 | 0.4647 | 0.3489 | 0.00% | 5.16% |
| **Logistic Regression** | 48.86% | 40.66% | 0.3056 | 0.5746 | 0.3514 | 0.4066 | 32.26% | 38.68% |

> **Model Selection Decision**: **Random Forest** is selected as the best overall classifier based on **Macro F1-Score (0.4704)**, successfully balancing minority class detection with high overall precision.

---

## 🚀 How to Run Locally

### 1. Clone / Navigate to Project Directory
```powershell
cd "c:\Users\Tanma\OneDrive\Desktop\ML-Project"
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Re-train and Verify Model Pipelines
```powershell
python -c "from src.train_models import train_and_save_all; train_and_save_all(force_retrain=True)"
```

### 4. Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```

The application will launch in your browser at: **`http://localhost:8501`**
