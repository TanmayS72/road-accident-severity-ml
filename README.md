# Road Traffic Accident Severity Prediction & Risk Analysis Using Machine Learning
### 2nd-Year Engineering Machine Learning Lab Project

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)](https://plotly.com/)

[![Report](https://img.shields.io/badge/Academic%20Report-10%20Pages%20%7C%2015%20Marks-10b981.svg)](reports/PROJECT_REPORT.md)

An interactive machine learning analytics application that analyzes historical road traffic collision logs, predicts injury severity using standard syllabus-aligned classifiers, benchmarks model architectures under severe class imbalance, and provides data-driven road safety risk intelligence.

> 📄 **Official Course Submission**: View the complete 10-page academic project report with literature review, performance figures, and viva defense notes: [**`reports/PROJECT_REPORT.md`**](reports/PROJECT_REPORT.md).

---

## 1. Project Overview & Problem Statement

Road traffic collisions represent a critical public health and infrastructure challenge globally. Predicting accident severity at the time of collision occurrence is vital for:
- Informing emergency trauma medical response dispatching.
- Identifying high-risk infrastructural blackspots (unlit intersections, undivided medians).
- Evaluating behavioral driving infractions (tailgating, overspeeding).

### Problem Statement
Given 25 pre-crash environmental, vehicular, road geometry, and driver demographic attributes recorded by traffic authorities, classify the resulting collision into one of three injury severity tiers:
1. **Fatal injury**
2. **Serious Injury**
3. **Slight Injury**

This problem presents severe **class imbalance** (fatal injuries constitute only 1.28% of records), requiring careful evaluation beyond raw classification accuracy.

---

## 2. Project Objectives

1. **Curriculum Alignment**: Implement and compare exactly four standard, syllabus-friendly machine learning classification algorithms taught in second-year engineering:
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Decision Tree
   - Random Forest
2. **Strict Leakage Prevention**: Build self-contained Scikit-Learn `Pipeline` architectures that fit transformers exclusively on training partitions and eliminate post-incident casualty outcome leakage.
3. **Imbalance-Aware Benchmarking**: Demonstrate the *Accuracy Paradox* and evaluate models using **Macro F1-Score** and **Balanced Accuracy**.
4. **Interactive Analytics Interface**: Deliver a multi-page Streamlit web dashboard adhering to a modern dark navy, slate, cyan, and teal design system.

---

## 3. Dataset Description

The project utilizes two historical accident datasets located in the `data/` directory:

### Primary Dataset: `data/RTA Dataset.csv`
- **Total Records**: 12,316 historical collision entries
- **Total Features**: 32 raw attributes
- **Missing Values**: 20,057 total missing cells (e.g., `Defect_of_vehicle`: 4,427, `Service_year_of_vehicle`: 3,928)
- **Duplicate Rows**: 0 duplicate records
- **Target Variable**: `Accident_severity`

### Verification Dataset: `data/cleaned.csv`
- **Total Records**: 12,316 rows × 15 columns
- **Purpose**: Verified benchmark dataset used for numeric target label confirmation:
  - `0` &rarr; **Fatal injury** (158 records, 1.28%)
  - `1` &rarr; **Serious Injury** (1,743 records, 14.15%)
  - `2` &rarr; **Slight Injury** (10,415 records, 84.57%)

---

## 4. Target Classes & Class Imbalance

The primary target column `Accident_severity` exhibits severe class skew:

| Severity Class | Definition & Trauma Level | Total Records | Percentage |
| :--- | :--- | :---: | :---: |
| **Slight Injury** | Minor cuts, bruising, ambulatory outpatient care | 10,415 | **84.57%** |
| **Serious Injury** | Severe trauma requiring hospitalization, fractures | 1,743 | **14.15%** |
| **Fatal injury** | Mortality sustained within 30 days of collision | 158 | **1.28%** |

> [!IMPORTANT]
> Because 84.57% of cases are minor, a trivial majority classifier achieves ~84.6% accuracy while failing 100% of fatal cases. Therefore, **Macro F1-Score** and **Balanced Accuracy** are the governing optimization metrics.

---

## 5. Machine Learning Algorithms

Four standard, syllabus-approved Scikit-Learn classifiers are implemented:

1. **K-Nearest Neighbors (KNN)**:
   - *Architecture*: Instance-based lazy learner ($k=7$) using Euclidean distance over standardized numerical and one-hot encoded vectors with distance weighting (`weights='distance'`).
   - *Viva Note*: Does not natively support cost-sensitive class weights, leading it to favor the majority class.
2. **Logistic Regression**:
   - *Architecture*: Multinomial linear log-odds classifier with L2 regularization (`max_iter=1000`) and cost-sensitive balanced weighting (`class_weight='balanced'`).
   - *Viva Note*: Penalizes misclassifications on rare fatal injuries inversely proportional to their class frequency.
3. **Decision Tree**:
   - *Architecture*: Non-parametric recursive splitting model using Gini impurity with tree regularization (`max_depth=10`, `class_weight='balanced'`).
   - *Viva Note*: Produces interpretable decision rules with high sensitivity to minority branches.
4. **Random Forest (Selected Best Model)**:
   - *Architecture*: Bagging ensemble of 100 de-correlated decision trees (`n_estimators=100`, `max_depth=12`, `class_weight='balanced'`).
   - *Viva Note*: Achieves the highest **Macro F1-Score (0.4704)** by combining predictions across bootstrap subsets, reducing variance and over-fitting.

> [!NOTE]
> Advanced boosting libraries (XGBoost, LightGBM, CatBoost) and neural networks are strictly excluded to maintain 2nd-year engineering ML lab syllabus compliance.

---

## 6. Preprocessing & Leakage Prevention

All preprocessing is encapsulated inside Scikit-Learn `Pipeline` and `ColumnTransformer` objects:

1. **Numerical Features** (`Number_of_vehicles_involved`, `Number_of_casualties`, `Hour`):
   - Imputed using `SimpleImputer(strategy='median')`.
   - Scaled using `StandardScaler` for distance/linear models (KNN, Logistic Regression).
2. **Categorical Features** (22 features: `Age_band_of_driver`, `Light_conditions`, `Cause_of_accident`, etc.):
   - Imputed using `SimpleImputer(strategy='most_frequent')`.
   - Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to ensure unseen categories in inference never trigger crashes.
3. **Strict Target & Casualty Leakage Prevention**:
   - `Accident_severity`: Target variable; strictly isolated from inputs.
   - `Casualty_severity`: Direct post-collision triage outcome; excluded to prevent 100% target leakage.
   - `Casualty_class`, `Sex_of_casualty`, `Age_band_of_casualty`, `Work_of_casuality`, `Fitness_of_casuality`: Post-accident victim logs (36% missing); excluded.
   - `Time`: Replaced by numerical `Hour` (0–23) to capture diurnal cyclic traffic dynamics.

---

## 7. Train/Test Methodology & Evaluation Metrics

- **Partitioning**: Stratified 80% train ($N=9,852$) and 20% test ($N=2,464$) split (`random_state=42`), preserving class ratios in both splits.
- **Leak-Free Discipline**: Preprocessors are fitted **strictly on `X_train`**. `X_test` is transformed only during inference evaluation.
- **Evaluation Metrics**:
  - **Overall Accuracy**: Total correct predictions over total samples.
  - **Balanced Accuracy**: Unweighted mean of recall across each of the three classes.
  - **Macro F1-Score**: Unweighted mean of F1-scores across all three classes (penalizes models that fail on fatal accidents).
  - **Disaggregated Confusion Matrix & Class Support**: Explicit verification across minority classes (Test Support: Fatal = 31, Serious = 349, Slight = 2,084).

---

## 8. Official Benchmark Results

Evaluated on the held-out stratified test partition ($N=2,464$ cases):

| Classifier | Overall Accuracy | Balanced Accuracy | Macro F1-Score | Weighted F1 | Macro Precision | Macro Recall | Fatal Recall | Serious Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ | **78.25%** | **46.26%** | **0.4704** | **0.7848** | **0.4832** | **0.4626** | **19.35%** | **33.24%** |
| **Decision Tree** | 62.87% | **51.40%** | 0.3829 | 0.6893 | 0.3932 | **0.5140** | **48.39%** | **38.68%** |
| **K-Nearest Neighbors** | **84.70%** | 34.89% | 0.3385 | 0.7891 | 0.4647 | 0.3489 | 0.00% | 5.16% |
| **Logistic Regression** | 48.86% | 40.66% | 0.3056 | 0.5746 | 0.3514 | 0.4066 | 32.26% | 38.68% |

> **Viva Takeaway**: **Random Forest** is selected as the best overall model based on **Macro F1 (0.4704)**. While KNN has the highest raw accuracy (84.70%), it completely fails on fatal collisions (0.00% recall), illustrating the classic **Accuracy Paradox**.

---

## 9. Project Directory Structure

```text
ML-Project/
├── .streamlit/
│   └── config.toml                  # Dark navy, slate, cyan & teal design tokens
├── data/
│   ├── RTA Dataset.csv              # Primary collision registry (12,316 rows × 32 columns)
│   └── cleaned.csv                  # Imputed benchmark dataset (12,316 rows × 15 columns)
├── models/
│   ├── k_nearest_neighbors.joblib   # Persisted KNN pipeline
│   ├── logistic_regression.joblib   # Persisted Logistic Regression pipeline
│   ├── decision_tree.joblib         # Persisted Decision Tree pipeline
│   ├── random_forest.joblib         # Persisted Random Forest pipeline
│   ├── best_model.joblib            # Serialized top model (Random Forest)
│   ├── metrics_cache.joblib         # Precomputed multi-class test benchmark metrics
│   ├── test_data.joblib             # Stratified test partition (2,464 records)
│   └── metadata.joblib              # Feature columns, class orders & split metadata
├── pages/
│   ├── 1_Dashboard.py               # Operational KPIs, diurnal curves & executive metrics
│   ├── 2_Exploratory_Data_Analysis.py # Univariate, bivariate, diurnal & missingness profiling
│   ├── 3_Severity_Prediction.py     # Interactive real-time ML inference & probability engine
│   ├── 4_Model_Comparison.py        # 4-model benchmark, confusion matrices & syllabus guides
│   └── 5_Risk_Analysis.py           # Temporal heatmaps, environmental & vehicular hazard profiles
├── src/
│   ├── __init__.py                  # Package marker
│   ├── data_loader.py               # Dataset loading, temporal extraction & caching
│   ├── preprocessing.py             # ColumnTransformer & Scikit-learn Pipeline builders
│   ├── train_models.py              # Stratified training & model persistence
│   ├── evaluation.py                # Multi-class scoring, confusion matrices & classification reports
│   └── visualizations.py            # Responsive Plotly charts & styled metric cards
├── reports/
│   ├── figures/                     # 6 publication-grade high-res evaluation plots
│   └── PROJECT_REPORT.md            # Comprehensive 10-page academic lab report (15 marks)
├── scripts/
│   └── generate_report_figures.py   # Script to reproduce publication figures
├── app.py                           # Executive portal & pipeline overview
├── requirements.txt                 # Minimum verified dependencies
├── .gitignore                       # Git exclusions for Python caches & logs
└── README.md                        # Academic lab documentation
```

---

## 10. Multi-Page Application Features

1. **Executive Portal (`app.py`)**:
   - High-level system overview, workflow architecture, dataset preview, and navigation cards.
2. **Dashboard (`pages/1_Dashboard.py`)**:
   - 4 dynamic KPI cards (Total Collisions, Serious/Fatal Proportion, Peak Collision Hour, Daylight Share).
   - Diurnal accident frequency distribution and top recorded accident causes.
3. **Exploratory Data Analysis (`pages/2_Exploratory_Data_Analysis.py`)**:
   - 4 dynamic registry KPIs (12,316 records, 32 features, 20,057 missing cells, 0 duplicates).
   - Expandable data schema, missingness audit table, and first 15 records.
   - Interactive severity distribution switcher (Donut vs. Bar chart).
   - Driver demographics and road condition distribution and 100% normalized stacked bar charts.
   - Serious-or-fatal percentage ranking with small sample warnings.
   - Pearson correlation heatmap strictly on continuous/count numerical variables.
4. **Severity Prediction (`pages/3_Severity_Prediction.py`)**:
   - Interactive inference engine across 25 pre-crash features organized into 3 sections (Driver, Road/Environment, Vehicle/Incident).
   - Dynamic model selector supporting all 4 trained pipelines without retraining.
   - `Load Sample Accident` button that pulls genuine historical records with target blinded.
   - Color-coded result card, Plotly probability breakdown, and held-out test performance scorecards.
5. **Model Comparison (`pages/4_Model_Comparison.py`)**:
   - Comparative benchmark table and grouped bar charts across all 4 classifiers.
   - Multi-class confusion matrices and disaggregated per-class precision/recall/F1 metrics.
   - Feature importance attribution chart for tree/linear models.
   - Syllabus-aligned educational guides for each algorithm.
6. **Risk Analysis (`pages/5_Risk_Analysis.py`)**:
   - Day-of-week vs. Hour-of-day collision concentration heatmap.
   - Severity breakdown across environmental conditions and vehicle classifications.
   - Driver experience vs. age band vulnerability cross-tabulation.
   - Non-causal observational disclaimer.

---

## 11. Installation Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Setup
```powershell
# 1. Clone repository
git clone https://github.com/TanmayS72/road-accident-severity-ml.git
cd road-accident-severity-ml

# 2. (Optional) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 12. How to Run the Application

Launch the Streamlit web server from the project root directory:

```powershell
streamlit run app.py
```
*Or via python module:*
```powershell
python -m streamlit run app.py
```

The application will open automatically at:
**`http://localhost:8501`**

---

## 13. How to Retrain Models (Optional)

The trained model pipelines and benchmark cache are already serialized in `models/`. To retrain the 4 standard classifiers from scratch on updated splits:

```powershell
python -c "from src.train_models import train_and_save_all; train_and_save_all(force_retrain=True)"
```

---

## 14. Academic Limitations & Disclaimer

> [!WARNING]
> **Educational Prototype Disclaimer**:
> This software is an academic engineering machine learning prototype developed strictly for demonstration and educational purposes.
> - **Observational Data**: Historical patterns documented in police accident registries represent empirical correlations and do not demonstrate deterministic physical causation.
> - **Imbalanced Minority Class**: Fatal accidents represent 1.28% of recorded events; model sensitivity to rare fatal events is subject to variance and trade-offs.
> - **Safety Notice**: Predictions must **NOT** be used for emergency dispatch routing, official legal accident classification, insurance liability determinations, or real-world safety decisions.
