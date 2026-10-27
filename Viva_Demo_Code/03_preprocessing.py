"""
03_preprocessing.py
===================
Viva Demonstration Script 3: Preprocessing, Feature Engineering & Encoding
Author: 2nd-Year Engineering Machine Learning Project

PURPOSE:
Demonstrate how raw data is transformed into a clean mathematical matrix for ML:
1. Separate features (X) and target variable (y).
2. Feature Selection & Preventing Target Leakage:
   - Exclude post-crash casualty columns that disclose severity outcomes.
3. Feature Engineering:
   - Extract numeric 'Hour' (0-23) from the raw timestamp string 'Time'.
4. Impute missing values (median for numerical, mode for categorical).
5. Categorical Encoding:
   - Apply One-Hot Encoding (pd.get_dummies) to convert text categories into 0/1 columns.
6. Feature Scaling (Standardization):
   - Apply Scikit-Learn's StandardScaler (mean=0, std=1) to numerical features.
   - Explain why KNN & Logistic Regression require scaling, while Decision Trees do not.

HOW TO RUN:
- From project root:       python Viva_Demo_Code/03_preprocessing.py
- From Viva_Demo_Code dir: python 03_preprocessing.py
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# STEP 1: Load Raw Dataset
# -------------------------------------------------------------------------
data_path = "data/RTA Dataset.csv" if os.path.exists("data/RTA Dataset.csv") else "../data/RTA Dataset.csv"

print("=" * 70)
print("  STEP 3: PREPROCESSING, ENCODING & FEATURE SCALING")
print("=" * 70)
print(f"Loading raw dataset from: {data_path}")

df = pd.read_csv(data_path)
print(f"Raw dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# -------------------------------------------------------------------------
# STEP 2: Feature Engineering - Extract Hour from Time String
# -------------------------------------------------------------------------
# The 'Time' column contains strings like "17:02:00" or "1:06:00".
# Machine learning algorithms cannot perform calculations on raw time strings.
# We convert Time into a clean numerical feature 'Hour' (0 to 23).
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12).astype(int)
print("\n[1] FEATURE ENGINEERING:")
print(f"    - Converted raw 'Time' strings to numerical 'Hour' feature (range: {df['Hour'].min()} to {df['Hour'].max()}).")
print("    - Why? Captures diurnal traffic risk cycles (rush hour vs midnight accidents).")

# -------------------------------------------------------------------------
# STEP 3: Feature Selection & Eliminating Target Leakage
# -------------------------------------------------------------------------
# TARGET VARIABLE:
target_col = "Accident_severity"
y = df[target_col].copy()

# EXCLUDED COLUMNS:
# We MUST exclude post-collision casualty triage columns:
# - 'Casualty_severity': directly discloses victim injury severity (severe leakage!)
# - 'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty', 'Work_of_casuality', 'Fitness_of_casuality'
# - 'Time': replaced by engineered 'Hour'
# - 'Accident_severity': ground truth target itself
excluded_cols = [
    "Accident_severity",
    "Time",
    "Casualty_severity",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
]

feature_cols = [col for col in df.columns if col not in excluded_cols]
X = df[feature_cols].copy()

print("\n[2] TARGET LEAKAGE PREVENTION & FEATURE SELECTION:")
print(f"    - Ground Truth Target (y) : '{target_col}' (3 classes)")
print(f"    - Selected Features (X)   : {len(feature_cols)} features retained")
print(f"    - Excluded Leakage Columns: {len(excluded_cols)} columns dropped")
print("      (e.g., 'Casualty_severity' is recorded AFTER hospital triage; using it")
print("       would cheat the model since injury outcome discloses accident severity!)")

# -------------------------------------------------------------------------
# STEP 4: Impute Missing Values in Features
# -------------------------------------------------------------------------
num_features = ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]
cat_features = [col for col in feature_cols if col not in num_features]

# Impute numerical features with median
for col in num_features:
    X[col] = X[col].fillna(X[col].median())

# Impute categorical features with mode (most frequent)
for col in cat_features:
    X[col] = X[col].fillna(X[col].mode()[0])

print("\n[3] IMPUTATION:")
print(f"    - Numerical features ({len(num_features)}) imputed using MEDIAN.")
print(f"    - Categorical features ({len(cat_features)}) imputed using MODE.")
print(f"    - Remaining missing values in X: {X.isnull().sum().sum()}")

# -------------------------------------------------------------------------
# STEP 5: Categorical Encoding (One-Hot Encoding)
# -------------------------------------------------------------------------
# Machine learning algorithms are mathematical equations. They cannot multiply
# text like "Male" or "Rainy". We convert text categories into binary (0 or 1)
# dummy columns using One-Hot Encoding (pd.get_dummies).
# drop_first=True prevents the 'Dummy Variable Trap' (multicollinearity).
X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)

print("\n[4] ONE-HOT ENCODING (CATEGORICAL TO NUMERICAL):")
print(f"    - Features before encoding: {X.shape[1]} columns")
print(f"    - Features after encoding : {X_encoded.shape[1]} columns (binary dummy columns)")
print("    - Why One-Hot Encoding instead of Label Encoding?")
print("      Label Encoding assigns numbers (0, 1, 2) which implies an artificial")
print("      order (e.g. Rainy > Sunny), whereas nominal categories have no order.")

# -------------------------------------------------------------------------
# STEP 6: Feature Scaling (Standardization)
# -------------------------------------------------------------------------
# StandardScaler transforms values so Mean = 0 and Standard Deviation = 1.
# z = (x - mean) / std_dev
scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[num_features] = scaler.fit_transform(X_encoded[num_features])

print("\n[5] FEATURE SCALING (STANDARDIZATION):")
print("    - Standardized numerical features using StandardScaler (z = (x - mean) / std):")
for col in num_features:
    mean_val = X_scaled[col].mean()
    std_val = X_scaled[col].std()
    print(f"      * {col:<30}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")

# -------------------------------------------------------------------------
# VIVA EXPLANATION NOTE FOR TEACHER / EXAMINER
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  VIVA EXPLANATION NOTES FOR THE TEACHER:")
print("  1. Why scaling is necessary for some models:")
print("     - K-Nearest Neighbors (KNN): uses Euclidean distance. An unscaled feature")
print("       with large numbers (e.g., Hour: 0-23) would dominate small numbers")
print("       (e.g., Number_of_casualties: 1-2). Scaling gives all features equal weight.")
print("     - Logistic Regression: gradient descent converges faster with scaled features.")
print("  2. Why Tree models do NOT need scaling:")
print("     - Decision Tree & Random Forest make threshold splits (e.g., Hour > 18).")
print("       Monotonic scaling does not change the order of values or the split point.")
print("  3. Production Architecture Difference:")
print("     In this demo script, we scale directly to show the math.")
print("     In production (src/preprocessing.py), we use a Scikit-Learn 'Pipeline'")
print("     with 'ColumnTransformer' to fit scalers ONLY on training partitions.")
print("=" * 70)
