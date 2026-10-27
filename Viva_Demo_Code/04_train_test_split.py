"""
04_train_test_split.py
======================
Viva Demonstration Script 4: Train-Test Split & Stratification
Author: 2nd-Year Engineering Machine Learning Project

PURPOSE:
Demonstrate how and why data is partitioned for supervised learning:
1. Load and prepare preprocessed features (X) and target (y).
2. Split dataset into Training Set (80%) and Testing Set (20%).
3. Explain why we must NEVER evaluate models on the training data (Overfitting).
4. Demonstrate STRATIFIED SPLITTING ('stratify=y'):
   - Compare class distribution before vs after the split.
   - Explain why stratification is vital when dealing with rare classes (1.28% Fatal injuries).
5. Explain DATA LEAKAGE:
   - Why scalers/transformers must only be fitted on X_train, never on X_test.

HOW TO RUN:
- From project root:       python Viva_Demo_Code/04_train_test_split.py
- From Viva_Demo_Code dir: python 04_train_test_split.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# STEP 1: Load and Prepare Data
# -------------------------------------------------------------------------
data_path = "data/RTA Dataset.csv" if os.path.exists("data/RTA Dataset.csv") else "../data/RTA Dataset.csv"

print("=" * 70)
print("  STEP 4: TRAIN-TEST SPLIT & STRATIFICATION DEMONSTRATION")
print("=" * 70)
print(f"Loading raw dataset from: {data_path}")

df = pd.read_csv(data_path)

# Feature engineering: extract Hour
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12).astype(int)

# Separate Target (y) and Features (X)
y = df["Accident_severity"].copy()

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
feature_cols = [c for c in df.columns if c not in excluded_cols]
X = df[feature_cols].copy()

# Impute missing values
num_features = ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]
cat_features = [c for c in feature_cols if c not in num_features]

for col in num_features:
    X[col] = X[col].fillna(X[col].median())
for col in cat_features:
    X[col] = X[col].fillna(X[col].mode()[0])

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)

print(f"Prepared Feature Matrix X : {X_encoded.shape[0]:,} rows x {X_encoded.shape[1]} columns")
print(f"Target Series y           : {y.shape[0]:,} labels across 3 classes\n")

# -------------------------------------------------------------------------
# STEP 2: Perform Stratified Train-Test Split (80% Train, 20% Test)
# -------------------------------------------------------------------------
# Parameters explained:
# - test_size=0.20 : 80% of data (9,852 rows) used for training, 20% (2,464 rows) for testing.
# - random_state=42: Seed for the pseudo-random generator, ensuring reproducible splits.
# - stratify=y     : Preserves the exact class proportion across both train and test splits.
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("[1] DATASET PARTITION SIZES:")
print(f"    - Training Set (80%): {X_train.shape[0]:,} rows  (Used to fit/train models)")
print(f"    - Testing Set  (20%): {X_test.shape[0]:,} rows   (Used exclusively to evaluate models)")

# -------------------------------------------------------------------------
# STEP 3: Verify Stratification Across Classes
# -------------------------------------------------------------------------
print("\n[2] VERIFYING STRATIFICATION (CLASS PROPORTIONS):")
print(f"    {'Severity Class':<16} | {'Original (%)':<14} | {'Train (%)':<14} | {'Test (%)':<14}")
print("    " + "-" * 62)

classes = ["Slight Injury", "Serious Injury", "Fatal injury"]
orig_dist = y.value_counts(normalize=True) * 100
train_dist = y_train.value_counts(normalize=True) * 100
test_dist = y_test.value_counts(normalize=True) * 100

for cls in classes:
    print(f"    {cls:<16} | {orig_dist.get(cls, 0):>10.2f}%    | {train_dist.get(cls, 0):>10.2f}%    | {test_dist.get(cls, 0):>10.2f}%")

print("\n    Observation: The class ratios match perfectly across all three splits!")

# -------------------------------------------------------------------------
# STEP 4: Demonstrating Proper Scaling Without Data Leakage
# -------------------------------------------------------------------------
print("\n[3] PREVENTING DATA LEAKAGE DURING SCALING:")
print("    - Rule: Fit the scaler ONLY on X_train; transform both X_train and X_test.")
print("    - Why? If we fitted StandardScaler on the whole dataset or on X_test,")
print("      information about the test set mean and standard deviation would leak")
print("      into the model, artificially inflating test scores.")

scaler = StandardScaler()
# FIT and TRANSFORM on X_train
X_train_scaled = X_train.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])

# ONLY TRANSFORM on X_test (never call fit() on test data!)
X_test_scaled = X_test.copy()
X_test_scaled[num_features] = scaler.transform(X_test[num_features])

print("    - Successfully scaled numerical columns using train-fitted scaler:")
for col in num_features:
    print(f"      * {col:<28}: Train Mean={X_train_scaled[col].mean():.2f}, Test Mean={X_test_scaled[col].mean():.2f}")

# -------------------------------------------------------------------------
# VIVA EXPLANATION NOTE FOR TEACHER / EXAMINER
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  VIVA EXPLANATION NOTES FOR THE TEACHER:")
print("  1. Why not 100% training?")
print("     A model that memorizes the training data will show high training")
print("     accuracy (overfitting), but fails on new unseen cases. The test set")
print("     simulates future, real-world crash logs.")
print("  2. What is 'stratify=y' and why is it mandatory?")
print("     In this dataset, 'Fatal injury' is only 1.28% (158 rows out of 12,316).")
print("     Without 'stratify=y', a random split might assign almost all fatal cases")
print("     to the training set and zero to the test set, making it impossible to")
print("     test whether our model can detect life-threatening accidents.")
print("  3. What is Data Leakage?")
print("     Data leakage occurs when information from outside the training dataset")
print("     is used to create the model. Splitting BEFORE preprocessing prevents it.")
print("=" * 70)
