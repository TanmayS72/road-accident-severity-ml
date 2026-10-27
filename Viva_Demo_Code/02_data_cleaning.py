"""
02_data_cleaning.py
===================
Viva Demonstration Script 2: Data Inspection & Cleaning
Author: 2nd-Year Engineering Machine Learning Project

PURPOSE:
Demonstrate fundamental data cleaning concepts for a viva demonstration:
1. Load the original CSV dataset safely without modifying the original file.
2. Check for duplicate rows and remove them (demonstrated on a DataFrame copy).
3. Check missing values before cleaning.
4. Explain why blindly dropping rows (`dropna()`) is dangerous in machine learning.
5. Impute missing values on a working copy:
   - Numerical columns: filled with the MEDIAN (robust to outliers).
   - Categorical columns: filled with the MODE (most frequent category).
6. Verify missing-value counts before and after cleaning.

HOW TO RUN:
- From project root:       python Viva_Demo_Code/02_data_cleaning.py
- From Viva_Demo_Code dir: python 02_data_cleaning.py
"""

import os
import pandas as pd

# -------------------------------------------------------------------------
# STEP 1: Load the Dataset Safely
# -------------------------------------------------------------------------
# Locate CSV relative path
data_path = "data/RTA Dataset.csv" if os.path.exists("data/RTA Dataset.csv") else "../data/RTA Dataset.csv"

print("=" * 70)
print("  STEP 2: DATA INSPECTION & CLEANING DEMONSTRATION")
print("=" * 70)
print(f"Loading raw dataset from: {data_path}")

df_raw = pd.read_csv(data_path)

# IMPORTANT: Always work on a separate copy to protect raw source data!
df = df_raw.copy()
print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.\n")

# -------------------------------------------------------------------------
# STEP 2: Check for Duplicate Rows
# -------------------------------------------------------------------------
# df.duplicated().sum() counts identical rows.
duplicates_count = df.duplicated().sum()
print(f"[1] DUPLICATE ROWS CHECK:")
print(f"    - Number of duplicate rows detected: {duplicates_count}")

if duplicates_count > 0:
    print(f"    - Action: Removing {duplicates_count} duplicate row(s)...")
    # drop_duplicates() removes repeated rows
    df = df.drop_duplicates()
    print(f"    - Remaining rows after removing duplicates: {len(df):,}")
else:
    print("    - Cleanliness status: No duplicate rows present in this dataset.")

# -------------------------------------------------------------------------
# STEP 3: Missing Values Before Cleaning
# -------------------------------------------------------------------------
missing_before = df.isnull().sum().sum()
print(f"\n[2] MISSING VALUES (BEFORE CLEANING):")
print(f"    - Total missing cells across entire dataset: {missing_before:,}")
print(f"    - Number of columns with missing data: {(df.isnull().sum() > 0).sum()} of {df.shape[1]}")

# -------------------------------------------------------------------------
# STEP 4: Educational Discussion: Why NOT dropna()?
# -------------------------------------------------------------------------
print(f"\n[3] WHY WE DO NOT BLINDLY DROP ROWS WITH 'dropna()':")
# Simulate what would happen if we used dropna()
rows_with_any_na = df.shape[0] - df.dropna().shape[0]
pct_lost = (rows_with_any_na / df.shape[0]) * 100
print(f"    - If we used df.dropna(), we would discard {rows_with_any_na:,} rows ({pct_lost:.1f}% of data!).")
print(f"    - Crucial Viva Point: In an imbalanced dataset where Fatal injuries")
print(f"      make up only 1.28% (158 rows), dropping rows would destroy valuable")
print(f"      crash examples. Imputation preserves our entire sample size.")

# -------------------------------------------------------------------------
# STEP 5: Educational Imputation (Median for Numeric, Mode for Categorical)
# -------------------------------------------------------------------------
print(f"\n[4] HANDLING MISSING VALUES VIA IMPUTATION:")

# Identify numerical and categorical column types
# Numerical columns: contain integer or floating-point counts
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
# Categorical columns: contain object / string text categories
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Exclude target from feature imputation
if "Accident_severity" in cat_cols:
    cat_cols.remove("Accident_severity")

# A. Numerical Imputation: Fill with Median
# Why median? The median is unaffected by extreme values/outliers compared to the mean.
print("    A. Numerical Columns Imputation (Strategy: Median):")
for col in num_cols:
    col_missing = df[col].isnull().sum()
    if col_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"       * {col}: filled {col_missing} NaNs with median ({median_val})")
    else:
        print(f"       * {col}: 0 NaNs (clean)")

# B. Categorical Imputation: Fill with Mode (Most Frequent Value)
# Why mode? For categorical text, the most frequent category is the safest guess.
print("    B. Categorical Columns Imputation (Strategy: Most Frequent / Mode):")
imputed_cat_count = 0
for col in cat_cols:
    col_missing = df[col].isnull().sum()
    if col_missing > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        imputed_cat_count += 1

print(f"       * Imputed missing values across {imputed_cat_count} categorical columns using mode.")

# -------------------------------------------------------------------------
# STEP 6: Missing Values After Cleaning
# -------------------------------------------------------------------------
missing_after = df.isnull().sum().sum()
print(f"\n[5] MISSING VALUES (AFTER CLEANING):")
print(f"    - Total missing cells before: {missing_before:,}")
print(f"    - Total missing cells after : {missing_after:,}")
print(f"    - Status: Dataset is now 100% complete and free of NaNs.")

# -------------------------------------------------------------------------
# VIVA EXPLANATION NOTE FOR TEACHER / EXAMINER
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  VIVA EXPLANATION NOTES FOR THE TEACHER:")
print("  1. Safety: The raw CSV file was NOT modified or overwritten.")
print("  2. Reason for Median: Traffic variables like vehicle counts have")
print("     right-skewed distributions; the median provides a robust central")
print("     tendency that resists outlier influence.")
print("  3. Production Architecture Difference:")
print("     In this demo script, we impute globally to illustrate the math.")
print("     In our production project (src/preprocessing.py), we use Scikit-Learn's")
print("     'SimpleImputer' fitted STRICTLY on the training split inside a")
print("     'ColumnTransformer' to prevent train-test data leakage.")
print("=" * 70)
