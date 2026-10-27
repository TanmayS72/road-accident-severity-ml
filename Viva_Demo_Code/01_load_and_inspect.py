"""
01_load_and_inspect.py
=======================
Viva Demonstration Script 1: Loading & Inspecting the Dataset
Author: 2nd-Year Engineering Machine Learning Project

PURPOSE:
Demonstrate basic dataset inspection using Pandas:
1. Import Pandas library.
2. Load the original Road Traffic Accident (RTA) CSV dataset.
3. Display the first 5 rows (head).
4. Print the dataset dimensions (shape: rows and columns).
5. Display missing-value counts per column.
6. Display the distribution of the target variable ('Accident_severity').

HOW TO RUN:
- From project root:       python Viva_Demo_Code/01_load_and_inspect.py
- From Viva_Demo_Code dir: python 01_load_and_inspect.py
"""

import os
import pandas as pd

# -------------------------------------------------------------------------
# STEP 1: Locate and Load the CSV Dataset
# -------------------------------------------------------------------------
# We dynamically locate 'data/RTA Dataset.csv' whether executed from the
# project root folder or directly inside the 'Viva_Demo_Code' folder.
data_path = "data/RTA Dataset.csv" if os.path.exists("data/RTA Dataset.csv") else "../data/RTA Dataset.csv"

print("=" * 70)
print("  STEP 1: LOAD AND INSPECT THE RAW ACCIDENT DATASET")
print("=" * 70)
print(f"Loading dataset from: {data_path}\n")

# Load CSV into a Pandas DataFrame
df = pd.read_csv(data_path)

# -------------------------------------------------------------------------
# STEP 2: Dataset Dimensions (Shape)
# -------------------------------------------------------------------------
# df.shape returns a tuple: (number_of_rows, number_of_columns)
num_rows, num_cols = df.shape
print(f"[1] DATASET DIMENSIONS:")
print(f"    - Total Records (Rows)   : {num_rows:,}")
print(f"    - Total Features (Columns): {num_cols}")

# -------------------------------------------------------------------------
# STEP 3: Preview First 5 Rows (Head)
# -------------------------------------------------------------------------
# df.head(5) shows the first 5 records. We display the first 6 columns
# so the terminal output remains readable and clean for the examiner.
print(f"\n[2] PREVIEW OF FIRST 5 ROWS (First 6 columns shown for display):")
print(df.iloc[:5, :6].to_string())

# -------------------------------------------------------------------------
# STEP 4: Missing Values Count
# -------------------------------------------------------------------------
# df.isnull().sum() counts how many NaN (empty) values exist in each column.
missing_series = df.isnull().sum()
cols_with_missing = missing_series[missing_series > 0].sort_values(ascending=False)

print(f"\n[3] MISSING VALUES COUNT:")
print(f"    - Columns with missing values ({len(cols_with_missing)} out of {num_cols} columns):")
for col_name, count in cols_with_missing.items():
    pct = (count / num_rows) * 100
    print(f"      * {col_name:<28}: {count:>5} missing ({pct:>5.1f}%)")

print(f"    - Total missing cells across entire dataset: {missing_series.sum():,}")

# -------------------------------------------------------------------------
# STEP 5: Target Variable Distribution ('Accident_severity')
# -------------------------------------------------------------------------
# value_counts() calculates how many accidents fall into each severity level.
print(f"\n[4] TARGET VARIABLE DISTRIBUTION ('Accident_severity'):")
severity_counts = df["Accident_severity"].value_counts()
severity_percent = df["Accident_severity"].value_counts(normalize=True) * 100

for label in severity_counts.index:
    count = severity_counts[label]
    pct = severity_percent[label]
    print(f"    - {label:<16}: {count:>6,} rows ({pct:>5.2f}%)")

# -------------------------------------------------------------------------
# VIVA EXPLANATION NOTE FOR TEACHER / EXAMINER
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  VIVA EXPLANATION NOTES FOR THE TEACHER:")
print("  1. The dataset contains 12,316 accident records and 32 raw features.")
print("  2. Severe Class Imbalance: 'Slight Injury' represents 84.57% of data,")
print("     while 'Fatal injury' is only 1.28%.")
print("  3. Key takeaway: A naive model predicting 'Slight Injury' for every")
print("     case would reach 84.6% accuracy, but fail 100% of fatal cases!")
print("     That is why we must evaluate using Balanced Accuracy & Macro F1.")
print("=" * 70)
