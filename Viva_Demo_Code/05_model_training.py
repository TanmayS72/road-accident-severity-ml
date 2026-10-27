"""
05_model_training.py
====================
Viva Demonstration Script 5: Model Training on 4 Syllabus Algorithms
Author: 2nd-Year Engineering Machine Learning Project

PURPOSE:
Demonstrate how to instantiate, train, and test the exact 4 machine learning
classification algorithms permitted in second-year engineering:
1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Tree Classifier
4. Random Forest Classifier

DEMONSTRATES:
- Calling .fit(X_train, y_train) to train each model.
- Calling .predict(X_test) to generate predictions.
- Demonstrating why KNN and Logistic Regression use SCALED features,
  while Decision Tree and Random Forest use UNROUNDED/UNSCALED features.
- Understanding key hyperparameters: n_neighbors, class_weight='balanced',
  max_depth, and n_estimators.
- Previewing sample predictions vs actual ground truth.

HOW TO RUN:
- From project root:       python Viva_Demo_Code/05_model_training.py
- From Viva_Demo_Code dir: python 05_model_training.py
"""

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# STEP 1: Load and Prepare Training & Testing Partitions
# -------------------------------------------------------------------------
data_path = "data/RTA Dataset.csv" if os.path.exists("data/RTA Dataset.csv") else "../data/RTA Dataset.csv"

print("=" * 70)
print("  STEP 5: TRAINING THE 4 STANDARD MACHINE LEARNING ALGORITHMS")
print("=" * 70)
print(f"Loading raw dataset from: {data_path}...")

df = pd.read_csv(data_path)

# Feature engineering: extract Hour
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12).astype(int)

# Separate Target and Features
y = df["Accident_severity"].copy()
excluded_cols = [
    "Accident_severity", "Time", "Casualty_severity", "Casualty_class",
    "Sex_of_casualty", "Age_band_of_casualty", "Work_of_casuality", "Fitness_of_casuality"
]
feature_cols = [c for c in df.columns if c not in excluded_cols]
X = df[feature_cols].copy()

# Simple imputation
num_features = ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]
cat_features = [c for c in feature_cols if c not in num_features]
for col in num_features:
    X[col] = X[col].fillna(X[col].median())
for col in cat_features:
    X[col] = X[col].fillna(X[col].mode()[0])

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)

# Stratified Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize numerical features for distance & gradient models
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_test_scaled[num_features] = scaler.transform(X_test[num_features])

print(f"Dataset split complete: {X_train.shape[0]:,} training samples, {X_test.shape[0]:,} test samples.")
print(f"Feature count: {X_train.shape[1]} inputs\n")

# -------------------------------------------------------------------------
# STEP 2: Instantiate the 4 Syllabus Algorithms
# -------------------------------------------------------------------------
# We define each algorithm with its standard, syllabus-friendly hyperparameters:
models = {
    # 1. K-Nearest Neighbors: Instance-based learner; k=7 closest points vote
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(n_neighbors=7),
        "X_tr": X_train_scaled,
        "X_te": X_test_scaled,
        "note": "Uses scaled features for Euclidean distance. k=7 neighbors.",
    },
    # 2. Logistic Regression: Linear log-odds classifier with balanced class weighting
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "X_tr": X_train_scaled,
        "X_te": X_test_scaled,
        "note": "Uses scaled features for gradient solver. class_weight='balanced'.",
    },
    # 3. Decision Tree: Recursive binary splitting based on Gini Impurity
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
        "X_tr": X_train,
        "X_te": X_test,
        "note": "Scale-invariant. max_depth=10 controls tree depth to avoid overfitting.",
    },
    # 4. Random Forest: Bagging ensemble of 100 de-correlated decision trees
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=1),
        "X_tr": X_train,
        "X_te": X_test,
        "note": "Ensemble of 100 trees voting together. Reduces variance and overfitting.",
    },
}

# -------------------------------------------------------------------------
# STEP 3: Train Each Model and Make Predictions
# -------------------------------------------------------------------------
print("[1] TRAINING PROGRESS:")
trained_models = {}
predictions = {}

for name, config in models.items():
    clf = config["model"]
    xtr = config["X_tr"]
    xte = config["X_te"]

    print(f"\n    --> Training: {name}...")
    print(f"        Setup: {config['note']}")

    # Measure fitting time
    start_time = time.time()
    # .fit() is the core Scikit-Learn function where the model learns
    clf.fit(xtr, y_train)
    elapsed = time.time() - start_time

    # .predict() applies the learned model to held-out test data
    y_pred = clf.predict(xte)

    trained_models[name] = clf
    predictions[name] = y_pred

    print(f"        Status: Training finished in {elapsed:.2f} seconds.")

# -------------------------------------------------------------------------
# STEP 4: Sample Predictions vs Ground Truth
# -------------------------------------------------------------------------
print("\n" + "-" * 70)
print("[2] SAMPLE PREDICTIONS PREVIEW (First 5 Test Samples):")
print(f"    {'Index':<6} | {'Actual Severity':<16} | {'Decision Tree':<16} | {'Random Forest':<16}")
print("    " + "-" * 62)

actual_samples = y_test.iloc[:5].values
dt_samples = predictions["Decision Tree"][:5]
rf_samples = predictions["Random Forest"][:5]

for i in range(5):
    print(f"    {i+1:<6} | {actual_samples[i]:<16} | {dt_samples[i]:<16} | {rf_samples[i]:<16}")

# -------------------------------------------------------------------------
# VIVA EXPLANATION NOTE FOR TEACHER / EXAMINER
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  VIVA EXPLANATION NOTES FOR THE TEACHER:")
print("  1. What is .fit()?")
print("     It is the method where the model estimates its internal parameters")
print("     from the training data (e.g., weights for Logistic Regression, split")
print("     thresholds for Decision Trees).")
print("  2. What is class_weight='balanced'?")
print("     Because 'Fatal injury' is only 1.28% of the data, standard algorithms")
print("     would ignore it to maximize overall accuracy. 'balanced' applies an")
print("     inversely proportional penalty, making mistakes on fatal cases much")
print("     more costly, forcing the model to detect rare severe accidents.")
print("  3. Production Architecture Difference:")
print("     In this demo script, models are trained on the fly.")
print("     In production (src/train_models.py), trained models are serialized")
print("     to disk using 'joblib' (.joblib files) so the Streamlit application")
print("     can serve real-time predictions with sub-second latency.")
print("=" * 70)
