# Viva Demonstration Suite: Machine Learning Process Guide
### 2nd-Year Engineering Machine Learning Lab Viva Defense

> 🎓 **Purpose**: This folder (`Viva_Demo_Code/`) contains self-contained, educational Python scripts designed specifically to demonstrate the end-to-end Machine Learning lifecycle to your college teacher/examiner during a lab viva.
>
> ⚠️ **Architecture Notice**: These scripts are written in simple, linear Pandas and Scikit-Learn code suitable for second-year engineering explanation. The production application in the parent directory uses modular Scikit-Learn `Pipeline` architectures, serialized `.joblib` models, and an interactive multi-page Streamlit web dashboard.

---

## 1. Directory Structure & Demonstration Workflow

```text
Viva_Demo_Code/
│
├── 01_load_and_inspect.py    # Step 1: Dataset loading, shape, head, missing values, class distribution
├── 02_data_cleaning.py       # Step 2: Duplicate check, safe imputation (median/mode), why avoid dropna()
├── 03_preprocessing.py       # Step 3: Hour extraction, leakage prevention, One-Hot Encoding, StandardScaler
├── 04_train_test_split.py    # Step 4: 80/20 train-test split, 'stratify=y' proof, preventing data leakage
├── 05_model_training.py      # Step 5: .fit() and .predict() across the 4 syllabus ML algorithms
├── 06_model_comparison.py    # Step 6: Multi-metric benchmark table, confusion matrix, Accuracy Paradox
└── README.md                 # This comprehensive viva defense and question-answer guide
```

---

## 2. Quick Execution Guide

All scripts are configured to work whether executed from the project root or from inside the `Viva_Demo_Code` directory.

### Running from Project Root:
```bash
python Viva_Demo_Code/01_load_and_inspect.py
python Viva_Demo_Code/02_data_cleaning.py
python Viva_Demo_Code/03_preprocessing.py
python Viva_Demo_Code/04_train_test_split.py
python Viva_Demo_Code/05_model_training.py
python Viva_Demo_Code/06_model_comparison.py
```

### Running from Inside `Viva_Demo_Code/`:
```bash
cd Viva_Demo_Code
python 01_load_and_inspect.py
python 02_data_cleaning.py
python 03_preprocessing.py
python 04_train_test_split.py
python 05_model_training.py
python 06_model_comparison.py
```

---

## 3. Step-by-Step Script Walkthrough for Examiner

### Script 1: `01_load_and_inspect.py`
* **What it demonstrates**:
  * Loading `data/RTA Dataset.csv` via `pd.read_csv()`.
  * Verifying dataset dimensions: **12,316 rows** and **32 columns**.
  * Checking missing values: 16 columns have missing values (total 20,057 missing cells).
  * Checking class imbalance in target `Accident_severity`:
    * **Slight Injury**: 10,415 rows (84.57%)
    * **Serious Injury**: 1,743 rows (14.15%)
    * **Fatal injury**: 158 rows (1.28%)
* **Key Line to Show Teacher**:
  ```python
  df["Accident_severity"].value_counts(normalize=True) * 100
  ```
* **What to say**: *"Sir/Madam, the dataset has severe class imbalance: over 84% of crashes are minor, while fatal accidents account for only 1.28%. This directly dictates why accuracy alone cannot be our evaluation metric."*

---

### Script 2: `02_data_cleaning.py`
* **What it demonstrates**:
  * Safe data handling on a DataFrame copy (`df.copy()`), leaving the original CSV intact.
  * Checking duplicate records (`df.duplicated().sum() = 0`).
  * Explaining why `df.dropna()` is dangerous: dropping rows with any missing value discards **9,427 rows (76.5% of data)** and removes precious fatal crash examples.
  * Imputation on working copy:
    * **Numerical features**: filled with **Median** (robust against outliers).
    * **Categorical features**: filled with **Mode** (most frequent category).
* **Key Line to Show Teacher**:
  ```python
  df[col] = df[col].fillna(df[col].median())     # For numeric columns
  df[col] = df[col].fillna(df[col].mode()[0])    # For categorical columns
  ```
* **What to say**: *"We preserve 100% of the sample records using median and mode imputation rather than dropping rows, which would wipe out over 76% of our collision data."*

---

### Script 3: `03_preprocessing.py`
* **What it demonstrates**:
  * **Feature Engineering**: Extracting numerical `Hour` (0–23) from time strings (`"17:02:00"`).
  * **Target Leakage Prevention**: Dropping post-collision triage features (`Casualty_severity`, `Casualty_class`, `Sex_of_casualty`, etc.).
  * **One-Hot Encoding**: Converting text categories into binary dummy variables (`pd.get_dummies(..., drop_first=True)`), expanding features from 25 to 156 numeric inputs.
  * **Standard Scaling**: Applying `StandardScaler` ($z = \frac{x - \mu}{\sigma}$) to numerical columns.
* **Key Line to Show Teacher**:
  ```python
  X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
  scaler = StandardScaler()
  X_scaled[num_features] = scaler.fit_transform(X_encoded[num_features])
  ```
* **What to say**: *"Algorithms require numeric matrices. We use One-Hot Encoding to avoid implying an arbitrary rank order on nominal categories. We scale numerical features for distance-based models like KNN."*

---

### Script 4: `04_train_test_split.py`
* **What it demonstrates**:
  * Splitting dataset into **80% Training (9,852 rows)** and **20% Testing (2,464 rows)**.
  * The necessity of `stratify=y`: showing that both splits maintain the exact 84.56% / 14.15% / 1.28% class distribution.
  * Preventing Data Leakage: fitting `StandardScaler` strictly on `X_train`, then using `.transform()` on `X_test`.
* **Key Line to Show Teacher**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, y, test_size=0.20, random_state=42, stratify=y
  )
  ```
* **What to say**: *"Because fatal crashes make up only 1.28%, a standard random split risks placing too few fatal cases in the test set. 'stratify=y' guarantees identical representation in both splits."*

---

### Script 5: `05_model_training.py`
* **What it demonstrates**:
  * Training the 4 permitted syllabus classification algorithms:
    1. **K-Nearest Neighbors**: `n_neighbors=7` on scaled features.
    2. **Logistic Regression**: `max_iter=1000`, `class_weight='balanced'` on scaled features.
    3. **Decision Tree**: `max_depth=10`, `class_weight='balanced'` on unscaled features.
    4. **Random Forest**: `n_estimators=100`, `max_depth=12`, `class_weight='balanced'` on unscaled features.
  * Inspecting training execution times and viewing sample predictions against ground truth.
* **Key Line to Show Teacher**:
  ```python
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  ```
* **What to say**: *"We train each model using .fit() and predict on unseen data with .predict(). We use class_weight='balanced' to penalize mistakes on rare fatal accidents."*

---

### Script 6: `06_model_comparison.py`
* **What it demonstrates**:
  * Side-by-side benchmark comparison table on the identical test set:
    * **Accuracy**, **Balanced Accuracy**, **Macro Precision**, **Macro Recall**, **Macro F1-Score**.
  * Demonstrating the **Accuracy Paradox**:
    * KNN achieves **84.29% accuracy**, but only **34.10% balanced accuracy** and **0.3242 Macro F1** (majority-class bias).
    * Random Forest achieves **78.33% accuracy**, **47.35% balanced accuracy**, and **0.4729 Macro F1** (best multi-class balance).
  * Displaying the multi-class Confusion Matrix for the best model.
* **Key Line to Show Teacher**:
  ```python
  cm = confusion_matrix(y_test, y_pred, labels=['Fatal injury', 'Serious Injury', 'Slight Injury'])
  ```
* **What to say**: *"This table proves the Accuracy Paradox: KNN appears accurate at 84.3%, but fails to detect fatal crashes. Random Forest achieves the highest Macro F1-score (~0.47), making it our selected model."*

---

## 4. Benchmark Results Comparison Table

These are the exact results produced by running `06_model_comparison.py`:

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (k=7)** | 84.29% | 34.10% | 0.4052 | 0.3410 | 0.3242 | Majority class bias |
| **Logistic Regression** | 48.86% | 40.59% | 0.3509 | 0.4059 | 0.3052 | High fatal recall, low precision |
| **Decision Tree (depth=10)** | 67.57% | 52.94% | 0.3997 | 0.5294 | 0.4073 | Interpretable rule tree |
| **Random Forest (100 trees)** | **78.33%** | **47.35%** | **0.4725** | **47.35%** | **0.4729** | **Selected Best Model** |

> 📌 **Note on Production Scores**: The production project reports **0.4704 Macro F1** for Random Forest (trained inside Scikit-Learn `Pipeline` with `ColumnTransformer`). This educational script achieves **0.4729 Macro F1** using direct preprocessing. Both verify that Random Forest is the best-performing model.

---

## 5. Comprehensive Viva Q&A Guide (Examiner Questions & Answers)

### SECTION A: DATA UNDERSTANDING & EDA

#### Q1: What is the problem statement of your machine learning project?
**Answer**: "Given 25 pre-crash environmental, vehicular, road infrastructure, and driver demographic attributes recorded by traffic authorities, classify the collision outcome into one of three injury severity tiers: *Fatal injury*, *Serious Injury*, or *Slight Injury*."

#### Q2: What is the shape of your dataset?
**Answer**: "The raw RTA Dataset contains **12,316 rows** and **32 raw columns**."

#### Q3: What is the distribution of the target variable `Accident_severity`?
**Answer**: 
- Slight Injury: **10,415 records (84.57%)**
- Serious Injury: **1,743 records (14.15%)**
- Fatal injury: **158 records (1.28%)**

---

### SECTION B: DATA CLEANING & IMPUTATION

#### Q4: Why didn't you just use `df.dropna()` to remove missing values?
**Answer**: "If we used `dropna()`, we would lose **9,427 rows (76.5% of the data)**. In a severely imbalanced dataset where fatal crashes are already only 158 rows, dropping rows would eliminate crucial minority-class examples and degrade model training."

#### Q5: Why did you impute numerical features with Median instead of Mean?
**Answer**: "Numerical traffic features (like `Number_of_vehicles_involved` and `Number_of_casualties`) are right-skewed with extreme values/outliers. The median represents the 50th percentile and is robust against outlier distortion, whereas the mean is pulled by extreme values."

#### Q6: Why did you impute categorical features with Mode?
**Answer**: "Categorical features consist of discrete string labels. The mode represents the most frequent category in historical records, making it the statistically sound default assumption when data is missing."

---

### SECTION C: FEATURE ENGINEERING & DATA LEAKAGE

#### Q7: What is Target Leakage (Data Leakage), and how did you prevent it?
**Answer**: "Target leakage happens when an input feature includes information that would only be available *after* the target event has occurred. We strictly excluded post-collision casualty triage columns (`Casualty_severity`, `Casualty_class`, `Work_of_casuality`, etc.) because casualty triage is conducted at the hospital *after* the crash. Including them would artificially inflate accuracy during training but fail on real-time crash prediction."

#### Q8: How did you handle the `Time` feature?
**Answer**: "The raw `Time` column contains strings (e.g. `'17:02:00'`). ML algorithms cannot process string timestamps. We extracted the numerical `Hour` (0 to 23), which allows the models to learn diurnal risk dynamics such as rush hour congestion and midnight fatigue."

---

### SECTION D: ENCODING & FEATURE SCALING

#### Q9: Why did you use One-Hot Encoding instead of Label Encoding for categorical features?
**Answer**: "Label Encoding assigns arbitrary integers (e.g., Sunny=0, Rainy=1, Foggy=2). Linear models and distance metrics would interpret this as an ordinal relationship (Foggy > Rainy > Sunny), which is mathematically incorrect for nominal data. One-Hot Encoding creates separate binary 0/1 indicators for each category, avoiding false ordering."

#### Q10: Why did you scale numerical features for KNN and Logistic Regression, but not for Decision Trees?
**Answer**:
- **KNN**: Calculates geometric distances ($\sqrt{\sum (x_i - y_i)^2}$). An unscaled feature with large values (like `Hour`: 0–23) would completely dominate small features (like `Number_of_casualties`: 1–2).
- **Logistic Regression**: Uses gradient descent; feature scaling ensures spherical loss contours and faster convergence.
- **Decision Trees / Random Forest**: Make axis-parallel threshold splits ($x_i \le \theta$). Scaling is monotonic and does not alter the ranking or split points of features.

---

### SECTION E: TRAIN-TEST SPLIT & STRATIFICATION

#### Q11: What is the purpose of `train_test_split`?
**Answer**: "To evaluate the model's ability to generalize to unseen data. Evaluating on training data only measures memorization (overfitting). A test set simulates real-world future collisions."

#### Q12: What does `stratify=y` do, and why is it mandatory here?
**Answer**: "In our dataset, 'Fatal injury' is only 1.28%. A standard random split might assign almost all fatal cases to the training set and none to the test set by chance. `stratify=y` guarantees that the training set and testing set have the exact same class proportions (84.56% Slight, 14.15% Serious, 1.28% Fatal)."

#### Q13: Why must you fit the scaler ONLY on the training set?
**Answer**: "If we fit the scaler on the whole dataset or on the test set, information about the test distribution's mean and variance leaks into the model prior to testing. To prevent leakage, we call `scaler.fit_transform(X_train)` and only `scaler.transform(X_test)`."

---

### SECTION F: THE 4 MACHINE LEARNING ALGORITHMS

#### Q14: Explain K-Nearest Neighbors (KNN).
**Answer**: "KNN is an instance-based, non-parametric lazy learner. It stores all training instances. To classify a new query point, it calculates Euclidean distances to all training points, identifies the $k=7$ nearest neighbors, and assigns the class by majority vote."

#### Q15: Explain Logistic Regression for multi-class classification.
**Answer**: "Logistic Regression models the log-odds of each class as a linear combination of input features: $\log\left(\frac{P(Y=k)}{P(Y=K)}\right) = \beta_0 + \sum \beta_i X_i$. It uses the Softmax function to convert raw scores into probabilities that sum to 1.0."

#### Q16: Explain Decision Tree Classifier.
**Answer**: "A Decision Tree recursively partitions the feature space into rectangular regions using greedy binary splits. At each node, it selects the feature and threshold that maximizes purity (minimizes **Gini Impurity** = $1 - \sum p_i^2$). We constrain `max_depth=10` to prevent the tree from growing leaves with single noisy samples."

#### Q17: Explain Random Forest Classifier.
**Answer**: "Random Forest is a bagging ensemble of $B=100$ de-correlated decision trees. Each tree is trained on a bootstrap sample (random sampling with replacement). At each split, only a random subset of features is considered ($\sqrt{p}$). The forest combines all 100 trees via majority voting, which dramatically reduces model variance."

#### Q18: What is `class_weight='balanced'`?
**Answer**: "In standard loss calculation, all misclassifications carry equal weight. With `class_weight='balanced'`, Scikit-Learn weights each class inversely proportional to its frequency:
$$w_j = \frac{N}{K \cdot n_j}$$
For Fatal injury (1.28%), mistakes are penalized ~65 times higher than Slight injury, forcing the model to learn decision boundaries for rare fatal crashes."

---

### SECTION G: EVALUATION METRICS & ACCURACY PARADOX

#### Q19: What is the "Accuracy Paradox"?
**Answer**: "When a dataset has extreme class imbalance, a trivial classifier that always predicts the majority class ('Slight Injury') achieves **84.57% accuracy** while having **0% recall** on fatal crashes. Accuracy creates the illusion of high performance while failing on the most critical cases."

#### Q20: What is the difference between Macro F1-Score and Weighted F1-Score?
**Answer**:
- **Macro F1-Score**: Calculates the unweighted arithmetic mean of F1 scores across all classes:
  $$\text{Macro F1} = \frac{\text{F1}_{\text{Fatal}} + \text{F1}_{\text{Serious}} + \text{F1}_{\text{Slight}}}{3}$$
  It treats all classes equally, regardless of frequency.
- **Weighted F1-Score**: Weights each class's F1 score by its support (sample count). Because 84.57% of samples are Slight Injury, Weighted F1 reflects mostly the majority class.
- **Conclusion**: Macro F1 is the honest, decisive metric for imbalanced problems.

#### Q21: What is a Confusion Matrix?
**Answer**: "A $3 \times 3$ grid where rows represent actual ground-truth classes and columns represent predicted classes. Diagonal cells represent correct predictions (True Positives), while off-diagonal cells indicate false positive and false negative classification errors."

---

### SECTION H: PRODUCTION vs DEMONSTRATION CODE

#### Q22: What is the difference between these demo scripts and your main project code?
**Answer**:
1. **Pipelines**: The main project uses `sklearn.pipeline.Pipeline` and `ColumnTransformer` to bundle imputation, encoding, and scaling into a single object, ensuring zero data leakage.
2. **Model Persistence**: The main project saves trained pipelines to `.joblib` files inside the `models/` folder using `joblib.dump()`.
3. **Interactive UI**: The main project includes a multi-page **Streamlit** dashboard (`app.py`, `pages/`) with interactive risk calculators and Plotly visualizations.
4. **Demonstration Code**: These demo scripts are kept linear, readable, and free of abstractions so that every step can be inspected and explained line-by-line during viva defense.

---

## 6. Checklist Before Your Viva

- [x] Run `python Viva_Demo_Code/01_load_and_inspect.py` and verify dataset load and shape.
- [x] Run `python Viva_Demo_Code/02_data_cleaning.py` and verify missing values drop to 0.
- [x] Run `python Viva_Demo_Code/03_preprocessing.py` and explain One-Hot Encoding and Scaling.
- [x] Run `python Viva_Demo_Code/04_train_test_split.py` and explain `stratify=y`.
- [x] Run `python Viva_Demo_Code/05_model_training.py` and explain `.fit()` and `class_weight='balanced'`.
- [x] Run `python Viva_Demo_Code/06_model_comparison.py` and explain the Accuracy Paradox table.
