"""
Model training and pipeline persistence for the 4 permitted standard ML algorithms:
1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Tree
4. Random Forest
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_rta_data, SEVERITY_ORDER
from src.preprocessing import (
    build_model_pipeline,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.evaluation import compare_all_models, evaluate_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILENAMES = {
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
}

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_CACHE_PATH = MODELS_DIR / "metrics_cache.joblib"
TEST_DATA_PATH = MODELS_DIR / "test_data.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"


def get_model_instances() -> Dict[str, Any]:
    """
    Instantiate the 4 standard syllabus-friendly ML classification pipelines.
    - KNN & Logistic Regression: use scaled numerical features.
    - Decision Tree & Random Forest: use unscaled numerical features.
    - Class weighting: 'balanced' applied to LR, DT, RF. KNN does not support class weights.
    """
    return {
        "K-Nearest Neighbors": build_model_pipeline(
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            scale_numeric=True,
        ),
        "Logistic Regression": build_model_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            scale_numeric=True,
        ),
        "Decision Tree": build_model_pipeline(
            DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=42),
            scale_numeric=False,
        ),
        "Random Forest": build_model_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                max_depth=12,
                random_state=42,
                n_jobs=1,
            ),
            scale_numeric=False,
        ),
    }


def train_and_save_all(
    test_size: float = 0.20,
    random_state: int = 42,
    force_retrain: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame, str]:
    """
    Train all 4 permitted models on primary RTA dataset, evaluate on identical test split,
    and persist trained pipelines and benchmark artifacts.
    """
    if not force_retrain and check_models_exist():
        return load_saved_models_and_artifacts()

    # Load primary dataset
    df = load_rta_data()
    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratified Train/Test Split (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = get_model_instances()

    # Train each pipeline on training partition only
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        # Save individual model pipeline
        filename = MODEL_FILENAMES[name]
        joblib.dump(pipeline, MODELS_DIR / filename)

    # Evaluate on held-out test split
    comparison_df, metrics_cache, best_model_name = compare_all_models(models, X_test, y_test)

    # Save best model pipeline separately
    best_pipeline = models[best_model_name]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    # Save test dataset and metrics
    joblib.dump({"X_test": X_test, "y_test": y_test}, TEST_DATA_PATH)
    joblib.dump(
        {"comparison_df": comparison_df, "metrics_cache": metrics_cache, "best_model_name": best_model_name},
        METRICS_CACHE_PATH,
    )

    # Save metadata
    metadata = {
        "train_records": len(X_train),
        "test_records": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "feature_columns": ALL_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_order": SEVERITY_ORDER,
        "best_model_name": best_model_name,
        "algorithms": list(models.keys()),
    }
    joblib.dump(metadata, METADATA_PATH)

    return models, X_test, y_test, comparison_df, best_model_name


def check_models_exist() -> bool:
    """Verify that all 4 model pipelines and benchmark artifacts exist."""
    if not TEST_DATA_PATH.exists() or not METRICS_CACHE_PATH.exists():
        return False
    for filename in MODEL_FILENAMES.values():
        if not (MODELS_DIR / filename).exists():
            return False
    return True


def load_saved_models_and_artifacts() -> Tuple[
    Dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame, str
]:
    """Load pre-trained model pipelines, test data, and precomputed evaluation results."""
    test_data = joblib.load(TEST_DATA_PATH)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    cached = joblib.load(METRICS_CACHE_PATH)
    comparison_df = cached["comparison_df"]
    best_model_name = cached["best_model_name"]

    models = {}
    for name, filename in MODEL_FILENAMES.items():
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)

    return models, X_test, y_test, comparison_df, best_model_name


def get_cached_metrics() -> Dict[str, Any]:
    """Retrieve precomputed metrics cache."""
    if METRICS_CACHE_PATH.exists():
        cached = joblib.load(METRICS_CACHE_PATH)
        return cached.get("metrics_cache", {})
    return {}
