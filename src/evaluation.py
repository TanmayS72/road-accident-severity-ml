"""
Evaluation metrics, confusion matrices, and model benchmark utilities.
Supports consistent multi-class scoring across all permitted algorithms.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data_loader import SEVERITY_ORDER


def evaluate_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model pipeline on a held-out test set.
    Computes overall scalar metrics, confusion matrix, classification report,
    and disaggregated per-class precision/recall/F1/support.
    """
    labels = class_order or SEVERITY_ORDER

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    # Overall Scalar Metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))

    # Confusion Matrix: rows=Actual, cols=Predicted
    cm_raw = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    # Classification Report
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    # Disaggregated Per-Class Metrics DataFrame
    per_class_records = []
    for cls_name in labels:
        if cls_name in report_dict:
            c_data = report_dict[cls_name]
            per_class_records.append({
                "Severity Class": cls_name,
                "Precision": float(c_data["precision"]),
                "Recall (Sensitivity)": float(c_data["recall"]),
                "F1-Score": float(c_data["f1-score"]),
                "Support": int(c_data["support"]),
            })
    per_class_df = pd.DataFrame(per_class_records)

    # Feature Importances Extraction (if model supports it)
    feature_importances = None
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if classifier is not None and preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
                feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            elif hasattr(classifier, "coef_"):
                # Average absolute coefficient weight across multi-class outputs
                coef_abs = np.mean(np.abs(classifier.coef_), axis=0)
                feature_importances = pd.Series(coef_abs, index=feature_names).sort_values(ascending=False)
        except Exception:
            feature_importances = None

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm_raw,
        "confusion_matrix_norm": cm_norm,
        "classification_report": report_dict,
        "per_class_df": per_class_df,
        "feature_importances": feature_importances,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "class_labels": labels,
    }


def compare_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Evaluate all models and produce a comparison DataFrame sorted by Macro F1.
    Also returns detailed metrics cache and the name of the selected best model.
    """
    rows = []
    metrics_cache = {}

    for name, pipeline in models_dict.items():
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        metrics_cache[name] = metrics
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Balanced Accuracy": metrics["balanced_accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Macro Precision": metrics["macro_precision"],
            "Macro Recall": metrics["macro_recall"],
        })

    comp_df = pd.DataFrame(rows).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    best_model_name = comp_df.iloc[0]["Model"]

    return comp_df, metrics_cache, best_model_name
