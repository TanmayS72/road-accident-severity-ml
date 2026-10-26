"""
Data loading and dataset validation utilities for Road Traffic Accident Severity project.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, List
import pandas as pd
import streamlit as st

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RTA_PATH = DATA_DIR / "RTA Dataset.csv"
CLEANED_PATH = DATA_DIR / "cleaned.csv"

# Consistent, readable class ordering
SEVERITY_ORDER: List[str] = ["Fatal injury", "Serious Injury", "Slight Injury"]

# Default label mappings (verified against real data)
SEVERITY_MAP: Dict[int, str] = {
    0: "Fatal injury",
    1: "Serious Injury",
    2: "Slight Injury",
}

INV_SEVERITY_MAP: Dict[str, int] = {v: k for k, v in SEVERITY_MAP.items()}

# UI color accents for severity tiers
SEVERITY_COLORS: Dict[str, str] = {
    "Fatal injury": "#ef4444",      # Rose / Red
    "Serious Injury": "#f59e0b",    # Amber / Orange
    "Slight Injury": "#06b6d4",     # Cyan / Teal
}


def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric hour (0-23) and standard diurnal time band from the Time column."""
    if "Time" not in df.columns:
        return df

    df = df.copy()
    try:
        time_parsed = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
        mask = time_parsed.isna()
        if mask.any():
            time_parsed = time_parsed.fillna(
                pd.to_datetime(df.loc[mask, "Time"], format="%H:%M", errors="coerce")
            )

        df["Hour"] = time_parsed.dt.hour.fillna(12).astype(int)

        def assign_time_band(hour: int) -> str:
            if 6 <= hour < 12:
                return "Morning (06:00-11:59)"
            elif 12 <= hour < 17:
                return "Afternoon (12:00-16:59)"
            elif 17 <= hour < 22:
                return "Evening (17:00-21:59)"
            else:
                return "Night (22:00-05:59)"

        df["Time_band"] = df["Hour"].apply(assign_time_band)
    except Exception:
        df["Hour"] = 12
        df["Time_band"] = "Afternoon (12:00-16:59)"

    return df


@st.cache_data(show_spinner="Loading primary RTA dataset...")
def load_rta_data() -> pd.DataFrame:
    """
    Load the primary original RTA Dataset (12,316 rows, 32 columns).
    Extracts Hour and Time_band, validates target column presence.
    """
    if not RTA_PATH.exists():
        raise FileNotFoundError(
            f"Primary accident dataset not found at '{RTA_PATH}'. "
            "Please ensure 'RTA Dataset.csv' is present in the data/ directory."
        )

    df = pd.read_csv(RTA_PATH)
    if df.empty:
        raise ValueError("Primary dataset 'RTA Dataset.csv' is empty.")

    if "Accident_severity" not in df.columns:
        raise KeyError(
            "Expected target column 'Accident_severity' not found in RTA Dataset.csv."
        )

    df = _extract_time_features(df)

    # Standardize Day_of_week order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "Day_of_week" in df.columns:
        df["Day_of_week"] = pd.Categorical(df["Day_of_week"], categories=day_order, ordered=True)

    return df


@st.cache_data(show_spinner="Loading cleaned verification dataset...")
def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned dataset (12,316 rows, 15 columns) used for secondary verification.
    """
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{CLEANED_PATH}'. "
            "Please ensure 'cleaned.csv' is present in the data/ directory."
        )

    df = pd.read_csv(CLEANED_PATH)
    if df.empty:
        raise ValueError("Dataset 'cleaned.csv' is empty.")

    if "Accident_severity" not in df.columns:
        raise KeyError(
            "Expected target column 'Accident_severity' not found in cleaned.csv."
        )

    # Attach readable severity label dynamically based on verified mapping
    if "Severity_label" not in df.columns:
        df["Severity_label"] = df["Accident_severity"].map(SEVERITY_MAP).fillna("Unknown")

    return df


def validate_datasets() -> Dict[str, Any]:
    """
    Conduct thorough validation across both datasets:
    - Dimensions
    - Target column presence and counts
    - Duplicate records
    - Missing values
    - Verification of numeric mapping in cleaned.csv
    """
    rta_df = load_rta_data()
    clean_df = load_cleaned_data()

    # Dynamic target distribution from raw dataset
    raw_target_counts = rta_df["Accident_severity"].value_counts().to_dict()
    clean_target_counts = clean_df["Accident_severity"].value_counts().to_dict()

    # Dynamic verification of numeric mapping:
    # We compare crosstab between original text target and cleaned integer target
    mapping_verification = pd.crosstab(
        rta_df["Accident_severity"], clean_df["Accident_severity"]
    ).to_dict()

    # Derived verified mapping: {clean_int: original_str}
    verified_mapping = {}
    for orig_label in rta_df["Accident_severity"].unique():
        # Find which integer matches this original label
        sub_mask = rta_df["Accident_severity"] == orig_label
        matched_ints = clean_df.loc[sub_mask, "Accident_severity"].unique()
        if len(matched_ints) == 1:
            verified_mapping[int(matched_ints[0])] = str(orig_label)

    return {
        "rta_shape": rta_df.shape,
        "clean_shape": clean_df.shape,
        "rta_duplicates": int(rta_df.duplicated().sum()),
        "clean_duplicates": int(clean_df.drop(columns=["Severity_label"], errors="ignore").duplicated().sum()),
        "rta_missing_cells": int(rta_df.isnull().sum().sum()),
        "clean_missing_cells": int(clean_df.drop(columns=["Severity_label"], errors="ignore").isnull().sum().sum()),
        "raw_target_counts": raw_target_counts,
        "clean_target_counts": clean_target_counts,
        "verified_mapping": verified_mapping,
        "mapping_crosstab": mapping_verification,
        "rta_columns": list(rta_df.columns),
        "clean_columns": [c for c in clean_df.columns if c != "Severity_label"],
    }


def get_dataset_summary() -> Dict[str, Any]:
    """Return high-level summary KPIs for the UI."""
    rta_df = load_rta_data()
    total_records = len(rta_df)
    counts = rta_df["Accident_severity"].value_counts().to_dict()

    return {
        "total_records": total_records,
        "rta_shape": rta_df.shape,
        "fatal_count": counts.get("Fatal injury", 0),
        "serious_count": counts.get("Serious Injury", 0),
        "slight_count": counts.get("Slight Injury", 0),
        "fatal_pct": (counts.get("Fatal injury", 0) / total_records) * 100,
        "serious_pct": (counts.get("Serious Injury", 0) / total_records) * 100,
        "slight_pct": (counts.get("Slight Injury", 0) / total_records) * 100,
    }
