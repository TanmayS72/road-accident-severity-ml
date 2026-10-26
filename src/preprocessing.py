"""
Preprocessing module using Scikit-Learn ColumnTransformer and Pipeline.
Prevents data leakage by fitting transformers strictly on training partitions.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target Column
TARGET_COLUMN = "Accident_severity"

# Numerical Feature Columns (from primary RTA dataset)
NUMERICAL_FEATURES: List[str] = [
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Hour",
]

# Categorical Feature Columns (from primary RTA dataset)
CATEGORICAL_FEATURES: List[str] = [
    "Day_of_week",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Educational_level",
    "Vehicle_driver_relation",
    "Driving_experience",
    "Type_of_vehicle",
    "Owner_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
    "Area_accident_occured",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Vehicle_movement",
    "Pedestrian_movement",
    "Cause_of_accident",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Excluded Columns Documentation
EXCLUDED_COLUMNS_RATIONALE: Dict[str, str] = {
    "Accident_severity": (
        "Ground-truth prediction target; must be strictly excluded from input features."
    ),
    "Casualty_severity": (
        "Direct post-incident injury severity triage outcome; introducing this creates "
        "severe target leakage since victim severity directly discloses accident severity."
    ),
    "Casualty_class": (
        "Post-collision casualty documentation recorded after victim triage; contains 36% 'na' values."
    ),
    "Sex_of_casualty": (
        "Post-collision casualty demographic recorded after victim triage; contains 36% 'na' values."
    ),
    "Age_band_of_casualty": (
        "Post-collision casualty demographic recorded after victim triage; contains 36% 'na' values."
    ),
    "Work_of_casuality": (
        "Post-collision casualty occupational profile with high missingness (>26%); not available at crash time."
    ),
    "Fitness_of_casuality": (
        "Post-collision medical fitness assessment with high missingness (>21%); not available at crash time."
    ),
    "Time": (
        "Raw high-cardinality string timestamp (1,074 unique HH:MM:SS values). Replaced with "
        "engineered numerical 'Hour' (0-23) to capture diurnal cyclic traffic dynamics cleanly."
    ),
}


def get_feature_lists() -> Tuple[List[str], List[str], List[str]]:
    """Return (all_features, numerical_features, categorical_features)."""
    return ALL_FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def get_excluded_columns_documentation() -> Dict[str, str]:
    """Return dictionary of excluded columns and academic rationale."""
    return EXCLUDED_COLUMNS_RATIONALE


def create_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer.
    - Numerical: SimpleImputer(median) -> optional StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    
    scale_numeric=True: Used for distance/gradient based models (KNN, Logistic Regression)
    scale_numeric=False: Used for tree-based models (Decision Tree, Random Forest)
    """
    if scale_numeric:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(classifier: Any, scale_numeric: bool = True) -> Pipeline:
    """
    Wrap the ColumnTransformer preprocessor and classifier into a single self-contained Pipeline.
    Prevents data leakage and ensures seamless inference from raw dataframes.
    """
    preprocessor = create_preprocessor(scale_numeric=scale_numeric)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# Standardized Test Scenarios for Interactive Prediction
PRESET_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Low-Risk Urban Daylight Incident": {
        "Number_of_vehicles_involved": 2,
        "Number_of_casualties": 1,
        "Hour": 14,
        "Day_of_week": "Monday",
        "Age_band_of_driver": "31-50",
        "Sex_of_driver": "Male",
        "Educational_level": "Junior high school",
        "Vehicle_driver_relation": "Employee",
        "Driving_experience": "5-10yr",
        "Type_of_vehicle": "Automobile",
        "Owner_of_vehicle": "Owner",
        "Service_year_of_vehicle": "5-10yrs",
        "Defect_of_vehicle": "No defect",
        "Area_accident_occured": "Office areas",
        "Lanes_or_Medians": "Two-way (divided with broken lines road marking)",
        "Road_allignment": "Tangent road with flat terrain",
        "Types_of_Junction": "No junction",
        "Road_surface_type": "Asphalt roads",
        "Road_surface_conditions": "Dry",
        "Light_conditions": "Daylight",
        "Weather_conditions": "Normal",
        "Type_of_collision": "Vehicle with vehicle collision",
        "Vehicle_movement": "Going straight",
        "Pedestrian_movement": "Not a Pedestrian",
        "Cause_of_accident": "No distancing",
    },
    "High-Risk Night Highway Incident": {
        "Number_of_vehicles_involved": 2,
        "Number_of_casualties": 3,
        "Hour": 23,
        "Day_of_week": "Friday",
        "Age_band_of_driver": "18-30",
        "Sex_of_driver": "Male",
        "Educational_level": "Elementary school",
        "Vehicle_driver_relation": "Owner",
        "Driving_experience": "Below 1yr",
        "Type_of_vehicle": "Lorry (41?100Q)",
        "Owner_of_vehicle": "Owner",
        "Service_year_of_vehicle": "Above 10yr",
        "Defect_of_vehicle": "No defect",
        "Area_accident_occured": "Outside rural areas",
        "Lanes_or_Medians": "Undivided Two way",
        "Road_allignment": "Tangent road with flat terrain",
        "Types_of_Junction": "Y Shape",
        "Road_surface_type": "Earth roads",
        "Road_surface_conditions": "Wet or damp",
        "Light_conditions": "Darkness - no lighting",
        "Weather_conditions": "Raining",
        "Type_of_collision": "Rollover",
        "Vehicle_movement": "Going straight",
        "Pedestrian_movement": "Not a Pedestrian",
        "Cause_of_accident": "Driving at high speed",
    },
    "Critical Reckless & Impaired Driving Scenario": {
        "Number_of_vehicles_involved": 1,
        "Number_of_casualties": 2,
        "Hour": 2,
        "Day_of_week": "Saturday",
        "Age_band_of_driver": "18-30",
        "Sex_of_driver": "Male",
        "Educational_level": "Junior high school",
        "Vehicle_driver_relation": "Employee",
        "Driving_experience": "1-2yr",
        "Type_of_vehicle": "Automobile",
        "Owner_of_vehicle": "Owner",
        "Service_year_of_vehicle": "Unknown",
        "Defect_of_vehicle": "No defect",
        "Area_accident_occured": "Recreational areas",
        "Lanes_or_Medians": "Undivided Two way",
        "Road_allignment": "Escarpments",
        "Types_of_Junction": "Crossing",
        "Road_surface_type": "Asphalt roads with some distress",
        "Road_surface_conditions": "Wet or damp",
        "Light_conditions": "Darkness - lights unlit",
        "Weather_conditions": "Fog or mist",
        "Type_of_collision": "Collision with roadside objects",
        "Vehicle_movement": "Turnover",
        "Pedestrian_movement": "Not a Pedestrian",
        "Cause_of_accident": "Drunk driving",
    },
    "Pedestrian Conflict in Low Visibility": {
        "Number_of_vehicles_involved": 1,
        "Number_of_casualties": 1,
        "Hour": 19,
        "Day_of_week": "Wednesday",
        "Age_band_of_driver": "31-50",
        "Sex_of_driver": "Male",
        "Educational_level": "Junior high school",
        "Vehicle_driver_relation": "Employee",
        "Driving_experience": "2-5yr",
        "Type_of_vehicle": "Automobile",
        "Owner_of_vehicle": "Owner",
        "Service_year_of_vehicle": "2-5yrs",
        "Defect_of_vehicle": "No defect",
        "Area_accident_occured": "Residential areas",
        "Lanes_or_Medians": "Two-way (divided with broken lines road marking)",
        "Road_allignment": "Tangent road with flat terrain",
        "Types_of_Junction": "Crossing",
        "Road_surface_type": "Asphalt roads",
        "Road_surface_conditions": "Dry",
        "Light_conditions": "Darkness - lights lit",
        "Weather_conditions": "Normal",
        "Type_of_collision": "Collision with pedestrians",
        "Vehicle_movement": "Going straight",
        "Pedestrian_movement": "Crossing from driver's nearside",
        "Cause_of_accident": "No priority to pedestrian",
    },
}
