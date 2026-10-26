"""
Page 3: Severity Prediction
Interactive Machine Learning Inference using standard syllabus ML pipelines:
1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Tree
4. Random Forest
"""

from pathlib import Path
import random
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from src.data_loader import (
    load_rta_data,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
)
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PRESET_SCENARIOS,
)
from src.train_models import (
    MODELS_DIR,
    MODEL_FILENAMES,
    get_cached_metrics,
)
from src.visualizations import (
    apply_custom_css,
    plot_probability_breakdown,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
    COLOR_TEAL,
    COLOR_SKY,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Severity Prediction | Road Accident Intelligence",
    page_icon="🎯",
    layout="wide",
)

apply_custom_css()

# -----------------------------------------------------------------------------
# DATA LOADING & CACHED RESOURCE HELPERS
# -----------------------------------------------------------------------------
try:
    df_raw = load_rta_data()
    dataset_ready = True
except Exception as e:
    st.error(f"Failed to load primary collision registry: {str(e)}")
    st.stop()


@st.cache_resource
def load_model_pipeline(filename: str):
    """Load and cache an individual sklearn model pipeline from disk."""
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# Helper to get clean unique categories from df_raw for selectboxes
@st.cache_data
def get_column_options(col_name: str) -> list:
    """Return sorted list of non-null string options for a categorical column."""
    if col_name not in df_raw.columns:
        return ["Unknown"]
    vals = [
        str(x).strip()
        for x in df_raw[col_name].dropna().unique()
        if str(x).strip() != "" and str(x).strip().lower() != "nan"
    ]
    return sorted(vals)


# Model mapping definition
MODEL_OPTIONS = [
    "K-Nearest Neighbors (KNN)",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]

MODEL_SPECS = {
    "K-Nearest Neighbors (KNN)": {
        "key": "K-Nearest Neighbors",
        "file": "k_nearest_neighbors.joblib",
        "icon": "📍",
        "syllabus_desc": "Predicts the class using nearby examples in the training data.",
        "details": "Calculates Euclidean distance across scaled numerical features and one-hot vectors, weighting nearest neighbors by inverse distance.",
    },
    "Logistic Regression": {
        "key": "Logistic Regression",
        "file": "logistic_regression.joblib",
        "icon": "📈",
        "syllabus_desc": "Uses a learned classification function to estimate class probabilities.",
        "details": "Fits a multinomial linear log-odds decision surface with balanced class weighting to compensate for dataset injury skew.",
    },
    "Decision Tree": {
        "key": "Decision Tree",
        "file": "decision_tree.joblib",
        "icon": "🌳",
        "syllabus_desc": "Predicts a class using learned decision rules.",
        "details": "Recursively partitions feature space using Gini impurity thresholds down to depth 10 with balanced class weighting.",
    },
    "Random Forest": {
        "key": "Random Forest",
        "file": "random_forest.joblib",
        "icon": "🌲",
        "syllabus_desc": "Combines predictions from multiple decision trees.",
        "details": "Aggregates an ensemble of 100 de-correlated bootstrap trees with random feature subspacing to reduce variance.",
    },
}

# -----------------------------------------------------------------------------
# 1. PAGE HEADER & 2. MODEL SELECTION
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([2.1, 1.4])

with header_col1:
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h1 style="color: #ffffff; font-size: 2.1rem; font-weight: 800; margin: 0 0 4px 0; letter-spacing: -0.02em;">
                Accident Severity Prediction
            </h1>
            <div style="color: #06b6d4; font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
                Interactive Machine Learning Inference
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; line-height: 1.5; margin: 0; max-width: 900px;">
                Enter recorded accident characteristics to obtain a severity prediction from a trained classification model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    selected_model_display = st.selectbox(
        "Select Classification Model Pipeline:",
        MODEL_OPTIONS,
        index=3,  # Default to Random Forest
        help="Select one of the four standard syllabus-approved classification pipelines.",
    )

    selected_spec = MODEL_SPECS[selected_model_display]
    current_pipeline = load_model_pipeline(selected_spec["file"])
    model_is_ready = current_pipeline is not None

    if model_is_ready:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 7px 12px; margin-top: 4px; display: flex; align-items: center; justify-content: space-between;">
                <span style="color: #10b981; font-size: 0.82rem; font-weight: 700;">
                    ● Pipeline Active: Ready for Inference
                </span>
                <span style="color: #94a3b8; font-size: 0.76rem; font-family: monospace;">
                    {selected_spec['file']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 7px 12px; margin-top: 4px;">
                <span style="color: #ef4444; font-size: 0.82rem; font-weight: 700;">
                    ● Pipeline Unavailable: Artifact Missing ({selected_spec['file']})
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. SAMPLE SCENARIO & PRESET LOADER
# -----------------------------------------------------------------------------
# Initialize form values in session_state if not present
if "form_values" not in st.session_state:
    # Use standard default or row 0
    row0 = df_raw[ALL_FEATURE_COLUMNS].iloc[0].to_dict()
    default_vals = {}
    for col in ALL_FEATURE_COLUMNS:
        v = row0.get(col)
        if pd.isna(v):
            opts = get_column_options(col)
            default_vals[col] = opts[0] if opts else "Unknown"
        else:
            default_vals[col] = v
    st.session_state.form_values = default_vals

sample_box_col1, sample_box_col2 = st.columns([1.2, 2.5])

with sample_box_col1:
    btn_load_sample = st.button(
        "📋 Load Sample Accident",
        use_container_width=True,
        help="Populate the form using a genuine historical collision record from the primary RTA dataset (target severity is blinded).",
    )

with sample_box_col2:
    # Additional preset shortcuts for demonstrating different risk tiers in viva
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🏙️ Urban Daylight", use_container_width=True):
            st.session_state.form_values = PRESET_SCENARIOS["Low-Risk Urban Daylight Incident"].copy()
            st.rerun()
    with p2:
        if st.button("🛣️ Night Highway", use_container_width=True):
            st.session_state.form_values = PRESET_SCENARIOS["High-Risk Night Highway Incident"].copy()
            st.rerun()
    with p3:
        if st.button("⚠️ Reckless / Impaired", use_container_width=True):
            st.session_state.form_values = PRESET_SCENARIOS["Critical Reckless & Impaired Driving Scenario"].copy()
            st.rerun()

if btn_load_sample:
    try:
        # Sample a random historical row from df_raw
        random_idx = random.randint(0, len(df_raw) - 1)
        sampled_row = df_raw[ALL_FEATURE_COLUMNS].iloc[random_idx].to_dict()
        new_form_vals = {}
        for col in ALL_FEATURE_COLUMNS:
            raw_v = sampled_row.get(col)
            if pd.isna(raw_v):
                opts = get_column_options(col)
                new_form_vals[col] = opts[0] if opts else "Unknown"
            else:
                if col in NUMERICAL_FEATURES:
                    new_form_vals[col] = int(raw_v)
                else:
                    new_form_vals[col] = str(raw_v).strip()
        st.session_state.form_values = new_form_vals
        st.session_state["sample_case_id"] = random_idx + 1
        st.rerun()
    except Exception as ex:
        st.error(f"Could not load historical sample record: {str(ex)}")

if "sample_case_id" in st.session_state:
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #38bdf8; margin-top: -6px; margin-bottom: 12px;">
            ℹ️ Populated inputs from Historical Case <b>#{st.session_state['sample_case_id']}</b> in <code>data/RTA Dataset.csv</code>.
            (Target severity excluded to ensure blind evaluation).
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 3. INPUT FORM (STREAMLIT FORM WITH 3 STRUCTURED SECTIONS)
# -----------------------------------------------------------------------------
cur_vals = st.session_state.form_values

with st.form(key="severity_prediction_form"):
    form_c1, form_c2, form_c3 = st.columns(3)

    # -------------------------------------------------------------------------
    # SECTION A — DRIVER INFORMATION
    # -------------------------------------------------------------------------
    with form_c1:
        st.markdown(
            """
            <div class="rta-card" style="margin-bottom: 14px; padding: 12px 16px; border-left: 4px solid #06b6d4;">
                <h4 style="color: #06b6d4; margin: 0; font-size: 0.98rem; text-transform: uppercase; letter-spacing: 0.04em;">
                    Section A — Driver Information
                </h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Age_band_of_driver
        opts_age = get_column_options("Age_band_of_driver")
        val_age = str(cur_vals.get("Age_band_of_driver", opts_age[0]))
        idx_age = opts_age.index(val_age) if val_age in opts_age else 0
        input_age = st.selectbox("Age Band of Driver", opts_age, index=idx_age)

        # Sex_of_driver
        opts_sex = get_column_options("Sex_of_driver")
        val_sex = str(cur_vals.get("Sex_of_driver", opts_sex[0]))
        idx_sex = opts_sex.index(val_sex) if val_sex in opts_sex else 0
        input_sex = st.selectbox("Sex of Driver", opts_sex, index=idx_sex)

        # Driving_experience
        opts_exp = get_column_options("Driving_experience")
        val_exp = str(cur_vals.get("Driving_experience", opts_exp[0]))
        idx_exp = opts_exp.index(val_exp) if val_exp in opts_exp else 0
        input_exp = st.selectbox("Driving Experience", opts_exp, index=idx_exp)

        # Educational_level
        opts_edu = get_column_options("Educational_level")
        val_edu = str(cur_vals.get("Educational_level", opts_edu[0]))
        idx_edu = opts_edu.index(val_edu) if val_edu in opts_edu else 0
        input_edu = st.selectbox("Educational Level", opts_edu, index=idx_edu)

        # Vehicle_driver_relation
        opts_rel = get_column_options("Vehicle_driver_relation")
        val_rel = str(cur_vals.get("Vehicle_driver_relation", opts_rel[0]))
        idx_rel = opts_rel.index(val_rel) if val_rel in opts_rel else 0
        input_rel = st.selectbox("Vehicle Driver Relation", opts_rel, index=idx_rel)

    # -------------------------------------------------------------------------
    # SECTION B — ROAD AND ENVIRONMENTAL CONDITIONS
    # -------------------------------------------------------------------------
    with form_c2:
        st.markdown(
            """
            <div class="rta-card" style="margin-bottom: 14px; padding: 12px 16px; border-left: 4px solid #38bdf8;">
                <h4 style="color: #38bdf8; margin: 0; font-size: 0.98rem; text-transform: uppercase; letter-spacing: 0.04em;">
                    Section B — Road & Environment
                </h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Light_conditions
        opts_light = get_column_options("Light_conditions")
        val_light = str(cur_vals.get("Light_conditions", opts_light[0]))
        idx_light = opts_light.index(val_light) if val_light in opts_light else 0
        input_light = st.selectbox("Light Conditions", opts_light, index=idx_light)

        # Weather_conditions
        opts_weath = get_column_options("Weather_conditions")
        val_weath = str(cur_vals.get("Weather_conditions", opts_weath[0]))
        idx_weath = opts_weath.index(val_weath) if val_weath in opts_weath else 0
        input_weath = st.selectbox("Weather Conditions", opts_weath, index=idx_weath)

        # Road_surface_conditions
        opts_rcond = get_column_options("Road_surface_conditions")
        val_rcond = str(cur_vals.get("Road_surface_conditions", opts_rcond[0]))
        idx_rcond = opts_rcond.index(val_rcond) if val_rcond in opts_rcond else 0
        input_rcond = st.selectbox("Road Surface Conditions", opts_rcond, index=idx_rcond)

        # Road_surface_type
        opts_rtype = get_column_options("Road_surface_type")
        val_rtype = str(cur_vals.get("Road_surface_type", opts_rtype[0]))
        idx_rtype = opts_rtype.index(val_rtype) if val_rtype in opts_rtype else 0
        input_rtype = st.selectbox("Road Surface Type", opts_rtype, index=idx_rtype)

        # Road_allignment
        opts_align = get_column_options("Road_allignment")
        val_align = str(cur_vals.get("Road_allignment", opts_align[0]))
        idx_align = opts_align.index(val_align) if val_align in opts_align else 0
        input_align = st.selectbox("Road Alignment", opts_align, index=idx_align)

        # Types_of_Junction
        opts_junc = get_column_options("Types_of_Junction")
        val_junc = str(cur_vals.get("Types_of_Junction", opts_junc[0]))
        idx_junc = opts_junc.index(val_junc) if val_junc in opts_junc else 0
        input_junc = st.selectbox("Types of Junction", opts_junc, index=idx_junc)

        # Lanes_or_Medians
        opts_lanes = get_column_options("Lanes_or_Medians")
        val_lanes = str(cur_vals.get("Lanes_or_Medians", opts_lanes[0]))
        idx_lanes = opts_lanes.index(val_lanes) if val_lanes in opts_lanes else 0
        input_lanes = st.selectbox("Lanes or Medians", opts_lanes, index=idx_lanes)

        # Area_accident_occured
        opts_area = get_column_options("Area_accident_occured")
        val_area = str(cur_vals.get("Area_accident_occured", opts_area[0]))
        idx_area = opts_area.index(val_area) if val_area in opts_area else 0
        input_area = st.selectbox("Area Accident Occurred", opts_area, index=idx_area)

        # Day_of_week
        opts_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        val_day = str(cur_vals.get("Day_of_week", "Monday"))
        idx_day = opts_day.index(val_day) if val_day in opts_day else 0
        input_day = st.selectbox("Day of Week", opts_day, index=idx_day)

    # -------------------------------------------------------------------------
    # SECTION C — VEHICLE AND INCIDENT DETAILS
    # -------------------------------------------------------------------------
    with form_c3:
        st.markdown(
            """
            <div class="rta-card" style="margin-bottom: 14px; padding: 12px 16px; border-left: 4px solid #14b8a6;">
                <h4 style="color: #14b8a6; margin: 0; font-size: 0.98rem; text-transform: uppercase; letter-spacing: 0.04em;">
                    Section C — Vehicle & Incident Details
                </h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Type_of_vehicle
        opts_vtype = get_column_options("Type_of_vehicle")
        val_vtype = str(cur_vals.get("Type_of_vehicle", opts_vtype[0]))
        idx_vtype = opts_vtype.index(val_vtype) if val_vtype in opts_vtype else 0
        input_vtype = st.selectbox("Type of Vehicle", opts_vtype, index=idx_vtype)

        # Owner_of_vehicle
        opts_owner = get_column_options("Owner_of_vehicle")
        val_owner = str(cur_vals.get("Owner_of_vehicle", opts_owner[0]))
        idx_owner = opts_owner.index(val_owner) if val_owner in opts_owner else 0
        input_owner = st.selectbox("Owner of Vehicle", opts_owner, index=idx_owner)

        # Service_year_of_vehicle
        opts_syear = get_column_options("Service_year_of_vehicle")
        val_syear = str(cur_vals.get("Service_year_of_vehicle", opts_syear[0]))
        idx_syear = opts_syear.index(val_syear) if val_syear in opts_syear else 0
        input_syear = st.selectbox("Service Year of Vehicle", opts_syear, index=idx_syear)

        # Defect_of_vehicle
        opts_defect = get_column_options("Defect_of_vehicle")
        val_defect = str(cur_vals.get("Defect_of_vehicle", opts_defect[0]))
        idx_defect = opts_defect.index(val_defect) if val_defect in opts_defect else 0
        input_defect = st.selectbox("Defect of Vehicle", opts_defect, index=idx_defect)

        # Type_of_collision
        opts_coll = get_column_options("Type_of_collision")
        val_coll = str(cur_vals.get("Type_of_collision", opts_coll[0]))
        idx_coll = opts_coll.index(val_coll) if val_coll in opts_coll else 0
        input_coll = st.selectbox("Type of Collision", opts_coll, index=idx_coll)

        # Vehicle_movement
        opts_move = get_column_options("Vehicle_movement")
        val_move = str(cur_vals.get("Vehicle_movement", opts_move[0]))
        idx_move = opts_move.index(val_move) if val_move in opts_move else 0
        input_move = st.selectbox("Vehicle Movement", opts_move, index=idx_move)

        # Pedestrian_movement
        opts_ped = get_column_options("Pedestrian_movement")
        val_ped = str(cur_vals.get("Pedestrian_movement", opts_ped[0]))
        idx_ped = opts_ped.index(val_ped) if val_ped in opts_ped else 0
        input_ped = st.selectbox("Pedestrian Movement", opts_ped, index=idx_ped)

        # Cause_of_accident
        opts_cause = get_column_options("Cause_of_accident")
        val_cause = str(cur_vals.get("Cause_of_accident", opts_cause[0]))
        idx_cause = opts_cause.index(val_cause) if val_cause in opts_cause else 0
        input_cause = st.selectbox("Cause of Accident", opts_cause, index=idx_cause)

        # Numerical Fields with Validation
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            val_nveh = int(cur_vals.get("Number_of_vehicles_involved", 2))
            input_nveh = st.number_input("Vehicles", min_value=1, max_value=10, value=val_nveh, step=1)
        with sub_c2:
            val_ncas = int(cur_vals.get("Number_of_casualties", 1))
            input_ncas = st.number_input("Casualties", min_value=1, max_value=20, value=val_ncas, step=1)
        with sub_c3:
            val_hour = int(cur_vals.get("Hour", 14))
            input_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=val_hour, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. PREDICTION BUTTON
    # -------------------------------------------------------------------------
    submit_prediction = st.form_submit_button(
        "Predict Accident Severity",
        type="primary",
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# 5. MODEL INFERENCE & 6. PREDICTION RESULTS
# -----------------------------------------------------------------------------
if submit_prediction:
    if not model_is_ready:
        st.error(
            f"Cannot perform inference: The selected model pipeline artifact "
            f"('{selected_spec['file']}') could not be loaded from the models/ directory."
        )
        st.stop()

    # Input validation
    validation_passed = True
    if input_nveh < 1 or input_nveh > 10:
        st.error("Validation error: Number of vehicles involved must be between 1 and 10.")
        validation_passed = False
    if input_ncas < 1 or input_ncas > 20:
        st.error("Validation error: Number of casualties must be between 1 and 20.")
        validation_passed = False
    if input_hour < 0 or input_hour > 23:
        st.error("Validation error: Hour must be between 0 and 23.")
        validation_passed = False

    if not validation_passed:
        st.stop()

    # Construct clean payload containing EXACTLY the 25 expected features
    payload = {
        "Number_of_vehicles_involved": int(input_nveh),
        "Number_of_casualties": int(input_ncas),
        "Hour": int(input_hour),
        "Day_of_week": input_day,
        "Age_band_of_driver": input_age,
        "Sex_of_driver": input_sex,
        "Educational_level": input_edu,
        "Vehicle_driver_relation": input_rel,
        "Driving_experience": input_exp,
        "Type_of_vehicle": input_vtype,
        "Owner_of_vehicle": input_owner,
        "Service_year_of_vehicle": input_syear,
        "Defect_of_vehicle": input_defect,
        "Area_accident_occured": input_area,
        "Lanes_or_Medians": input_lanes,
        "Road_allignment": input_align,
        "Types_of_Junction": input_junc,
        "Road_surface_type": input_rtype,
        "Road_surface_conditions": input_rcond,
        "Light_conditions": input_light,
        "Weather_conditions": input_weath,
        "Type_of_collision": input_coll,
        "Vehicle_movement": input_move,
        "Pedestrian_movement": input_ped,
        "Cause_of_accident": input_cause,
    }

    # Pass the raw DataFrame directly to the saved sklearn Pipeline
    input_df = pd.DataFrame([payload])

    try:
        # Pipeline performs its own imputation, one-hot encoding, and scaling
        pred_label = current_pipeline.predict(input_df)[0]

        # Extract class probabilities if supported
        has_proba = hasattr(current_pipeline, "predict_proba")
        if has_proba:
            raw_probs = current_pipeline.predict_proba(input_df)[0]
            raw_classes = list(current_pipeline.classes_)
            prob_map = {cls_name: float(p) for cls_name, p in zip(raw_classes, raw_probs)}
            # Map probabilities to standardized display order
            display_order = ["Fatal injury", "Serious Injury", "Slight Injury"]
            ordered_probs = np.array([prob_map.get(c, 0.0) for c in display_order])
        else:
            ordered_probs = None

        # Severity visual indicator styling
        if pred_label == "Fatal injury":
            badge_color = COLOR_ROSE
            badge_text = "FATAL INJURY (CRITICAL SEVERITY)"
            badge_bg = "rgba(239, 68, 68, 0.15)"
            explanation = (
                "The classifier predicted a <b>Fatal injury</b> outcome. In historical collision registries, "
                "fatal collisions represent an acute minority (1.28% of cases, 158 records) and are strongly associated "
                "with extreme kinetic trauma, unlit nighttime darkness, or excessive speed."
            )
        elif pred_label == "Serious Injury":
            badge_color = COLOR_AMBER
            badge_text = "SERIOUS INJURY (MODERATE / SEVERE TRAUMA)"
            badge_bg = "rgba(245, 158, 11, 0.15)"
            explanation = (
                "The classifier predicted a <b>Serious Injury</b> outcome, indicating major physical trauma "
                "requiring inpatient hospitalization or surgical intervention. Serious injuries account for 14.15% "
                "(1,743 records) of the historical dataset."
            )
        else:
            badge_color = COLOR_CYAN
            badge_text = "SLIGHT INJURY (MINOR TRAUMA)"
            badge_bg = "rgba(6, 182, 212, 0.15)"
            explanation = (
                "The classifier predicted a <b>Slight Injury</b> outcome, denoting superficial injuries, abrasions, "
                "or outpatient ambulatory care. Minor injuries constitute the vast majority (84.57%, 10,415 records) "
                "of historical traffic collision records."
            )

        # ---------------------------------------------------------------------
        # DISPLAY PREDICTION RESULT CARD
        # ---------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="rta-card" style="border-left: 6px solid {badge_color}; background: linear-gradient(135deg, rgba(28, 37, 65, 0.98), rgba(15, 23, 42, 0.95)); padding: 22px 24px; margin-top: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
                    <div>
                        <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; padding: 4px 14px; border-radius: 20px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                            {badge_text}
                        </span>
                        <h2 style="color: #ffffff; margin: 10px 0 6px 0; font-size: 1.8rem; letter-spacing: -0.02em;">
                            Predicted Severity: <span style="color: {badge_color};">{pred_label}</span>
                        </h2>
                        <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 10px;">
                            Active Algorithm: <b style="color: #ffffff;">{selected_model_display}</b>
                        </div>
                        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.55; margin: 0; max-width: 850px;">
                            {explanation}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------------------
        # 7. CLASS PROBABILITIES
        # ---------------------------------------------------------------------
        prob_col, context_col = st.columns([1.4, 1])

        with prob_col:
            if has_proba and ordered_probs is not None:
                st.markdown(
                    """
                    <div style="font-size: 0.88rem; font-weight: 700; color: #06b6d4; text-transform: uppercase; margin-bottom: 6px;">
                        Estimated Class Probabilities
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_prob = plot_probability_breakdown(ordered_probs, ["Fatal injury", "Serious Injury", "Slight Injury"])
                st.plotly_chart(fig_prob, use_container_width=True)

                st.markdown(
                    """
                    <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45; margin-top: -6px; margin-bottom: 12px;">
                        ℹ️ <b>Statistical Interpretation:</b> Model probabilities represent mathematical decision scores computed by the classifier's
                        decision function on the input features. They are model-derived probability estimates and should not be interpreted as guaranteed real-world collision outcome probabilities.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"The selected model ({selected_model_display}) does not produce continuous class probability estimates.")

        with context_col:
            st.markdown(
                """
                <div class="rta-card" style="padding: 18px 20px; height: 100%;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px;">
                        Scenario Configuration Summary
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <ul style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.6; padding-left: 18px; margin: 0;">
                    <li><b>Collision Type:</b> {input_coll}</li>
                    <li><b>Primary Cause:</b> {input_cause}</li>
                    <li><b>Lighting & Visibility:</b> {input_light}</li>
                    <li><b>Road Surface:</b> {input_rcond} ({input_rtype})</li>
                    <li><b>Driver Demographics:</b> Age {input_age}, {input_sex} ({input_exp} exp)</li>
                    <li><b>Impact Scope:</b> {input_nveh} vehicle(s), {input_ncas} casualt(ies) at {input_hour:02d}:00</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Inference execution failed: {str(e)}")

# -----------------------------------------------------------------------------
# 8. MODEL INFORMATION & HELD-OUT EVALUATION METRICS
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Model Architecture & Benchmark Metrics</h3>
        <span class="section-badge">Academic Lab Profile</span>
    </div>
    """,
    unsafe_allow_html=True,
)

m_info_col1, m_info_col2 = st.columns([1.4, 1.2])

with m_info_col1:
    st.markdown(
        f"""
        <div class="rta-card" style="padding: 20px 22px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #06b6d4; text-transform: uppercase; margin-bottom: 6px;">
                Algorithm Overview
            </div>
            <h4 style="color: #ffffff; margin: 0 0 8px 0; font-size: 1.15rem;">
                {selected_spec['icon']} {selected_model_display}
            </h4>
            <div style="font-size: 0.9rem; color: #38bdf8; font-weight: 600; margin-bottom: 8px;">
                "{selected_spec['syllabus_desc']}"
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.5; margin: 0 0 10px 0;">
                {selected_spec['details']}
            </p>
            <div style="font-size: 0.78rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                <b>Syllabus Standard:</b> Built strictly with Scikit-Learn; no boosting libraries, complex neural architectures, or external inference dependencies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_info_col2:
    # Display saved held-out test set metrics
    all_metrics = get_cached_metrics()
    model_metrics = all_metrics.get(selected_spec["key"], {})

    if model_metrics:
        acc_pct = model_metrics.get("accuracy", 0.0) * 100
        bal_acc_pct = model_metrics.get("balanced_accuracy", 0.0) * 100
        macro_f1 = model_metrics.get("macro_f1", 0.0)

        st.markdown(
            f"""
            <div class="rta-card" style="padding: 20px 22px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #14b8a6; text-transform: uppercase; margin-bottom: 6px;">
                    Held-Out Test Set Performance (N=2,464)
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0;">
                    <div style="background: rgba(11, 19, 43, 0.6); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">Overall Acc</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff;">{acc_pct:.1f}%</div>
                    </div>
                    <div style="background: rgba(11, 19, 43, 0.6); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(56, 189, 248, 0.2);">
                        <div style="font-size: 0.74rem; color: #38bdf8; text-transform: uppercase;">Balanced Acc</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #38bdf8;">{bal_acc_pct:.1f}%</div>
                    </div>
                    <div style="background: rgba(11, 19, 43, 0.6); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(6, 182, 212, 0.2);">
                        <div style="font-size: 0.74rem; color: #06b6d4; text-transform: uppercase;">Macro F1</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #06b6d4;">{macro_f1:.4f}</div>
                    </div>
                </div>
                <div style="font-size: 0.76rem; color: #94a3b8; line-height: 1.4;">
                    * <b>Note:</b> These metrics represent held-out test-set generalization on historical accident records (20% stratified split), not a confidence score for this individual prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Evaluation benchmark metrics not cached.")

# -----------------------------------------------------------------------------
# 10. EDUCATIONAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 14px 18px; margin-top: 10px;">
        <div style="display: flex; align-items: flex-start; gap: 10px;">
            <span style="font-size: 1.2rem; color: #f59e0b;">⚠️</span>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
                <b style="color: #f59e0b;">Educational Machine Learning Prototype Disclaimer:</b><br>
                This application is an educational machine learning prototype trained on historical accident records.
                Predictions may be incorrect and should not be used for emergency dispatch, official accident classification,
                or real-world safety decisions.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer callout
st.markdown(
    """
    <div style="text-align: center; padding: 22px 0 10px 0; color: #64748b; font-size: 0.78rem;">
        Road Accident Intelligence • Severity Inference Engine • 2nd-Year Engineering Machine Learning Lab
    </div>
    """,
    unsafe_allow_html=True,
)
