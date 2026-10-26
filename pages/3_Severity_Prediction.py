"""
Page 3: Severity Prediction
Interactive real-time incident severity forecasting using standard syllabus ML pipelines.
"""

import streamlit as st
import numpy as np
import pandas as pd

from src.data_loader import load_rta_data, SEVERITY_COLORS, SEVERITY_ORDER
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PRESET_SCENARIOS,
)
from src.train_models import (
    load_saved_models_and_artifacts,
    check_models_exist,
    get_cached_metrics,
)
from src.visualizations import (
    apply_custom_css,
    render_page_header,
    render_kpi_card,
    plot_probability_breakdown,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
)

st.set_page_config(
    page_title="Severity Prediction | RTA Analytics",
    page_icon="🎯",
    layout="wide",
)

apply_custom_css()

render_page_header(
    title="Real-Time Accident Severity Prediction",
    subtitle="Simulate collision scenarios across 25 environmental, vehicle, and driver features using standard Scikit-Learn pipelines.",
    badge="Inference Engine",
)

# Load primary dataset to dynamically extract feature category choices
df_raw = load_rta_data()

# Ensure trained models are present
if not check_models_exist():
    st.error(
        "Trained models not found. Please navigate to the Model Comparison page or "
        "execute model training to initialize the pipelines."
    )
    st.stop()

models, X_test, y_test, comparison_df, best_model_name = load_saved_models_and_artifacts()
metrics_cache = get_cached_metrics()

# Quick Preset Scenarios Selector
st.markdown(
    """
    <div style="background: rgba(28, 37, 65, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
        <span style="color: #06b6d4; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">
            ⚡ Quick-Load Scenario Presets:
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

p_col1, p_col2, p_col3, p_col4 = st.columns(4)

# Initialize session state for input fields
if "input_state" not in st.session_state:
    st.session_state.input_state = PRESET_SCENARIOS["Low-Risk Urban Daylight Incident"].copy()

with p_col1:
    if st.button("🏙️ Low-Risk Urban Daylight", use_container_width=True):
        st.session_state.input_state = PRESET_SCENARIOS["Low-Risk Urban Daylight Incident"].copy()
        st.rerun()

with p_col2:
    if st.button("🛣️ High-Risk Night Highway", use_container_width=True):
        st.session_state.input_state = PRESET_SCENARIOS["High-Risk Night Highway Incident"].copy()
        st.rerun()

with p_col3:
    if st.button("⚠️ Drunk & Reckless Driving", use_container_width=True):
        st.session_state.input_state = PRESET_SCENARIOS["Critical Reckless & Impaired Driving Scenario"].copy()
        st.rerun()

with p_col4:
    if st.button("🚶 Pedestrian Night Conflict", use_container_width=True):
        st.session_state.input_state = PRESET_SCENARIOS["Pedestrian Conflict in Low Visibility"].copy()
        st.rerun()

# Model Selector & Selected Model Evaluation Display
st.markdown("<br>", unsafe_allow_html=True)

# Format model choices to highlight the best model
model_options = list(models.keys())
default_idx = model_options.index(best_model_name) if best_model_name in model_options else 0

c_model, c_metrics = st.columns([1.3, 2.2])

with c_model:
    selected_model_name = st.selectbox(
        "Select Classification Engine:",
        model_options,
        index=default_idx,
        help="Four syllabus-friendly algorithms trained on identical stratified partitions.",
    )

with c_metrics:
    # Display the selected model's actual name and evaluation results
    sel_metrics = metrics_cache.get(selected_model_name, {})
    if sel_metrics:
        m_acc = sel_metrics.get("accuracy", 0) * 100
        m_bal = sel_metrics.get("balanced_accuracy", 0) * 100
        m_f1 = sel_metrics.get("macro_f1", 0)
        is_best = " ⭐ Selected Best Model" if selected_model_name == best_model_name else ""
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 10px 16px; margin-top: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                    <span style="color: #ffffff; font-weight: 700;">{selected_model_name}{is_best}</span>
                    <span style="color: #94a3b8;">
                        Macro F1: <b style="color: #06b6d4;">{m_f1:.4f}</b> &nbsp;|&nbsp; 
                        Balanced Acc: <b style="color: #38bdf8;">{m_bal:.1f}%</b> &nbsp;|&nbsp; 
                        Accuracy: <b style="color: #ffffff;">{m_acc:.1f}%</b>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Helper function to get categorical options safely
def get_options(col_name: str) -> list:
    opts = sorted([str(x) for x in df_raw[col_name].dropna().unique().tolist()])
    return opts

# Main Input Form: 3 Distinct Responsive Columns
form_col1, form_col2, form_col3 = st.columns(3)

# Column 1: Driver Profile
with form_col1:
    st.markdown(
        """
        <div class="rta-card" style="margin-bottom: 12px; padding: 14px 18px;">
            <h4 style="color: #06b6d4; margin: 0; font-size: 1rem;">👤 Driver Profile</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    age_opts = get_options("Age_band_of_driver")
    cur_age = st.session_state.input_state.get("Age_band_of_driver", age_opts[0])
    age_idx = age_opts.index(cur_age) if cur_age in age_opts else 0
    age_band = st.selectbox("Age Band of Driver", age_opts, index=age_idx)

    sex_opts = get_options("Sex_of_driver")
    cur_sex = st.session_state.input_state.get("Sex_of_driver", sex_opts[0])
    sex_idx = sex_opts.index(cur_sex) if cur_sex in sex_opts else 0
    sex_driver = st.selectbox("Sex of Driver", sex_opts, index=sex_idx)

    edu_opts = get_options("Educational_level")
    cur_edu = st.session_state.input_state.get("Educational_level", edu_opts[0])
    edu_idx = edu_opts.index(cur_edu) if cur_edu in edu_opts else 0
    edu_level = st.selectbox("Educational Level", edu_opts, index=edu_idx)

    rel_opts = get_options("Vehicle_driver_relation")
    cur_rel = st.session_state.input_state.get("Vehicle_driver_relation", rel_opts[0])
    rel_idx = rel_opts.index(cur_rel) if cur_rel in rel_opts else 0
    driver_relation = st.selectbox("Vehicle Driver Relation", rel_opts, index=rel_idx)

    exp_opts = get_options("Driving_experience")
    cur_exp = st.session_state.input_state.get("Driving_experience", exp_opts[0])
    exp_idx = exp_opts.index(cur_exp) if cur_exp in exp_opts else 0
    experience = st.selectbox("Driving Experience", exp_opts, index=exp_idx)

# Column 2: Road & Environmental Conditions
with form_col2:
    st.markdown(
        """
        <div class="rta-card" style="margin-bottom: 12px; padding: 14px 18px;">
            <h4 style="color: #38bdf8; margin: 0; font-size: 1rem;">🛣️ Road & Environment</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    day_opts = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    cur_day = st.session_state.input_state.get("Day_of_week", "Monday")
    day_idx = day_opts.index(cur_day) if cur_day in day_opts else 0
    day_of_week = st.selectbox("Day of Week", day_opts, index=day_idx)

    hour_val = int(st.session_state.input_state.get("Hour", 14))
    hour = st.slider("Hour of Collision (0 - 23)", min_value=0, max_value=23, value=hour_val)

    area_opts = get_options("Area_accident_occured")
    cur_area = st.session_state.input_state.get("Area_accident_occured", area_opts[0])
    area_idx = area_opts.index(cur_area) if cur_area in area_opts else 0
    area_accident = st.selectbox("Area Accident Occurred", area_opts, index=area_idx)

    lanes_opts = get_options("Lanes_or_Medians")
    cur_lanes = st.session_state.input_state.get("Lanes_or_Medians", lanes_opts[0])
    lanes_idx = lanes_opts.index(cur_lanes) if cur_lanes in lanes_opts else 0
    lanes = st.selectbox("Lanes or Medians", lanes_opts, index=lanes_idx)

    align_opts = get_options("Road_allignment")
    cur_align = st.session_state.input_state.get("Road_allignment", align_opts[0])
    align_idx = align_opts.index(cur_align) if cur_align in align_opts else 0
    road_alignment = st.selectbox("Road Alignment", align_opts, index=align_idx)

    junc_opts = get_options("Types_of_Junction")
    cur_junc = st.session_state.input_state.get("Types_of_Junction", junc_opts[0])
    junc_idx = junc_opts.index(cur_junc) if cur_junc in junc_opts else 0
    junction = st.selectbox("Types of Junction", junc_opts, index=junc_idx)

    surf_opts = get_options("Road_surface_type")
    cur_surf = st.session_state.input_state.get("Road_surface_type", surf_opts[0])
    surf_idx = surf_opts.index(cur_surf) if cur_surf in surf_opts else 0
    surface = st.selectbox("Road Surface Type", surf_opts, index=surf_idx)

    cond_opts = get_options("Road_surface_conditions")
    cur_cond = st.session_state.input_state.get("Road_surface_conditions", cond_opts[0])
    cond_idx = cond_opts.index(cur_cond) if cur_cond in cond_opts else 0
    surface_conditions = st.selectbox("Surface Conditions", cond_opts, index=cond_idx)

    light_opts = get_options("Light_conditions")
    cur_light = st.session_state.input_state.get("Light_conditions", light_opts[0])
    light_idx = light_opts.index(cur_light) if cur_light in light_opts else 0
    light = st.selectbox("Light Conditions", light_opts, index=light_idx)

    weath_opts = get_options("Weather_conditions")
    cur_weath = st.session_state.input_state.get("Weather_conditions", weath_opts[0])
    weath_idx = weath_opts.index(cur_weath) if cur_weath in weath_opts else 0
    weather = st.selectbox("Weather Conditions", weath_opts, index=weath_idx)

# Column 3: Vehicle & Conflict Mechanics
with form_col3:
    st.markdown(
        """
        <div class="rta-card" style="margin-bottom: 12px; padding: 14px 18px;">
            <h4 style="color: #14b8a6; margin: 0; font-size: 1rem;">💥 Collision Dynamics</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    num_veh = int(st.session_state.input_state.get("Number_of_vehicles_involved", 2))
    num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=num_veh)

    num_cas = int(st.session_state.input_state.get("Number_of_casualties", 1))
    num_casualties = st.number_input("Number of Casualties", min_value=1, max_value=15, value=num_cas)

    vtype_opts = get_options("Type_of_vehicle")
    cur_vtype = st.session_state.input_state.get("Type_of_vehicle", vtype_opts[0])
    vtype_idx = vtype_opts.index(cur_vtype) if cur_vtype in vtype_opts else 0
    vehicle_type = st.selectbox("Type of Vehicle", vtype_opts, index=vtype_idx)

    owner_opts = get_options("Owner_of_vehicle")
    cur_owner = st.session_state.input_state.get("Owner_of_vehicle", owner_opts[0])
    owner_idx = owner_opts.index(cur_owner) if cur_owner in owner_opts else 0
    vehicle_owner = st.selectbox("Owner of Vehicle", owner_opts, index=owner_idx)

    syear_opts = get_options("Service_year_of_vehicle")
    cur_syear = st.session_state.input_state.get("Service_year_of_vehicle", syear_opts[0])
    syear_idx = syear_opts.index(cur_syear) if cur_syear in syear_opts else 0
    service_year = st.selectbox("Service Year of Vehicle", syear_opts, index=syear_idx)

    defect_opts = get_options("Defect_of_vehicle")
    cur_defect = st.session_state.input_state.get("Defect_of_vehicle", defect_opts[0])
    defect_idx = defect_opts.index(cur_defect) if cur_defect in defect_opts else 0
    defect_vehicle = st.selectbox("Defect of Vehicle", defect_opts, index=defect_idx)

    coll_opts = get_options("Type_of_collision")
    cur_coll = st.session_state.input_state.get("Type_of_collision", coll_opts[0])
    coll_idx = coll_opts.index(cur_coll) if cur_coll in coll_opts else 0
    collision_type = st.selectbox("Type of Collision", coll_opts, index=coll_idx)

    move_opts = get_options("Vehicle_movement")
    cur_move = st.session_state.input_state.get("Vehicle_movement", move_opts[0])
    move_idx = move_opts.index(cur_move) if cur_move in move_opts else 0
    movement = st.selectbox("Vehicle Movement", move_opts, index=move_idx)

    ped_opts = get_options("Pedestrian_movement")
    cur_ped = st.session_state.input_state.get("Pedestrian_movement", ped_opts[0])
    ped_idx = ped_opts.index(cur_ped) if cur_ped in ped_opts else 0
    pedestrian = st.selectbox("Pedestrian Movement", ped_opts, index=ped_idx)

    cause_opts = get_options("Cause_of_accident")
    cur_cause = st.session_state.input_state.get("Cause_of_accident", cause_opts[0])
    cause_idx = cause_opts.index(cur_cause) if cur_cause in cause_opts else 0
    cause = st.selectbox("Cause of Accident", cause_opts, index=cause_idx)

st.markdown("<br>", unsafe_allow_html=True)

# Build current payload matching the 25 features expected by the pipeline
current_payload = {
    "Number_of_vehicles_involved": int(num_vehicles),
    "Number_of_casualties": int(num_casualties),
    "Hour": int(hour),
    "Day_of_week": day_of_week,
    "Age_band_of_driver": age_band,
    "Sex_of_driver": sex_driver,
    "Educational_level": edu_level,
    "Vehicle_driver_relation": driver_relation,
    "Driving_experience": experience,
    "Type_of_vehicle": vehicle_type,
    "Owner_of_vehicle": vehicle_owner,
    "Service_year_of_vehicle": service_year,
    "Defect_of_vehicle": defect_vehicle,
    "Area_accident_occured": area_accident,
    "Lanes_or_Medians": lanes,
    "Road_allignment": road_alignment,
    "Types_of_Junction": junction,
    "Road_surface_type": surface,
    "Road_surface_conditions": surface_conditions,
    "Light_conditions": light,
    "Weather_conditions": weather,
    "Type_of_collision": collision_type,
    "Vehicle_movement": movement,
    "Pedestrian_movement": pedestrian,
    "Cause_of_accident": cause,
}

# Run Prediction CTA
btn_predict = st.button("🚀 Run Severity Inference & Risk Assessment", type="primary", use_container_width=True)

if btn_predict:
    selected_pipeline = models[selected_model_name]
    input_df = pd.DataFrame([current_payload])

    try:
        # Pass directly into the self-contained pipeline
        pred_label = selected_pipeline.predict(input_df)[0]

        # Probabilities
        if hasattr(selected_pipeline, "predict_proba"):
            probs = selected_pipeline.predict_proba(input_df)[0]
            classes = list(selected_pipeline.classes_)
            # Map probabilities to explicit SEVERITY_ORDER
            prob_dict = {cls_name: prob for cls_name, prob in zip(classes, probs)}
            ordered_probs = np.array([prob_dict.get(c, 0.0) for c in SEVERITY_ORDER])
        else:
            ordered_probs = np.array([0.0, 0.0, 0.0])
            if pred_label in SEVERITY_ORDER:
                ordered_probs[SEVERITY_ORDER.index(pred_label)] = 1.0

        p_fatal = ordered_probs[0]
        p_serious = ordered_probs[1]
        p_slight = ordered_probs[2]

        # Composite Hazard Risk Index (0 - 100)
        risk_score = min(100.0, (p_fatal * 100.0 * 2.5) + (p_serious * 100.0 * 0.8) + (p_slight * 100.0 * 0.1))

        # Dynamic Styling for Result Banner
        if pred_label == "Fatal injury":
            border_col = COLOR_ROSE
            badge_bg = "rgba(239, 68, 68, 0.15)"
            threat_level = "High Hazard (Tier 1 Priority)"
        elif pred_label == "Serious Injury":
            border_col = COLOR_AMBER
            badge_bg = "rgba(245, 158, 11, 0.15)"
            threat_level = "Elevated Hazard (Tier 2 Priority)"
        else:
            border_col = COLOR_CYAN
            badge_bg = "rgba(6, 182, 212, 0.15)"
            threat_level = "Low-to-Moderate Hazard (Tier 3 Priority)"

        st.markdown(
            f"""
            <div class="rta-card" style="border-left: 6px solid {border_col}; background: linear-gradient(135deg, rgba(28, 37, 65, 0.98), rgba(15, 23, 42, 0.95)); margin-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="background: {badge_bg}; color: {border_col}; border: 1px solid {border_col}; padding: 3px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">
                            {threat_level}
                        </span>
                        <h2 style="color: #ffffff; margin: 8px 0 4px 0; font-size: 1.6rem; letter-spacing: -0.02em;">
                            Forecasted Severity: <span style="color: {border_col};">{pred_label}</span>
                        </h2>
                        <div style="color: #94a3b8; font-size: 0.88rem;">
                            Algorithm: <b style="color: #ffffff;">{selected_model_name}</b> &nbsp;|&nbsp; 
                            Confidence: <b style="color: #ffffff;">{ordered_probs[SEVERITY_ORDER.index(pred_label)] * 100:.1f}%</b>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Composite Risk Index</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {border_col};">{risk_score:.1f}<span style="font-size: 1rem; color: #64748b;"> / 100</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Output Breakdown Details
        res_col1, res_col2 = st.columns([1.3, 1])

        with res_col1:
            fig_probs = plot_probability_breakdown(ordered_probs, SEVERITY_ORDER)
            st.plotly_chart(fig_probs, use_container_width=True)

        with res_col2:
            st.markdown(
                """
                <div class="rta-card" style="height: 250px; overflow-y: auto;">
                    <h4 style="color: #06b6d4; margin-top: 0; font-size: 0.95rem;">
                        ⚠️ Active High-Risk Factor Flags
                    </h4>
                    <ul style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                """,
                unsafe_allow_html=True,
            )

            risk_flags = []
            if light in ["Darkness - no lighting", "Darkness - lights unlit"]:
                risk_flags.append(f"<b>Lighting Hazard:</b> Driving under {light.lower()} severely impairs braking distance.")
            if cause in ["Drunk driving", "Driving under the influence of drugs", "Driving at high speed"]:
                risk_flags.append(f"<b>Behavioral Violation:</b> {cause} represents acute kinetic danger.")
            if experience in ["Below 1yr", "No Licence"]:
                risk_flags.append(f"<b>Driver Inexperience:</b> Driver has {experience} experience.")
            if collision_type in ["Rollover", "Collision with pedestrians"]:
                risk_flags.append(f"<b>High-Impact Collision:</b> {collision_type} causes extreme trauma transfer.")
            if num_casualties >= 2:
                risk_flags.append(f"<b>Mass Casualty Load:</b> {num_casualties} victims involved in collision.")

            if not risk_flags:
                st.markdown("<li>No extreme hazard flags triggered in this specific configuration. Baseline risk applies.</li>", unsafe_allow_html=True)
            else:
                for flag in risk_flags:
                    st.markdown(f"<li>{flag}</li>", unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)

        # Operational Guidance
        st.markdown(
            """
            <div class="rta-card" style="margin-top: 14px;">
                <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">
                    🛡️ Emergency Response & Traffic Management Guidance
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; font-size: 0.82rem; color: #94a3b8;">
                    <div>
                        <b style="color: #06b6d4;">Dispatch Recommendation:</b><br>
                        Deploy emergency medical responders equipped with advanced life support if fatal or serious injury probability exceeds 25%.
                    </div>
                    <div>
                        <b style="color: #06b6d4;">Traffic Engineering Countermeasure:</b><br>
                        Evaluate road segment for physical speed dampeners, median dividers, or high-luminance street lighting retrofits.
                    </div>
                    <div>
                        <b style="color: #06b6d4;">Enforcement Focus:</b><br>
                        Increase automated radar cameras and sobriety checkpoints during evening hours along similar arterial segments.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Inference pipeline encountered an unexpected issue: {str(e)}")
