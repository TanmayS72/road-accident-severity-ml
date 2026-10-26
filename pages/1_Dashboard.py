"""
Page 1: Dashboard
Road Accident Intelligence: Operational metrics, incident distributions,
pattern analytics, syllabus model performance preview, and automated data insights.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_rta_data,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
)
from src.train_models import (
    check_models_exist,
    load_saved_models_and_artifacts,
    get_cached_metrics,
)
from src.visualizations import (
    apply_custom_css,
    render_kpi_card,
    plot_severity_donut,
    plot_hourly_accidents_24h,
    plot_top_5_causes,
    plot_dashboard_model_preview,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
    COLOR_TEAL,
    COLOR_SKY,
)

# Page configuration
st.set_page_config(
    page_title="Dashboard | Road Accident Intelligence",
    page_icon="📈",
    layout="wide",
)

# Apply cohesive dark navy and slate styling
apply_custom_css()

# -----------------------------------------------------------------------------
# 1. DATA INGESTION & DYNAMIC SYSTEM STATUS
# -----------------------------------------------------------------------------
try:
    df_raw = load_rta_data()
    dataset_loaded = True
    dataset_name = "RTA Dataset.csv"
    total_records = len(df_raw)
except Exception as e:
    df_raw = pd.DataFrame()
    dataset_loaded = False
    dataset_name = "Unavailable"
    total_records = 0

# Check trained models and evaluation artifacts dynamically
models_ready = check_models_exist()
trained_models_count = 0
comparison_df = None
best_model_name = "N/A"
best_macro_f1 = None

if models_ready:
    try:
        models, X_test, y_test, comparison_df, best_model_name = load_saved_models_and_artifacts()
        trained_models_count = len(models)
        if comparison_df is not None and not comparison_df.empty and "Macro F1" in comparison_df.columns:
            best_row = comparison_df.loc[comparison_df["Macro F1"].idxmax()]
            best_model_name = str(best_row["Model"])
            best_macro_f1 = float(best_row["Macro F1"])
    except Exception:
        models_ready = False

# -----------------------------------------------------------------------------
# 2. HEADER SECTION & DYNAMIC STATUS BADGES
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <h1 style="color: #ffffff; font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                Road Accident Intelligence
            </h1>
        </div>
        <div style="color: #06b6d4; font-size: 1.05rem; font-weight: 600; margin-bottom: 10px;">
            Machine Learning-Based Severity Prediction & Risk Analysis
        </div>
        <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5; margin: 0 0 16px 0; max-width: 950px;">
            This application analyzes historical road accident records and compares machine learning classifiers
            to predict recorded accident severity, assess collision patterns, and deliver decision support for road safety management.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compact Status Indicators (Determined Dynamically)
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if dataset_loaded:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 7px 12px; border-radius: 6px;">
                <span style="color: #10b981; font-weight: 600; font-size: 0.82rem;">✓ Primary Dataset Loaded</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">Source: data/{dataset_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; padding: 7px 12px; border-radius: 6px;">
                <span style="color: #ef4444; font-weight: 600; font-size: 0.82rem;">✕ Dataset Missing</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">data/RTA Dataset.csv not found</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with status_col2:
    st.markdown(
        f"""
        <div style="background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; padding: 7px 12px; border-radius: 6px;">
            <span style="color: #06b6d4; font-weight: 600; font-size: 0.82rem;">📊 Total Records Available</span><br>
            <span style="color: #94a3b8; font-size: 0.75rem;">{total_records:,} Documented Collisions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col3:
    if models_ready and trained_models_count > 0:
        st.markdown(
            f"""
            <div style="background: rgba(20, 184, 166, 0.1); border-left: 3px solid #14b8a6; padding: 7px 12px; border-radius: 6px;">
                <span style="color: #14b8a6; font-weight: 600; font-size: 0.82rem;">⚡ {trained_models_count} Trained Classifiers Ready</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">KNN, Logistic Reg, Decision Tree, Random Forest</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 7px 12px; border-radius: 6px;">
                <span style="color: #f59e0b; font-weight: 600; font-size: 0.82rem;">⚠️ Models Not Initialized</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">Run train_models.py to initialize</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

if not dataset_loaded or df_raw.empty:
    st.error("Cannot render dashboard: Primary accident dataset is unavailable.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS (Optional Exploration Slice)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 Interactive Filters")
    st.markdown(
        "<span style='color: #94a3b8; font-size: 0.78rem;'>Filter records to inspect specific operational subsets.</span>",
        unsafe_allow_html=True,
    )

    days = ["All Days"] + [d for d in df_raw["Day_of_week"].dropna().unique().tolist() if str(d).strip()]
    selected_day = st.selectbox("Day of Week", days)

    weathers = ["All Weather"] + sorted([str(w) for w in df_raw["Weather_conditions"].dropna().unique() if str(w).strip()])
    selected_weather = st.selectbox("Weather Condition", weathers)

    lights = ["All Lighting"] + sorted([str(l) for l in df_raw["Light_conditions"].dropna().unique() if str(l).strip()])
    selected_light = st.selectbox("Light Condition", lights)

    surfaces = ["All Surfaces"] + sorted([str(s) for s in df_raw["Road_surface_type"].dropna().unique() if str(s).strip()])
    selected_surface = st.selectbox("Road Surface Type", surfaces)

    st.markdown("---")
    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()

# Apply Filters
filtered_df = df_raw.copy()

if selected_day != "All Days":
    filtered_df = filtered_df[filtered_df["Day_of_week"] == selected_day]

if selected_weather != "All Weather":
    filtered_df = filtered_df[filtered_df["Weather_conditions"] == selected_weather]

if selected_light != "All Lighting":
    filtered_df = filtered_df[filtered_df["Light_conditions"] == selected_light]

if selected_surface != "All Surfaces":
    filtered_df = filtered_df[filtered_df["Road_surface_type"] == selected_surface]

filtered_count = len(filtered_df)
if filtered_count == 0:
    st.warning("No records match the selected filter combination. Please reset your filters.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. KEY PERFORMANCE INDICATORS (4 CARDS)
# -----------------------------------------------------------------------------
# Severity counts derived from target column Accident_severity
sev_counts = filtered_df["Accident_severity"].value_counts().to_dict()
slight_count = sev_counts.get("Slight Injury", 0)
serious_count = sev_counts.get("Serious Injury", 0)
fatal_count = sev_counts.get("Fatal injury", 0)

# Card 1: Total Accidents
actual_total_accidents = len(df_raw) if selected_day == "All Days" and selected_weather == "All Weather" and selected_light == "All Lighting" and selected_surface == "All Surfaces" else filtered_count
total_subtitle = "Historical Collisions in RTA Dataset" if filtered_count == len(df_raw) else f"Filtered Subset ({filtered_count:,} of {len(df_raw):,})"

# Card 2: Slight Injury Rate
slight_rate = (slight_count / filtered_count) * 100 if filtered_count > 0 else 0.0

# Card 3: Serious and Fatal Accidents (Combined)
combined_serious_fatal = serious_count + fatal_count
combined_sf_rate = (combined_serious_fatal / filtered_count) * 100 if filtered_count > 0 else 0.0

# Card 4: Best Model Macro F1
if best_macro_f1 is not None:
    f1_display = f"{best_macro_f1:.4f}"
    f1_subtitle = f"Top Classifier: {best_model_name}"
else:
    f1_display = "N/A"
    f1_subtitle = "Evaluation results pending"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    render_kpi_card(
        title="Total Accidents",
        value=f"{actual_total_accidents:,}",
        subtitle=total_subtitle,
        accent_color="#ffffff",
    )

with kpi2:
    render_kpi_card(
        title="Slight Injury Rate",
        value=f"{slight_rate:.2f}%",
        subtitle=f"{slight_count:,} recorded cases",
        accent_color=COLOR_CYAN,
    )

with kpi3:
    render_kpi_card(
        title="Serious & Fatal Accidents",
        value=f"{combined_serious_fatal:,}",
        subtitle=f"{combined_sf_rate:.2f}% ({serious_count:,} serious + {fatal_count:,} fatal)",
        accent_color=COLOR_AMBER,
    )

with kpi4:
    render_kpi_card(
        title="Best Model Macro F1",
        value=f1_display,
        subtitle=f1_subtitle,
        accent_color=COLOR_TEAL,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ACCIDENT SEVERITY DISTRIBUTION & 5. ACCIDENT PATTERN ANALYSIS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Collision Severity & Diurnal Pattern Analysis</h3>
        <span class="section-badge">Distribution & Telemetry</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_col1, chart_col2 = st.columns([1, 1.4])

with chart_col1:
    # 4. Large Interactive Plotly Donut Chart
    fig_donut = plot_severity_donut(filtered_df)
    st.plotly_chart(fig_donut, use_container_width=True)

with chart_col2:
    # 5. Chart A — Accidents by Hour (Chronologically sorted 0 to 23)
    fig_hourly = plot_hourly_accidents_24h(filtered_df)
    st.plotly_chart(fig_hourly, use_container_width=True)

# Chart B — Top 5 Accident Causes (Full width or paired)
st.markdown("<br>", unsafe_allow_html=True)
c_causes, c_info = st.columns([1.5, 1])

with c_causes:
    # 5. Chart B — Top 5 Accident Causes
    fig_causes = plot_top_5_causes(filtered_df)
    st.plotly_chart(fig_causes, use_container_width=True)

with c_info:
    # Supplementary context regarding top accident cause distribution
    top_cause_series = filtered_df["Cause_of_accident"].dropna().value_counts()
    if not top_cause_series.empty:
        top_cause_name = top_cause_series.index[0]
        top_cause_cnt = top_cause_series.iloc[0]
        top_cause_pct = (top_cause_cnt / filtered_count) * 100
    else:
        top_cause_name = "N/A"
        top_cause_cnt = 0
        top_cause_pct = 0.0

    st.markdown(
        f"""
        <div class="rta-card" style="height: 340px; padding: 20px 22px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Leading Collision Cause
            </div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #06b6d4; margin-bottom: 8px;">
                {top_cause_name}
            </div>
            <p style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.55; margin-bottom: 12px;">
                In the selected dataset, <b>{top_cause_cnt:,} collisions</b> ({top_cause_pct:.1f}%) were recorded under <b>{top_cause_name}</b>, making it the most frequent contributing factor.
            </p>
            <div style="font-size: 0.82rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                Headway maintenance violations and improper lane transitions together account for the majority of multi-vehicle collisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 6. MODEL PERFORMANCE PREVIEW
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Model Performance Preview</h3>
        <span class="section-badge">Benchmarking 4 Classifiers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if comparison_df is not None and not comparison_df.empty:
    m_chart_col, m_table_col = st.columns([1.5, 1])

    with m_chart_col:
        # Grouped Plotly Bar Chart comparing Macro F1, Accuracy, Balanced Accuracy
        fig_model_preview = plot_dashboard_model_preview(comparison_df)
        st.plotly_chart(fig_model_preview, use_container_width=True)

    with m_table_col:
        st.markdown(
            f"""
            <div class="rta-card" style="height: 360px; overflow-y: auto; padding: 18px 20px;">
                <h4 style="color: #06b6d4; margin: 0 0 8px 0; font-size: 1rem;">
                    Held-Out Test Set Metrics
                </h4>
                <p style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 12px;">
                    Evaluated on 2,464 test collisions (20% stratified test split).
                </p>
            """,
            unsafe_allow_html=True,
        )

        # Show concise scorecard table
        display_cols = [c for c in ["Model", "Macro F1", "Accuracy", "Balanced Accuracy"] if c in comparison_df.columns]
        st.dataframe(
            comparison_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Macro F1": st.column_config.NumberColumn(format="%.4f"),
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Balanced Accuracy": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.markdown(
            f"""
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    ⭐ <b>Selected Model:</b> <b style="color: #ffffff;">{best_model_name}</b> achieved the highest Macro F1-score (<b>{best_macro_f1:.4f}</b>), optimizing sensitivity on minority fatal and serious cases.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info(
        "Model evaluation results are not currently loaded. "
        "Please visit the Model Comparison page or run model training to generate the performance scorecard."
    )

# -----------------------------------------------------------------------------
# 7. AUTOMATIC DATA INSIGHTS (3 Concise Dynamic Findings)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Automated Telemetry Insights</h3>
        <span class="section-badge">Data-Driven Observations</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dynamically compute the 3 insights from the actual dataset
# Insight 1: Most common accident severity
top_sev = sev_counts
most_common_sev = max(top_sev, key=top_sev.get) if top_sev else "Slight Injury"
most_common_sev_count = top_sev.get(most_common_sev, 0)
most_common_sev_pct = (most_common_sev_count / filtered_count) * 100 if filtered_count > 0 else 0.0

# Insight 2: Most frequently recorded accident cause
top_causes = filtered_df["Cause_of_accident"].dropna().value_counts()
top_cause_str = top_causes.index[0] if not top_causes.empty else "N/A"
top_cause_val = top_causes.iloc[0] if not top_causes.empty else 0
top_cause_pct_val = (top_cause_val / filtered_count) * 100 if filtered_count > 0 else 0.0

# Insight 3: Hour with highest recorded accident count
valid_hrs = filtered_df["Hour"].dropna().astype(int)
hourly_series = valid_hrs.value_counts()
if not hourly_series.empty:
    peak_h = hourly_series.idxmax()
    peak_h_cnt = hourly_series.max()
    peak_h_pct = (peak_h_cnt / filtered_count) * 100 if filtered_count > 0 else 0.0
    peak_h_label = f"{peak_h:02d}:00 ({'12 AM' if peak_h == 0 else f'{peak_h} AM' if peak_h < 12 else '12 PM' if peak_h == 12 else f'{peak_h - 12} PM'})"
else:
    peak_h_label = "17:00 (5 PM)"
    peak_h_cnt = 0
    peak_h_pct = 0.0

in_col1, in_col2, in_col3 = st.columns(3)

with in_col1:
    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; height: 160px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #06b6d4; text-transform: uppercase; margin-bottom: 6px;">
                1. Severity Imbalance Observation
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                {most_common_sev} Dominance
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                Most recorded accidents resulted in <b>{most_common_sev}</b>, accounting for <b>{most_common_sev_pct:.1f}%</b> ({most_common_sev_count:,} records), reflecting pronounced class skew.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with in_col2:
    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; height: 160px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #38bdf8; text-transform: uppercase; margin-bottom: 6px;">
                2. Leading Incident Factor
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                "{top_cause_str}"
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                The most frequently documented accident cause in the registry was <b>{top_cause_str}</b>, representing <b>{top_cause_pct_val:.1f}%</b> ({top_cause_val:,} records) of all incidents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with in_col3:
    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; height: 160px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #14b8a6; text-transform: uppercase; margin-bottom: 6px;">
                3. Diurnal Peak Window
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                {peak_h_label} Collision Peak
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                Most recorded accidents occurred during the late afternoon and evening, with highest frequency recorded at <b>{peak_h_label}</b> ({peak_h_cnt:,} records, {peak_h_pct:.1f}%).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 8. PROJECT WORKFLOW (Compact Plain-English Flowchart)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">End-to-End Machine Learning Workflow</h3>
        <span class="section-badge">Standard 5-Stage Pipeline</span>
    </div>
    """,
    unsafe_allow_html=True,
)

wf1, wf2, wf3, wf4, wf5 = st.columns(5)

with wf1:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 14px 10px; height: 165px;">
            <div style="color: #06b6d4; font-weight: 700; font-size: 0.85rem; margin-bottom: 3px;">STAGE 1</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Historical Accident Data</div>
            <div style="color: #94a3b8; font-size: 0.74rem; line-height: 1.35;">
                Ingestion of 12,316 police-reported collision records with 32 environmental, vehicle, and driver attributes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with wf2:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 14px 10px; height: 165px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 0.85rem; margin-bottom: 3px;">STAGE 2</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Data Preprocessing</div>
            <div style="color: #94a3b8; font-size: 0.74rem; line-height: 1.35;">
                ColumnTransformer with median and mode imputation, One-Hot encoding, and scaling fitted strictly on train data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with wf3:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 14px 10px; height: 165px;">
            <div style="color: #14b8a6; font-weight: 700; font-size: 0.85rem; margin-bottom: 3px;">STAGE 3</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Model Training</div>
            <div style="color: #94a3b8; font-size: 0.74rem; line-height: 1.35;">
                Training 4 standard syllabus classifiers (KNN, Logistic Regression, Decision Tree, Random Forest) with class balancing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with wf4:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 14px 10px; height: 165px;">
            <div style="color: #f59e0b; font-weight: 700; font-size: 0.85rem; margin-bottom: 3px;">STAGE 4</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Model Evaluation</div>
            <div style="color: #94a3b8; font-size: 0.74rem; line-height: 1.35;">
                Benchmarking across Macro F1, Balanced Accuracy, confusion matrices, and minority-class sensitivity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with wf5:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 14px 10px; height: 165px;">
            <div style="color: #ef4444; font-weight: 700; font-size: 0.85rem; margin-bottom: 3px;">STAGE 5</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Severity Prediction</div>
            <div style="color: #94a3b8; font-size: 0.74rem; line-height: 1.35;">
                Real-time interactive incident forecasting with probability confidence, risk indexing, and safety guidance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer callout
st.markdown(
    """
    <div style="text-align: center; padding: 16px 0 6px 0; color: #64748b; font-size: 0.78rem;">
        Road Accident Intelligence • 2nd-Year Engineering Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
