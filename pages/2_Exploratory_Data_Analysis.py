"""
Page 2: Exploratory Data Analysis (EDA)
Discovering Patterns in Road Traffic Accidents: Comprehensive statistical auditing,
univariate and bivariate profiling, diurnal patterns, association rankings, and numerical analysis.
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
from src.visualizations import (
    apply_custom_css,
    render_kpi_card,
    get_base_plotly_layout,
    plot_severity_distribution,
    plot_hourly_accidents_24h,
    plot_weekday_accidents,
    plot_feature_distribution,
    plot_feature_severity_stacked,
    compute_serious_fatal_ranking,
    plot_numerical_correlation_matrix,
    plot_numerical_distribution_by_severity,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
    COLOR_TEAL,
    COLOR_SKY,
)

# Page Configuration
st.set_page_config(
    page_title="EDA | Road Accident Intelligence",
    page_icon="🔬",
    layout="wide",
)

apply_custom_css()

# Load primary raw dataset
try:
    df_raw = load_rta_data()
    dataset_loaded = True
except Exception as e:
    st.error(f"Failed to load dataset: {str(e)}")
    st.stop()

# Identify original CSV columns (excluding engineered Hour/Time_band for schema counts)
raw_csv_path = Path(__file__).resolve().parent.parent / "data" / "RTA Dataset.csv"
if raw_csv_path.exists():
    raw_csv_df = pd.read_csv(raw_csv_path)
    total_raw_cols = raw_csv_df.shape[1]
    total_raw_rows = raw_csv_df.shape[0]
    total_duplicates = int(raw_csv_df.duplicated().sum())
    total_missing_cells = int(raw_csv_df.isnull().sum().sum())
else:
    raw_csv_df = df_raw
    total_raw_cols = 32
    total_raw_rows = len(df_raw)
    total_duplicates = 0
    total_missing_cells = int(df_raw.drop(columns=["Hour", "Time_band"], errors="ignore").isnull().sum().sum())

# -----------------------------------------------------------------------------
# 1. PAGE HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <h1 style="color: #ffffff; font-size: 2.1rem; font-weight: 800; margin: 0 0 4px 0; letter-spacing: -0.02em;">
            Exploratory Data Analysis
        </h1>
        <div style="color: #06b6d4; font-size: 1.05rem; font-weight: 600; margin-bottom: 8px;">
            Discovering Patterns in Road Traffic Accidents
        </div>
        <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5; margin: 0; max-width: 950px;">
            Explore accident severity, driver characteristics, environmental conditions, road features,
            and recorded accident causes using historical data from the primary collision registry.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. DATASET OVERVIEW (4 KPI CARDS & AUDIT EXPANDER)
# -----------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    render_kpi_card(
        title="Total Records",
        value=f"{total_raw_rows:,}",
        subtitle="Historical Police Collision Cases",
        accent_color="#ffffff",
    )

with kpi_col2:
    render_kpi_card(
        title="Total Features",
        value=f"{total_raw_cols}",
        subtitle="Attributes in Original Registry",
        accent_color=COLOR_SKY,
    )

with kpi_col3:
    render_kpi_card(
        title="Missing Values",
        value=f"{total_missing_cells:,}",
        subtitle=f"{total_missing_cells / (total_raw_rows * total_raw_cols) * 100:.2f}% of total data matrix",
        accent_color=COLOR_AMBER,
    )

with kpi_col4:
    render_kpi_card(
        title="Duplicate Records",
        value=f"{total_duplicates}",
        subtitle="Zero Duplicate Rows in Raw Registry",
        accent_color="#10b981",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Expandable Dataset Schema & Missingness Audit
with st.expander("📋 Detailed Dataset Schema, Missingness Audit & Target Definition", expanded=False):
    st.markdown(
        """
        <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin-bottom: 14px;">
            <b style="color: #ffffff;">Data Integrity Note:</b> No records were deleted, modified, or imputed during exploratory analysis.
            The metrics below represent the unaltered historical collision registry.
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns([1.4, 1])

    with t_col1:
        st.markdown("<h5 style='color: #06b6d4; margin-top: 0;'>Attribute Schema & Missingness Audit</h5>", unsafe_allow_html=True)
        # Compute missingness per column
        audit_records = []
        for col in raw_csv_df.columns:
            m_count = int(raw_csv_df[col].isnull().sum())
            m_pct = (m_count / total_raw_rows) * 100
            audit_records.append({
                "Column Name": col,
                "Data Type": str(raw_csv_df[col].dtype),
                "Missing Count": m_count,
                "Missing Pct": m_pct,
                "Unique Values": int(raw_csv_df[col].nunique()),
            })
        audit_df = pd.DataFrame(audit_records)
        st.dataframe(
            audit_df,
            use_container_width=True,
            height=280,
            hide_index=True,
            column_config={
                "Missing Count": st.column_config.NumberColumn(format="%d"),
                "Missing Pct": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    with t_col2:
        st.markdown("<h5 style='color: #06b6d4; margin-top: 0;'>Target Variable: Accident_severity</h5>", unsafe_allow_html=True)
        sev_counts = df_raw["Accident_severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0)
        sev_audit_df = pd.DataFrame({
            "Severity Category": sev_counts.index,
            "Record Count": sev_counts.values.astype(int),
            "Percentage": (sev_counts.values / total_raw_rows) * 100,
        })
        st.dataframe(
            sev_audit_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Record Count": st.column_config.NumberColumn(format="%d"),
                "Percentage": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.45; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                <b>Target Explanation:</b> <code>Accident_severity</code> denotes the maximum injury trauma level sustained by any casualty in the collision event.
                The categories are defined by police traffic recording protocols:
                <b>Slight Injury</b> (minor cuts, bruises, ambulatory),
                <b>Serious Injury</b> (hospitalization, fractures, severe trauma), and
                <b>Fatal injury</b> (mortality within 30 days of the crash).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # First 15 Rows Preview
    st.markdown("<h5 style='color: #ffffff; margin-top: 14px;'>First 15 Registry Records</h5>", unsafe_allow_html=True)
    st.dataframe(raw_csv_df.head(15), use_container_width=True, height=260)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. ACCIDENT SEVERITY DISTRIBUTION (DONUT VS BAR SWITCHER)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Accident Severity Distribution</h3>
        <span class="section-badge">Target Imbalance Analysis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

sev_c1, sev_c2 = st.columns([1.6, 1])

with sev_c1:
    chart_view = st.radio(
        "Select Visualization Mode:",
        ["Donut Chart", "Bar Chart"],
        horizontal=True,
        label_visibility="collapsed",
    )
    chart_type_param = "bar" if chart_view == "Bar Chart" else "donut"
    fig_sev = plot_severity_distribution(df_raw, chart_type=chart_type_param)
    st.plotly_chart(fig_sev, use_container_width=True)

with sev_c2:
    st.markdown(
        """
        <div class="rta-card" style="height: 380px; padding: 22px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #06b6d4; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Class Imbalance Dynamics
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
                Skewed Injury Distribution
            </div>
            <p style="font-size: 0.86rem; color: #cbd5e1; line-height: 1.55; margin-bottom: 12px;">
                The dataset displays pronounced class imbalance: <b>84.57%</b> of recorded accidents (10,415 records) are minor (<b>Slight Injury</b>),
                while <b>14.15%</b> (1,743 records) are <b>Serious</b>, and only <b>1.28%</b> (158 records) are <b>Fatal</b>.
            </p>
            <div style="font-size: 0.82rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                <b>Modeling Consequence:</b> Classifiers that optimize only overall accuracy tend to predict the majority class exclusively.
                Evaluating balanced accuracy and macro F1-score is necessary to assess minority-class detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. DRIVER CHARACTERISTICS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Driver Characteristics</h3>
        <span class="section-badge">Demographics & Experience</span>
    </div>
    """,
    unsafe_allow_html=True,
)

driver_features = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Educational_level",
    "Vehicle_driver_relation",
]

sel_driver_feat = st.selectbox(
    "Select Driver Characteristic to Explore:",
    driver_features,
    index=0,
    format_func=lambda x: x.replace("_", " "),
)

d_col1, d_col2 = st.columns(2)

with d_col1:
    fig_d_dist = plot_feature_distribution(df_raw, sel_driver_feat)
    st.plotly_chart(fig_d_dist, use_container_width=True)

with d_col2:
    fig_d_stacked = plot_feature_severity_stacked(df_raw, sel_driver_feat)
    st.plotly_chart(fig_d_stacked, use_container_width=True)

# Missing values note for driver characteristics
missing_driver_cnt = int(df_raw[sel_driver_feat].isnull().sum())
if missing_driver_cnt > 0:
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: -10px; margin-bottom: 15px;">
            ℹ️ <b>Missing Values:</b> <code>{sel_driver_feat}</code> contains <b>{missing_driver_cnt:,} unrecorded entries</b> ({missing_driver_cnt / len(df_raw) * 100:.2f}%), labeled as 'Missing / Unknown'.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. ROAD AND ENVIRONMENTAL CONDITIONS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Road and Environmental Conditions</h3>
        <span class="section-badge">Infrastructure & Weather Factors</span>
    </div>
    """,
    unsafe_allow_html=True,
)

road_env_features = [
    "Light_conditions",
    "Weather_conditions",
    "Road_surface_conditions",
    "Types_of_Junction",
    "Road_allignment",
    "Lanes_or_Medians",
    "Area_accident_occured",
]

sel_env_feat = st.selectbox(
    "Select Road / Environmental Condition to Analyze:",
    road_env_features,
    index=0,
    format_func=lambda x: x.replace("_", " "),
)

e_col1, e_col2 = st.columns(2)

with e_col1:
    fig_e_dist = plot_feature_distribution(df_raw, sel_env_feat)
    st.plotly_chart(fig_e_dist, use_container_width=True)

with e_col2:
    fig_e_stacked = plot_feature_severity_stacked(df_raw, sel_env_feat)
    st.plotly_chart(fig_e_stacked, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. TIME-BASED ACCIDENT ANALYSIS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Time-Based Accident Analysis</h3>
        <span class="section-badge">Diurnal & Weekly Dynamics</span>
    </div>
    """,
    unsafe_allow_html=True,
)

t_chart1, t_chart2 = st.columns(2)

with t_chart1:
    fig_h24 = plot_hourly_accidents_24h(df_raw)
    st.plotly_chart(fig_h24, use_container_width=True)

with t_chart2:
    fig_wday = plot_weekday_accidents(df_raw)
    st.plotly_chart(fig_wday, use_container_width=True)

# Dynamic Peak Hour Observation & Month Scope Note
hourly_counts = df_raw["Hour"].dropna().astype(int).value_counts()
peak_h = int(hourly_counts.idxmax())
peak_h_cnt = int(hourly_counts.max())
peak_h_pct = (peak_h_cnt / len(df_raw)) * 100
peak_h_str = f"{peak_h:02d}:00 ({'12 AM' if peak_h == 0 else f'{peak_h} AM' if peak_h < 12 else '12 PM' if peak_h == 12 else f'{peak_h - 12} PM'})"

st.markdown(
    f"""
    <div class="rta-card" style="padding: 14px 18px; margin-top: 5px;">
        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;">
            ⏰ <b>Diurnal Observation:</b> Most recorded accidents occurred during the evening rush hour, with the highest collision frequency observed at <b>{peak_h_str}</b> ({peak_h_cnt:,} records, {peak_h_pct:.2f}% of total).<br>
            📅 <b>Calendar Scope Note:</b> The historical police incident registry records collision timestamps (HH:MM:SS) and day of the week, but does not record calendar dates or months. Temporal analysis is evaluated across the 24-hour diurnal cycle and weekly schedules without fabricating missing calendar variables.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. ACCIDENT CAUSES AND VEHICLE CHARACTERISTICS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Accident Causes and Vehicle Characteristics</h3>
        <span class="section-badge">Collision Dynamics</span>
    </div>
    """,
    unsafe_allow_html=True,
)

cause_veh_features = [
    "Cause_of_accident",
    "Type_of_collision",
    "Type_of_vehicle",
    "Vehicle_movement",
    "Number_of_vehicles_involved",
    "Number_of_casualties",
]

sel_cause_feat = st.selectbox(
    "Select Collision / Vehicular Attribute:",
    cause_veh_features,
    index=0,
    format_func=lambda x: x.replace("_", " "),
)

cv_col1, cv_col2 = st.columns([1.4, 1])

with cv_col1:
    fig_cv = plot_feature_distribution(df_raw, sel_cause_feat, top_n=10)
    st.plotly_chart(fig_cv, use_container_width=True)

with cv_col2:
    # Summary Table for the selected attribute
    clean_series = df_raw[sel_cause_feat].fillna("Missing / Unknown").astype(str)
    cv_counts = clean_series.value_counts().reset_index()
    cv_counts.columns = [sel_cause_feat.replace("_", " "), "Count"]
    cv_counts["Percentage"] = (cv_counts["Count"] / len(df_raw)) * 100

    st.markdown(
        f"""
        <div class="rta-card" style="height: 380px; overflow-y: auto; padding: 18px;">
            <h4 style="color: #06b6d4; margin: 0 0 8px 0; font-size: 1rem;">
                Frequency Summary: {sel_cause_feat.replace('_', ' ')}
            </h4>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        cv_counts,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Count": st.column_config.NumberColumn(format="%d"),
            "Percentage": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 8. SEVERITY COMPARISON (SERIOUS-OR-FATAL PERCENTAGE RANKING)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Severity Comparison & High-Risk Category Ranking</h3>
        <span class="section-badge">Serious & Fatal Injury Proportion</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.4; margin-bottom: 12px;">
        Compare injury severity across categories of any selected attribute.
        Categories are ranked by <b>Serious-or-Fatal Percentage</b>:
        <code>(Serious Injury Count + Fatal Injury Count) / Total Accidents × 100</code>.
    </p>
    """,
    unsafe_allow_html=True,
)

ranking_candidate_cols = [
    "Cause_of_accident",
    "Type_of_collision",
    "Driving_experience",
    "Age_band_of_driver",
    "Light_conditions",
    "Weather_conditions",
    "Road_surface_conditions",
    "Types_of_Junction",
    "Road_allignment",
    "Lanes_or_Medians",
    "Type_of_vehicle",
]

sel_rank_col = st.selectbox(
    "Select Feature for Serious-or-Fatal Severity Comparison:",
    ranking_candidate_cols,
    index=0,
    format_func=lambda x: x.replace("_", " "),
)

rank_df, fig_rank = compute_serious_fatal_ranking(df_raw, sel_rank_col)

r_col1, r_col2 = st.columns([1.4, 1])

with r_col1:
    st.plotly_chart(fig_rank, use_container_width=True)

with r_col2:
    st.markdown(
        f"""
        <div class="rta-card" style="height: {max(360, min(700, len(rank_df) * 32))}px; overflow-y: auto; padding: 18px;">
            <h4 style="color: #06b6d4; margin: 0 0 6px 0; font-size: 0.95rem;">
                Comparison Scorecard
            </h4>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 10px;">
                Actual counts and computed rates per category.
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        rank_df[[sel_rank_col, "Total", "Serious_or_Fatal", "Serious_or_Fatal_Pct", "Fatal", "Serious", "Slight"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            sel_rank_col: st.column_config.TextColumn("Category"),
            "Total": st.column_config.NumberColumn("Total", format="%d"),
            "Serious_or_Fatal": st.column_config.NumberColumn("Serious/Fatal", format="%d"),
            "Serious_or_Fatal_Pct": st.column_config.NumberColumn("Serious/Fatal %", format="%.2f%%"),
            "Fatal": st.column_config.NumberColumn("Fatal", format="%d"),
            "Serious": st.column_config.NumberColumn("Serious", format="%d"),
            "Slight": st.column_config.NumberColumn("Slight", format="%d"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Small sample size warning & Non-causal association explanation
small_sample_categories = rank_df[rank_df["Total"] < 20][sel_rank_col].tolist()
if small_sample_categories:
    st.markdown(
        f"""
        <div style="font-size: 0.8rem; color: #f59e0b; margin-top: 4px;">
            ⚠️ <b>Small Sample Warning:</b> The categories <code>{', '.join(small_sample_categories[:4])}</code> have fewer than 20 recorded incidents.
            Percentages calculated on small samples can be volatile.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="callout-info" style="margin-top: 10px;">
        📌 <b>Association vs. Causation:</b> These comparisons demonstrate observational empirical associations
        documented in historical accident logs. They do not demonstrate that an environmental or vehicle attribute directly caused
        injury severity. Compounding factors (such as speed, seatbelt usage, or infrastructure) contribute jointly to crash outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. NUMERICAL FEATURE ANALYSIS (ACTUAL NUMERICAL COLUMNS ONLY)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Numerical Feature Analysis</h3>
        <span class="section-badge">Continuous & Count Attributes</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Identify actual numerical columns dynamically
actual_num_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]

n_col1, n_col2 = st.columns([1.1, 1.3])

with n_col1:
    st.markdown(
        """
        <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 8px;">
            Pearson correlation matrix calculated strictly across actual numerical variables:
            <code>Number_of_vehicles_involved</code>, <code>Number_of_casualties</code>, and <code>Hour</code>.
            (Nominal categorical variables are excluded from the correlation matrix).
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_num_corr = plot_numerical_correlation_matrix(df_raw, actual_num_cols)
    st.plotly_chart(fig_num_corr, use_container_width=True)

with n_col2:
    sel_num_col = st.selectbox(
        "Select Numerical Attribute to Inspect by Severity Tier:",
        actual_num_cols,
        index=1 if "Number_of_casualties" in actual_num_cols else 0,
        format_func=lambda x: x.replace("_", " "),
    )
    fig_num_box = plot_numerical_distribution_by_severity(df_raw, sel_num_col)
    st.plotly_chart(fig_num_box, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 10. AUTOMATIC DATA INSIGHTS (3-5 DYNAMICALLY COMPUTED OBSERVATIONS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Automated Telemetry Insights</h3>
        <span class="section-badge">Data-Driven Observations</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Most common accident severity
top_sev_name = sev_counts.idxmax()
top_sev_count = int(sev_counts.max())
top_sev_pct = (top_sev_count / len(df_raw)) * 100

# 2. Most frequent recorded cause
top_cause_series = df_raw["Cause_of_accident"].dropna().value_counts()
top_cause_name = str(top_cause_series.index[0])
top_cause_count = int(top_cause_series.iloc[0])
top_cause_pct = (top_cause_count / len(df_raw)) * 100

# 3. Most common driver age group
top_age_series = df_raw["Age_band_of_driver"].dropna().value_counts()
top_age_name = str(top_age_series.index[0])
top_age_count = int(top_age_series.iloc[0])
top_age_pct = (top_age_count / len(df_raw)) * 100

# 4. Most frequent lighting condition
top_light_series = df_raw["Light_conditions"].dropna().value_counts()
top_light_name = str(top_light_series.index[0])
top_light_count = int(top_light_series.iloc[0])
top_light_pct = (top_light_count / len(df_raw)) * 100

# 5. Highest serious-or-fatal rate cause with at least 20 records
cause_ranking, _ = compute_serious_fatal_ranking(df_raw, "Cause_of_accident")
stable_causes = cause_ranking[cause_ranking["Total"] >= 20]
if not stable_causes.empty:
    top_sf_cause = stable_causes.iloc[0]["Cause_of_accident"]
    top_sf_rate = stable_causes.iloc[0]["Serious_or_Fatal_Pct"]
    top_sf_total = stable_causes.iloc[0]["Total"]
    top_sf_sf = stable_causes.iloc[0]["Serious_or_Fatal"]
else:
    top_sf_cause = "N/A"
    top_sf_rate = 0.0
    top_sf_total = 0
    top_sf_sf = 0

ins_col1, ins_col2 = st.columns(2)

with ins_col1:
    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 14px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #06b6d4; text-transform: uppercase; margin-bottom: 4px;">
                1. Target Severity Concentration
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                {top_sev_name} ({top_sev_pct:.1f}%)
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                Most recorded accidents resulted in <b>{top_sev_name}</b>, accounting for <b>{top_sev_count:,} out of {len(df_raw):,} cases</b> ({top_sev_pct:.2f}%).
                Fatal injuries constitute a small minority (158 records, 1.28%).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 14px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px;">
                2. Leading Primary Accident Cause
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                "{top_cause_name}" ({top_cause_pct:.1f}%)
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                The most frequently documented accident cause in the police registry was <b>{top_cause_name}</b> (following too closely),
                with <b>{top_cause_count:,} incidents</b> ({top_cause_pct:.2f}% of all recorded events).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #14b8a6; text-transform: uppercase; margin-bottom: 4px;">
                3. Primary Driver Demographics
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                Age Band {top_age_name} ({top_age_pct:.1f}%)
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                The driver age group with the highest collision representation was <b>{top_age_name} years</b>,
                accounting for <b>{top_age_count:,} collisions</b> ({top_age_pct:.2f}%), followed closely by drivers aged 31–50.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with ins_col2:
    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 14px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #f59e0b; text-transform: uppercase; margin-bottom: 4px;">
                4. Environmental Visibility
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                {top_light_name} ({top_light_pct:.1f}%)
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                The vast majority of collisions occurred during <b>{top_light_name}</b> ({top_light_count:,} records, {top_light_pct:.2f}%).
                However, unlit nighttime darkness exhibits a substantially higher serious-and-fatal injury rate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 14px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #ef4444; text-transform: uppercase; margin-bottom: 4px;">
                5. Diurnal Rush-Hour Peak
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                {peak_h_str} ({peak_h_cnt:,} records)
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                Most recorded collisions occurred in the late afternoon, peaking at <b>{peak_h_str}</b> with
                <b>{peak_h_cnt:,} recorded accidents</b>, coinciding with the evening commuter egress window.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rta-card" style="padding: 16px 18px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px;">
                6. Cause Severity Association
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                "{top_sf_cause}" ({top_sf_rate:.1f}% Serious/Fatal)
            </div>
            <p style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45; margin: 0;">
                Among accident causes with at least 20 documented events, <b>{top_sf_cause}</b> demonstrated the highest
                serious-or-fatal injury rate at <b>{top_sf_rate:.2f}%</b> ({top_sf_sf:,} of {top_sf_total:,} crashes).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer callout
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0 10px 0; color: #64748b; font-size: 0.78rem;">
        Road Accident Intelligence • Exploratory Data Analysis Module • 2nd-Year Engineering Machine Learning Lab
    </div>
    """,
    unsafe_allow_html=True,
)
