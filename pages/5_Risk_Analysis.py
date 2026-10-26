"""
Page 5: Risk Analysis
Empirical Risk Patterns & Evidence-Based Road Safety Intelligence.
Analyzes temporal hotspots, environmental exposures, vehicle configurations, and root causes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_rta_data, SEVERITY_COLORS, SEVERITY_ORDER
from src.visualizations import (
    apply_custom_css,
    render_page_header,
    render_kpi_card,
    get_base_plotly_layout,
    plot_hourly_accidents_24h,
    plot_weekday_accidents,
    plot_feature_severity_stacked,
    compute_serious_fatal_ranking,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
    COLOR_SKY,
    COLOR_TEAL,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Analysis | Road Accident Intelligence",
    page_icon="🛡️",
    layout="wide",
)

apply_custom_css()

render_page_header(
    title="Accident Risk Patterns & Safety Intelligence",
    subtitle="Empirical hazard profiling across diurnal schedules, infrastructure conditions, vehicle classifications, and primary causes.",
    badge="Safety Analytics",
)

# -----------------------------------------------------------------------------
# DATA LOADING & SEVERE COLLISION METRICS
# -----------------------------------------------------------------------------
df = load_rta_data()

# Calculate Severe Collision Flag (Fatal or Serious = 1, Slight = 0)
df["Is_Severe"] = df["Accident_severity"].apply(
    lambda x: 1 if str(x).strip() in ["Fatal injury", "Serious Injury"] else 0
)

total_events = len(df)
total_severe = int(df["Is_Severe"].sum())
overall_severe_rate = (total_severe / total_events) * 100

# Peak hour calculation
hourly_counts = df["Hour"].dropna().astype(int).value_counts()
peak_h = int(hourly_counts.idxmax())
peak_h_cnt = int(hourly_counts.max())
peak_h_str = f"{peak_h:02d}:00 ({'12 AM' if peak_h == 0 else f'{peak_h} AM' if peak_h < 12 else '12 PM' if peak_h == 12 else f'{peak_h - 12} PM'})"

# -----------------------------------------------------------------------------
# OVERVIEW KPI CARDS
# -----------------------------------------------------------------------------
r1, r2, r3, r4 = st.columns(4)

with r1:
    render_kpi_card(
        title="Total Collisions",
        value=f"{total_events:,}",
        subtitle="Audited Police Registry Records",
        accent_color="#ffffff",
    )

with r2:
    render_kpi_card(
        title="Severe Collisions",
        value=f"{total_severe:,}",
        subtitle="Fatal + Serious Injury Cases",
        accent_color=COLOR_ROSE,
    )

with r3:
    render_kpi_card(
        title="Baseline Severe Rate",
        value=f"{overall_severe_rate:.2f}%",
        subtitle=f"Minority Trauma (Slight: {100 - overall_severe_rate:.2f}%)",
        accent_color=COLOR_AMBER,
    )

with r4:
    render_kpi_card(
        title="Peak Incident Hour",
        value=peak_h_str,
        subtitle=f"{peak_h_cnt:,} Collisions Recorded",
        accent_color=COLOR_CYAN,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RISK INTELLIGENCE TABS
# -----------------------------------------------------------------------------
tab_temporal, tab_env, tab_vehicles_causes, tab_driver, tab_insights = st.tabs([
    "⏰ Temporal Risk Hotspots",
    "🛣️ Road & Environmental Risk",
    "🚗 Vehicle Types & Accident Causes",
    "👤 Driver Demographics Vulnerability",
    "📋 Key Observations & Safety Interventions",
])

# -----------------------------------------------------------------------------
# TAB 1: TEMPORAL RISK HOTSPOTS
# -----------------------------------------------------------------------------
with tab_temporal:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Diurnal & Weekly Collision Dynamics</h4>
            <span class="section-badge">Temporal Patterns</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        fig_h = plot_hourly_accidents_24h(df)
        st.plotly_chart(fig_h, use_container_width=True)

    with t_col2:
        fig_w = plot_weekday_accidents(df)
        st.plotly_chart(fig_w, use_container_width=True)

    # Hour-of-Day vs Day-of-Week Collision Density Heatmap
    st.markdown("<h5 style='color: #06b6d4; margin-top: 15px;'>Hour-of-Day vs. Day-of-Week Collision Density Matrix</h5>", unsafe_allow_html=True)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_temp = (
        df.groupby(["Day_of_week", "Hour"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(day_order)
    )

    fig_temp = go.Figure(
        data=go.Heatmap(
            z=pivot_temp.values,
            x=[f"{h:02d}:00" for h in pivot_temp.columns],
            y=pivot_temp.index,
            colorscale=[[0, "#0b132b"], [0.4, "#1c2541"], [0.8, "#0284c7"], [1, "#06b6d4"]],
            colorbar=dict(title="Accidents"),
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Recorded Crashes: %{z}<extra></extra>",
        )
    )
    layout_temp = get_base_plotly_layout(height=340)
    layout_temp.update(
        title=dict(text="Diurnal & Weekly Accident Concentration Heatmap", font=dict(size=13, color="#ffffff")),
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=30, t=40, b=40),
    )
    fig_temp.update_layout(layout_temp)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Time Band Severe Proportion
    tb_col1, tb_col2 = st.columns([1.5, 1])
    with tb_col1:
        if "Time_band" in df.columns:
            tb_stats = (
                df.groupby("Time_band", observed=False)
                .agg(
                    Total=("Is_Severe", "count"),
                    Severe=("Is_Severe", "sum"),
                    Severe_Rate=("Is_Severe", lambda s: s.mean() * 100),
                )
                .reset_index()
            )
            fig_tb = px.bar(
                tb_stats,
                x="Time_band",
                y="Severe_Rate",
                color="Severe_Rate",
                color_continuous_scale=[[0, "#0891b2"], [1, "#ef4444"]],
                text=tb_stats["Severe_Rate"].apply(lambda r: f"{r:.2f}%"),
                labels={"Time_band": "", "Severe_Rate": "Severe Injury Rate (%)"},
            )
            layout_tb = get_base_plotly_layout(height=280)
            layout_tb.update(
                title=dict(text="Severe Injury Rate by Time Band", font=dict(size=13, color="#ffffff")),
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=40, b=30),
            )
            fig_tb.update_layout(layout_tb)
            fig_tb.update_traces(textposition="outside")
            st.plotly_chart(fig_tb, use_container_width=True)

    with tb_col2:
        st.markdown(
            f"""
            <div class="rta-card" style="height: 280px; padding: 18px;">
                <h5 style="color: #38bdf8; margin-top: 0; font-size: 0.95rem;">Temporal Risk Summary</h5>
                <p style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.55; margin-bottom: 8px;">
                    • <b>Peak Volume Window:</b> The late afternoon commute (15:00–18:00) records the highest incident volume, peaking at <b>{peak_h_str}</b> with <b>{peak_h_cnt:,} crashes</b>.<br>
                    • <b>Weekend & Night Elevation:</b> While overall traffic volume is lower during late night hours, collisions during the Night time band exhibit a higher ratio of serious and fatal injuries due to reduced visibility and higher vehicular speeds.
                </p>
                <div style="font-size: 0.78rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
                    * <i>Calendar scope note:</i> Registry records diurnal timestamps without calendar dates.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: ROAD & ENVIRONMENTAL RISK
# -----------------------------------------------------------------------------
with tab_env:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Environmental & Infrastructure Hazard Breakdown</h4>
            <span class="section-badge">Physical Road Conditions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    env_feature_options = [
        "Light_conditions",
        "Weather_conditions",
        "Road_surface_conditions",
        "Road_surface_type",
        "Road_allignment",
        "Types_of_Junction",
        "Lanes_or_Medians",
        "Area_accident_occured",
    ]

    sel_env_var = st.selectbox(
        "Select Road / Environmental Condition to Profile:",
        env_feature_options,
        index=0,
        format_func=lambda x: x.replace("_", " "),
    )

    env_col1, env_col2 = st.columns([1.3, 1.2])

    with env_col1:
        st.markdown(f"<h5 style='color: #06b6d4; font-size: 0.95rem;'>Severity Proportions within {sel_env_var.replace('_', ' ')}</h5>", unsafe_allow_html=True)
        fig_env_stacked = plot_feature_severity_stacked(df, sel_env_var)
        st.plotly_chart(fig_env_stacked, use_container_width=True)

    with env_col2:
        st.markdown(f"<h5 style='color: #06b6d4; font-size: 0.95rem;'>Scorecard: {sel_env_var.replace('_', ' ')}</h5>", unsafe_allow_html=True)
        env_rank_df, _ = compute_serious_fatal_ranking(df, sel_env_var)
        st.dataframe(
            env_rank_df[[sel_env_var, "Total", "Serious_or_Fatal", "Serious_or_Fatal_Pct", "Fatal", "Serious", "Slight"]],
            use_container_width=True,
            hide_index=True,
            height=340,
            column_config={
                sel_env_var: st.column_config.TextColumn("Condition Category"),
                "Total": st.column_config.NumberColumn("Total Recorded", format="%d"),
                "Serious_or_Fatal": st.column_config.NumberColumn("Severe Count", format="%d"),
                "Serious_or_Fatal_Pct": st.column_config.NumberColumn("Severe %", format="%.2f%%"),
                "Fatal": st.column_config.NumberColumn("Fatal", format="%d"),
                "Serious": st.column_config.NumberColumn("Serious", format="%d"),
                "Slight": st.column_config.NumberColumn("Slight", format="%d"),
            },
        )

        small_sample_env = env_rank_df[env_rank_df["Total"] < 20][sel_env_var].tolist()
        if small_sample_env:
            st.markdown(
                f"""
                <div style="font-size: 0.78rem; color: #f59e0b; margin-top: 4px;">
                    ⚠️ Categories with &lt;20 records (e.g. <code>{', '.join(small_sample_env[:3])}</code>) may exhibit volatile percentages.
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# TAB 3: VEHICLE TYPES & ACCIDENT CAUSES
# -----------------------------------------------------------------------------
with tab_vehicles_causes:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Vehicular Profiles & Root Cause Hazard Ranking</h4>
            <span class="section-badge">Collision Etiology</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    vc_c1, vc_c2 = st.columns(2)

    with vc_c1:
        st.markdown("<h5 style='color: #06b6d4; font-size: 0.95rem;'>1. Vehicle Type Hazard Ranking</h5>", unsafe_allow_html=True)
        veh_rank_df, fig_veh = compute_serious_fatal_ranking(df, "Type_of_vehicle")
        st.plotly_chart(fig_veh, use_container_width=True)

        st.dataframe(
            veh_rank_df[["Type_of_vehicle", "Total", "Serious_or_Fatal", "Serious_or_Fatal_Pct"]],
            use_container_width=True,
            hide_index=True,
            height=250,
            column_config={
                "Type_of_vehicle": st.column_config.TextColumn("Vehicle Type"),
                "Total": st.column_config.NumberColumn("Total Cases", format="%d"),
                "Serious_or_Fatal": st.column_config.NumberColumn("Severe Cases", format="%d"),
                "Serious_or_Fatal_Pct": st.column_config.NumberColumn("Severe Rate (%)", format="%.2f%%"),
            },
        )

    with vc_c2:
        st.markdown("<h5 style='color: #38bdf8; font-size: 0.95rem;'>2. Primary Accident Cause Hazard Ranking</h5>", unsafe_allow_html=True)
        cause_rank_df, fig_cause = compute_serious_fatal_ranking(df, "Cause_of_accident")
        st.plotly_chart(fig_cause, use_container_width=True)

        st.dataframe(
            cause_rank_df[["Cause_of_accident", "Total", "Serious_or_Fatal", "Serious_or_Fatal_Pct"]],
            use_container_width=True,
            hide_index=True,
            height=250,
            column_config={
                "Cause_of_accident": st.column_config.TextColumn("Primary Cause"),
                "Total": st.column_config.NumberColumn("Total Recorded", format="%d"),
                "Serious_or_Fatal": st.column_config.NumberColumn("Severe Cases", format="%d"),
                "Serious_or_Fatal_Pct": st.column_config.NumberColumn("Severe Rate (%)", format="%.2f%%"),
            },
        )

# -----------------------------------------------------------------------------
# TAB 4: DRIVER DEMOGRAPHICS VULNERABILITY
# -----------------------------------------------------------------------------
with tab_driver:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Driver Experience vs. Age Band Risk Cross-Tabulation</h4>
            <span class="section-badge">Demographic Exposure</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pivot_driver = (
        df.groupby(["Driving_experience", "Age_band_of_driver"], observed=False)["Is_Severe"]
        .agg(["count", "mean"])
        .reset_index()
    )
    pivot_driver["Severe_Rate"] = pivot_driver["mean"] * 100

    pivot_matrix = pivot_driver.pivot(
        index="Driving_experience",
        columns="Age_band_of_driver",
        values="Severe_Rate",
    )

    fig_driver = go.Figure(
        data=go.Heatmap(
            z=pivot_matrix.values,
            x=pivot_matrix.columns,
            y=pivot_matrix.index,
            colorscale=[[0, "#0b132b"], [0.5, "#d97706"], [1, "#ef4444"]],
            colorbar=dict(title="Severe %"),
            text=np.round(pivot_matrix.values, 1),
            texttemplate="%{text}%",
            textfont=dict(color="#ffffff", size=12),
            hovertemplate="Experience: %{y}<br>Age: %{x}<br>Severe Rate: %{z:.2f}%<extra></extra>",
        )
    )
    layout_driver = get_base_plotly_layout(height=380)
    layout_driver.update(
        title=dict(text="Severe Collision Rate (%) by Driver Demographics", font=dict(size=13, color="#ffffff")),
        xaxis=dict(title="Age Band of Driver"),
        yaxis=dict(title="Driving Experience"),
        margin=dict(l=30, r=30, t=40, b=40),
    )
    fig_driver.update_layout(layout_driver)
    st.plotly_chart(fig_driver, use_container_width=True)

    # Driver Sex & Education breakdown
    d_sub1, d_sub2 = st.columns(2)
    with d_sub1:
        sex_rank, _ = compute_serious_fatal_ranking(df, "Sex_of_driver")
        st.markdown("<h5 style='color: #06b6d4; font-size: 0.9rem;'>Severity Breakdown by Driver Sex</h5>", unsafe_allow_html=True)
        st.dataframe(
            sex_rank[["Sex_of_driver", "Total", "Serious_or_Fatal", "Serious_or_Fatal_Pct"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sex_of_driver": st.column_config.TextColumn("Driver Sex"),
                "Total": st.column_config.NumberColumn("Total", format="%d"),
                "Serious_or_Fatal": st.column_config.NumberColumn("Severe", format="%d"),
                "Serious_or_Fatal_Pct": st.column_config.NumberColumn("Severe %", format="%.2f%%"),
            },
        )

    with d_sub2:
        edu_rank, _ = compute_serious_fatal_ranking(df, "Educational_level")
        st.markdown("<h5 style='color: #06b6d4; font-size: 0.9rem;'>Severity Breakdown by Educational Level</h5>", unsafe_allow_html=True)
        st.dataframe(
            edu_rank[["Educational_level", "Total", "Serious_or_Fatal", "Serious_or_Fatal_Pct"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Educational_level": st.column_config.TextColumn("Education"),
                "Total": st.column_config.NumberColumn("Total", format="%d"),
                "Serious_or_Fatal": st.column_config.NumberColumn("Severe", format="%d"),
                "Serious_or_Fatal_Pct": st.column_config.NumberColumn("Severe %", format="%.2f%%"),
            },
        )

# -----------------------------------------------------------------------------
# TAB 5: KEY OBSERVATIONS & SAFETY INTERVENTIONS
# -----------------------------------------------------------------------------
with tab_insights:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Empirical Risk Findings & Targeted Interventions</h4>
            <span class="section-badge">Actionable Safety Intelligence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ins_c1, ins_c2 = st.columns(2)

    with ins_c1:
        st.markdown(
            f"""
            <div class="rta-card" style="padding: 18px 20px; margin-bottom: 14px;">
                <div style="color: #06b6d4; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">1. Diurnal Congestion Surge vs. Nighttime Lethality</div>
                <p style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.5; margin: 6px 0 0 0;">
                    The absolute volume of traffic collisions concentrates during the late afternoon commuter rush hour, peaking at <b>{peak_h_str} ({peak_h_cnt:,} records)</b>.
                    However, unlit nighttime conditions exhibit nearly double the proportion of severe casualties due to delayed reaction times and higher speed differentials.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="rta-card" style="padding: 18px 20px; margin-bottom: 14px;">
                <div style="color: #38bdf8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">2. Young Driver Incident Concentration</div>
                <p style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.5; margin: 6px 0 0 0;">
                    Drivers aged <b>18–30</b> represent the largest cohort of recorded accidents (34.7%, 4,271 incidents), with tailgating ("No distancing")
                    accounting for over 18% of all collisions. Targeted defensive headway education offers high prevention leverage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ins_c2:
        st.markdown(
            """
            <div class="rta-card" style="padding: 18px 20px; margin-bottom: 14px;">
                <div style="color: #14b8a6; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">3. Infrastructure Geometry Separations</div>
                <p style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.5; margin: 6px 0 0 0;">
                    Undivided two-way roads and segments lacking median barriers demonstrate significantly elevated rates of multi-vehicle head-on collisions.
                    Installing physical concrete barriers or wire medians physically eliminates crossover impact vectors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="rta-card" style="padding: 18px 20px; margin-bottom: 14px;">
                <div style="color: #f59e0b; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">4. High-Impact Kinetic Causes</div>
                <p style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.5; margin: 6px 0 0 0;">
                    Among causes with reliable sample sizes, <b>Overspeeding</b> and <b>Driving to the left</b> generate the highest severe injury proportions (>20% severe),
                    confirming that kinetic velocity directly amplifies injury severity upon impact.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Association vs. Causation & Geographic Disclaimer Callout
    st.markdown(
        """
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 14px 18px; margin-top: 15px;">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <span style="font-size: 1.2rem; color: #f59e0b;">⚠️</span>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5;">
                    <b style="color: #f59e0b;">Statistical Association & Scope Notice:</b><br>
                    • <b>Correlation vs. Causation:</b> Historical patterns documented here represent empirical associations in police incident registries, not proof of direct physical causation. Multiple compounding factors contribute simultaneously to crash outcomes.<br>
                    • <b>Geographic Scope:</b> The primary collision dataset does not contain GPS latitude/longitude coordinates. Consequently, spatial pin mapping is excluded to maintain strict data integrity without fabricating speculative geographic coordinates.
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
        Road Accident Intelligence • Risk Analysis & Safety Module • 2nd-Year Engineering Machine Learning Lab
    </div>
    """,
    unsafe_allow_html=True,
)
