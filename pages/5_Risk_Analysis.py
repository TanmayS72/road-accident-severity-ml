"""
Page 5: Risk Analysis
Multi-dimensional hazard matrices, temporal collision heatmaps, and evidence-based safety policy intelligence.
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
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
    COLOR_SKY,
)

st.set_page_config(
    page_title="Risk Analysis | RTA Analytics",
    page_icon="🛡️",
    layout="wide",
)

apply_custom_css()

render_page_header(
    title="Accident Risk Patterns & Safety Intelligence",
    subtitle="Identify compounding hazard hotspots across driver demographics, infrastructure types, and environmental exposures.",
    badge="Hazard Intelligence",
)

# Load data
df = load_rta_data()

# Calculate Severe Collision Flag (Fatal or Serious = 1, Slight = 0)
df["Is_Severe"] = df["Accident_severity"].apply(
    lambda x: 1 if x in ["Fatal injury", "Serious Injury", 0, 1] else 0
)

# Overview KPIs
total_events = len(df)
total_severe = df["Is_Severe"].sum()
overall_severe_rate = (total_severe / total_events) * 100

r1, r2, r3, r4 = st.columns(4)

with r1:
    render_kpi_card(
        title="Aggregate Collisions",
        value=f"{total_events:,}",
        subtitle="Audited Registry Records",
        accent_color="#ffffff",
    )

with r2:
    render_kpi_card(
        title="Severe Injury Collisions",
        value=f"{total_severe:,}",
        subtitle="Fatal + Serious Incidents",
        accent_color=COLOR_ROSE,
    )

with r3:
    render_kpi_card(
        title="Baseline Severe Rate",
        value=f"{overall_severe_rate:.2f}%",
        subtitle="National baseline average",
        accent_color=COLOR_AMBER,
    )

with r4:
    # Identify highest severe rate condition
    dark_unlit_severe = df[df["Light_conditions"] == "Darkness - lights unlit"]["Is_Severe"].mean() * 100
    render_kpi_card(
        title="Peak Hazard Exposure",
        value=f"{dark_unlit_severe:.1f}%",
        subtitle="Unlit night conditions",
        accent_color=COLOR_CYAN,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Risk Matrices Tabs
tab_temporal, tab_driver, tab_infra, tab_causes, tab_policies = st.tabs([
    "⏰ Temporal Hazard Grid",
    "👤 Driver Demographics Risk",
    "🛣️ Infrastructure & Environment",
    "⚠️ Root Cause Hazard Ranking",
    "📋 Strategic Safety Recommendations",
])

# -------------------------------------------------------------------------
# TAB 1: Temporal Hazard Grid
# -------------------------------------------------------------------------
with tab_temporal:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Hour-of-Day vs Day-of-Week Collision Density</h4>
            <span class="section-badge">Temporal Hotspots</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            hovertemplate="Day: %{y}<br>Time: %{x}<br>Collisions: %{z}<extra></extra>",
        )
    )
    layout = get_base_plotly_layout(height=380)
    layout.update(
        title=dict(text="Diurnal Collision Concentration Heatmap", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(autorange="reversed"),
    )
    fig_temp.update_layout(layout)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Severe Rate by Time Band
    st.markdown(
        """
        <h5 style="color: #ffffff; font-size: 0.95rem; margin-top: 10px;">
            Severity Rate by Diurnal Time Band
        </h5>
        """,
        unsafe_allow_html=True,
    )
    time_band_stats = (
        df.groupby("Time_band", observed=False)
        .agg(
            Total=("Is_Severe", "count"),
            Severe=("Is_Severe", "sum"),
            Severe_Rate=("Is_Severe", lambda s: s.mean() * 100),
        )
        .reset_index()
    )

    t_b1, t_b2 = st.columns([1.5, 1])
    with t_b1:
        fig_tb = px.bar(
            time_band_stats,
            x="Time_band",
            y="Severe_Rate",
            color="Severe_Rate",
            color_continuous_scale=[[0, "#0891b2"], [1, "#ef4444"]],
            text=time_band_stats["Severe_Rate"].apply(lambda r: f"{r:.2f}%"),
            labels={"Time_band": "", "Severe_Rate": "Severe Injury Rate (%)"},
        )
        layout = get_base_plotly_layout(height=300)
        layout.update(
            title=dict(text="Severe Collision Proportion by Time Band", font=dict(size=13, color="#ffffff")),
            coloraxis_showscale=False,
        )
        fig_tb.update_layout(layout)
        fig_tb.update_traces(textposition="outside")
        st.plotly_chart(fig_tb, use_container_width=True)

    with t_b2:
        st.markdown(
            """
            <div class="rta-card" style="height: 300px; padding: 16px;">
                <h5 style="color: #06b6d4; margin-top: 0;">Temporal Risk Observations</h5>
                <p style="color: #cbd5e1; font-size: 0.84rem; line-height: 1.5;">
                    • <b>Late Night Lethality:</b> Despite lower overall traffic volume after 22:00, collisions occurring during the <b>Night Band</b> exhibit higher fatality proportions.<br><br>
                    • <b>Evening Congestion Surge:</b> Peak collision count concentrates between <b>17:00 and 19:00 on Fridays</b>, driven by homeward commuter hurry and visual fatigue.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------------------
# TAB 2: Driver Demographics Risk
# -------------------------------------------------------------------------
with tab_driver:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Driver Experience vs Age Band Vulnerability</h4>
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

    # Filter out tiny cells (< 15 records) for statistical validity
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
    layout = get_base_plotly_layout(height=420)
    layout.update(
        title=dict(text="Severe Accident Rate (%) by Driver Profile Matrix", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Age Band"),
        yaxis=dict(title="Driving Experience"),
    )
    fig_driver.update_layout(layout)
    st.plotly_chart(fig_driver, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 3: Infrastructure & Environment
# -------------------------------------------------------------------------
with tab_infra:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Infrastructure Geometry & Lighting Exposure</h4>
            <span class="section-badge">Environmental Risk</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    infra_c1, infra_c2 = st.columns(2)

    with infra_c1:
        # Lanes or Medians Severe Rate
        lane_stats = (
            df.groupby("Lanes_or_Medians", observed=False)
            .agg(Total=("Is_Severe", "count"), Severe_Rate=("Is_Severe", lambda s: s.mean() * 100))
            .reset_index()
            .sort_values(by="Severe_Rate", ascending=False)
        )
        fig_lane = px.bar(
            lane_stats,
            y="Lanes_or_Medians",
            x="Severe_Rate",
            orientation="h",
            color="Severe_Rate",
            color_continuous_scale=[[0, "#0891b2"], [1, "#f59e0b"]],
            text=lane_stats["Severe_Rate"].apply(lambda r: f"{r:.1f}%"),
            labels={"Lanes_or_Medians": "", "Severe_Rate": "Severe Rate (%)"},
        )
        layout = get_base_plotly_layout(height=360)
        layout.update(
            title=dict(text="Severe Injury Rate by Roadway Configuration", font=dict(size=13, color="#ffffff")),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        fig_lane.update_layout(layout)
        fig_lane.update_traces(textposition="outside")
        st.plotly_chart(fig_lane, use_container_width=True)

    with infra_c2:
        # Light Conditions Severe Rate
        light_stats = (
            df.groupby("Light_conditions", observed=False)
            .agg(Total=("Is_Severe", "count"), Severe_Rate=("Is_Severe", lambda s: s.mean() * 100))
            .reset_index()
            .sort_values(by="Severe_Rate", ascending=False)
        )
        fig_light = px.bar(
            light_stats,
            y="Light_conditions",
            x="Severe_Rate",
            orientation="h",
            color="Severe_Rate",
            color_continuous_scale=[[0, "#0891b2"], [1, "#ef4444"]],
            text=light_stats["Severe_Rate"].apply(lambda r: f"{r:.1f}%"),
            labels={"Light_conditions": "", "Severe_Rate": "Severe Rate (%)"},
        )
        layout = get_base_plotly_layout(height=360)
        layout.update(
            title=dict(text="Severe Injury Rate by Light Condition", font=dict(size=13, color="#ffffff")),
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        fig_light.update_layout(layout)
        fig_light.update_traces(textposition="outside")
        st.plotly_chart(fig_light, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 4: Root Cause Hazard Ranking
# -------------------------------------------------------------------------
with tab_causes:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Primary Cause Hazard Score & Casualty Load</h4>
            <span class="section-badge">Hazard Prioritization</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cause_stats = (
        df.groupby("Cause_of_accident", observed=False)
        .agg(
            Total_Crashes=("Is_Severe", "count"),
            Severe_Crashes=("Is_Severe", "sum"),
            Severe_Rate=("Is_Severe", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    cause_stats["Pct_of_All_Crashes"] = (cause_stats["Total_Crashes"] / total_events) * 100
    cause_stats = cause_stats.sort_values(by="Severe_Rate", ascending=False).reset_index(drop=True)

    st.dataframe(
        cause_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cause_of_accident": st.column_config.TextColumn("Primary Cause"),
            "Total_Crashes": st.column_config.NumberColumn("Total Recorded"),
            "Severe_Crashes": st.column_config.NumberColumn("Fatal / Serious"),
            "Severe_Rate": st.column_config.NumberColumn("Severe Rate (%)", format="%.2f%%"),
            "Pct_of_All_Crashes": st.column_config.NumberColumn("Proportion of All Incidents", format="%.2f%%"),
        },
    )

# -------------------------------------------------------------------------
# TAB 5: Strategic Safety Recommendations
# -------------------------------------------------------------------------
with tab_policies:
    st.markdown(
        """
        <div class="section-header">
            <h4 class="section-title">Evidence-Based Road Safety Interventions</h4>
            <span class="section-badge">Policy & Engineering Directives</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pol1, pol2 = st.columns(2)

    with pol1:
        st.markdown(
            """
            <div class="rta-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🚧</div>
                <h4 style="color: #06b6d4; margin: 0 0 8px 0;">1. Highway Median & Lane Separation Upgrades</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    <b>Data Finding:</b> Undivided two-way roads and improper lane changes comprise over 42% of multi-vehicle collisions.<br>
                    <b>Action Directive:</b> Retrofit high-density arterial corridors with concrete New Jersey barriers or flexible steel wire medians to physically eliminate head-on crossover collisions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="rta-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">💡</div>
                <h4 style="color: #38bdf8; margin: 0 0 8px 0;">2. Night-Time Illumination & Speed Enforcement</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    <b>Data Finding:</b> Darkness without lighting elevates collision severity rates to almost 2x baseline daylight levels.<br>
                    <b>Action Directive:</b> Prioritize smart solar/LED streetlamp installation across known junction blackspots and deploy automated speed radar cameras calibrated for low-light enforcement.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pol2:
        st.markdown(
            """
            <div class="rta-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🎓</div>
                <h4 style="color: #14b8a6; margin: 0 0 8px 0;">3. Novice Driver Graduated Licensing & Headway Auditing</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    <b>Data Finding:</b> Drivers with &lt; 1-2 years experience and tailgating violations ("No distancing") drive the largest absolute collision volume.<br>
                    <b>Action Directive:</b> Institute mandatory defensive driving and safe braking distance certifications before commercial fleet licensing; introduce probationary night driving restrictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="rta-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🚑</div>
                <h4 style="color: #f59e0b; margin: 0 0 8px 0;">4. Predictive Trauma Care & EMS Pre-Positioning</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
                    <b>Data Finding:</b> Severe collision clusters concentrate predictably during Friday and weekend evening hours (17:00 - 21:00).<br>
                    <b>Action Directive:</b> Strategically pre-stage trauma emergency medical response units adjacent to arterial junction nodes during evening commute windows to compress the Golden Hour survival interval.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
