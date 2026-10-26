"""
Visualizations, styling, and Plotly interactive chart components.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import SEVERITY_COLORS, SEVERITY_ORDER, SEVERITY_MAP

# Core Palette Constants
COLOR_BG = "#0b132b"
COLOR_CARD = "#1c2541"
COLOR_CARD_BORDER = "rgba(56, 189, 248, 0.18)"
COLOR_TEXT_MAIN = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_CYAN = "#06b6d4"
COLOR_TEAL = "#14b8a6"
COLOR_SKY = "#38bdf8"
COLOR_AMBER = "#f59e0b"
COLOR_ROSE = "#ef4444"


def apply_custom_css():
    """Inject cohesive modern dark navy, slate, and cyan CSS styles."""
    css = f"""
    <style>
    /* Global Body & Backgrounds */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT_MAIN};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    /* Modern Card Container */
    .rta-card {{
        background-color: {COLOR_CARD};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .rta-card:hover {{
        border-color: rgba(6, 182, 212, 0.45);
    }}

    /* KPI Metric Cards */
    .kpi-container {{
        background: linear-gradient(135deg, rgba(28, 37, 65, 0.95), rgba(15, 23, 42, 0.85));
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    .kpi-title {{
        color: {COLOR_TEXT_MUTED};
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {COLOR_TEXT_MAIN};
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    .kpi-subtitle {{
        color: {COLOR_CYAN};
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 6px;
    }}

    /* Section Subheadings */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .section-title {{
        color: {COLOR_TEXT_MAIN};
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0;
    }}
    .section-badge {{
        background: rgba(6, 182, 212, 0.15);
        color: {COLOR_CYAN};
        border: 1px solid rgba(6, 182, 212, 0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    /* Hero Banner */
    .hero-banner {{
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }}
    .hero-title {{
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }}
    .hero-highlight {{
        color: {COLOR_CYAN};
    }}
    .hero-desc {{
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.5;
        max-width: 860px;
        margin-bottom: 0;
    }}

    /* Sidebar Customization */
    [data-testid="stSidebar"] {{
        background-color: #080e21;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }}
    [data-testid="stSidebarNav"] {{
        padding-top: 10px;
    }}

    /* Alert / Callout styles */
    .callout-info {{
        background: rgba(6, 182, 212, 0.08);
        border-left: 4px solid {COLOR_CYAN};
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #e2e8f0;
    }}
    .callout-warning {{
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid {COLOR_AMBER};
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #fef3c7;
    }}
    .callout-danger {{
        background: rgba(239, 68, 68, 0.08);
        border-left: 4px solid {COLOR_ROSE};
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #fecaca;
    }}

    /* Streamlit widget tweaks */
    div[data-baseweb="select"] > div {{
        background-color: {COLOR_CARD};
        border-color: rgba(255, 255, 255, 0.12);
        color: #ffffff;
    }}
    .stButton>button {{
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(6, 182, 212, 0.3);
        transition: all 0.15s ease-in-out;
    }}
    .stButton>button:hover {{
        border-color: {COLOR_CYAN};
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, badge: Optional[str] = None):
    """Render a modern standardized page header."""
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div style="margin-bottom: 22px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
                <h1 style="color: #ffffff; font-size: 1.85rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;">
                    {title}
                </h1>
                {badge_html}
            </div>
            <p style="color: {COLOR_TEXT_MUTED}; font-size: 0.98rem; margin: 0; line-height: 1.4;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: Any,
    subtitle: str = "",
    accent_color: str = COLOR_CYAN,
):
    """Render a KPI metric card with consistent styling."""
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="color: {accent_color};">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_base_plotly_layout(height: int = 400) -> Dict[str, Any]:
    """Return dark navy layout dictionary for Plotly charts."""
    return dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28, 37, 65, 0.4)",
        font=dict(color=COLOR_TEXT_MAIN, family="sans-serif", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
        ),
    )


def plot_severity_donut(df: pd.DataFrame) -> go.Figure:
    """Create an interactive donut chart of accident severity."""
    return plot_severity_distribution(df, chart_type="donut")


def plot_severity_distribution(df: pd.DataFrame, chart_type: str = "donut") -> go.Figure:
    """
    Create an interactive Plotly chart of accident severity distribution.
    Supports switching between 'donut' and 'bar' chart with identical color coding.
    """
    col = "Accident_severity" if "Accident_severity" in df.columns else "Severity_label"
    counts = df[col].value_counts().reindex(SEVERITY_ORDER).fillna(0).reset_index()
    counts.columns = ["Severity", "Count"]
    counts["Count"] = counts["Count"].astype(int)
    total = counts["Count"].sum()
    counts["Percentage"] = (counts["Count"] / total * 100) if total > 0 else 0.0

    colors = [SEVERITY_COLORS.get(s, COLOR_CYAN) for s in counts["Severity"]]

    if chart_type.lower() == "bar":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts["Severity"],
                    y=counts["Count"],
                    marker=dict(
                        color=colors,
                        line=dict(color="rgba(255, 255, 255, 0.15)", width=1.5),
                    ),
                    text=[
                        f"<b>{row['Count']:,}</b><br>({row['Percentage']:.2f}%)"
                        for _, row in counts.iterrows()
                    ],
                    textposition="outside",
                    textfont=dict(color="#ffffff", size=12),
                    hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata:.2f}%<extra></extra>",
                    customdata=counts["Percentage"],
                )
            ]
        )
        layout = get_base_plotly_layout(height=380)
        layout.update(
            title=dict(text="Accident Severity Distribution (Bar View)", font=dict(size=14, color="#ffffff")),
            xaxis=dict(title="Severity Category"),
            yaxis=dict(title="Incident Count", range=[0, max(counts["Count"]) * 1.18]),
            margin=dict(l=30, r=30, t=50, b=40),
        )
        fig.update_layout(layout)
        return fig
    else:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts["Severity"],
                    values=counts["Count"],
                    hole=0.58,
                    marker=dict(colors=colors, line=dict(color="#0b132b", width=2)),
                    textinfo="percent+label",
                    textfont=dict(size=12, color="#ffffff"),
                    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Proportion: %{percent}<extra></extra>",
                    sort=False,
                )
            ]
        )
        layout = get_base_plotly_layout(height=380)
        layout.update(
            title=dict(text="Accident Severity Breakdown (Donut View)", font=dict(size=14, color="#ffffff")),
            showlegend=False,
            annotations=[
                dict(
                    text=f"Total Records<br><b>{total:,}</b>",
                    x=0.5,
                    y=0.5,
                    font_size=14,
                    font_color="#ffffff",
                    showarrow=False,
                )
            ],
            margin=dict(l=30, r=30, t=50, b=40),
        )
        fig.update_layout(layout)
        return fig


def plot_hourly_accidents(df: pd.DataFrame) -> go.Figure:
    """Plot hourly accident distribution with stacked severity breakdown."""
    if "Hour" not in df.columns:
        fig = go.Figure()
        return fig

    # Group by Hour and Severity
    sev_col = "Severity_label" if "Severity_label" in df.columns else "Accident_severity"
    hourly_df = (
        df.groupby(["Hour", sev_col])
        .size()
        .reset_index(name="Count")
    )

    color_map = {
        "Fatal injury": COLOR_ROSE,
        "Serious Injury": COLOR_AMBER,
        "Slight Injury": COLOR_CYAN,
        0: COLOR_ROSE,
        1: COLOR_AMBER,
        2: COLOR_CYAN,
    }

    fig = px.bar(
        hourly_df,
        x="Hour",
        y="Count",
        color=sev_col,
        color_discrete_map=color_map,
        category_orders={sev_col: SEVERITY_ORDER},
        labels={"Hour": "Hour of Day (00:00 - 23:00)", "Count": "Incident Count"},
    )

    layout = get_base_plotly_layout(height=340)
    layout.update(
        title=dict(text="Accident Frequency by Hour of Day", font=dict(size=14, color="#ffffff")),
        barmode="stack",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
    )
    fig.update_layout(layout)
    return fig


def plot_top_causes(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Horizontal stacked bar chart showing top primary accident causes."""
    if "Cause_of_accident" not in df.columns:
        return go.Figure()

    sev_col = "Severity_label" if "Severity_label" in df.columns else "Accident_severity"
    top_causes = df["Cause_of_accident"].value_counts().head(top_n).index.tolist()
    filtered_df = df[df["Cause_of_accident"].isin(top_causes)]

    grouped = (
        filtered_df.groupby(["Cause_of_accident", sev_col])
        .size()
        .reset_index(name="Count")
    )

    color_map = {
        "Fatal injury": COLOR_ROSE,
        "Serious Injury": COLOR_AMBER,
        "Slight Injury": COLOR_CYAN,
        0: COLOR_ROSE,
        1: COLOR_AMBER,
        2: COLOR_CYAN,
    }

    fig = px.bar(
        grouped,
        y="Cause_of_accident",
        x="Count",
        color=sev_col,
        color_discrete_map=color_map,
        orientation="h",
        category_orders={
            "Cause_of_accident": top_causes[::-1],
            sev_col: SEVERITY_ORDER,
        },
        labels={"Cause_of_accident": "", "Count": "Accident Count"},
    )

    layout = get_base_plotly_layout(height=360)
    layout.update(
        title=dict(text=f"Top {top_n} Primary Accident Causes", font=dict(size=14, color="#ffffff")),
        barmode="stack",
    )
    fig.update_layout(layout)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """Plot high-contrast annotated confusion matrix."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # Annotated text with raw count and percentage
    text_annotations = []
    for i in range(len(class_names)):
        row_text = []
        for j in range(len(class_names)):
            raw_cnt = cm[i, j]
            pct = cm_norm[i, j] * 100
            row_text.append(f"<b>{raw_cnt:,}</b><br>({pct:.1f}%)")
        text_annotations.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=[f"Pred: {c}" for c in class_names],
            y=[f"Actual: {c}" for c in class_names],
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(color="#ffffff", size=13),
            colorscale=[
                [0.0, "#0b132b"],
                [0.3, "#1c2541"],
                [0.7, "#0891b2"],
                [1.0, "#06b6d4"],
            ],
            showscale=True,
            colorbar=dict(title="Normalized", tickfont=dict(color="#ffffff")),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Ratio: %{z:.3f}<extra></extra>",
        )
    )

    layout = get_base_plotly_layout(height=420)
    layout.update(
        title=dict(text="Confusion Matrix (Normalized & Counts)", font=dict(size=14, color="#ffffff")),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_layout(layout)
    return fig


def plot_model_comparison_bars(df_comp: pd.DataFrame) -> go.Figure:
    """Grouped interactive bar chart comparing performance metrics across models."""
    metrics_to_plot = ["Macro F1", "Balanced Accuracy", "Accuracy", "Weighted F1"]

    melted = df_comp.melt(
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    palette = {
        "Macro F1": COLOR_CYAN,
        "Balanced Accuracy": COLOR_TEAL,
        "Accuracy": COLOR_SKY,
        "Weighted F1": COLOR_AMBER,
    }

    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_map=palette,
        labels={"Score": "Score (0 - 1)", "Model": ""},
    )

    layout = get_base_plotly_layout(height=400)
    layout.update(
        title=dict(text="Model Performance Comparison Benchmark", font=dict(size=14, color="#ffffff")),
        yaxis=dict(range=[0, 1.0], dtick=0.2),
    )
    fig.update_layout(layout)
    return fig


def plot_top_5_causes(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing the five most frequent recorded accident causes."""
    if "Cause_of_accident" not in df.columns or df["Cause_of_accident"].dropna().empty:
        return go.Figure()

    # Drop nulls and take top 5
    causes_clean = df["Cause_of_accident"].dropna().astype(str)
    top_5 = causes_clean.value_counts().head(5).reset_index()
    top_5.columns = ["Cause", "Count"]

    # Reverse order so highest count is at the top of the horizontal chart
    top_5 = top_5.iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=top_5["Count"],
            y=top_5["Cause"],
            orientation="h",
            marker=dict(
                color=top_5["Count"],
                colorscale=[[0, "#0e7490"], [0.5, "#0891b2"], [1, "#06b6d4"]],
                line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
            ),
            text=top_5["Count"].apply(lambda v: f"<b>{v:,}</b>"),
            textposition="auto",
            textfont=dict(color="#ffffff", size=12),
            hovertemplate="<b>%{y}</b><br>Recorded Collisions: %{x:,}<extra></extra>",
        )
    )

    layout = get_base_plotly_layout(height=340)
    layout.update(
        title=dict(text="Top 5 Primary Accident Causes", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Number of Recorded Accidents"),
        yaxis=dict(title=""),
        margin=dict(l=20, r=20, t=45, b=35),
    )
    fig.update_layout(layout)
    return fig


def plot_hourly_accidents_24h(df: pd.DataFrame) -> go.Figure:
    """Chronologically sorted (0 to 23) bar chart displaying accident frequency across the 24-hour day."""
    if "Hour" not in df.columns or df["Hour"].dropna().empty:
        return go.Figure()

    # Group by valid numeric hour and count records without fabricating missing hours
    valid_hours = df["Hour"].dropna().astype(int)
    counts = valid_hours.value_counts().sort_index()

    # Prepare DataFrame sorted chronologically from 0 to 23
    hourly_df = pd.DataFrame({
        "Hour": counts.index,
        "Hour_Label": [f"{h:02d}:00" for h in counts.index],
        "Count": counts.values,
    }).sort_values(by="Hour").reset_index(drop=True)

    peak_hour = hourly_df.loc[hourly_df["Count"].idxmax()]["Hour"]

    # Highlight peak hour distinctly
    colors = [
        COLOR_AMBER if h == peak_hour else COLOR_CYAN
        for h in hourly_df["Hour"]
    ]

    fig = go.Figure(
        go.Bar(
            x=hourly_df["Hour_Label"],
            y=hourly_df["Count"],
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.12)", width=1),
            ),
            text=hourly_df["Count"].apply(lambda v: f"{v:,}"),
            textposition="outside",
            textfont=dict(color="#cbd5e1", size=10),
            hovertemplate="Time: <b>%{x}</b><br>Collisions: <b>%{y:,}</b><extra></extra>",
        )
    )

    layout = get_base_plotly_layout(height=340)
    layout.update(
        title=dict(text="Accident Frequency by Hour (24-Hour Day)", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickangle=-30),
        yaxis=dict(title="Recorded Accidents"),
        margin=dict(l=20, r=20, t=45, b=45),
    )
    fig.update_layout(layout)
    return fig


def plot_dashboard_model_preview(df_comp: pd.DataFrame) -> go.Figure:
    """Compact grouped comparison of Macro F1, Accuracy, and Balanced Accuracy across the 4 classifiers."""
    metrics_to_plot = ["Macro F1", "Balanced Accuracy", "Accuracy"]
    
    # Filter only available metrics
    available_metrics = [m for m in metrics_to_plot if m in df_comp.columns]
    
    melted = df_comp.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    # Identify best model dynamically
    best_row = df_comp.loc[df_comp["Macro F1"].idxmax()]
    best_model_name = best_row["Model"]
    best_f1 = best_row["Macro F1"]

    palette = {
        "Macro F1": COLOR_CYAN,
        "Balanced Accuracy": COLOR_TEAL,
        "Accuracy": COLOR_SKY,
    }

    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_map=palette,
        labels={"Score": "Metric Score (0.0 - 1.0)", "Model": ""},
    )

    layout = get_base_plotly_layout(height=360)
    layout.update(
        title=dict(
            text=f"Model Performance Comparison (Top Model: {best_model_name} • Macro F1: {best_f1:.4f})",
            font=dict(size=14, color="#ffffff"),
        ),
        yaxis=dict(range=[0, 1.05], dtick=0.2),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    fig.update_layout(layout)
    return fig



def plot_feature_importance(importances: pd.Series, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of top predictive features."""
    top_s = importances.head(top_n).iloc[::-1]

    # Clean feature names for display
    clean_names = [name.replace("_", " ") for name in top_s.index]

    fig = go.Figure(
        go.Bar(
            x=top_s.values,
            y=clean_names,
            orientation="h",
            marker=dict(
                color=top_s.values,
                colorscale=[[0, "#1e3a8a"], [0.5, "#0284c7"], [1, "#06b6d4"]],
            ),
            hovertemplate="Feature: %{y}<br>Relative Importance: %{x:.4f}<extra></extra>",
        )
    )

    layout = get_base_plotly_layout(height=400)
    layout.update(
        title=dict(text=f"Top {top_n} Predictive Feature Attributions", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Importance Weight"),
    )
    fig.update_layout(layout)
    return fig


def plot_probability_breakdown(probs: np.ndarray, class_names: List[str]) -> go.Figure:
    """Horizontal bar chart of predicted probabilities for each severity tier."""
    colors = [COLOR_ROSE, COLOR_AMBER, COLOR_CYAN]
    pcts = [p * 100 for p in probs]

    fig = go.Figure(
        go.Bar(
            x=pcts,
            y=class_names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"<b>{p:.1f}%</b>" for p in pcts],
            textposition="auto",
            textfont=dict(color="#ffffff", size=13),
            hovertemplate="Severity: %{y}<br>Probability: %{x:.2f}%<extra></extra>",
        )
    )

    layout = get_base_plotly_layout(height=230)
    layout.update(
        title=dict(text="Class Probability Distribution", font=dict(size=13, color="#ffffff")),
        xaxis=dict(range=[0, 100], title="Probability (%)", dtick=20),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_layout(layout)
    return fig


def plot_roc_curves(roc_data: Dict[str, Dict[str, List[float]]], macro_auc: Optional[float] = None) -> go.Figure:
    """Plot multi-class ROC curves."""
    fig = go.Figure()

    colors = {
        "Fatal injury": COLOR_ROSE,
        "Serious Injury": COLOR_AMBER,
        "Slight Injury": COLOR_CYAN,
    }

    for cls_name, coords in roc_data.items():
        fpr = coords["fpr"]
        tpr = coords["tpr"]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=cls_name,
                line=dict(color=colors.get(cls_name, COLOR_CYAN), width=2.5),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )

    # Reference diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="#64748b", dash="dash", width=1.5),
            hoverinfo="none",
        )
    )

    title_text = "Multi-Class One-vs-Rest ROC Curves"
    if macro_auc is not None:
        title_text += f" (Macro AUC: {macro_auc:.3f})"

    layout = get_base_plotly_layout(height=400)
    layout.update(
        title=dict(text=title_text, font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="False Positive Rate", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", range=[0, 1]),
    )
    fig.update_layout(layout)
    return fig


def plot_feature_distribution(df: pd.DataFrame, col_name: str, top_n: int = 12) -> go.Figure:
    """Frequency bar chart for categorical or discrete features with percentage annotations."""
    if col_name not in df.columns:
        return go.Figure()

    series = df[col_name].fillna("Missing / Unknown").astype(str)
    counts = series.value_counts().head(top_n).reset_index()
    counts.columns = [col_name, "Count"]
    total = len(df)
    counts["Percentage"] = (counts["Count"] / total * 100) if total > 0 else 0.0

    fig = go.Figure(
        go.Bar(
            x=counts[col_name],
            y=counts["Count"],
            marker=dict(
                color=counts["Count"],
                colorscale=[[0, "#0891b2"], [1, "#06b6d4"]],
                line=dict(color="rgba(255, 255, 255, 0.12)", width=1),
            ),
            text=[f"<b>{row['Count']:,}</b> ({row['Percentage']:.1f}%)" for _, row in counts.iterrows()],
            textposition="outside",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate="Category: <b>%{x}</b><br>Incidents: %{y:,}<br>Percentage: %{customdata:.2f}%<extra></extra>",
            customdata=counts["Percentage"],
        )
    )

    clean_title = col_name.replace("_", " ")
    layout = get_base_plotly_layout(height=380)
    layout.update(
        title=dict(text=f"Distribution of {clean_title}", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title=clean_title, tickangle=-25),
        yaxis=dict(title="Incident Count", range=[0, max(counts["Count"]) * 1.18]),
        margin=dict(l=30, r=30, t=50, b=50),
    )
    fig.update_layout(layout)
    return fig


def plot_feature_severity_stacked(df: pd.DataFrame, col_name: str, top_n: int = 10) -> go.Figure:
    """
    100% normalized stacked bar chart displaying the proportion of Slight, Serious,
    and Fatal injuries across the top categories of a given feature.
    """
    if col_name not in df.columns:
        return go.Figure()

    clean_s = df[col_name].fillna("Missing / Unknown").astype(str)
    top_categories = clean_s.value_counts().head(top_n).index.tolist()

    sub_df = df[clean_s.isin(top_categories)].copy()
    sub_df[col_name] = clean_s

    # Contingency cross-tab normalized across rows (percentages within each category)
    crosstab = pd.crosstab(
        sub_df[col_name],
        sub_df["Accident_severity"],
        normalize="index",
    ) * 100

    # Ensure consistent column ordering
    for s_label in SEVERITY_ORDER:
        if s_label not in crosstab.columns:
            crosstab[s_label] = 0.0
    crosstab = crosstab[SEVERITY_ORDER]

    # Reindex to match top categories order
    crosstab = crosstab.reindex(top_categories)

    melted = crosstab.reset_index().melt(
        id_vars=[col_name],
        var_name="Severity",
        value_name="Percentage",
    )

    fig = px.bar(
        melted,
        x=col_name,
        y="Percentage",
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"Severity": SEVERITY_ORDER},
        labels={"Percentage": "Proportion Within Category (%)", col_name: col_name.replace("_", " ")},
    )

    layout = get_base_plotly_layout(height=400)
    layout.update(
        title=dict(
            text=f"Injury Severity Breakdown Across {col_name.replace('_', ' ')} (100% Normalized)",
            font=dict(size=14, color="#ffffff"),
        ),
        barmode="stack",
        xaxis=dict(tickangle=-25),
        yaxis=dict(range=[0, 100], title="Percentage within Category (%)"),
        margin=dict(l=30, r=30, t=50, b=50),
    )
    fig.update_layout(layout)
    return fig


def plot_weekday_accidents(df: pd.DataFrame) -> go.Figure:
    """Chronologically sorted accident frequency by day of the week (Monday to Sunday)."""
    if "Day_of_week" not in df.columns:
        return go.Figure()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = df["Day_of_week"].value_counts().reindex(day_order).fillna(0).reset_index()
    counts.columns = ["Day", "Count"]
    counts["Count"] = counts["Count"].astype(int)
    total = counts["Count"].sum()
    counts["Percentage"] = (counts["Count"] / total * 100) if total > 0 else 0.0

    peak_day = counts.loc[counts["Count"].idxmax()]["Day"]
    colors = [COLOR_AMBER if d == peak_day else COLOR_CYAN for d in counts["Day"]]

    fig = go.Figure(
        go.Bar(
            x=counts["Day"],
            y=counts["Count"],
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.12)", width=1),
            ),
            text=[f"<b>{row['Count']:,}</b> ({row['Percentage']:.1f}%)" for _, row in counts.iterrows()],
            textposition="outside",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate="Day: <b>%{x}</b><br>Accidents: %{y:,}<br>Share: %{customdata:.2f}%<extra></extra>",
            customdata=counts["Percentage"],
        )
    )

    layout = get_base_plotly_layout(height=340)
    layout.update(
        title=dict(text="Accident Distribution by Day of Week (Mon - Sun)", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Day of Week"),
        yaxis=dict(title="Incident Count", range=[0, max(counts["Count"]) * 1.18]),
        margin=dict(l=30, r=30, t=45, b=40),
    )
    fig.update_layout(layout)
    return fig


def compute_serious_fatal_ranking(df: pd.DataFrame, col_name: str) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Calculate Serious-or-Fatal percentage for each category of col_name:
    (Serious + Fatal) / Total * 100
    Returns: (summary_df, horizontal_bar_fig)
    """
    clean_s = df[col_name].fillna("Missing / Unknown").astype(str)
    temp_df = df.copy()
    temp_df[col_name] = clean_s

    # Aggregate counts
    grouped = temp_df.groupby(col_name)["Accident_severity"].agg(
        Total="count",
        Fatal=lambda s: (s == "Fatal injury").sum(),
        Serious=lambda s: (s == "Serious Injury").sum(),
        Slight=lambda s: (s == "Slight Injury").sum(),
    ).reset_index()

    grouped["Serious_or_Fatal"] = grouped["Serious"] + grouped["Fatal"]
    grouped["Serious_or_Fatal_Pct"] = (grouped["Serious_or_Fatal"] / grouped["Total"]) * 100

    # Sort descending by serious-or-fatal rate
    grouped = grouped.sort_values(by="Serious_or_Fatal_Pct", ascending=False).reset_index(drop=True)

    # Horizontal bar chart (reversed so highest rate is on top)
    plot_df = grouped.iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=plot_df["Serious_or_Fatal_Pct"],
            y=plot_df[col_name],
            orientation="h",
            marker=dict(
                color=plot_df["Serious_or_Fatal_Pct"],
                colorscale=[[0, "#d97706"], [0.5, "#ea580c"], [1, "#ef4444"]],
                line=dict(color="rgba(255, 255, 255, 0.12)", width=1),
            ),
            text=[
                f"<b>{pct:.1f}%</b> ({sf:,}/{tot:,})"
                for pct, sf, tot in zip(plot_df["Serious_or_Fatal_Pct"], plot_df["Serious_or_Fatal"], plot_df["Total"])
            ],
            textposition="auto",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate="<b>%{y}</b><br>Serious/Fatal Rate: %{x:.2f}%<extra></extra>",
        )
    )

    clean_title = col_name.replace("_", " ")
    layout = get_base_plotly_layout(height=max(360, min(700, len(grouped) * 32)))
    layout.update(
        title=dict(text=f"Serious & Fatal Collision Rate by {clean_title} (%)", font=dict(size=14, color="#ffffff")),
        xaxis=dict(title="Serious or Fatal Rate (%)", range=[0, max(grouped["Serious_or_Fatal_Pct"].max() * 1.15, 10)]),
        yaxis=dict(title=""),
        margin=dict(l=20, r=20, t=45, b=35),
    )
    fig.update_layout(layout)

    return grouped, fig


def plot_numerical_correlation_matrix(df: pd.DataFrame, num_cols: List[str]) -> go.Figure:
    """Calculate Pearson correlation strictly on actual numerical variables."""
    valid_num_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_num_cols) < 2:
        return go.Figure()

    corr_df = df[valid_num_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=[c.replace("_", " ") for c in corr_df.columns],
            y=[c.replace("_", " ") for c in corr_df.index],
            colorscale=[[0, "#0b132b"], [0.5, "#1c2541"], [1, "#06b6d4"]],
            zmin=-0.15,
            zmax=0.5,
            text=np.round(corr_df.values, 3),
            texttemplate="%{text}",
            textfont=dict(color="#ffffff", size=13),
            colorbar=dict(title="Pearson r", tickfont=dict(color="#ffffff")),
        )
    )

    layout = get_base_plotly_layout(height=360)
    layout.update(
        title=dict(text="Numerical Attributes Correlation Matrix (Pearson)", font=dict(size=14, color="#ffffff")),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=30, t=50, b=40),
    )
    fig.update_layout(layout)
    return fig


def plot_numerical_distribution_by_severity(df: pd.DataFrame, num_col: str) -> go.Figure:
    """Box plot of numerical variable across the 3 severity tiers."""
    if num_col not in df.columns or "Accident_severity" not in df.columns:
        return go.Figure()

    fig = px.box(
        df,
        x="Accident_severity",
        y=num_col,
        color="Accident_severity",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"Accident_severity": SEVERITY_ORDER},
        labels={"Accident_severity": "Severity Category", num_col: num_col.replace("_", " ")},
    )

    clean_title = num_col.replace("_", " ")
    layout = get_base_plotly_layout(height=360)
    layout.update(
        title=dict(text=f"{clean_title} by Severity Tier", font=dict(size=14, color="#ffffff")),
        showlegend=False,
        margin=dict(l=30, r=30, t=45, b=40),
    )
    fig.update_layout(layout)
    return fig

