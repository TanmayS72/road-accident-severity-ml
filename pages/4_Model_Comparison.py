"""
Page 4: Model Comparison & Benchmarking
Comparative evaluation across the 4 permitted standard ML algorithms:
1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Tree
4. Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import SEVERITY_ORDER
from src.train_models import (
    load_saved_models_and_artifacts,
    check_models_exist,
    train_and_save_all,
    get_cached_metrics,
)
from src.visualizations import (
    apply_custom_css,
    render_page_header,
    render_kpi_card,
    plot_model_comparison_bars,
    plot_confusion_matrix,
    plot_feature_importance,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
    COLOR_SKY,
)

st.set_page_config(
    page_title="Model Comparison | RTA Analytics",
    page_icon="⚖️",
    layout="wide",
)

apply_custom_css()

render_page_header(
    title="Model Comparison & Algorithmic Benchmarking",
    subtitle="Comparative evaluation across 4 syllabus-aligned ML classifiers evaluated on identical 20% stratified test partitions (2,464 collisions).",
    badge="Model Lab",
)

# Ensure models are loaded
if not check_models_exist():
    with st.spinner("Pre-training the 4 permitted ML models..."):
        train_and_save_all(force_retrain=False)

models, X_test, y_test, comparison_df, best_model_name = load_saved_models_and_artifacts()
metrics_cache = get_cached_metrics()

# Top Performance KPI Cards
best_macro_row = comparison_df.iloc[0]
best_acc_row = comparison_df.sort_values(by="Accuracy", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)

with k1:
    render_kpi_card(
        title="Best Model (Macro F1)",
        value=best_model_name,
        subtitle=f"Macro F1: {best_macro_row['Macro F1']:.4f}",
        accent_color=COLOR_CYAN,
    )

with k2:
    render_kpi_card(
        title="Best Balanced Accuracy",
        value=f"{best_macro_row['Balanced Accuracy'] * 100:.1f}%",
        subtitle=f"Model: {best_macro_row['Model']}",
        accent_color="#38bdf8",
    )

with k3:
    render_kpi_card(
        title="Peak Raw Accuracy",
        value=f"{best_acc_row['Accuracy'] * 100:.1f}%",
        subtitle=f"Model: {best_acc_row['Model']} (Imbalanced)",
        accent_color="#10b981",
    )

with k4:
    render_kpi_card(
        title="Held-out Test Cases",
        value=f"{len(y_test):,}",
        subtitle="20% Stratified Split (Leak-Free)",
        accent_color="#ffffff",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Algorithmic Benchmark Table
st.markdown(
    """
    <div class="section-header">
        <h4 class="section-title">Comparative Performance Scorecard</h4>
        <span class="section-badge">Standard Syllabus Classifiers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "Balanced Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "Macro F1": st.column_config.NumberColumn(format="%.4f"),
        "Weighted F1": st.column_config.NumberColumn(format="%.4f"),
        "Macro Precision": st.column_config.NumberColumn(format="%.4f"),
        "Macro Recall": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Benchmark Visual Comparison Chart
st.markdown("<br>", unsafe_allow_html=True)
fig_bars = plot_model_comparison_bars(comparison_df)
st.plotly_chart(fig_bars, use_container_width=True)

# Technical Insight Callout on the Accuracy Paradox & Syllabus Relevance
st.markdown(
    """
    <div class="callout-warning" style="margin: 20px 0;">
        <b style="color: #f59e0b; font-size: 0.95rem;">💡 Academic Insight: The "Accuracy Paradox" & Why Random Forest Was Selected:</b><br>
        • <b>K-Nearest Neighbors (KNN)</b> achieved the highest raw accuracy (<b>84.70%</b>), but standard KNN does not support <code>class_weight='balanced'</code>. Consequently, KNN simply assigned 99% of samples to the majority class ("Slight Injury") and failed completely on fatal collisions (<b>0% Fatal Recall</b>).<br>
        • <b>Decision Tree</b> and <b>Random Forest</b> leverage cost-sensitive balanced weighting, actively penalizing errors on rare fatal and serious classes.<br>
        • <b>Random Forest</b> achieved the highest <b>Macro F1-Score (0.4704)</b> and balanced accuracy, making it objectively the best-performing architecture for accident severity detection.
    </div>
    """,
    unsafe_allow_html=True,
)

# Deep-Dive Model Diagnostic Inspector
st.markdown(
    """
    <div class="section-header">
        <h4 class="section-title">Deep-Dive Algorithmic Diagnostics</h4>
        <span class="section-badge">Model Inspector</span>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_inspect_name = st.selectbox(
    "Select Model for Diagnostic Decomposition:",
    list(models.keys()),
    index=list(models.keys()).index(best_model_name) if best_model_name in models else 0,
)

inspect_metrics = metrics_cache.get(selected_inspect_name)
if inspect_metrics is None:
    st.error("Diagnostic metrics not available for this model.")
    st.stop()

# Two columns: Confusion Matrix + Per-Class Classification Report
diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    cm = inspect_metrics["confusion_matrix"]
    fig_cm = plot_confusion_matrix(cm, SEVERITY_ORDER)
    st.plotly_chart(fig_cm, use_container_width=True)

with diag_col2:
    st.markdown(
        """
        <div class="rta-card" style="height: 420px; overflow-y: auto;">
            <h4 style="color: #06b6d4; margin-top: 0; font-size: 1rem;">
                Disaggregated Per-Class Metrics
            </h4>
            <p style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 12px;">
                Evaluates sensitivity and precision across each specific injury severity tier.
            </p>
        """,
        unsafe_allow_html=True,
    )
    per_class_df = inspect_metrics.get("per_class_df")
    if per_class_df is not None:
        st.dataframe(
            per_class_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall (Sensitivity)": st.column_config.NumberColumn(format="%.4f"),
                "F1-Score": st.column_config.NumberColumn(format="%.4f"),
                "Support": st.column_config.NumberColumn(format="%d"),
            },
        )
    st.markdown(
        f"""
        <hr style="border-color: rgba(255,255,255,0.08); margin: 14px 0;">
        <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.5;">
            <b>Minority Class Focus:</b><br>
            • Fatal Injury Support: <b>31 cases</b> (1.28% of test set)<br>
            • Serious Injury Support: <b>349 cases</b> (14.16% of test set)<br>
            • Slight Injury Support: <b>2,084 cases</b> (84.58% of test set)
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Feature Importance Section
st.markdown(
    """
    <div class="section-header">
        <h4 class="section-title">Feature Importance & Attribution Analysis</h4>
        <span class="section-badge">Explainability</span>
    </div>
    """,
    unsafe_allow_html=True,
)

feat_imp = inspect_metrics.get("feature_importances")
if feat_imp is not None:
    fig_feat = plot_feature_importance(feat_imp, top_n=14)
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.markdown(
        f"""
        <div class="callout-info">
            ℹ️ <b>Explainability Note for {selected_inspect_name}:</b><br>
            K-Nearest Neighbors is an instance-based lazy learner based on multi-dimensional feature distance.
            It does not construct an explicit internal hypothesis or feature weights (unlike Decision Trees, Random Forests, or Logistic Regression coefficients).
        </div>
        """,
        unsafe_allow_html=True,
    )

# Algorithm Educational Reference (Syllabus Guide)
st.markdown(
    """
    <div class="section-header">
        <h4 class="section-title">Standard Syllabus Algorithms Overview</h4>
        <span class="section-badge">Academic Reference</span>
    </div>
    """,
    unsafe_allow_html=True,
)

algo_c1, algo_c2 = st.columns(2)

with algo_c1:
    st.markdown(
        """
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 12px;">
            <div style="color: #06b6d4; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">1. K-Nearest Neighbors (KNN)</div>
            <div style="color: #ffffff; font-size: 1.05rem; font-weight: 700; margin: 4px 0;">Instance-Based Lazy Learner</div>
            <p style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.5; margin: 0;">
                <b>Core Principle:</b> Classifies an unknown collision by computing Euclidean distances across scaled numerical attributes and one-hot encoded vectors, taking a distance-weighted vote among the <i>k=7</i> nearest neighbors.<br>
                <b>Viva Insight:</b> KNN achieves high raw accuracy (84.70%) because it memorizes the 84.57% majority class. However, without native cost-sensitive class weighting, it achieved <b>0.00% recall on fatal collisions</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 12px;">
            <div style="color: #38bdf8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">2. Logistic Regression</div>
            <div style="color: #ffffff; font-size: 1.05rem; font-weight: 700; margin: 4px 0;">Linear Probabilistic Classifier</div>
            <p style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.5; margin: 0;">
                <b>Core Principle:</b> Learns a linear log-odds decision boundary using the multinomial softmax function. Features are standardized, and inverse class weights (<code>class_weight='balanced'</code>) penalize errors on rare classes.<br>
                <b>Viva Insight:</b> While raw accuracy is lower (48.86%), its balanced accuracy (40.66%) demonstrates active sensitivity to serious and fatal collisions rather than default majority prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with algo_c2:
    st.markdown(
        """
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 12px;">
            <div style="color: #14b8a6; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">3. Decision Tree</div>
            <div style="color: #ffffff; font-size: 1.05rem; font-weight: 700; margin: 4px 0;">Non-Parametric Rule-Based Model</div>
            <p style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.5; margin: 0;">
                <b>Core Principle:</b> Recursively splits the feature space on feature thresholds that maximize Gini impurity reduction. Constrained to <code>max_depth=10</code> with balanced class weights to control tree variance.<br>
                <b>Viva Insight:</b> Achieves the highest balanced accuracy (51.40%) by aggressively prioritizing minority branches, producing highly interpretable hierarchical rules.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="rta-card" style="padding: 16px 18px; margin-bottom: 12px;">
            <div style="color: #f59e0b; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">4. Random Forest (Selected Best Model)</div>
            <div style="color: #ffffff; font-size: 1.05rem; font-weight: 700; margin: 4px 0;">Bootstrap Aggregation (Bagging) Ensemble</div>
            <p style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.5; margin: 0;">
                <b>Core Principle:</b> Ensembles 100 de-correlated decision trees trained on bootstrap samples with random feature subsets. Combines individual tree probability estimates through soft voting.<br>
                <b>Viva Insight:</b> Selected as the optimal model because it achieves the peak <b>Macro F1-Score (0.4704)</b> and 78.25% accuracy, balancing minority trauma detection with low false positive rates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Retraining & Parameter Tuning Hub
with st.expander("⚙️ Pipeline Retraining & Experimentation Controls"):
    st.markdown(
        """
        <p style="color: #94a3b8; font-size: 0.85rem;">
            Adjust train/test split proportion or random seed, and re-train all 4 standard models dynamically.
            Results will update cached artifacts.
        </p>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        split_ratio = st.slider("Test Split Proportion", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
    with t_col2:
        seed_val = st.number_input("Random Generator Seed", min_value=1, max_value=999, value=42)

    if st.button("🔄 Retrain All 4 Models on New Parameters", type="primary"):
        with st.spinner("Retraining standard syllabus classifiers across updated dataset splits..."):
            train_and_save_all(test_size=split_ratio, random_state=int(seed_val), force_retrain=True)
            st.success("All 4 models successfully retrained and persistent cache refreshed!")
            st.rerun()
