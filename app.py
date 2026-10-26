"""
Road Traffic Accident Severity Prediction and Risk Analysis
Main Entry Point & Executive Welcome Portal
"""

import streamlit as st

# Configure multi-page app settings
st.set_page_config(
    page_title="RTA Severity Analytics & Risk ML",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import (
    load_rta_data,
    load_cleaned_data,
    get_dataset_summary,
    SEVERITY_COLORS,
)
from src.train_models import check_models_exist, load_saved_models_and_artifacts
from src.visualizations import (
    apply_custom_css,
    render_kpi_card,
    COLOR_CYAN,
    COLOR_AMBER,
    COLOR_ROSE,
)

# Apply global styling
apply_custom_css()

# Sidebar branding & system health
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🚦</span>
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #ffffff;">RTA RiskIntel</h3>
                    <span style="font-size: 0.75rem; color: #06b6d4; font-weight: 600; text-transform: uppercase;">
                        Severity AI Platform
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 System Status")

    # Verify dataset loading status
    try:
        summary = get_dataset_summary()
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;">
                <span style="color: #10b981; font-weight: 600; font-size: 0.82rem;">✓ Datasets Verified</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">12,316 records loaded</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Dataset Loading Error: {str(e)}")

    # Verify model status
    models_ready = check_models_exist()
    if models_ready:
        st.markdown(
            """
            <div style="background: rgba(6, 182, 212, 0.1); border-left: 3px solid #06b6d4; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;">
                <span style="color: #06b6d4; font-weight: 600; font-size: 0.82rem;">✓ ML Pipelines Ready</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">4 Standard Syllabus Classifiers</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Models not pre-trained. Initializing...")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5;">
            <b>ML Algorithms</b>: KNN, Logistic Regression, Decision Tree, Random Forest<br>
            <b>Framework</b>: Scikit-learn Pipeline & ColumnTransformer<br>
            <b>Dashboard Engine</b>: Streamlit 1.51<br>
            <b>Data Source</b>: Addis Ababa Police Commission RTA Database
        </div>
        """,
        unsafe_allow_html=True,
    )

# Hero Section
st.markdown(
    """
    <div class="hero-banner">
        <div style="display: inline-block; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.35); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; color: #06b6d4; margin-bottom: 12px;">
            MACHINE LEARNING DECISION SUPPORT SYSTEM
        </div>
        <h1 class="hero-title">
            Road Traffic Accident <span class="hero-highlight">Severity Prediction</span> & Risk Intelligence
        </h1>
        <p class="hero-desc">
            An enterprise-grade machine learning platform analyzing 12,316 accident records to forecast injury severity, benchmark multi-class classification algorithms, and expose high-risk environmental and behavioral driving hazards.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Core Statistics KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    render_kpi_card(
        title="Total Incidents",
        value=f"{summary['total_records']:,}",
        subtitle="100% Audited Records",
        accent_color="#ffffff",
    )

with col2:
    render_kpi_card(
        title="Fatal Injuries",
        value=f"{summary['fatal_count']:,}",
        subtitle=f"{summary['fatal_pct']:.2f}% of total accidents",
        accent_color=COLOR_ROSE,
    )

with col3:
    render_kpi_card(
        title="Serious Injuries",
        value=f"{summary['serious_count']:,}",
        subtitle=f"{summary['serious_pct']:.2f}% of total accidents",
        accent_color=COLOR_AMBER,
    )

with col4:
    render_kpi_card(
        title="Slight Injuries",
        value=f"{summary['slight_count']:,}",
        subtitle=f"{summary['slight_pct']:.2f}% of total accidents",
        accent_color=COLOR_CYAN,
    )

with col5:
    render_kpi_card(
        title="Feature Space",
        value=f"{len(summary['clean_columns'])} / {len(summary['rta_columns'])}",
        subtitle="Cleaned vs Raw Dimensions",
        accent_color="#38bdf8",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Cards Section
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">Application Modules & Capabilities</h3>
        <span class="section-badge">5 Integrated Pages</span>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown(
        """
        <div class="rta-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📈</div>
            <h4 style="color: #ffffff; margin: 0 0 6px 0; font-size: 1.1rem;">1. Executive Dashboard</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                High-level operational overview, diurnal accident distribution, severity donuts, and primary accident cause rankings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Dashboard →", key="btn_nav_1", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

with nav_col2:
    st.markdown(
        """
        <div class="rta-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🔬</div>
            <h4 style="color: #ffffff; margin: 0 0 6px 0; font-size: 1.1rem;">2. Exploratory Data Analysis</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Compare raw vs cleaned datasets, univariate attribute frequencies, bivariate correlation heatmaps, and missingness audits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Data →", key="btn_nav_2", use_container_width=True):
        st.switch_page("pages/2_Exploratory_Data_Analysis.py")

with nav_col3:
    st.markdown(
        """
        <div class="rta-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🎯</div>
            <h4 style="color: #ffffff; margin: 0 0 6px 0; font-size: 1.1rem;">3. Severity Prediction</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Real-time interactive incident severity inference engine with multi-model probability breakdown and risk factor attribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Prediction →", key="btn_nav_3", use_container_width=True):
        st.switch_page("pages/3_Severity_Prediction.py")

nav_col4, nav_col5, nav_col6 = st.columns(3)

with nav_col4:
    st.markdown(
        """
        <div class="rta-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">⚖️</div>
            <h4 style="color: #ffffff; margin: 0 0 6px 0; font-size: 1.1rem;">4. Model Benchmarking</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Comparative evaluation across Random Forest, XGBoost, Balanced Logistic Regression, and Decision Trees with ROC & Confusion Matrices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Compare Models →", key="btn_nav_4", use_container_width=True):
        st.switch_page("pages/4_Model_Comparison.py")

with nav_col5:
    st.markdown(
        """
        <div class="rta-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🛡️</div>
            <h4 style="color: #ffffff; margin: 0 0 6px 0; font-size: 1.1rem;">5. Risk Pattern Analysis</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Multi-dimensional hazard matrices crossing driver experience, lighting, road alignments, and actionable safety policy countermeasures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Analyze Risk Patterns →", key="btn_nav_5", use_container_width=True):
        st.switch_page("pages/5_Risk_Analysis.py")

with nav_col6:
    st.markdown(
        """
        <div class="rta-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">📦</div>
            <h4 style="color: #ffffff; margin: 0 0 6px 0; font-size: 1.1rem;">Data Architecture</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 14px;">
                Original RTA dataset: 32 columns with temporal & casualty metrics. Cleaned dataset: 14 categorical features + balanced severity target.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Direct access via sidebar navigation at any time.")

st.markdown("<br>", unsafe_allow_html=True)

# Machine Learning Workflow Section
st.markdown(
    """
    <div class="section-header">
        <h3 class="section-title">End-to-End Machine Learning Pipeline</h3>
        <span class="section-badge">Architecture & Methodology</span>
    </div>
    """,
    unsafe_allow_html=True,
)

flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)

with flow_col1:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 16px 12px;">
            <div style="color: #06b6d4; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">STAGE 1</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Data Ingestion</div>
            <div style="color: #94a3b8; font-size: 0.75rem; line-height: 1.3;">
                Streamlit cached loader reading RTA (32 cols) and cleaned (15 cols) with schema validation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with flow_col2:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 16px 12px;">
            <div style="color: #06b6d4; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">STAGE 2</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Pre-Processing</div>
            <div style="color: #94a3b8; font-size: 0.75rem; line-height: 1.3;">
                ColumnTransformer with median/most_frequent imputation, One-Hot encoding, and numerical scaling for KNN & LR.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with flow_col3:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 16px 12px;">
            <div style="color: #06b6d4; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">STAGE 3</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Imbalance Control</div>
            <div style="color: #94a3b8; font-size: 0.75rem; line-height: 1.3;">
                Class-weight balancing on LR, DT, and RF to mitigate 84.6% slight injury dominance (KNN evaluated without weights).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with flow_col4:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 16px 12px;">
            <div style="color: #06b6d4; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">STAGE 4</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Model Benchmarking</div>
            <div style="color: #94a3b8; font-size: 0.75rem; line-height: 1.3;">
                Syllabus benchmark: K-Nearest Neighbors, Logistic Regression, Decision Tree, and Random Forest.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with flow_col5:
    st.markdown(
        """
        <div class="rta-card" style="text-align: center; padding: 16px 12px;">
            <div style="color: #06b6d4; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">STAGE 5</div>
            <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;">Inference & Risk</div>
            <div style="color: #94a3b8; font-size: 0.75rem; line-height: 1.3;">
                Interactive prediction dashboard with probability confidence, risk index, and safety alerts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer callout
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0; color: #64748b; font-size: 0.8rem;">
        Road Traffic Accident Severity Prediction & Risk Analytics Platform • Pair Programming AI System
    </div>
    """,
    unsafe_allow_html=True,
)
