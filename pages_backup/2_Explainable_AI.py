"""
Explainable AI (XAI) Page using SHAP.

Provides transparent attribution of model decisions at both the individual
prediction level and global model population level using genuine SHAP TreeExplainer values.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.ui_helpers import apply_custom_css, render_header, render_disclaimer
from src.prediction import load_trained_pipeline, DEFAULT_MODEL_PATH
from src.explainability import (
    compute_single_prediction_shap,
    compute_global_feature_importance,
    plot_single_shap_waterfall_plotly,
    plot_global_importance_plotly,
    generate_shap_summary_figure_matplotlib,
)
from src.preprocessing import ALL_FEATURE_COLUMNS, VALID_DOMAINS

st.set_page_config(
    page_title="Explainable AI (SHAP)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="Explainable AI (XAI) with SHAP",
    subtitle="Deconstructing Random Forest decision boundaries into transparent, mathematically principled feature contributions.",
    icon="🔍"
)

render_disclaimer()

# Verify model pipeline
if not os.path.exists(DEFAULT_MODEL_PATH):
    st.error("Trained model pipeline not found. Please navigate to the Home page or run train_model.py first.")
    st.stop()

pipeline = load_trained_pipeline(DEFAULT_MODEL_PATH)

# Tab selection for Individual vs Global explanations
tab_indiv, tab_global = st.tabs(["🎯 Individual Prediction SHAP", "🌐 Global Model Feature Importance"])

# ==========================================
# TAB 1: INDIVIDUAL PREDICTION EXPLANATION
# ==========================================
with tab_indiv:
    st.markdown("### 🎯 Local Feature Attribution for Candidate Profile")
    st.markdown(
        "<span style='color: #B8B1CC;'>Understand exactly why the model predicted a specific probability for this candidate. "
        "Positive values (green) push the probability towards <strong>Selected</strong>, while negative values (red) pull it towards <strong>Not Selected</strong>.</span>",
        unsafe_allow_html=True
    )

    # Use profile from session state or allow quick custom inputs
    candidate_profile = st.session_state.get("latest_profile", None)

    if candidate_profile is None:
        st.info("💡 No student profile submitted yet. You can use the default profile below or fill the form on the Prediction page.")
        candidate_profile = {
            "CGPA": 8.2,
            "Number of Relevant Projects": 4,
            "Average Project Complexity": 4,
            "Previous Internship Experience": "Yes",
            "Relevant Certifications": 2,
            "Number of Programming Languages": 3,
            "Number of Technical Skills": 8,
            "Internship Domain Applied": "Artificial Intelligence and Machine Learning",
        }

    with st.expander("📝 View or Adjust Current Profile Inputs for Explanation", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cgpa_in = st.number_input("CGPA", 0.0, 10.0, float(candidate_profile["CGPA"]), 0.1, key="shap_cgpa")
            proj_in = st.number_input("Projects", 0, 20, int(candidate_profile["Number of Relevant Projects"]), 1, key="shap_proj")
        with c2:
            comp_in = st.selectbox("Complexity (1-5)", [1, 2, 3, 4, 5], int(candidate_profile["Average Project Complexity"]) - 1, key="shap_comp")
            intern_in = st.selectbox("Internship Exp", ["No", "Yes"], 1 if candidate_profile["Previous Internship Experience"] == "Yes" else 0, key="shap_intern")
        with c3:
            certs_in = st.number_input("Certs", 0, 15, int(candidate_profile["Relevant Certifications"]), 1, key="shap_certs")
            langs_in = st.number_input("Languages", 1, 15, int(candidate_profile["Number of Programming Languages"]), 1, key="shap_langs")
        with c4:
            skills_in = st.number_input("Skills", 1, 30, int(candidate_profile["Number of Technical Skills"]), 1, key="shap_skills")
            domain_in = st.selectbox("Domain", VALID_DOMAINS, VALID_DOMAINS.index(candidate_profile["Internship Domain Applied"]) if candidate_profile["Internship Domain Applied"] in VALID_DOMAINS else 0, key="shap_domain")

        active_profile = {
            "CGPA": cgpa_in,
            "Number of Relevant Projects": proj_in,
            "Average Project Complexity": comp_in,
            "Previous Internship Experience": intern_in,
            "Relevant Certifications": certs_in,
            "Number of Programming Languages": langs_in,
            "Number of Technical Skills": skills_in,
            "Internship Domain Applied": domain_in,
        }
    else:
        active_profile = candidate_profile

    # Compute SHAP
    with st.spinner("Calculating Shapley values using TreeExplainer..."):
        try:
            shap_result = compute_single_prediction_shap(pipeline, active_profile)
        except Exception as e:
            st.error(f"SHAP calculation encountered an issue: {str(e)}")
            st.stop()

    # Narrative explanation banner
    st.markdown(
        f"""
        <div class="dashboard-card" style="border-left: 4px solid #A78BFA;">
            <h4 style="margin: 0 0 0.5rem 0; color: #A78BFA;">💬 Model Reasoning Summary</h4>
            <p style="color: #F5F3FF; font-size: 0.95rem; margin: 0;">
                {shap_result['narrative']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Plotly Waterfall / Bar Chart
    chart_col, table_col = st.columns([1.3, 1])

    with chart_col:
        fig_shap = plot_single_shap_waterfall_plotly(shap_result)
        st.plotly_chart(fig_shap, use_container_width=True)

    with table_col:
        st.markdown("#### 📋 Detailed Feature Attribution Breakdown")
        df_contribs = pd.DataFrame(shap_result["contributions"])
        df_contribs = df_contribs[["feature", "shap_value", "impact"]]
        df_contribs.columns = ["Feature Attribute", "SHAP Impact", "Effect Direction"]
        df_contribs["SHAP Impact"] = df_contribs["SHAP Impact"].apply(lambda v: f"{v:+.4f}")
        st.dataframe(df_contribs, use_container_width=True, hide_index=True, height=380)

    st.markdown(
        """
        <div style="background-color: rgba(28, 23, 48, 0.6); padding: 0.8rem 1rem; border-radius: 8px; border: 1px solid #2D254D; font-size: 0.85rem; color: #B8B1CC;">
            💡 <strong>Attribution vs. Causality:</strong> SHAP values explain how the trained Random Forest model arrived at its probability estimate given the features provided. They represent mathematical model feature importance, not guaranteed real-world causal guarantees.
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# TAB 2: GLOBAL FEATURE IMPORTANCE
# ==========================================
with tab_global:
    st.markdown("### 🌐 Global Population-Level Feature Importance")
    st.markdown(
        "<span style='color: #B8B1CC;'>Examines which student attributes have the greatest overall influence across the entire applicant population.</span>",
        unsafe_allow_html=True
    )

    DATA_PATH = "data/internship_dataset.csv"
    if not os.path.exists(DATA_PATH):
        st.warning("Dataset CSV not found. Please run generate_dataset.py.")
        st.stop()

    @st.cache_data
    def load_cached_global_shap():
        raw_df = pd.read_csv(DATA_PATH)
        sample_df = raw_df.sample(min(200, len(raw_df)), random_state=42)
        return compute_global_feature_importance(pipeline, sample_df)

    with st.spinner("Computing global SHAP values across dataset sample..."):
        importance_df, X_trans, shap_vals_selected, feature_names = load_cached_global_shap()

    gcol1, gcol2 = st.columns([1.2, 1])

    with gcol1:
        fig_global = plot_global_importance_plotly(importance_df)
        st.plotly_chart(fig_global, use_container_width=True)

    with gcol2:
        st.markdown(
            """
            <div class="dashboard-card">
                <h4 style="color: #A78BFA; margin-top: 0;">🔬 Key Global Insights</h4>
                <p style="color: #B8B1CC; font-size: 0.9rem;">
                    Across all evaluated student profiles in the synthetic dataset:
                </p>
                <ul style="color: #F5F3FF; font-size: 0.88rem; line-height: 1.7;">
                    <li><strong>Project Complexity & Volume:</strong> Consistently rank among the strongest predictors, indicating that architectural depth and completed builds strongly influence model decisions.</li>
                    <li><strong>Academic Performance (CGPA):</strong> Operates as a foundational gatekeeper, where high CGPA consistently exerts positive SHAP push.</li>
                    <li><strong>Prior Internship Experience:</strong> Provides a distinct non-linear positive lift, differentiating candidates with real-world exposure.</li>
                    <li><strong>Technical Stack Depth:</strong> Skills and certifications provide domain-specific bonus attributions, especially in specialized tracks.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 📊 Ranked Importance Values")
        st.dataframe(
            importance_df[["Feature", "Mean Absolute SHAP"]].style.format({"Mean Absolute SHAP": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
            height=200
        )

    # Optional Native SHAP Beeswarm Plot
    st.markdown("### 🐝 SHAP Summary (Beeswarm) Distribution Plot")
    st.markdown(
        "<span style='color: #B8B1CC;'>Shows both the magnitude and the direction of impact. Red dots indicate high feature values, while blue dots indicate low feature values.</span>",
        unsafe_allow_html=True
    )
    if st.checkbox("Render Matplotlib SHAP Beeswarm Plot", value=False):
        with st.spinner("Rendering Matplotlib beeswarm distribution..."):
            fig_beeswarm = generate_shap_summary_figure_matplotlib(X_trans, shap_vals_selected, feature_names)
            st.pyplot(fig_beeswarm, clear_figure=True)

