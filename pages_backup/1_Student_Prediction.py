"""
Student Internship Prediction Page.

Collects student academic, technical, and experiential profile data,
validates inputs, and runs inference through the trained Random Forest pipeline.
Displays probabilistic outcome, profile scores, domain readiness, candidate vs.
ideal profile gap analysis, and tailored recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ui_helpers import (
    apply_custom_css,
    render_header,
    render_disclaimer,
    create_probability_gauge_plotly,
)
from src.preprocessing import VALID_DOMAINS, validate_student_profile
from src.prediction import predict_student_success

st.set_page_config(
    page_title="Internship Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="Internship Selection Prediction & Diagnostic Report",
    subtitle="Evaluate your student profile against industry internship benchmarks using our trained Random Forest model.",
    icon="🎓"
)

render_disclaimer()

# Setup default profile if in session
default_profile = st.session_state.get("latest_profile", {
    "CGPA": 7.85,
    "Number of Relevant Projects": 3,
    "Average Project Complexity": 3,
    "Previous Internship Experience": "No",
    "Relevant Certifications": 1,
    "Number of Programming Languages": 3,
    "Number of Technical Skills": 6,
    "Internship Domain Applied": "Artificial Intelligence and Machine Learning",
})

# Student Input Form
st.markdown("### 📋 Student Candidate Profile Input")
st.markdown(
    "<span style='color: #B8B1CC;'>Fill in your current academic standing, practical projects, and technical skills below:</span>",
    unsafe_allow_html=True
)

with st.form("student_profile_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📚 Academic & Experiential Standing")
        cgpa_val = st.number_input(
            "Cumulative Grade Point Average (CGPA) (0.0 to 10.0)",
            min_value=0.0,
            max_value=10.0,
            value=float(default_profile.get("CGPA", 7.85)),
            step=0.05,
            format="%.2f",
            help="Your official university CGPA scaled out of 10.0."
        )

        internship_exp = st.radio(
            "Previous Internship Experience",
            options=["No", "Yes"],
            index=1 if default_profile.get("Previous Internship Experience") == "Yes" else 0,
            horizontal=True,
            help="Have you previously completed an industry or corporate internship?"
        )

        domain_applied = st.selectbox(
            "Target Internship Domain Applied",
            options=VALID_DOMAINS,
            index=VALID_DOMAINS.index(default_profile.get("Internship Domain Applied", VALID_DOMAINS[0]))
            if default_profile.get("Internship Domain Applied") in VALID_DOMAINS else 0,
            help="Select the technical domain or role you are targeting."
        )

        relevant_certs = st.number_input(
            "Relevant Industry Certifications (Count)",
            min_value=0,
            max_value=15,
            value=int(default_profile.get("Relevant Certifications", 1)),
            step=1,
            help="Number of relevant course certificates or vendor credentials (AWS, Meta, Google, etc.)."
        )

    with col2:
        st.markdown("#### 💻 Technical & Project Portfolio")
        relevant_projects = st.number_input(
            "Number of Relevant Projects (Count)",
            min_value=0,
            max_value=20,
            value=int(default_profile.get("Number of Relevant Projects", 3)),
            step=1,
            help="Practical software, research, or development projects relevant to the target domain."
        )

        complexity_labels = {
            1: "1 - Basic (Tutorial or single-file scripts)",
            2: "2 - Intermediate (Basic CRUD, basic UI/CLI)",
            3: "3 - Moderate (Database, REST APIs, client-server)",
            4: "4 - Advanced (Full-stack, authentication, CI/CD, testing)",
            5: "5 - Production-grade (Microservices, deployed cloud scale, real users)"
        }
        complexity_val = st.selectbox(
            "Average Project Complexity",
            options=[1, 2, 3, 4, 5],
            index=int(default_profile.get("Average Project Complexity", 3)) - 1,
            format_func=lambda x: complexity_labels[x],
            help="Self-assessed architectural and algorithmic depth of your projects."
        )

        prog_languages = st.number_input(
            "Number of Programming Languages Known",
            min_value=1,
            max_value=15,
            value=int(default_profile.get("Number of Programming Languages", 3)),
            step=1,
            help="Languages you can write and debug comfortably (e.g., Python, C++, Java, JS)."
        )

        tech_skills = st.number_input(
            "Number of Technical Tools & Skills",
            min_value=1,
            max_value=30,
            value=int(default_profile.get("Number of Technical Skills", 6)),
            step=1,
            help="Tools, frameworks, and libraries mastered (e.g., Docker, Git, SQL, React, PyTorch)."
        )

    submit_button = st.form_submit_button("🔮 Predict Internship Success", use_container_width=True)

if submit_button:
    profile_candidate = {
        "CGPA": float(cgpa_val),
        "Number of Relevant Projects": int(relevant_projects),
        "Average Project Complexity": int(complexity_val),
        "Previous Internship Experience": str(internship_exp),
        "Relevant Certifications": int(relevant_certs),
        "Number of Programming Languages": int(prog_languages),
        "Number of Technical Skills": int(tech_skills),
        "Internship Domain Applied": str(domain_applied),
    }

    # Validate
    is_valid, validation_errors = validate_student_profile(profile_candidate)
    if not is_valid:
        st.error(f"❌ Input Validation Failed:\n" + "\n".join([f"- {e}" for e in validation_errors]))
        st.stop()

    with st.spinner("Processing profile through Scikit-learn Random Forest Pipeline..."):
        try:
            result = predict_student_success(profile_candidate)
            st.session_state["latest_profile"] = profile_candidate
            st.session_state["latest_result"] = result
        except Exception as e:
            st.error(f"Error during model prediction: {str(e)}")
            st.stop()

# If results are available in session
if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    prob_sel = result["probability_selected"]
    prob_not = result["probability_not_selected"]
    outcome = result["predicted_outcome"]
    overall_score = result["overall_score"]
    domain_score = result["domain_readiness_score"]
    suitability = result["domain_suitability"]
    summary = result["domain_summary"]

    st.markdown("<br><hr style='border-color: #2D254D;'><br>", unsafe_allow_html=True)
    st.markdown("## 🎯 Prediction & Diagnostic Results")

    # Primary Outcome Banner
    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        st.markdown(
            f"""
            <div class="dashboard-card" style="text-align: center; padding: 2rem 1.5rem;">
                <div style="font-size: 0.95rem; text-transform: uppercase; color: #B8B1CC; letter-spacing: 0.08em; margin-bottom: 0.8rem;">
                    Predicted Selection Status
                </div>
                <div>
                    <span class="{'badge-selected' if outcome == 'Selected' else 'badge-not-selected'}">
                        {'🎉 SELECTED' if outcome == 'Selected' else '⚠️ NOT SELECTED'}
                    </span>
                </div>
                <p style="color: #B8B1CC; font-size: 0.85rem; margin-top: 1.2rem; margin-bottom: 0;">
                    Decision based on Random Forest majority voting threshold (P &ge; 50%).
                </p>
                <div style="margin-top: 1.5rem; display: flex; justify-content: space-around;">
                    <div>
                        <div style="font-size: 0.8rem; color: #948BA8;">P(Selected)</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #10B981;">{prob_sel * 100:.1f}%</div>
                    </div>
                    <div style="border-left: 1px solid #3B2E68;"></div>
                    <div>
                        <div style="font-size: 0.8rem; color: #948BA8;">P(Not Selected)</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #EF4444;">{prob_not * 100:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col2:
        gauge_fig = create_probability_gauge_plotly(prob_sel)
        st.plotly_chart(gauge_fig, use_container_width=True)

    # Secondary Metrics: Overall Score & Domain Readiness
    score_col1, score_col2 = st.columns(2)

    with score_col1:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #A78BFA;">🏆 Overall Profile Score</h4>
                    <span style="font-size: 1.8rem; font-weight: 800; color: #F5F3FF;">{overall_score} <span style="font-size: 1rem; color: #948BA8;">/ 100</span></span>
                </div>
                <div style="background-color: #241D40; border-radius: 9999px; height: 10px; margin: 0.8rem 0; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%); width: {overall_score}%; height: 100%;"></div>
                </div>
                <p style="color: #B8B1CC; font-size: 0.85rem; margin-bottom: 0.6rem;">
                    Transparent heuristic evaluating academic record, project volume, depth, prior internships, certifications, and technical scope.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("🔍 View Profile Score Component Breakdown"):
            for comp_name, comp_val in result["overall_breakdown"].items():
                st.write(f"- **{comp_name}:** {comp_val} pts")

    with score_col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #A78BFA;">🎯 Domain Readiness Score</h4>
                    <span style="font-size: 1.8rem; font-weight: 800; color: #F5F3FF;">{domain_score} <span style="font-size: 1rem; color: #948BA8;">/ 100</span></span>
                </div>
                <div style="background-color: #241D40; border-radius: 9999px; height: 10px; margin: 0.8rem 0; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #059669 0%, #10B981 100%); width: {domain_score}%; height: 100%;"></div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.4rem;">
                    <span style="background-color: #2D254D; color: #A78BFA; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">
                        {suitability}
                    </span>
                    <span style="color: #B8B1CC; font-size: 0.85rem;">for {result['input_profile']['Internship Domain Applied']}</span>
                </div>
                <p style="color: #948BA8; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0;">
                    {summary}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Candidate vs. Ideal Profile Benchmark Comparison
    st.markdown("### ⚖️ Candidate Profile vs. Ideal Target Benchmark")
    st.markdown(
        "<span style='color: #B8B1CC;'>Target values represent aspirational reference goals for competitive applicants, not rigid hiring barriers:</span>",
        unsafe_allow_html=True
    )
    df_comparison = result["ideal_comparison"]
    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True
    )

    # Strengths & Improvement Areas
    col_str, col_imp = st.columns(2)

    with col_str:
        st.markdown(
            """
            <div class="dashboard-card" style="border-left: 4px solid #10B981;">
                <h4 style="color: #10B981; margin-top: 0;">💪 Demonstrated Strengths</h4>
            """,
            unsafe_allow_html=True
        )
        for s in result["recommendations"]["strengths"]:
            st.markdown(f"- ✅ {s}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_imp:
        st.markdown(
            """
            <div class="dashboard-card" style="border-left: 4px solid #F59E0B;">
                <h4 style="color: #F59E0B; margin-top: 0;">📈 Recommended Focus Areas</h4>
            """,
            unsafe_allow_html=True
        )
        for imp in result["recommendations"]["improvement_areas"]:
            st.markdown(f"- ⚠️ {imp}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Prioritized Action Items
    st.markdown("### 📌 Prioritized Action Items (Tailored to Domain)")
    for i, action in enumerate(result["recommendations"]["action_items"], 1):
        st.markdown(
            f"""
            <div style="background-color: #1C1730; border: 1px solid #2D254D; border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.8rem;">
                <span style="background-color: #7C3AED; color: #FFFFFF; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; flex-shrink: 0;">
                    {i}
                </span>
                <span style="color: #F5F3FF; font-size: 0.95rem;">{action}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Short-term Improvement Roadmap
    st.markdown("### 🗺️ Suggested 4-Phase Short-Term Roadmap")
    for phase_info in result["recommendations"]["roadmap"]:
        with st.expander(f"📍 {phase_info['phase']}"):
            st.write(f"**Focus:** {phase_info['focus']}")
            st.write(f"**Key Deliverable:** {phase_info['deliverable']}")

    # Quick next actions
    st.markdown("<br>", unsafe_allow_html=True)
    n_col1, n_col2 = st.columns(2)
    with n_col1:
        st.page_link("pages/2_Explainable_AI.py", label="🔍 View SHAP Model Explainability for this Profile", icon="📊")
    with n_col2:
        st.page_link("pages/4_Improvement_Simulator.py", label="⚡ Simulate Profile Changes in What-If Simulator", icon="🧪")

