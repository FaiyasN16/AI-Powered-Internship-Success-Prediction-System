"""
AI-Powered Internship Success Prediction System.

A clean, modern, and simple student internship decision-support web application.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import VALID_DOMAINS, validate_student_profile
from src.prediction import predict_student_success, DEFAULT_MODEL_PATH
from src.explainability import compute_single_prediction_shap
from train_model import train_and_evaluate

# Page configuration
st.set_page_config(
    page_title="Internship Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, modern, minimalist dark-purple styling
st.markdown("""
<style>
    /* Global theme */
    .stApp {
        background-color: #0E0B1A;
        color: #F5F3FF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container cards */
    .neat-card {
        background-color: #181329;
        border: 1px solid #2B2347;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }

    /* Big title */
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 40%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-sub {
        color: #B8B1CC;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge-selected {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 0.5rem 1.8rem;
        border-radius: 50px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }
    .badge-not-selected {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 0.5rem 1.8rem;
        border-radius: 50px;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #A78BFA !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        width: 100%;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(167, 139, 250, 0.5) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #2B2347;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #181329;
        border: 1px solid #2B2347;
        border-radius: 8px 8px 0 0;
        color: #B8B1CC;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2B2347 !important;
        color: #FFFFFF !important;
        border-color: #7C3AED !important;
    }

    /* KPI metric mini card */
    .kpi-box {
        background-color: #181329;
        border: 1px solid #2B2347;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #A78BFA;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #B8B1CC;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Ensure trained model exists
if not os.path.exists(DEFAULT_MODEL_PATH):
    st.info("⚙️ Initializing model and dataset for first-time setup...")
    with st.spinner("Training model pipeline..."):
        train_and_evaluate()
    st.rerun()

# Header
st.markdown("""
<div>
    <div class="app-title">🎓 AI Internship Success Predictor</div>
    <div class="app-sub">Check your internship selection chances, discover strengths, and simulate improvements.</div>
</div>
""", unsafe_allow_html=True)

# Clean Navigation Tabs
tab_predict, tab_simulator, tab_analytics, tab_about = st.tabs([
    "🎯 Predict Outcome",
    "⚡ What-If Simulator",
    "📊 Quick Analytics",
    "ℹ️ About Project"
])

# ==========================================
# TAB 1: PREDICT OUTCOME (MAIN CORE FEATURE)
# ==========================================
with tab_predict:
    st.markdown("### 📋 Enter Your Profile")
    st.markdown("<span style='color: #B8B1CC; font-size: 0.95rem;'>Fill in your details below and click Predict.</span>", unsafe_allow_html=True)

    with st.form("simple_prediction_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            cgpa = st.number_input("1. CGPA (0.0 to 10.0)", min_value=0.0, max_value=10.0, value=7.80, step=0.1, help="Your academic score out of 10.0")
            internship = st.radio("2. Previous Internship Experience?", options=["No", "Yes"], index=0, horizontal=True)
            domain = st.selectbox("3. Target Internship Domain", options=VALID_DOMAINS, index=0)
            certs = st.number_input("4. Relevant Certifications (Count)", min_value=0, max_value=15, value=1, step=1)

        with col_right:
            projects = st.number_input("5. Number of Relevant Projects", min_value=0, max_value=20, value=3, step=1)
            complexity_opts = {
                1: "Level 1 - Basic (Starter scripts)",
                2: "Level 2 - Intermediate (Simple apps)",
                3: "Level 3 - Moderate (Database & APIs)",
                4: "Level 4 - Advanced (Full-stack with auth)",
                5: "Level 5 - Production-grade (Deployed & scalable)"
            }
            complexity = st.selectbox("6. Average Project Complexity", options=[1, 2, 3, 4, 5], index=2, format_func=lambda x: complexity_opts[x])
            languages = st.number_input("7. Programming Languages Known", min_value=1, max_value=10, value=3, step=1)
            skills = st.number_input("8. Number of Technical Skills / Tools", min_value=1, max_value=25, value=6, step=1)

        submitted = st.form_submit_button("🔮 Predict My Internship Outcome")

    if submitted:
        profile = {
            "CGPA": float(cgpa),
            "Number of Relevant Projects": int(projects),
            "Average Project Complexity": int(complexity),
            "Previous Internship Experience": str(internship),
            "Relevant Certifications": int(certs),
            "Number of Programming Languages": int(languages),
            "Number of Technical Skills": int(skills),
            "Internship Domain Applied": str(domain),
        }

        is_valid, errors = validate_student_profile(profile)
        if not is_valid:
            st.error("❌ " + " ".join(errors))
        else:
            with st.spinner("Analyzing profile..."):
                result = predict_student_success(profile)
                st.session_state["active_profile"] = profile
                st.session_state["active_result"] = result

    # Display clean results if available
    if "active_result" in st.session_state:
        res = st.session_state["active_result"]
        prob_sel = res["probability_selected"]
        outcome = res["predicted_outcome"]
        overall_score = res["overall_score"]
        readiness_score = res["domain_readiness_score"]
        suitability = res["domain_suitability"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Your Prediction Results")

        # Top row: Big Badge + Progress + Scores
        r_col1, r_col2, r_col3 = st.columns([1.2, 1, 1])

        with r_col1:
            st.markdown(
                f"""
                <div class="neat-card" style="text-align: center; padding: 1.8rem 1rem;">
                    <div style="font-size: 0.85rem; color: #B8B1CC; text-transform: uppercase; margin-bottom: 0.6rem;">Estimated Outcome</div>
                    <div class="{'badge-selected' if outcome == 'Selected' else 'badge-not-selected'}">
                        {'🎉 Likely Selected' if outcome == 'Selected' else '⚠️ Needs Improvement'}
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F5F3FF; margin-top: 1rem;">
                        {prob_sel * 100:.1f}% <span style="font-size: 0.9rem; color: #B8B1CC;">Chance</span>
                    </div>
                    <div style="background-color: #2B2347; border-radius: 8px; height: 10px; margin-top: 0.5rem; overflow: hidden;">
                        <div style="background: {'linear-gradient(90deg, #059669, #10B981)' if prob_sel >= 0.5 else 'linear-gradient(90deg, #DC2626, #EF4444)'}; width: {prob_sel * 100}%; height: 100%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r_col2:
            st.markdown(
                f"""
                <div class="neat-card" style="text-align: center; padding: 1.8rem 1rem;">
                    <div style="font-size: 0.85rem; color: #B8B1CC; text-transform: uppercase; margin-bottom: 0.6rem;">Profile Score</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #A78BFA;">{overall_score} <span style="font-size: 1.1rem; color: #B8B1CC;">/ 100</span></div>
                    <div style="font-size: 0.88rem; color: #B8B1CC; margin-top: 0.5rem;">Overall Academic & Technical Strength</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r_col3:
            st.markdown(
                f"""
                <div class="neat-card" style="text-align: center; padding: 1.8rem 1rem;">
                    <div style="font-size: 0.85rem; color: #B8B1CC; text-transform: uppercase; margin-bottom: 0.6rem;">Domain Readiness</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #10B981;">{readiness_score} <span style="font-size: 1.1rem; color: #B8B1CC;">/ 100</span></div>
                    <div style="font-size: 0.88rem; color: #A78BFA; font-weight: 600; margin-top: 0.5rem;">{suitability}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Bottom row: Strengths & Next Action Steps
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            st.markdown(
                """
                <div class="neat-card">
                    <h4 style="color: #10B981; margin-top: 0;">🌟 Your Strengths</h4>
                """,
                unsafe_allow_html=True
            )
            for s in res["recommendations"]["strengths"]:
                st.markdown(f"- ✅ {s}")
            st.markdown("</div>", unsafe_allow_html=True)

        with f_col2:
            st.markdown(
                """
                <div class="neat-card">
                    <h4 style="color: #A78BFA; margin-top: 0;">🚀 Recommended Next Steps</h4>
                """,
                unsafe_allow_html=True
            )
            for a in res["recommendations"]["action_items"][:3]:
                st.markdown(f"- 📌 {a}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Simple plain-English reasoning
        try:
            from src.prediction import load_trained_pipeline
            pipe = load_trained_pipeline()
            shap_info = compute_single_prediction_shap(pipe, res["input_profile"])
            st.markdown(
                f"""
                <div style="background-color: #181329; border-left: 4px solid #7C3AED; padding: 0.8rem 1.2rem; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #F5F3FF; margin-top: 0.5rem;">
                    💡 <strong>What Influenced This Result:</strong> {shap_info['narrative']}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            pass

# ==========================================
# TAB 2: WHAT-IF SIMULATOR
# ==========================================
with tab_simulator:
    st.markdown("### ⚡ What-If Profile Simulator")
    st.markdown("<span style='color: #B8B1CC; font-size: 0.95rem;'>Adjust the sliders below to see how improving your profile changes your selection chances in real-time.</span>", unsafe_allow_html=True)

    base = st.session_state.get("active_profile", {
        "CGPA": 7.5,
        "Number of Relevant Projects": 2,
        "Average Project Complexity": 2,
        "Previous Internship Experience": "No",
        "Relevant Certifications": 0,
        "Number of Programming Languages": 2,
        "Number of Technical Skills": 4,
        "Internship Domain Applied": "Web Development",
    })

    sim_c1, sim_c2 = st.columns(2)

    with sim_c1:
        sim_cgpa = st.slider("Simulate CGPA:", 5.0, 10.0, float(base["CGPA"]), 0.1)
        sim_projects = st.slider("Simulate Number of Projects:", 0, 10, int(base["Number of Relevant Projects"]), 1)
        sim_complexity = st.slider("Simulate Project Complexity (1=Basic to 5=Production):", 1, 5, int(base["Average Project Complexity"]), 1)
        sim_intern = st.radio("Simulate Prior Internship:", ["No", "Yes"], index=1 if base["Previous Internship Experience"] == "Yes" else 0, horizontal=True)

    with sim_c2:
        sim_certs = st.slider("Simulate Certifications:", 0, 8, int(base["Relevant Certifications"]), 1)
        sim_skills = st.slider("Simulate Technical Skills:", 1, 20, int(base["Number of Technical Skills"]), 1)
        sim_langs = st.slider("Simulate Languages:", 1, 8, int(base["Number of Programming Languages"]), 1)
        sim_domain = st.selectbox("Simulate Domain:", VALID_DOMAINS, index=VALID_DOMAINS.index(base["Internship Domain Applied"]) if base["Internship Domain Applied"] in VALID_DOMAINS else 0)

    sim_profile = {
        "CGPA": float(sim_cgpa),
        "Number of Relevant Projects": int(sim_projects),
        "Average Project Complexity": int(sim_complexity),
        "Previous Internship Experience": str(sim_intern),
        "Relevant Certifications": int(sim_certs),
        "Number of Programming Languages": int(sim_langs),
        "Number of Technical Skills": int(sim_skills),
        "Internship Domain Applied": str(sim_domain),
    }

    # Recalculate
    base_res = predict_student_success(base)
    new_res = predict_student_success(sim_profile)

    base_p = base_res["probability_selected"] * 100
    new_p = new_res["probability_selected"] * 100
    diff = new_p - base_p

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Simulation Impact")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Baseline Chance</div>
                <div class="kpi-num" style="color: #B8B1CC;">{base_p:.1f}%</div>
                <div style="font-size: 0.8rem; color: #948BA8;">Original Profile</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Simulated Chance</div>
                <div class="kpi-num" style="color: {'#10B981' if new_p >= 50 else '#EF4444'};">{new_p:.1f}%</div>
                <div style="font-size: 0.8rem; color: #10B981;">{new_res['predicted_outcome']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        diff_color = "#10B981" if diff >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Change in Probability</div>
                <div class="kpi-num" style="color: {diff_color};">{diff:+.1f}%</div>
                <div style="font-size: 0.8rem; color: #B8B1CC;">Net Impact</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# TAB 3: QUICK ANALYTICS
# ==========================================
with tab_analytics:
    st.markdown("### 📊 Dataset Trends & Insights")
    st.markdown("<span style='color: #B8B1CC; font-size: 0.95rem;'>Simple charts showing what makes applicants competitive based on 1,000 student records.</span>", unsafe_allow_html=True)

    data_file = "data/internship_dataset.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)

        a_col1, a_col2 = st.columns(2)

        with a_col1:
            fig1 = px.histogram(
                df, x="CGPA", color="Internship Outcome",
                color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
                barmode="overlay", nbins=20,
                title="<b>CGPA vs. Selection Outcome</b>"
            )
            fig1.update_layout(paper_bgcolor="#181329", plot_bgcolor="#181329", font_color="#F5F3FF", height=320)
            st.plotly_chart(fig1, use_container_width=True)

        with a_col2:
            fig2 = px.histogram(
                df, x="Number of Relevant Projects", color="Internship Outcome",
                color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
                barmode="group",
                title="<b>Number of Projects vs. Selection</b>"
            )
            fig2.update_layout(paper_bgcolor="#181329", plot_bgcolor="#181329", font_color="#F5F3FF", height=320)
            st.plotly_chart(fig2, use_container_width=True)

        # Domain breakdown
        domain_stats = df.groupby("Internship Domain Applied").agg(
            Applicants=("Internship Outcome", "count"),
            Selected=("Internship Outcome", lambda x: (x == "Selected").sum())
        ).reset_index()
        domain_stats["Selection Rate %"] = (domain_stats["Selected"] / domain_stats["Applicants"] * 100).round(1)

        fig3 = px.bar(
            domain_stats, x="Internship Domain Applied", y="Selection Rate %",
            color="Selection Rate %",
            color_continuous_scale=[[0, "#6D28D9"], [1, "#10B981"]],
            title="<b>Selection Rate (%) Across Technical Domains</b>",
            text="Selection Rate %"
        )
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        fig3.update_layout(paper_bgcolor="#181329", plot_bgcolor="#181329", font_color="#F5F3FF", height=340, xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# TAB 4: ABOUT PROJECT
# ==========================================
with tab_about:
    st.markdown("### ℹ️ About the Project")

    metrics_file = "models/model_metrics.json"
    acc = 77.5
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            m = json.load(f)
            acc = m.get("accuracy", 0.775) * 100

    st.markdown(
        f"""
        <div class="neat-card">
            <h4 style="color: #A78BFA; margin-top: 0;">📌 Project Overview</h4>
            <p style="color: #F5F3FF;">
                The <strong>AI-Powered Internship Success Prediction System</strong> helps college students evaluate their readiness for competitive internships.
                It uses a machine learning model to estimate your chances based on your academic record, projects, certifications, and technical skills.
            </p>
            <h4 style="color: #A78BFA;">⚙️ Machine Learning Model</h4>
            <ul style="color: #B8B1CC; line-height: 1.8;">
                <li><strong>Algorithm:</strong> Random Forest Classifier (120 Decision Trees)</li>
                <li><strong>Trained Dataset:</strong> 1,000 synthetic student applicant profiles</li>
                <li><strong>Model Accuracy:</strong> <span style="color: #10B981; font-weight: 700;">{acc:.1f}%</span> on held-out test data</li>
                <li><strong>Explainability:</strong> Powered by SHAP (SHapley Additive exPlanations)</li>
            </ul>
            <div style="background-color: rgba(124, 58, 237, 0.1); border-left: 3px solid #7C3AED; padding: 0.6rem 1rem; border-radius: 4px; font-size: 0.85rem; color: #B8B1CC; margin-top: 1rem;">
                ⚖️ <strong>Academic Disclaimer:</strong> This application is a decision-support guide for educational self-assessment and does not guarantee real-world hiring outcomes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
