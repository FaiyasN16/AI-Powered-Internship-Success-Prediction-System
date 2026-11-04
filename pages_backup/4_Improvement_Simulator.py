"""
Profile Improvement Simulator (What-If Analysis) Page.

Enables students to interactively experiment with profile improvements,
recalculating genuine Random Forest predictions for both original and simulated
profiles side-by-side to observe estimated selection probability deltas.
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ui_helpers import apply_custom_css, render_header, render_disclaimer
from src.preprocessing import VALID_DOMAINS, validate_student_profile
from src.prediction import predict_student_success, DEFAULT_MODEL_PATH

st.set_page_config(
    page_title="Improvement Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="Profile Improvement Simulator (What-If Analysis)",
    subtitle="Simulate hypothetical profile advancements and quantify their estimated impact on selection likelihood.",
    icon="⚡"
)

render_disclaimer()

if not os.path.exists(DEFAULT_MODEL_PATH):
    st.error("Trained model pipeline not found. Please train the model from the Home page first.")
    st.stop()

# 1. Baseline profile resolution
default_baseline = {
    "CGPA": 7.20,
    "Number of Relevant Projects": 2,
    "Average Project Complexity": 2,
    "Previous Internship Experience": "No",
    "Relevant Certifications": 0,
    "Number of Programming Languages": 2,
    "Number of Technical Skills": 4,
    "Internship Domain Applied": "Web Development",
}

baseline_profile = st.session_state.get("latest_profile", default_baseline)

# Preset options
st.markdown("### 🎯 Baseline Profile Setup")
preset_col, info_col = st.columns([1, 2])

with preset_col:
    preset_choice = st.selectbox(
        "Load Profile Preset or Current Session:",
        options=["Current Session Profile", "Introductory Applicant", "Average Candidate", "Competitive Candidate"],
        index=0 if "latest_profile" in st.session_state else 1
    )

if preset_choice == "Introductory Applicant":
    baseline_profile = {
        "CGPA": 6.80,
        "Number of Relevant Projects": 1,
        "Average Project Complexity": 1,
        "Previous Internship Experience": "No",
        "Relevant Certifications": 0,
        "Number of Programming Languages": 2,
        "Number of Technical Skills": 3,
        "Internship Domain Applied": "Web Development",
    }
elif preset_choice == "Average Candidate":
    baseline_profile = {
        "CGPA": 7.60,
        "Number of Relevant Projects": 3,
        "Average Project Complexity": 3,
        "Previous Internship Experience": "No",
        "Relevant Certifications": 1,
        "Number of Programming Languages": 3,
        "Number of Technical Skills": 5,
        "Internship Domain Applied": "Software Engineering",
    }
elif preset_choice == "Competitive Candidate":
    baseline_profile = {
        "CGPA": 8.70,
        "Number of Relevant Projects": 4,
        "Average Project Complexity": 4,
        "Previous Internship Experience": "Yes",
        "Relevant Certifications": 2,
        "Number of Programming Languages": 4,
        "Number of Technical Skills": 9,
        "Internship Domain Applied": "Artificial Intelligence and Machine Learning",
    }
elif preset_choice == "Current Session Profile" and "latest_profile" in st.session_state:
    baseline_profile = st.session_state["latest_profile"]

with info_col:
    st.markdown(
        f"""
        <div style="background-color: #1C1730; border: 1px solid #2D254D; border-radius: 8px; padding: 0.8rem 1.2rem; font-size: 0.88rem; color: #B8B1CC;">
            📌 <strong>Active Baseline Profile:</strong> CGPA {baseline_profile['CGPA']:.2f} |
            {baseline_profile['Number of Relevant Projects']} Projects |
            Complexity Level {baseline_profile['Average Project Complexity']} |
            Internship: {baseline_profile['Previous Internship Experience']} |
            {baseline_profile['Relevant Certifications']} Certs |
            {baseline_profile['Number of Programming Languages']} Languages |
            {baseline_profile['Number of Technical Skills']} Skills |
            Domain: {baseline_profile['Internship Domain Applied']}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# 2. Simulation Controls
st.markdown("### 🎛️ Adjust Simulated Profile Attributes")
st.markdown(
    "<span style='color: #B8B1CC;'>Modify any combination of attributes below to observe how the trained Random Forest model recalculates probabilities:</span>",
    unsafe_allow_html=True
)

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    sim_cgpa = st.slider(
        "Simulated CGPA (0.0 to 10.0):",
        min_value=5.0,
        max_value=10.0,
        value=float(baseline_profile["CGPA"]),
        step=0.05,
        help="Simulate an increase in academic performance."
    )

    sim_projects = st.slider(
        "Simulated Relevant Projects Count:",
        min_value=0,
        max_value=12,
        value=int(baseline_profile["Number of Relevant Projects"]),
        step=1,
        help="Simulate completing additional projects."
    )

    complexity_labels = {
        1: "1 - Basic",
        2: "2 - Intermediate",
        3: "3 - Moderate",
        4: "4 - Advanced",
        5: "5 - Production-grade"
    }
    sim_complexity = st.select_slider(
        "Simulated Project Complexity:",
        options=[1, 2, 3, 4, 5],
        value=int(baseline_profile["Average Project Complexity"]),
        format_func=lambda x: complexity_labels[x],
        help="Simulate refactoring projects to higher architectural complexity."
    )

    sim_intern = st.radio(
        "Simulated Internship Experience:",
        options=["No", "Yes"],
        index=1 if baseline_profile["Previous Internship Experience"] == "Yes" else 0,
        horizontal=True,
        help="Simulate gaining a prior internship or cooperative placement."
    )

with sim_col2:
    sim_certs = st.slider(
        "Simulated Relevant Certifications:",
        min_value=0,
        max_value=10,
        value=int(baseline_profile["Relevant Certifications"]),
        step=1,
        help="Simulate acquiring certified vendor credentials."
    )

    sim_langs = st.slider(
        "Simulated Programming Languages:",
        min_value=1,
        max_value=10,
        value=int(baseline_profile["Number of Programming Languages"]),
        step=1,
        help="Simulate expanding coding language proficiency."
    )

    sim_skills = st.slider(
        "Simulated Technical Skills Count:",
        min_value=1,
        max_value=25,
        value=int(baseline_profile["Number of Technical Skills"]),
        step=1,
        help="Simulate mastering developer tools and frameworks."
    )

    sim_domain = st.selectbox(
        "Simulated Internship Domain Applied:",
        options=VALID_DOMAINS,
        index=VALID_DOMAINS.index(baseline_profile["Internship Domain Applied"])
        if baseline_profile["Internship Domain Applied"] in VALID_DOMAINS else 0,
        help="Simulate switching target internship track."
    )

simulated_profile = {
    "CGPA": float(sim_cgpa),
    "Number of Relevant Projects": int(sim_projects),
    "Average Project Complexity": int(sim_complexity),
    "Previous Internship Experience": str(sim_intern),
    "Relevant Certifications": int(sim_certs),
    "Number of Programming Languages": int(sim_langs),
    "Number of Technical Skills": int(sim_skills),
    "Internship Domain Applied": str(sim_domain),
}

# Run Predictions on both Original and Simulated profiles
with st.spinner("Recalculating predictions for both profiles..."):
    orig_res = predict_student_success(baseline_profile)
    sim_res = predict_student_success(simulated_profile)

orig_prob = orig_res["probability_selected"] * 100
sim_prob = sim_res["probability_selected"] * 100
prob_diff = sim_prob - orig_prob

orig_outcome = orig_res["predicted_outcome"]
sim_outcome = sim_res["predicted_outcome"]

orig_score = orig_res["overall_score"]
sim_score = sim_res["overall_score"]
score_diff = sim_score - orig_score

orig_domain_score = orig_res["domain_readiness_score"]
sim_domain_score = sim_res["domain_readiness_score"]
domain_score_diff = sim_domain_score - orig_domain_score

st.markdown("<br><hr style='border-color: #2D254D;'><br>", unsafe_allow_html=True)
st.markdown("## 📊 Simulation Outcome & Comparative Analysis")

# Top Delta Callout Cards
sc1, sc2, sc3 = st.columns(3)

with sc1:
    delta_color = "#10B981" if prob_diff >= 0 else "#EF4444"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selection Probability Delta</div>
            <div class="metric-value" style="color: {delta_color};">{prob_diff:+.1f}%</div>
            <div class="metric-sub">{orig_prob:.1f}% &rarr; {sim_prob:.1f}% Estimated Probability</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with sc2:
    if orig_outcome == sim_outcome:
        status_label = f"Status Maintained: {sim_outcome}"
        status_color = "#A78BFA"
    elif orig_outcome == "Not Selected" and sim_outcome == "Selected":
        status_label = "Outcome Flipped to SELECTED! 🎉"
        status_color = "#10B981"
    else:
        status_label = "Outcome Dropped to NOT SELECTED ⚠️"
        status_color = "#EF4444"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Outcome Transition</div>
            <div class="metric-value" style="font-size: 1.5rem; color: {status_color}; margin: 0.6rem 0;">{status_label}</div>
            <div class="metric-sub">From <strong>{orig_outcome}</strong> to <strong>{sim_outcome}</strong></div>
        </div>
        """,
        unsafe_allow_html=True
    )

with sc3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Profile Score Delta</div>
            <div class="metric-value" style="color: {'#10B981' if score_diff >= 0 else '#EF4444'};">{score_diff:+.1f} pts</div>
            <div class="metric-sub">{orig_score} &rarr; {sim_score} Overall | Domain: {orig_domain_score} &rarr; {sim_domain_score}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Side-by-side Comparative Visuals
v_col1, v_col2 = st.columns(2)

with v_col1:
    # Probability comparison bar
    fig_prob_comp = go.Figure()
    fig_prob_comp.add_trace(go.Bar(
        x=["Original Profile", "Simulated Profile"],
        y=[orig_prob, sim_prob],
        marker_color=["#6D28D9", "#10B981" if sim_prob >= orig_prob else "#EF4444"],
        text=[f"{orig_prob:.1f}%", f"{sim_prob:.1f}%"],
        textposition="outside",
        textfont=dict(color="#F5F3FF", size=14, family="Inter"),
    ))
    fig_prob_comp.update_layout(
        title="<b>P(Selected) Probability Comparison</b>",
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        yaxis=dict(range=[0, 105], title="Probability (%)", gridcolor="#282245"),
        xaxis=dict(gridcolor="#282245"),
        height=320,
    )
    st.plotly_chart(fig_prob_comp, use_container_width=True)

with v_col2:
    # Score comparison bar
    fig_score_comp = go.Figure()
    fig_score_comp.add_trace(go.Bar(
        name="Overall Profile Score",
        x=["Original Profile", "Simulated Profile"],
        y=[orig_score, sim_score],
        marker_color="#A78BFA",
        text=[f"{orig_score}", f"{sim_score}"],
        textposition="inside",
    ))
    fig_score_comp.add_trace(go.Bar(
        name="Domain Readiness Score",
        x=["Original Profile", "Simulated Profile"],
        y=[orig_domain_score, sim_domain_score],
        marker_color="#059669",
        text=[f"{orig_domain_score}", f"{sim_domain_score}"],
        textposition="inside",
    ))
    fig_score_comp.update_layout(
        title="<b>Profile Scores Comparison (out of 100)</b>",
        barmode="group",
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 105], title="Score", gridcolor="#282245"),
        xaxis=dict(gridcolor="#282245"),
        height=320,
    )
    st.plotly_chart(fig_score_comp, use_container_width=True)

# Side-by-Side Detailed Feature Differences Table
st.markdown("### 📋 Side-by-Side Attribute Comparison")

features_to_compare = [
    ("CGPA", f"{baseline_profile['CGPA']:.2f}", f"{simulated_profile['CGPA']:.2f}", f"{simulated_profile['CGPA'] - baseline_profile['CGPA']:+.2f}"),
    ("Relevant Projects", str(baseline_profile["Number of Relevant Projects"]), str(simulated_profile["Number of Relevant Projects"]), f"{simulated_profile['Number of Relevant Projects'] - baseline_profile['Number of Relevant Projects']:+d}"),
    ("Project Complexity (1-5)", str(baseline_profile["Average Project Complexity"]), str(simulated_profile["Average Project Complexity"]), f"{simulated_profile['Average Project Complexity'] - baseline_profile['Average Project Complexity']:+d}"),
    ("Previous Internship Experience", baseline_profile["Previous Internship Experience"], simulated_profile["Previous Internship Experience"], "Changed" if baseline_profile["Previous Internship Experience"] != simulated_profile["Previous Internship Experience"] else "No Change"),
    ("Relevant Certifications", str(baseline_profile["Relevant Certifications"]), str(simulated_profile["Relevant Certifications"]), f"{simulated_profile['Relevant Certifications'] - baseline_profile['Relevant Certifications']:+d}"),
    ("Programming Languages", str(baseline_profile["Number of Programming Languages"]), str(simulated_profile["Number of Programming Languages"]), f"{simulated_profile['Number of Programming Languages'] - baseline_profile['Number of Programming Languages']:+d}"),
    ("Technical Skills Count", str(baseline_profile["Number of Technical Skills"]), str(simulated_profile["Number of Technical Skills"]), f"{simulated_profile['Number of Technical Skills'] - baseline_profile['Number of Technical Skills']:+d}"),
    ("Internship Domain", baseline_profile["Internship Domain Applied"], simulated_profile["Internship Domain Applied"], "Changed" if baseline_profile["Internship Domain Applied"] != simulated_profile["Internship Domain Applied"] else "No Change"),
]

df_diff = pd.DataFrame(features_to_compare, columns=["Feature Attribute", "Original Value", "Simulated Value", "Difference / Status"])
st.dataframe(df_diff, use_container_width=True, hide_index=True)

# Simulation Guidance & Recommended Next Actions
st.markdown("### 💡 Recommended Next Steps to Realize Simulated Gains")
sim_actions = sim_res["recommendations"]["action_items"]
for i, act in enumerate(sim_actions, 1):
    st.markdown(
        f"""
        <div style="background-color: #1C1730; border: 1px solid #2D254D; border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.8rem;">
            <span style="background-color: #10B981; color: #FFFFFF; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.8rem; flex-shrink: 0;">
                {i}
            </span>
            <span style="color: #F5F3FF; font-size: 0.95rem;">{act}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="disclaimer-box" style="margin-top: 1.5rem;">
        ⚠️ <strong>Simulation Interpretation Note:</strong> Hypothetical changes in this simulator reflect the
        statistical response surface of the trained Random Forest model. In real recruitment contexts, human
        evaluators evaluate portfolio evidence, communication, and interviews directly. No simulated score
        guarantees an employment outcome.
    </div>
    """,
    unsafe_allow_html=True
)

