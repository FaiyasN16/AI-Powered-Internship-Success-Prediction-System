"""
Student Analytics Dashboard Page.

Provides comprehensive visual explorations of the student applicant population,
outcome distributions, feature interactions, and domain benchmarks using Plotly.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ui_helpers import apply_custom_css, render_header, render_disclaimer

st.set_page_config(
    page_title="Student Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="Student Applicant Analytics Dashboard",
    subtitle="Exploratory data analysis of student profile metrics, outcome correlations, and domain benchmarks.",
    icon="📈"
)

render_disclaimer()

DATA_PATH = "data/internship_dataset.csv"
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found at 'data/internship_dataset.csv'. Please generate it from the Home page.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Interactive Filters
st.markdown("### 🔍 Filter Analytics Data")
col_f1, col_f2 = st.columns([2, 1])

with col_f1:
    selected_domains = st.multiselect(
        "Filter by Internship Domain:",
        options=sorted(df["Internship Domain Applied"].unique()),
        default=sorted(df["Internship Domain Applied"].unique())
    )

with col_f2:
    selected_intern_status = st.multiselect(
        "Filter by Outcome Status:",
        options=["Selected", "Not Selected"],
        default=["Selected", "Not Selected"]
    )

filtered_df = df[
    (df["Internship Domain Applied"].isin(selected_domains)) &
    (df["Internship Outcome"].isin(selected_intern_status))
]

if filtered_df.empty:
    st.warning("⚠️ No student records match the selected filters. Please expand your filter selection.")
    st.stop()

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_count = len(filtered_df)
sel_count = (filtered_df["Internship Outcome"] == "Selected").sum()
sel_rate = (sel_count / total_count * 100) if total_count > 0 else 0
avg_cgpa = filtered_df["CGPA"].mean()
avg_proj = filtered_df["Number of Relevant Projects"].mean()
prev_intern_pct = (filtered_df["Previous Internship Experience"] == "Yes").mean() * 100

with kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Filtered Students</div>
            <div class="metric-value">{total_count:,}</div>
            <div class="metric-sub">Out of {len(df):,} Total</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Selection Rate</div>
            <div class="metric-value">{sel_rate:.1f}%</div>
            <div class="metric-sub">{sel_count} Selected Candidates</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Mean CGPA</div>
            <div class="metric-value">{avg_cgpa:.2f}</div>
            <div class="metric-sub">Out of 10.0 Scale</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Projects</div>
            <div class="metric-value">{avg_proj:.1f}</div>
            <div class="metric-sub">Per Student Profile</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Prior Internship</div>
            <div class="metric-value">{prev_intern_pct:.1f}%</div>
            <div class="metric-sub">Reported Workplace Exp</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ROW 1: CGPA Distribution & Outcome Breakdown
row1_col1, row1_col2 = st.columns([1.4, 1])

with row1_col1:
    fig_cgpa = px.histogram(
        filtered_df,
        x="CGPA",
        color="Internship Outcome",
        barmode="overlay",
        marginal="box",
        nbins=25,
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>1. CGPA Distribution by Selection Outcome</b>",
        labels={"CGPA": "Cumulative GPA", "count": "Student Count"}
    )
    fig_cgpa.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#282245"),
        yaxis=dict(gridcolor="#282245"),
        height=380,
    )
    st.plotly_chart(fig_cgpa, use_container_width=True)

with row1_col2:
    outcome_counts = filtered_df["Internship Outcome"].value_counts().reset_index()
    outcome_counts.columns = ["Outcome", "Count"]
    fig_outcome = px.pie(
        outcome_counts,
        values="Count",
        names="Outcome",
        hole=0.55,
        color="Outcome",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>2. Overall Outcome Distribution</b>"
    )
    fig_outcome.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        height=380,
    )
    st.plotly_chart(fig_outcome, use_container_width=True)

# ROW 2: Projects & Complexity vs. Outcome
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    proj_grouped = filtered_df.groupby(["Number of Relevant Projects", "Internship Outcome"]).size().reset_index(name="Count")
    fig_proj = px.bar(
        proj_grouped,
        x="Number of Relevant Projects",
        y="Count",
        color="Internship Outcome",
        barmode="group",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>3. Relevant Projects Count vs. Selection Outcome</b>",
        labels={"Number of Relevant Projects": "Projects Completed", "Count": "Students"}
    )
    fig_proj.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        xaxis=dict(gridcolor="#282245", dtick=1),
        yaxis=dict(gridcolor="#282245"),
        height=360,
    )
    st.plotly_chart(fig_proj, use_container_width=True)

with row2_col2:
    comp_grouped = filtered_df.groupby(["Average Project Complexity", "Internship Outcome"]).size().reset_index(name="Count")
    comp_labels = {1: "1-Basic", 2: "2-Intermediate", 3: "3-Moderate", 4: "4-Advanced", 5: "5-Production"}
    comp_grouped["Complexity Level"] = comp_grouped["Average Project Complexity"].map(comp_labels)

    fig_comp = px.bar(
        comp_grouped,
        x="Complexity Level",
        y="Count",
        color="Internship Outcome",
        barmode="group",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>4. Project Complexity vs. Selection Outcome</b>",
        labels={"Count": "Students"}
    )
    fig_comp.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        xaxis=dict(gridcolor="#282245"),
        yaxis=dict(gridcolor="#282245"),
        height=360,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ROW 3: Prior Internship & Certifications
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    intern_grouped = filtered_df.groupby(["Previous Internship Experience", "Internship Outcome"]).size().reset_index(name="Count")
    fig_intern = px.bar(
        intern_grouped,
        x="Previous Internship Experience",
        y="Count",
        color="Internship Outcome",
        barmode="stack",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>5. Previous Internship Experience vs. Selection Outcome</b>",
        labels={"Previous Internship Experience": "Prior Internship Completed?", "Count": "Students"}
    )
    fig_intern.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        xaxis=dict(gridcolor="#282245"),
        yaxis=dict(gridcolor="#282245"),
        height=360,
    )
    st.plotly_chart(fig_intern, use_container_width=True)

with row3_col2:
    cert_grouped = filtered_df.groupby(["Relevant Certifications", "Internship Outcome"]).size().reset_index(name="Count")
    fig_cert = px.bar(
        cert_grouped,
        x="Relevant Certifications",
        y="Count",
        color="Internship Outcome",
        barmode="group",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>6. Certifications Count vs. Selection Outcome</b>",
        labels={"Relevant Certifications": "Certifications Earned", "Count": "Students"}
    )
    fig_cert.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        xaxis=dict(gridcolor="#282245", dtick=1),
        yaxis=dict(gridcolor="#282245"),
        height=360,
    )
    st.plotly_chart(fig_cert, use_container_width=True)

# ROW 4: Technical Skills & Programming Languages Distributions
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    fig_skills = px.box(
        filtered_df,
        x="Internship Outcome",
        y="Number of Technical Skills",
        color="Internship Outcome",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>7. Technical Skills Breadth by Outcome</b>"
    )
    fig_skills.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        xaxis=dict(gridcolor="#282245"),
        yaxis=dict(gridcolor="#282245"),
        height=360,
    )
    st.plotly_chart(fig_skills, use_container_width=True)

with row4_col2:
    fig_langs = px.histogram(
        filtered_df,
        x="Number of Programming Languages",
        color="Internship Outcome",
        barmode="group",
        color_discrete_map={"Selected": "#10B981", "Not Selected": "#EF4444"},
        title="<b>8. Programming Languages Count by Outcome</b>",
        labels={"Number of Programming Languages": "Programming Languages", "count": "Students"}
    )
    fig_langs.update_layout(
        paper_bgcolor="#161226",
        plot_bgcolor="#161226",
        font=dict(color="#F5F3FF", family="Inter"),
        xaxis=dict(gridcolor="#282245", dtick=1),
        yaxis=dict(gridcolor="#282245"),
        height=360,
    )
    st.plotly_chart(fig_langs, use_container_width=True)

# ROW 5: Domain-wise Outcome Comparison
st.markdown("### 🌐 9. Domain-Wise Applicant & Selection Comparison")
domain_summary = df.groupby("Internship Domain Applied").agg(
    Total_Applicants=("Internship Outcome", "count"),
    Selected_Count=("Internship Outcome", lambda x: (x == "Selected").sum()),
).reset_index()
domain_summary["Selection_Rate_Pct"] = (domain_summary["Selected_Count"] / domain_summary["Total_Applicants"] * 100).round(1)

fig_domain = px.bar(
    domain_summary,
    x="Internship Domain Applied",
    y="Total_Applicants",
    color="Selection_Rate_Pct",
    color_continuous_scale=[[0, "#4C1D95"], [0.5, "#7C3AED"], [1, "#10B981"]],
    title="<b>Domain-Wise Applicants and Selection Success Rates (%)</b>",
    labels={"Internship Domain Applied": "Domain", "Total_Applicants": "Total Applicants", "Selection_Rate_Pct": "Selection Rate (%)"},
    text="Selection_Rate_Pct"
)
fig_domain.update_traces(texttemplate="%{text}%", textposition="outside")
fig_domain.update_layout(
    paper_bgcolor="#161226",
    plot_bgcolor="#161226",
    font=dict(color="#F5F3FF", family="Inter"),
    xaxis=dict(gridcolor="#282245", tickangle=-25),
    yaxis=dict(gridcolor="#282245"),
    height=420,
)
st.plotly_chart(fig_domain, use_container_width=True)

