"""
Project Information, Methodology, and Documentation Page.

Comprehensive documentation detailing problem statement, architectural design,
machine learning workflow, synthetic dataset specifications, ethical limitations,
and future engineering enhancements.
"""

import streamlit as st
from src.ui_helpers import apply_custom_css, render_header, render_disclaimer

st.set_page_config(
    page_title="Project Information",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="Project Documentation & System Architecture",
    subtitle="Comprehensive academic specification of the AI-Powered Internship Success Prediction System.",
    icon="📖"
)

render_disclaimer()

# Expandable / Tabbed Documentation
tab_arch, tab_ml, tab_data, tab_ethics = st.tabs([
    "🏛️ Architecture & Objectives",
    "⚙️ ML Workflow & Evaluation",
    "📊 Dataset & Heuristic Rubrics",
    "⚖️ Ethical Considerations & Limitations"
])

# ==========================================
# TAB 1: ARCHITECTURE & OBJECTIVES
# ==========================================
with tab_arch:
    st.markdown("### 🎯 1. Problem Statement & Objectives")
    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="color: #A78BFA; margin-top: 0;">Problem Statement</h4>
            <p style="color: #F5F3FF;">
                Undergraduate and postgraduate students seeking technical internships frequently navigate recruitment
                processes without transparent feedback regarding their profile competitiveness. Traditional campus
                placements and corporate application portals function as "black boxes," leaving rejected applicants
                uncertain about whether their primary deficiency lies in academic grades (CGPA), project architectural
                depth, industry credentials, or lack of prior practical experience.
            </p>
            <h4 style="color: #A78BFA;">Project Objectives</h4>
            <ul style="color: #B8B1CC; line-height: 1.8;">
                <li><strong>Probabilistic Decision-Support:</strong> Deliver calibrated selection probability estimates using an ensemble Random Forest model trained on structured academic and technical metrics.</li>
                <li><strong>Algorithmic Explainability:</strong> Utilize SHAP (SHapley Additive exPlanations) to transparently isolate and rank exact feature contributions for individual candidate profiles.</li>
                <li><strong>Heuristic Profile Scoring:</strong> Provide transparent, rule-based Overall Profile (0-100) and Domain Readiness (0-100) scores distinct from statistical model likelihoods.</li>
                <li><strong>Actionable Roadmaps:</strong> Produce tailored 4-phase short-term skill development roadmaps addressing diagnosed gaps.</li>
                <li><strong>Interactive What-If Simulation:</strong> Enable students to explore hypothetical profile modifications and observe model response surfaces dynamically.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🏗️ 2. System Architecture")
    st.markdown(
        """
        ```mermaid
        flowchart TD
            A["Student Profile Input<br>(8 Features)"] --> B["Schema Validation<br>(Range & Type Guards)"]
            B --> C["Preprocessing Pipeline<br>(OneHotEncoder & StandardScaler)"]
            C --> D["Random Forest Classifier<br>(120 Balanced Decision Trees)"]
            D --> E["Probability & Class Prediction<br>P(Selected) & P(Not Selected)"]
            C --> F["SHAP TreeExplainer<br>(Shapley Value Attribution)"]
            A --> G["Heuristic Scoring Engine<br>(Overall & Domain Readiness)"]
            A --> H["Recommendation Engine<br>(Strengths, Gaps & Roadmaps)"]
            E --> I["Streamlit Interactive UI<br>(Dark-Purple Dashboard & Charts)"]
            F --> I
            G --> I
            H --> I
            I --> J["What-If Simulation Engine<br>(Side-by-Side Delta Recalculation)"]
        ```
        """,
        unsafe_allow_html=True
    )

# ==========================================
# TAB 2: ML WORKFLOW & EVALUATION
# ==========================================
with tab_ml:
    st.markdown("### 🔬 Machine Learning Pipeline & Evaluation")
    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="color: #A78BFA; margin-top: 0;">1. Algorithm Selection: Random Forest Classifier</h4>
            <p style="color: #F5F3FF;">
                We selected <strong>Random Forest (Breiman, 2001)</strong> as the primary classification engine for several reasons:
            </p>
            <ul style="color: #B8B1CC; line-height: 1.7;">
                <li><strong>Non-linear Feature Interactions:</strong> Captures complex interactions between project complexity, academic performance, and domain choices without requiring manual polynomial feature engineering.</li>
                <li><strong>Robustness to Outliers:</strong> Tree ensembles are invariant to monotonic transformations and resilient to extreme values in numerical features.</li>
                <li><strong>Inherent Regularization:</strong> Bagging (Bootstrap Aggregation) combined with random subspace feature sampling mitigates variance and overfitting.</li>
                <li><strong>Native SHAP Compatibility:</strong> TreeExplainer executes exact, polynomial-time Shapley value calculations on tree ensembles.</li>
            </ul>

            <h4 style="color: #A78BFA;">2. Data Preprocessing & Pipeline Integrity</h4>
            <p style="color: #B8B1CC;">
                To strictly prevent data leakage, all preprocessing transformations are encapsulated within a Scikit-learn
                <code>Pipeline</code> using a <code>ColumnTransformer</code>:
            </p>
            <ul style="color: #B8B1CC; line-height: 1.7;">
                <li><code>StandardScaler</code>: Scales numerical attributes (CGPA, Projects, Complexity, Certs, Languages, Skills) to zero mean and unit variance.</li>
                <li><code>OneHotEncoder</code>: Encodes categorical features (Internship Domain Applied, Previous Internship Experience) with <code>handle_unknown='ignore'</code>.</li>
                <li><strong>Pipeline Encapsulation:</strong> The preprocessor is fit strictly on the 80% training partition, ensuring zero information from the test partition leaks into the transformation statistics.</li>
            </ul>

            <h4 style="color: #A78BFA;">3. Evaluation Methodology</h4>
            <p style="color: #B8B1CC;">
                Evaluation is conducted on a stratified 20% held-out test set (200 records). Genuine evaluation metrics
                computed include:
            </p>
            <ul style="color: #B8B1CC; line-height: 1.7;">
                <li><strong>Accuracy:</strong> Overall proportion of correct classifications on held-out test data.</li>
                <li><strong>Precision & Recall:</strong> Targeted measurement on the minority/positive 'Selected' class.</li>
                <li><strong>F1-Score:</strong> Harmonic mean of precision and recall.</li>
                <li><strong>ROC-AUC:</strong> Area Under the Receiver Operating Characteristic curve measuring ranking quality across all decision thresholds.</li>
                <li><strong>Confusion Matrix:</strong> Full classification contingency breakdown (True Positives, False Positives, True Negatives, False Negatives).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# TAB 3: DATASET & HEURISTICS
# ==========================================
with tab_data:
    st.markdown("### 📊 Dataset Construction & Heuristic Scoring Rubrics")
    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="color: #A78BFA; margin-top: 0;">1. Synthetic Dataset Generation Methodology</h4>
            <p style="color: #F5F3FF;">
                The dataset contains 1,000 synthetic student profiles generated via <code>generate_dataset.py</code> with a fixed random seed (<code>random_state=42</code>).
            </p>
            <ul style="color: #B8B1CC; line-height: 1.7;">
                <li><strong>Distributions:</strong> CGPA follows a realistic truncated normal distribution (mean ~7.6, range 5.0 - 9.95). Projects and certifications follow Poisson distributions reflecting student activity counts.</li>
                <li><strong>Correlations:</strong> Technical skills and project complexities correlate realistically with project counts and academic standing.</li>
                <li><strong>Latent Propensity Function:</strong> A latent score integrates weighted academic, project, experiential, and domain-synergy components with controlled Gaussian noise (&sigma; = 0.65) to reflect unobserved interview dynamics and hiring variance.</li>
                <li><strong>No Target Leakage:</strong> Outcome labels are computed strictly downstream of the latent score and never looped back into input features.</li>
            </ul>

            <h4 style="color: #A78BFA;">2. Transparent Profile Scoring Rubric (100 Points)</h4>
            <p style="color: #B8B1CC;">
                Separate from the ML probability, an explicit, rule-based diagnostic rubric scores candidate profiles out of 100 points:
            </p>
            <table style="width: 100%; border-collapse: collapse; color: #F5F3FF; margin-top: 0.8rem;">
                <tr style="border-bottom: 1px solid #3B2E68; text-align: left;">
                    <th style="padding: 6px;">Evaluation Dimension</th>
                    <th style="padding: 6px;">Max Points</th>
                    <th style="padding: 6px;">Scoring Criterion</th>
                </tr>
                <tr style="border-bottom: 1px solid #2D254D;">
                    <td style="padding: 6px;">Academic Standing (CGPA)</td>
                    <td style="padding: 6px;">25</td>
                    <td style="padding: 6px;">(CGPA / 10.0) &times; 25</td>
                </tr>
                <tr style="border-bottom: 1px solid #2D254D;">
                    <td style="padding: 6px;">Project Portfolio Volume</td>
                    <td style="padding: 6px;">20</td>
                    <td style="padding: 6px;">min(Projects / 4.0, 1.0) &times; 20</td>
                </tr>
                <tr style="border-bottom: 1px solid #2D254D;">
                    <td style="padding: 6px;">Project Architectural Complexity</td>
                    <td style="padding: 6px;">15</td>
                    <td style="padding: 6px;">((Complexity - 1) / 4.0) &times; 15</td>
                </tr>
                <tr style="border-bottom: 1px solid #2D254D;">
                    <td style="padding: 6px;">Prior Professional Internship</td>
                    <td style="padding: 6px;">15</td>
                    <td style="padding: 6px;">15 pts if 'Yes', 0 pts if 'No'</td>
                </tr>
                <tr style="border-bottom: 1px solid #2D254D;">
                    <td style="padding: 6px;">Industry Certifications</td>
                    <td style="padding: 6px;">10</td>
                    <td style="padding: 6px;">min(Certs / 2.0, 1.0) &times; 10</td>
                </tr>
                <tr style="border-bottom: 1px solid #2D254D;">
                    <td style="padding: 6px;">Technical Skills Breadth</td>
                    <td style="padding: 6px;">10</td>
                    <td style="padding: 6px;">min(Skills / 8.0, 1.0) &times; 10</td>
                </tr>
                <tr>
                    <td style="padding: 6px;">Programming Languages</td>
                    <td style="padding: 6px;">5</td>
                    <td style="padding: 6px;">min(Langs / 3.0, 1.0) &times; 5</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# TAB 4: ETHICAL CONSIDERATIONS & LIMITATIONS
# ==========================================
with tab_ethics:
    st.markdown("### ⚖️ Limitations, Ethics & Future Enhancements")
    st.markdown(
        """
        <div class="dashboard-card">
            <h4 style="color: #EF4444; margin-top: 0;">1. System Limitations</h4>
            <ul style="color: #B8B1CC; line-height: 1.8;">
                <li><strong>Synthetic Dataset Context:</strong> The model is trained on a synthetic dataset created to simulate realistic recruitment criteria. It does not reflect proprietary candidate screening algorithms of any specific corporation.</li>
                <li><strong>Quantitative Proxy Limitations:</strong> Numerical features (e.g., number of projects or count of skills) serve as proxies and cannot fully measure code elegance, algorithmic novelty, or repository cleanliness.</li>
                <li><strong>Non-Causal Interpretability:</strong> SHAP values describe statistical model attribution, not causal guarantees. For instance, increasing the project count slider indicates what the model expects, not a guaranteed offer from a hiring team.</li>
                <li><strong>Absence of Qualitative Evaluation:</strong> The system does not evaluate soft skills, behavioral interview presence, or interpersonal communication.</li>
            </ul>

            <h4 style="color: #A78BFA;">2. Ethical Considerations</h4>
            <ul style="color: #B8B1CC; line-height: 1.8;">
                <li><strong>Not for Autonomous Hiring:</strong> This tool is strictly designed as an academic decision-support prototype for students, never as an automated screening or filtering tool for recruiters.</li>
                <li><strong>Algorithmic Fairness:</strong> Protected demographic attributes (gender, ethnicity, socio-economic background, geographic location) were intentionally excluded from feature schemas to prevent biased disparate impact.</li>
            </ul>

            <h4 style="color: #10B981;">3. Future Engineering Enhancements</h4>
            <ul style="color: #B8B1CC; line-height: 1.8;">
                <li><strong>Automated Resume & Portfolio Parsing:</strong> Integrate Large Language Models (LLMs) to parse PDF resumes and evaluate GitHub repository commit frequencies, unit test coverage, and documentation quality.</li>
                <li><strong>Simulated AI Mock Interviews:</strong> Incorporate audio/text conversational agents to assess technical problem-solving clarity and behavioral communication.</li>
                <li><strong>Domain-Specific Benchmarking APIs:</strong> Connect to real-time tech job market APIs (e.g., LinkedIn / Indeed) to dynamically calibrate demand weights for specific frameworks and cloud platforms.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

