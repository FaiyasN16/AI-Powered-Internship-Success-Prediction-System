# AI-Powered Internship Success Prediction System

[![Python Version](https://img.shields.io/badge/Python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An explainable machine learning decision-support platform designed to assess student readiness for technical internships. The system evaluates academic credentials, project depth, practical exposure, and technical skills to deliver calibrated selection probability estimates, transparent SHAP feature attributions, domain readiness diagnostics, and actionable career improvement roadmaps.

> **Academic Prototype Notice:** This system is an academic decision-support prototype built on reproducible synthetic data. It is intended for self-assessment, guidance, and career planning—not as a real recruitment screener or hiring guarantee.

---

## 📑 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Technology Stack](#-technology-stack)
5. [Project Structure](#-project-structure)
6. [Quick Start (Windows PowerShell)](#-quick-start-windows-powershell)
7. [Dataset Generation & Synthetic Assumptions](#-dataset-generation--synthetic-assumptions)
8. [Machine Learning Pipeline & Evaluation](#-machine-learning-pipeline--evaluation)
9. [Explainable AI (SHAP) Interpretation](#-explainable-ai-shap-interpretation)
10. [Profile Scoring & Domain Readiness Heuristics](#-profile-scoring--domain-readiness-heuristics)
11. [Profile Improvement Simulator](#-profile-improvement-simulator)
12. [Automated Testing](#-automated-testing)
13. [Limitations & Ethical Considerations](#-limitations--ethical-considerations)
14. [Troubleshooting Guide](#-troubleshooting-guide)

---

## 🎯 Problem Statement

Undergraduate and postgraduate students seeking technical internships frequently navigate recruitment processes with zero feedback regarding their profile competitiveness. Traditional campus placement drives and corporate application portals operate as "black boxes," leaving students uncertain about whether their primary gap lies in academic marks (CGPA), project architectural depth, lack of industry certifications, or absent prior experience.

This project delivers a multi-dimensional diagnostic platform that:
- Accurately classifies candidates into **Selected** or **Not Selected** with continuous probability estimates.
- Uncovers the exact mathematical factors driving the decision via **SHAP (SHapley Additive exPlanations)**.
- Compares candidates against benchmark target profiles.
- Provides personalized domain-specific action items and 4-phase short-term roadmaps.
- Enables interactive "What-If" simulation to visualize how future accomplishments shift selection probabilities.

---

## ✨ Key Features

- **🎯 Machine Learning Prediction Engine:**
  - Ensemble `RandomForestClassifier` trained within an isolated Scikit-learn `Pipeline`.
  - Automated `ColumnTransformer` with `OneHotEncoder` and `StandardScaler` ensuring zero data leakage.
  - Predicts both discrete classification and continuous class probabilities ($P(\text{Selected})$).

- **🔍 Explainable AI (SHAP TreeExplainer):**
  - **Local Attribution:** Interactive Plotly waterfall chart explaining how each feature pushes probability towards or away from selection for individual students.
  - **Global Importance:** Mean absolute SHAP ranking and optional Matplotlib beeswarm distribution plots.
  - Plain-English narrative summaries translating mathematical weights into clear guidance.

- **📊 Student Population Analytics Dashboard:**
  - 8+ interactive Plotly charts examining CGPA distributions, project volumes, complexity levels, certification impacts, prior internship conversions, and domain-wise acceptance rates.

- **⚡ What-If Profile Improvement Simulator:**
  - Side-by-side comparison of baseline vs. simulated student profiles.
  - Live recalculation through the saved pipeline with probability deltas ($\Delta \pm X\%$), outcome transitions, and score updates.

- **🏆 Heuristic Profile Scoring & Domain Readiness:**
  - Transparent 100-point Overall Profile Score.
  - Domain Readiness Score tailored to each of the 7 technical domains:
    - *Web Development*
    - *Artificial Intelligence and Machine Learning*
    - *Data Science and Analytics*
    - *Cybersecurity*
    - *Cloud Computing*
    - *Mobile App Development*
    - *Software Engineering*

- **⚖️ Candidate vs. Ideal Profile Benchmarks:**
  - Side-by-side gap analysis table comparing candidate metrics against industry reference benchmarks.

- **🗺️ Actionable Recommendations & 4-Phase Roadmap:**
  - Top strengths and improvement areas.
  - 3 to 5 prioritized, domain-aware action items.
  - Structured 4-phase timeline (Fundamentals &rarr; Capstone &rarr; Credentials &rarr; Interview Drills).

---

## 🏛️ System Architecture

```
+-------------------------------------------------------------+
|                      Streamlit Frontend                     |
|  - Home Dashboard (Metrics & Confusion Matrix)               |
|  - Student Prediction Form & Diagnostics                    |
|  - Explainable AI (SHAP Waterfall & Global Ranking)         |
|  - Student Analytics (Plotly Explorations)                  |
|  - What-If Improvement Simulator                            |
|  - Project Information & Architecture Documentation         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Application Core                        |
|  - src/preprocessing.py (Schema validation & Transformer)   |
|  - src/prediction.py    (Inference & Pipeline caching)      |
|  - src/explainability.py (SHAP TreeExplainer & Plotting)    |
|  - src/scoring.py       (100-pt Profile & Domain Rubrics)   |
|  - src/recommendations.py (Action items & Roadmap)          |
|  - src/ui_helpers.py    (Dark-purple theme CSS & widgets)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                Trained Pipeline & Artifacts                 |
|  - models/internship_success_model.joblib (Fitted Pipeline) |
|  - models/model_metrics.json (Measured Evaluation Results)  |
|  - data/internship_dataset.csv (1,000 Synthetic Records)    |
+-------------------------------------------------------------+
```

---

## 💻 Technology Stack

| Category | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13 / 3.14 | Core programming environment |
| **Frontend UI** | Streamlit | &ge; 1.30.0 | Reactive multi-page web application |
| **Data Processing** | Pandas & NumPy | &ge; 2.0.0 / &ge; 1.24.0 | Data structures, array operations, sampling |
| **Machine Learning**| Scikit-learn | &ge; 1.3.0 | ColumnTransformer, RandomForest, metrics |
| **Explainability** | SHAP | &ge; 0.42.0 | TreeExplainer, Shapley values, beeswarm plots |
| **Visualizations** | Plotly & Matplotlib | &ge; 5.18.0 / &ge; 3.7.0 | Interactive web charts and static beeswarm |
| **Model Persistence**| Joblib | &ge; 1.3.0 | Serialization of fitted pipeline |
| **Testing** | Pytest | &ge; 7.4.0 | Unit and integration test suite |

---

## 📁 Project Structure

```
AI-Powered-Internship-Success-Prediction-System/
├── app.py                         # Home Dashboard: KPI cards, confusion matrix, overview
├── generate_dataset.py            # Reproducible synthetic dataset generation script
├── train_model.py                 # Pipeline training, evaluation, and artifact serialization
├── requirements.txt               # Complete Python package dependencies
├── README.md                      # Comprehensive documentation and run instructions
├── .gitignore                     # Git exclusion rules
├── .streamlit/
│   └── config.toml                # Dark-purple application theme configuration
├── data/
│   └── internship_dataset.csv     # 1,000 synthetic student applicant records
├── models/
│   ├── internship_success_model.joblib  # Serialized fitted Scikit-learn Pipeline
│   └── model_metrics.json         # Genuine measured test evaluation metrics
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Feature definitions, schema validation, transformer
│   ├── prediction.py              # Pipeline loader, predict_proba, end-to-end evaluation
│   ├── explainability.py          # SHAP TreeExplainer, waterfall, global importance
│   ├── scoring.py                 # Transparent 100-pt profile and domain readiness heuristics
│   ├── recommendations.py         # Tailored strengths, gaps, prioritized actions, roadmap
│   └── ui_helpers.py              # CSS injection, dark-purple cards, probability gauge
├── pages/
│   ├── 1_Student_Prediction.py    # Form input, prediction, gauges, candidate vs ideal table
│   ├── 2_Explainable_AI.py        # Individual SHAP waterfall, global importance, beeswarm
│   ├── 3_Student_Analytics.py     # 8+ interactive Plotly exploratory analysis charts
│   ├── 4_Improvement_Simulator.py # Side-by-side What-If parameter modification tool
│   └── 5_Project_Information.py   # Complete project documentation & methodology
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py      # Schema, ranges, reproducibility, transformer tests
    ├── test_validation.py         # Range guards, negative values, domain validation tests
    └── test_prediction.py         # Inference, probability bounds, SHAP, heuristics, error tests
```

---

## 🚀 Quick Start (Windows PowerShell)

Follow these exact commands in Windows PowerShell to set up and run the system:

### 1. Clone or Open Workspace
```powershell
cd "c:\Users\faiya\OneDrive\Desktop\FOAI final"
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If script execution is disabled on your system, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Generate the Synthetic Dataset
```powershell
python generate_dataset.py
```
*Creates `data/internship_dataset.csv` with 1,000 reproducible student records.*

### 5. Train and Evaluate the Model Pipeline
```powershell
python train_model.py
```
*Fits the ColumnTransformer and RandomForestClassifier, computes actual test metrics, and saves `models/internship_success_model.joblib` and `models/model_metrics.json`.*

### 6. Run Automated Pytest Suite
```powershell
pytest -v
```

### 7. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
*Streamlit will automatically launch in your default web browser at `http://localhost:8501`.*

---

## 📊 Dataset Generation & Synthetic Assumptions

The dataset is generated via `generate_dataset.py` using a fixed seed (`random_state=42`).

### Features:
1. **CGPA:** Numerical float from 5.0 to 9.95 (Truncated Normal distribution, $\mu=7.6, \sigma=1.0$).
2. **Number of Relevant Projects:** Integer count from 0 to 10 (Poisson distribution, $\lambda=3.2$).
3. **Average Project Complexity:** Integer rating from 1 (Basic) to 5 (Production-grade).
4. **Previous Internship Experience:** Categorical ("Yes" / "No") (~32% baseline conversion).
5. **Relevant Certifications:** Integer count from 0 to 8 (Poisson distribution, $\lambda=1.6$).
6. **Number of Programming Languages:** Integer count from 1 to 8 (Poisson distribution, $\lambda=3.1$).
7. **Number of Technical Skills:** Integer count from 1 to 15 (Correlated with projects and languages).
8. **Internship Domain Applied:** Categorical selection across 7 industry tracks.

### Latent Score Logic & Non-Determinism:
Selection propensity is computed via a weighted combination of academic performance ($0.28$), projects ($0.24$), complexity ($0.22$), prior internship ($0.26$), certifications ($0.12$), languages ($0.08$), and technical skills ($0.16$), plus domain synergy adjustments.

Controlled Gaussian noise ($\epsilon \sim \mathcal{N}(0, 0.65)$) is injected to simulate unobserved factors (interview communication, soft skills, hiring manager subjectivity), ensuring outcomes are realistic and non-trivial.

---

## ⚙️ Machine Learning Pipeline & Evaluation

### Training Pipeline:
- **Partition:** 80% Training (800 records), 20% Held-out Testing (200 records) with stratification on the target class.
- **Model:** `RandomForestClassifier(n_estimators=120, max_depth=9, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42)`
- **Preprocessing:** `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder(handle_unknown='ignore')` on categorical features.

### Actual Measured Evaluation Results:
*Measured strictly on the held-out test set ($N=200$):*

| Metric | Measured Value | Percentage / Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | `0.7750` | **77.50%** correct classifications |
| **Precision (Selected)** | `0.8208` | **82.08%** true positive accuracy |
| **Recall (Selected)** | `0.7699` | **76.99%** positive class coverage |
| **F1-Score (Selected)** | `0.7945` | **79.45%** balanced harmonic score |
| **ROC - AUC** | `0.8755` | Excellent discrimination ability |

#### Actual Confusion Matrix ($N=200$):
$$\begin{bmatrix} 68 & 19 \\ 26 & 87 \end{bmatrix}$$
- **True Negatives:** 68
- **False Positives:** 19
- **False Negatives:** 26
- **True Positives:** 87

---

## 🔍 Explainable AI (SHAP) Interpretation

We employ **SHAP (SHapley Additive exPlanations)** based on cooperative game theory to solve the black-box nature of the Random Forest model:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

- **Individual Prediction Waterfall:** Quantifies how much each individual feature pushes the log-odds of selection up (green, $\phi > 0$) or down (red, $\phi < 0$) relative to the base expected value.
- **Global Feature Ranking:** Measures $\frac{1}{N} \sum_{k=1}^N |\phi_i^{(k)}|$, revealing that Project Complexity, CGPA, and Prior Internship are the most influential global determinants.
- **Attribution vs. Causality:** SHAP isolates statistical correlation within the model's learned response function; it does not claim direct physical causality in real-world recruitment.

---

## 🏆 Profile Scoring & Domain Readiness Heuristics

To provide actionable educational feedback independent of statistical class likelihoods, the system includes two rule-based rubrics:

### 1. Overall Profile Score (0 - 100):
- **Academic Standing (CGPA):** 25 points maximum
- **Project Portfolio Volume:** 20 points maximum
- **Project Architectural Complexity:** 15 points maximum
- **Prior Professional Internship:** 15 points
- **Industry Certifications:** 10 points maximum
- **Technical Skills Breadth:** 10 points maximum
- **Programming Languages:** 5 points maximum

### 2. Domain Readiness Score (0 - 100):
Tailors feature weights to domain demands:
- *AI/ML & Data Science:* Emphasizes project complexity ($35\%$) and technical skills ($25\%$).
- *Cloud & Cybersecurity:* Emphasizes vendor credentials ($35\%$) and architecture complexity ($25\%$).
- *Web & Mobile:* Emphasizes project portfolio volume ($35\%$) and practical toolkits ($25\%$).

---

## ⚡ Profile Improvement Simulator

The **What-If Simulator** allows students to model hypothetical progress:
- Adjust CGPA, add completed projects, increase complexity levels, toggle internship experience, add certifications, or switch domains.
- The simulator evaluates both the baseline profile and the modified profile through the identical saved pipeline.
- Calculates exact probability delta: $\Delta P = P_{\text{simulated}} - P_{\text{baseline}}$.
- Reports whether the outcome flips from *Not Selected* to *Selected*.

---

## 🧪 Automated Testing

Automated test suites are implemented in `tests/`:
- `test_preprocessing.py`: Validates synthetic dataset generation, column schema, non-null values, numerical boundaries, random seed reproducibility, and `ColumnTransformer` feature names.
- `test_validation.py`: Tests boundary guards for CGPA ($0-10$), negative project counts, invalid complexity ratings ($<1$ or $>5$), invalid internship values, and unrecognized domains.
- `test_prediction.py`: Verifies pipeline serialization, probability bounds ($0.0 \le P \le 1.0$), score calculations ($0-100$), SHAP contribution counts, recommendation roadmaps, and missing-file exception handling.

Run tests:
```powershell
pytest -v
```

---

## ⚖️ Limitations & Ethical Considerations

- **Academic Decision-Support Prototype:** Built exclusively for student educational advising. It must **never** be deployed as an automated recruitment screening or rejection gatekeeper.
- **Synthetic Data Baseline:** Patterns reflect the mathematical assumptions of the data generation generator. Real-world corporate selection involves company-specific requirements, university tiers, and recruiter preferences.
- **Qualitative Omissions:** The model cannot evaluate soft skills, behavioral suitability, written code quality, or live interview performance.
- **Fairness by Design:** Sensitive demographic attributes (gender, age, ethnicity, socio-economic background) are strictly excluded from the schema to prevent bias propagation.

---

## 🛠️ Troubleshooting Guide

### 1. Virtual Environment Activation Blocked
If PowerShell displays `Execution of scripts is disabled on this system`, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Missing Model or Dataset Error
If the dashboard displays a warning that the model is missing:
```powershell
python generate_dataset.py
python train_model.py
```
*Alternatively, click the **"Generate Synthetic Dataset & Train Model Pipeline"** button directly on the Home dashboard.*

### 3. Port Already in Use
If Streamlit reports port `8501` is occupied:
```powershell
streamlit run app.py --server.port 8502
```

---

## 📄 License
This project is licensed under the MIT License for academic and educational usage.

