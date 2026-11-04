"""
Explainable AI (XAI) Module using SHAP.

Computes genuine SHAP values from the fitted RandomForestClassifier pipeline
for both individual student predictions and global model importance.
Provides interactive Plotly visualizations and native SHAP summary plots.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import profile_to_dataframe, ALL_FEATURE_COLUMNS


FEATURE_NAME_MAP = {
    "num__CGPA": "CGPA (Academics)",
    "num__Number of Relevant Projects": "Relevant Projects Count",
    "num__Average Project Complexity": "Project Complexity (1-5)",
    "num__Relevant Certifications": "Certifications Count",
    "num__Number of Programming Languages": "Programming Languages Count",
    "num__Number of Technical Skills": "Technical Skills Count",
    "cat__Previous Internship Experience_Yes": "Prior Internship: Yes",
    "cat__Previous Internship Experience_No": "Prior Internship: No",
    "cat__Internship Domain Applied_Artificial Intelligence and Machine Learning": "Domain: AI & ML",
    "cat__Internship Domain Applied_Cloud Computing": "Domain: Cloud Computing",
    "cat__Internship Domain Applied_Cybersecurity": "Domain: Cybersecurity",
    "cat__Internship Domain Applied_Data Science and Analytics": "Domain: Data Science",
    "cat__Internship Domain Applied_Mobile App Development": "Domain: Mobile Dev",
    "cat__Internship Domain Applied_Software Engineering": "Domain: Software Eng",
    "cat__Internship Domain Applied_Web Development": "Domain: Web Dev",
}


def clean_feature_name(raw_name: str) -> str:
    """Map raw preprocessor feature names to human-readable labels."""
    if raw_name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[raw_name]
    clean = raw_name.replace("num__", "").replace("cat__", "").replace("Internship Domain Applied_", "Domain: ")
    clean = clean.replace("Previous Internship Experience_", "Prior Internship: ")
    return clean


def compute_single_prediction_shap(pipeline, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate SHAP contributions for a single student profile.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    profile : dict
        Validated student profile.

    Returns
    -------
    dict
        Structured SHAP results including individual contributions,
        base value, top positive/negative drivers, and narrative explanation.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform input dataframe
    df_input = profile_to_dataframe(profile)
    X_trans = preprocessor.transform(df_input)

    # Get feature names
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in raw_feature_names]

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(classifier)
    shap_explanation = explainer(X_trans)

    # Extract values for the "Selected" class (class 1)
    classes = list(pipeline.classes_)
    selected_idx = classes.index("Selected")

    if len(shap_explanation.shape) == 3:
        # Shape: (1, n_features, 2)
        shap_vals = shap_explanation.values[0, :, selected_idx]
        base_val = float(shap_explanation.base_values[0, selected_idx])
    else:
        shap_vals = shap_explanation.values[0]
        base_val = float(shap_explanation.base_values[0])

    # Structure feature contributions
    contributions = []
    for name, val in zip(feature_names, shap_vals):
        contributions.append({
            "feature": name,
            "shap_value": float(val),
            "abs_shap": abs(float(val)),
            "impact": "Positive (Boosts Selection)" if val > 0 else "Negative (Reduces Selection)",
        })

    # Sort by absolute SHAP impact
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    positive_drivers = [c["feature"] for c in contributions if c["shap_value"] > 0.01][:3]
    negative_drivers = [c["feature"] for c in contributions if c["shap_value"] < -0.01][:3]

    # Build plain English narrative
    narrative_parts = []
    if positive_drivers:
        narrative_parts.append(
            f"The primary attributes boosting your selection probability are **{', '.join(positive_drivers)}**."
        )
    if negative_drivers:
        narrative_parts.append(
            f"Conversely, the main factors dampening the estimated probability are **{', '.join(negative_drivers)}**."
        )
    if not narrative_parts:
        narrative_parts.append("Your profile features demonstrate balanced contributions across all measured criteria.")

    narrative = " ".join(narrative_parts)

    return {
        "feature_names": feature_names,
        "shap_values": shap_vals.tolist(),
        "base_value": base_val,
        "contributions": contributions,
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
        "narrative": narrative,
    }


def compute_global_feature_importance(pipeline, dataset_sample: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Compute global SHAP feature importance across a sample of student profiles.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    dataset_sample : pd.DataFrame
        Representative dataframe of student profiles.

    Returns
    -------
    (pd.DataFrame, np.ndarray, np.ndarray, list of str)
        DataFrame of ranked features, raw transformed data X, shap_values array, and feature names.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X = dataset_sample[ALL_FEATURE_COLUMNS].copy()
    X_trans = preprocessor.transform(X)

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in raw_feature_names]

    explainer = shap.TreeExplainer(classifier)
    shap_explanation = explainer(X_trans)

    classes = list(pipeline.classes_)
    selected_idx = classes.index("Selected")

    if len(shap_explanation.shape) == 3:
        shap_vals_selected = shap_explanation.values[:, :, selected_idx]
    else:
        shap_vals_selected = shap_explanation.values

    # Mean absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_vals_selected), axis=0)
    mean_direction = np.mean(shap_vals_selected, axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP": mean_abs_shap,
        "Average Direction": mean_direction,
    }).sort_values(by="Mean Absolute SHAP", ascending=False).reset_index(drop=True)

    return importance_df, X_trans, shap_vals_selected, feature_names


def plot_single_shap_waterfall_plotly(shap_result: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive Plotly horizontal bar chart illustrating individual SHAP contributions.
    """
    contributions = shap_result["contributions"]
    # Filter for non-trivial contributions (top 10)
    top_contribs = contributions[:10][::-1]  # Reverse for bottom-up horizontal bar

    features = [c["feature"] for c in top_contribs]
    vals = [c["shap_value"] for c in top_contribs]
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=features,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="#1C1730", width=1.5)
        ),
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#F5F3FF", size=12),
        hoverinfo="x+y",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Individual Feature SHAP Contributions (Impact on Selection Probability)</b>",
            font=dict(color="#F5F3FF", size=16),
        ),
        plot_bgcolor="#161226",
        paper_bgcolor="#161226",
        font=dict(color="#B8B1CC", family="sans-serif"),
        xaxis=dict(
            title="SHAP Value (Log-odds contribution to 'Selected')",
            zeroline=True,
            zerolinecolor="#A78BFA",
            zerolinewidth=2,
            gridcolor="#282245",
            tickfont=dict(color="#B8B1CC"),
        ),
        yaxis=dict(
            tickfont=dict(color="#F5F3FF", size=12),
        ),
        margin=dict(l=20, r=40, t=60, b=40),
        height=420,
    )

    return fig


def plot_global_importance_plotly(importance_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive Plotly horizontal bar chart for global feature importance.
    """
    # Top 12 features for readability
    top_df = importance_df.head(12).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=top_df["Mean Absolute SHAP"],
        y=top_df["Feature"],
        orientation="h",
        marker=dict(
            color=top_df["Mean Absolute SHAP"],
            colorscale=[[0, "#4C1D95"], [0.5, "#7C3AED"], [1, "#A78BFA"]],
            line=dict(color="#1C1730", width=1),
        ),
        text=[f"{v:.4f}" for v in top_df["Mean Absolute SHAP"]],
        textposition="outside",
        textfont=dict(color="#F5F3FF", size=11),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Global Feature Importance (Mean |SHAP Value|)</b>",
            font=dict(color="#F5F3FF", size=16),
        ),
        plot_bgcolor="#161226",
        paper_bgcolor="#161226",
        font=dict(color="#B8B1CC", family="sans-serif"),
        xaxis=dict(
            title="Mean |SHAP Value| (Impact on Model Predictions)",
            gridcolor="#282245",
            tickfont=dict(color="#B8B1CC"),
        ),
        yaxis=dict(
            tickfont=dict(color="#F5F3FF", size=12),
        ),
        margin=dict(l=20, r=50, t=60, b=40),
        height=440,
    )

    return fig


def generate_shap_summary_figure_matplotlib(X_trans: np.ndarray, shap_vals: np.ndarray, feature_names: List[str]) -> plt.Figure:
    """
    Generate native SHAP beeswarm / summary plot using Matplotlib.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#161226")
    ax.set_facecolor("#161226")

    shap.summary_plot(
        shap_vals,
        X_trans,
        feature_names=feature_names,
        show=False,
        max_display=10,
        plot_type="dot",
        color_bar=True
    )
    plt.tight_layout()
    return plt.gcf()

