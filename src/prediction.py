"""
Model Inference and Pipeline Loader Module.

Loads serialized pipeline artifacts and evaluates student profiles to generate
probabilistic selection outcomes, profile scores, and holistic diagnostic reports.
"""

import os
from functools import lru_cache
from typing import Dict, Any, Tuple
import joblib
import pandas as pd

from src.preprocessing import (
    validate_student_profile,
    profile_to_dataframe,
    ALL_FEATURE_COLUMNS,
)
from src.scoring import (
    calculate_overall_profile_score,
    calculate_domain_readiness,
)
from src.recommendations import (
    generate_recommendations,
    generate_ideal_profile_comparison,
)

DEFAULT_MODEL_PATH = "models/internship_success_model.joblib"
DEFAULT_METRICS_PATH = "models/model_metrics.json"


@lru_cache(maxsize=1)
def load_trained_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load and cache the trained Scikit-learn Pipeline from disk.

    Parameters
    ----------
    model_path : str
        Filepath to the saved joblib pipeline.

    Returns
    -------
    Pipeline
        Fitted Scikit-learn Pipeline.

    Raises
    ------
    FileNotFoundError
        If model artifact has not yet been trained and saved.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python train_model.py' to generate the dataset and train the model."
        )
    return joblib.load(model_path)


def predict_student_success(
    profile: Dict[str, Any],
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Run full end-to-end prediction and diagnostic evaluation on a student profile.

    Parameters
    ----------
    profile : dict
        Student profile attributes.
    model_path : str
        Path to the saved pipeline.

    Returns
    -------
    dict
        Comprehensive evaluation dictionary containing:
          - 'predicted_outcome': 'Selected' | 'Not Selected'
          - 'probability_selected': float (0.0 to 1.0)
          - 'probability_not_selected': float (0.0 to 1.0)
          - 'overall_score': float (0 to 100)
          - 'overall_breakdown': dict
          - 'domain_readiness_score': float (0 to 100)
          - 'domain_suitability': str
          - 'domain_summary': str
          - 'recommendations': dict
          - 'ideal_comparison': pd.DataFrame
    """
    # 1. Validate profile
    is_valid, errors = validate_student_profile(profile)
    if not is_valid:
        raise ValueError(f"Invalid student profile: {'; '.join(errors)}")

    # 2. Load model
    pipeline = load_trained_pipeline(model_path)

    # 3. Format input dataframe
    df_input = profile_to_dataframe(profile)

    # 4. Predict probabilities and outcome
    classes = list(pipeline.classes_)
    selected_idx = classes.index("Selected")
    not_selected_idx = classes.index("Not Selected")

    probabilities = pipeline.predict_proba(df_input)[0]
    prob_selected = float(probabilities[selected_idx])
    prob_not_selected = float(probabilities[not_selected_idx])

    predicted_outcome = "Selected" if prob_selected >= 0.50 else "Not Selected"

    # 5. Calculate scores and recommendations
    overall_score, overall_breakdown = calculate_overall_profile_score(profile)
    readiness_score, suitability, summary = calculate_domain_readiness(profile)
    recommendations = generate_recommendations(profile)
    ideal_comparison = generate_ideal_profile_comparison(profile)

    return {
        "predicted_outcome": predicted_outcome,
        "probability_selected": prob_selected,
        "probability_not_selected": prob_not_selected,
        "overall_score": overall_score,
        "overall_breakdown": overall_breakdown,
        "domain_readiness_score": readiness_score,
        "domain_suitability": suitability,
        "domain_summary": summary,
        "recommendations": recommendations,
        "ideal_comparison": ideal_comparison,
        "input_profile": profile,
    }

