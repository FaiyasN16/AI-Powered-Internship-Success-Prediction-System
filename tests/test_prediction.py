"""Tests for model loading, prediction inference, SHAP, heuristics, and error handling."""

import os
import pytest
import pandas as pd

from src.prediction import (
    predict_student_success,
    load_trained_pipeline,
    DEFAULT_MODEL_PATH,
)
from src.scoring import calculate_overall_profile_score, calculate_domain_readiness
from src.recommendations import generate_recommendations, generate_ideal_profile_comparison
from src.explainability import compute_single_prediction_shap


@pytest.fixture
def sample_profile():
    return {
        "CGPA": 8.4,
        "Number of Relevant Projects": 4,
        "Average Project Complexity": 4,
        "Previous Internship Experience": "Yes",
        "Relevant Certifications": 2,
        "Number of Programming Languages": 3,
        "Number of Technical Skills": 8,
        "Internship Domain Applied": "Artificial Intelligence and Machine Learning",
    }


def test_pipeline_loading():
    """Verify saved model pipeline loads properly."""
    assert os.path.exists(DEFAULT_MODEL_PATH), "Model file should exist before testing inference."
    pipeline = load_trained_pipeline(DEFAULT_MODEL_PATH)
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_predict_student_success_outputs(sample_profile):
    """Verify inference output structure, probability bounds, and class outcomes."""
    res = predict_student_success(sample_profile)

    assert res["predicted_outcome"] in ["Selected", "Not Selected"]
    assert 0.0 <= res["probability_selected"] <= 1.0
    assert 0.0 <= res["probability_not_selected"] <= 1.0
    assert abs((res["probability_selected"] + res["probability_not_selected"]) - 1.0) < 1e-4

    # Profile scoring bounds
    assert 0.0 <= res["overall_score"] <= 100.0
    assert 0.0 <= res["domain_readiness_score"] <= 100.0
    assert isinstance(res["domain_suitability"], str)

    # Recommendations structure
    recs = res["recommendations"]
    assert len(recs["strengths"]) > 0
    assert len(recs["action_items"]) >= 3
    assert len(recs["roadmap"]) == 4

    # Ideal profile comparison
    df_comp = res["ideal_comparison"]
    assert isinstance(df_comp, pd.DataFrame)
    assert not df_comp.empty
    assert "Profile Feature" in df_comp.columns
    assert "Target Benchmark" in df_comp.columns


def test_profile_scoring_bounds(sample_profile):
    """Verify score calculation boundaries across extremes."""
    # Minimum profile
    min_profile = {
        "CGPA": 0.0,
        "Number of Relevant Projects": 0,
        "Average Project Complexity": 1,
        "Previous Internship Experience": "No",
        "Relevant Certifications": 0,
        "Number of Programming Languages": 0,
        "Number of Technical Skills": 0,
        "Internship Domain Applied": "Web Development",
    }
    min_score, _ = calculate_overall_profile_score(min_profile)
    assert min_score == 0.0

    # Max profile
    max_profile = {
        "CGPA": 10.0,
        "Number of Relevant Projects": 10,
        "Average Project Complexity": 5,
        "Previous Internship Experience": "Yes",
        "Relevant Certifications": 5,
        "Number of Programming Languages": 5,
        "Number of Technical Skills": 15,
        "Internship Domain Applied": "Artificial Intelligence and Machine Learning",
    }
    max_score, _ = calculate_overall_profile_score(max_profile)
    assert max_score == 100.0


def test_shap_single_explanation(sample_profile):
    """Verify SHAP TreeExplainer produces expected feature contributions."""
    pipeline = load_trained_pipeline(DEFAULT_MODEL_PATH)
    shap_res = compute_single_prediction_shap(pipeline, sample_profile)

    assert "contributions" in shap_res
    assert len(shap_res["contributions"]) == 15
    assert "narrative" in shap_res
    assert len(shap_res["narrative"]) > 0
    assert isinstance(shap_res["base_value"], float)


def test_missing_model_file_error():
    """Verify appropriate FileNotFoundError is raised if model path is invalid."""
    with pytest.raises(FileNotFoundError):
        load_trained_pipeline("models/non_existent_model_xyz.joblib")

