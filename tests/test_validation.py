"""Tests for student profile schema validation."""

import pytest
from src.preprocessing import validate_student_profile, VALID_DOMAINS


@pytest.fixture
def valid_profile():
    return {
        "CGPA": 8.0,
        "Number of Relevant Projects": 3,
        "Average Project Complexity": 3,
        "Previous Internship Experience": "Yes",
        "Relevant Certifications": 2,
        "Number of Programming Languages": 3,
        "Number of Technical Skills": 6,
        "Internship Domain Applied": "Web Development",
    }


def test_valid_profile_passes(valid_profile):
    """Test that a compliant profile passes validation without errors."""
    is_valid, errors = validate_student_profile(valid_profile)
    assert is_valid is True
    assert len(errors) == 0


def test_invalid_cgpa_ranges(valid_profile):
    """Test that negative CGPA and CGPA > 10 are rejected."""
    bad_low = valid_profile.copy()
    bad_low["CGPA"] = -1.5
    is_valid, errors = validate_student_profile(bad_low)
    assert is_valid is False
    assert any("CGPA" in e for e in errors)

    bad_high = valid_profile.copy()
    bad_high["CGPA"] = 11.2
    is_valid, errors = validate_student_profile(bad_high)
    assert is_valid is False
    assert any("CGPA" in e for e in errors)


def test_negative_counts_rejected(valid_profile):
    """Test that negative project counts, certifications, languages, or skills are rejected."""
    for field in [
        "Number of Relevant Projects",
        "Relevant Certifications",
        "Number of Programming Languages",
        "Number of Technical Skills",
    ]:
        bad_profile = valid_profile.copy()
        bad_profile[field] = -1
        is_valid, errors = validate_student_profile(bad_profile)
        assert is_valid is False
        assert any(field in e for e in errors)


def test_invalid_complexity_range(valid_profile):
    """Test that complexity values outside 1-5 are rejected."""
    bad_profile = valid_profile.copy()
    bad_profile["Average Project Complexity"] = 0
    is_valid, errors = validate_student_profile(bad_profile)
    assert is_valid is False
    assert any("Complexity" in e for e in errors)

    bad_profile["Average Project Complexity"] = 6
    is_valid, errors = validate_student_profile(bad_profile)
    assert is_valid is False


def test_invalid_internship_exp(valid_profile):
    """Test that internship values other than 'Yes' or 'No' are rejected."""
    bad_profile = valid_profile.copy()
    bad_profile["Previous Internship Experience"] = "Maybe"
    is_valid, errors = validate_student_profile(bad_profile)
    assert is_valid is False
    assert any("Internship" in e for e in errors)


def test_invalid_domain_selection(valid_profile):
    """Test that unrecognized domain names are rejected."""
    bad_profile = valid_profile.copy()
    bad_profile["Internship Domain Applied"] = "Quantum Cooking"
    is_valid, errors = validate_student_profile(bad_profile)
    assert is_valid is False
    assert any("Domain" in e for e in errors)


def test_missing_required_keys(valid_profile):
    """Test that omitting required keys fails validation."""
    bad_profile = valid_profile.copy()
    del bad_profile["CGPA"]
    is_valid, errors = validate_student_profile(bad_profile)
    assert is_valid is False
    assert any("Missing required field" in e for e in errors)

