"""
Data Preprocessing and Validation Module.

Defines feature columns, schema validation, and Scikit-learn ColumnTransformer
pipelines for reproducible model training and inference.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical Domain Choices
VALID_DOMAINS = [
    "Web Development",
    "Artificial Intelligence and Machine Learning",
    "Data Science and Analytics",
    "Cybersecurity",
    "Cloud Computing",
    "Mobile App Development",
    "Software Engineering",
]

NUMERICAL_FEATURES = [
    "CGPA",
    "Number of Relevant Projects",
    "Average Project Complexity",
    "Relevant Certifications",
    "Number of Programming Languages",
    "Number of Technical Skills",
]

CATEGORICAL_FEATURES = [
    "Previous Internship Experience",
    "Internship Domain Applied",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Internship Outcome"
TARGET_CLASSES = ["Not Selected", "Selected"]


def validate_student_profile(profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate student profile input fields for type, presence, and valid domain/ranges.

    Parameters
    ----------
    profile : dict
        Dictionary of student profile attributes.

    Returns
    -------
    (bool, list of str)
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    # Check required keys
    for col in ALL_FEATURE_COLUMNS:
        if col not in profile or profile[col] is None:
            errors.append(f"Missing required field: '{col}'.")

    if errors:
        return False, errors

    # 1. CGPA
    try:
        cgpa = float(profile["CGPA"])
        if cgpa < 0.0 or cgpa > 10.0:
            errors.append(f"CGPA must be between 0.0 and 10.0 (received {cgpa}).")
    except (ValueError, TypeError):
        errors.append(f"CGPA must be a valid numerical value (received {profile.get('CGPA')}).")

    # 2. Number of Relevant Projects
    try:
        projects = int(profile["Number of Relevant Projects"])
        if projects < 0:
            errors.append(f"Number of Relevant Projects cannot be negative (received {projects}).")
        elif projects > 50:
            errors.append(f"Number of Relevant Projects is unrealistically high (received {projects}).")
    except (ValueError, TypeError):
        errors.append("Number of Relevant Projects must be an integer.")

    # 3. Average Project Complexity
    try:
        complexity = int(profile["Average Project Complexity"])
        if complexity < 1 or complexity > 5:
            errors.append(f"Average Project Complexity must be an integer between 1 and 5 (received {complexity}).")
    except (ValueError, TypeError):
        errors.append("Average Project Complexity must be an integer.")

    # 4. Previous Internship Experience
    prev_intern = str(profile.get("Previous Internship Experience")).strip()
    if prev_intern not in ["Yes", "No"]:
        errors.append(f"Previous Internship Experience must be 'Yes' or 'No' (received '{prev_intern}').")

    # 5. Relevant Certifications
    try:
        certs = int(profile["Relevant Certifications"])
        if certs < 0:
            errors.append(f"Relevant Certifications cannot be negative (received {certs}).")
        elif certs > 30:
            errors.append(f"Relevant Certifications count is unrealistically high (received {certs}).")
    except (ValueError, TypeError):
        errors.append("Relevant Certifications must be an integer.")

    # 6. Number of Programming Languages
    try:
        langs = int(profile["Number of Programming Languages"])
        if langs < 0:
            errors.append(f"Number of Programming Languages cannot be negative (received {langs}).")
        elif langs > 25:
            errors.append(f"Number of Programming Languages is unrealistically high (received {langs}).")
    except (ValueError, TypeError):
        errors.append("Number of Programming Languages must be an integer.")

    # 7. Number of Technical Skills
    try:
        skills = int(profile["Number of Technical Skills"])
        if skills < 0:
            errors.append(f"Number of Technical Skills cannot be negative (received {skills}).")
        elif skills > 50:
            errors.append(f"Number of Technical Skills is unrealistically high (received {skills}).")
    except (ValueError, TypeError):
        errors.append("Number of Technical Skills must be an integer.")

    # 8. Internship Domain Applied
    domain = str(profile.get("Internship Domain Applied")).strip()
    if domain not in VALID_DOMAINS:
        errors.append(
            f"Invalid Internship Domain Applied: '{domain}'. Must be one of: {', '.join(VALID_DOMAINS)}."
        )

    is_valid = len(errors) == 0
    return is_valid, errors


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a Scikit-learn ColumnTransformer for preprocessing.

    Encodes categorical features via OneHotEncoder and scales numerical
    features via StandardScaler.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor pipeline component.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract human-readable feature names after ColumnTransformer transformation.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer.

    Returns
    -------
    list of str
        Clean list of transformed feature names.
    """
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
            feature_names.extend(names)
        else:
            feature_names.extend(cols)
    return [str(f).replace("num__", "").replace("cat__", "") for f in feature_names]


def profile_to_dataframe(profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a single validated student profile dictionary into a single-row DataFrame.
    """
    return pd.DataFrame([{
        "CGPA": float(profile["CGPA"]),
        "Number of Relevant Projects": int(profile["Number of Relevant Projects"]),
        "Average Project Complexity": int(profile["Average Project Complexity"]),
        "Previous Internship Experience": str(profile["Previous Internship Experience"]).strip(),
        "Relevant Certifications": int(profile["Relevant Certifications"]),
        "Number of Programming Languages": int(profile["Number of Programming Languages"]),
        "Number of Technical Skills": int(profile["Number of Technical Skills"]),
        "Internship Domain Applied": str(profile["Internship Domain Applied"]).strip(),
    }])

