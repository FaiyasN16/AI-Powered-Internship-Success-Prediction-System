"""Tests for dataset schema, reproducibility, and ColumnTransformer preprocessing."""

import os
import pandas as pd
import numpy as np
import pytest

from generate_dataset import generate_internship_dataset, DOMAINS
from src.preprocessing import (
    build_preprocessor,
    get_feature_names_from_preprocessor,
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


def test_dataset_generation_and_schema():
    """Verify that dataset generation produces expected schema, types, and ranges."""
    df = generate_internship_dataset(n_samples=100, random_state=42, output_path=None)

    assert len(df) == 100
    assert TARGET_COLUMN in df.columns
    for col in ALL_FEATURE_COLUMNS:
        assert col in df.columns

    # Numerical ranges
    assert (df["CGPA"] >= 0.0).all() and (df["CGPA"] <= 10.0).all()
    assert (df["Number of Relevant Projects"] >= 0).all()
    assert (df["Average Project Complexity"] >= 1).all() and (df["Average Project Complexity"] <= 5).all()
    assert (df["Relevant Certifications"] >= 0).all()
    assert (df["Number of Programming Languages"] >= 0).all()
    assert (df["Number of Technical Skills"] >= 0).all()

    # Categorical domain checks
    assert set(df["Previous Internship Experience"].unique()).issubset({"Yes", "No"})
    assert set(df["Internship Domain Applied"].unique()).issubset(set(DOMAINS))
    assert set(df[TARGET_COLUMN].unique()).issubset({"Selected", "Not Selected"})


def test_dataset_reproducibility():
    """Verify that using the same random seed produces identical dataset records."""
    df1 = generate_internship_dataset(n_samples=50, random_state=99, output_path=None)
    df2 = generate_internship_dataset(n_samples=50, random_state=99, output_path=None)

    pd.testing.assert_frame_equal(df1, df2)


def test_preprocessor_fitting_and_transformation():
    """Test that ColumnTransformer fits correctly and produces consistent feature array."""
    df = generate_internship_dataset(n_samples=50, random_state=42, output_path=None)
    X = df[ALL_FEATURE_COLUMNS]

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 50
    # 6 numerical + 2 for previous internship (No, Yes) + 7 for domains = 15 columns
    assert X_trans.shape[1] == 15

    names = get_feature_names_from_preprocessor(preprocessor)
    assert len(names) == 15
    assert "CGPA" in names[0]

