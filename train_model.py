"""
Model Training and Evaluation Pipeline.

Loads student internship dataset, splits into stratified train/test sets,
fits ColumnTransformer preprocessing on train data, trains a RandomForestClassifier,
computes genuine evaluation metrics, and saves the complete fitted Pipeline
and metrics artifacts to the 'models/' directory.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.preprocessing import (
    build_preprocessor,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_CLASSES,
)
from generate_dataset import generate_internship_dataset


def train_and_evaluate(
    data_path: str = "data/internship_dataset.csv",
    model_output_path: str = "models/internship_success_model.joblib",
    metrics_output_path: str = "models/model_metrics.json",
    random_state: int = 42,
    test_size: float = 0.20,
) -> dict:
    """
    Execute full model training, evaluation, and serialization workflow.

    Parameters
    ----------
    data_path : str
        Path to student dataset CSV.
    model_output_path : str
        Destination path for saved joblib model pipeline.
    metrics_output_path : str
        Destination path for evaluation metrics JSON.
    random_state : int
        Seed for reproducibility.
    test_size : float
        Proportion of dataset reserved for evaluation.

    Returns
    -------
    dict
        Dictionary of measured performance metrics.
    """
    # 1. Ensure dataset exists
    if not os.path.exists(data_path):
        print(f"Dataset not found at '{data_path}'. Generating synthetic dataset now...")
        generate_internship_dataset(n_samples=1000, random_state=random_state, output_path=data_path)

    df = pd.read_csv(data_path)
    print(f"Loaded dataset from '{data_path}' with {len(df)} records.")

    # 2. Extract features and target
    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # 3. Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples.")

    # 4. Construct Scikit-learn Pipeline
    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # 5. Fit pipeline ONLY on training data
    print("Fitting preprocessing and Random Forest classifier on training data...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate on held-out test data
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Identify index for "Selected" class
    classes = list(pipeline.classes_)
    selected_idx = classes.index("Selected")
    y_test_binary = (y_test == "Selected").astype(int)
    y_proba_selected = y_proba[:, selected_idx]

    # Calculate exact metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="Selected"))
    rec = float(recall_score(y_test, y_pred, pos_label="Selected"))
    f1 = float(f1_score(y_test, y_pred, pos_label="Selected"))
    roc_auc = float(roc_auc_score(y_test_binary, y_proba_selected))
    cm = confusion_matrix(y_test, y_pred, labels=["Not Selected", "Selected"]).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "dataset_size": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_state": random_state,
        "classes": classes,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "model_parameters": {
            "n_estimators": classifier.n_estimators,
            "max_depth": classifier.max_depth,
            "min_samples_split": classifier.min_samples_split,
            "min_samples_leaf": classifier.min_samples_leaf,
            "class_weight": "balanced",
        }
    }

    # 7. Print evaluation results
    print("\n" + "=" * 55)
    print("ACTUAL MEASURED MODEL EVALUATION RESULTS:")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision: {prec:.4f} (Selected)")
    print(f"Recall:    {rec:.4f} (Selected)")
    print(f"F1-Score:  {f1:.4f} (Selected)")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix [Row: True, Col: Pred] (Labels: Not Selected, Selected):")
    print(np.array(cm))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 55)

    # 8. Persist artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    joblib.dump(pipeline, model_output_path)
    print(f"Saved complete trained pipeline to '{model_output_path}'.")

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to '{metrics_output_path}'.")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()

