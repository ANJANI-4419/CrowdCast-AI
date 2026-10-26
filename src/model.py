"""Model training, evaluation, and inference pipelines for CrowdCast AI.

Builds two separate pipelines:
1. RandomForestClassifier for crowd level categories (Low, Moderate, High).
2. RandomForestRegressor for continuous passenger count estimation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    CROWD_LEVEL_ORDER,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    build_preprocessor,
    validate_dataframe,
)


@dataclass
class ModelEvaluationReport:
    """Encapsulates comprehensive evaluation metrics for both models."""

    # Classification metrics
    accuracy: float
    classification_rep: Dict[str, Any]
    confusion_mat: np.ndarray
    classes: List[str]

    # Regression metrics
    mae: float
    rmse: float
    r2: float

    # Test set actuals vs predictions
    y_test_reg_actual: np.ndarray
    y_test_reg_pred: np.ndarray
    y_test_clf_actual: np.ndarray
    y_test_clf_pred: np.ndarray


def create_classifier_pipeline(random_state: int = 42) -> Pipeline:
    """Build full classification pipeline with preprocessor + RandomForestClassifier."""
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def create_regressor_pipeline(random_state: int = 42) -> Pipeline:
    """Build full regression pipeline with preprocessor + RandomForestRegressor."""
    preprocessor = build_preprocessor()
    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])


def train_and_evaluate_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Pipeline, ModelEvaluationReport, pd.DataFrame]:
    """Train both classification and regression pipelines on a shared train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset containing features and both target columns.
    test_size : float, optional
        Fraction of data reserved for held-out evaluation (default 0.2).
    random_state : int, optional
        Seed for split and estimators (default 42).

    Returns
    -------
    Tuple[Pipeline, Pipeline, ModelEvaluationReport, pd.DataFrame]
        clf_pipeline, reg_pipeline, evaluation_report, X_test
    """
    is_valid, msg = validate_dataframe(df, is_training=True)
    if not is_valid:
        raise ValueError(f"Training data validation failed: {msg}")

    X = df[ALL_FEATURE_COLUMNS]
    y_reg = df[TARGET_REGRESSION]
    y_clf = df[TARGET_CLASSIFICATION]

    # Perform train/test split (stratify on crowd_level to preserve balanced class proportions)
    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = (
        train_test_split(
            X,
            y_clf,
            y_reg,
            test_size=test_size,
            random_state=random_state,
            stratify=y_clf,
        )
    )

    # 1. Fit Classifier
    clf_pipeline = create_classifier_pipeline(random_state=random_state)
    clf_pipeline.fit(X_train, y_train_clf)

    # 2. Fit Regressor
    reg_pipeline = create_regressor_pipeline(random_state=random_state)
    reg_pipeline.fit(X_train, y_train_reg)

    # 3. Evaluate Classifier on held-out test set
    y_pred_clf = clf_pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test_clf, y_pred_clf))
    clf_report = classification_report(
        y_test_clf,
        y_pred_clf,
        labels=CROWD_LEVEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(
        y_test_clf,
        y_pred_clf,
        labels=CROWD_LEVEL_ORDER,
    )

    # 4. Evaluate Regressor on held-out test set
    y_pred_reg = reg_pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test_reg, y_pred_reg))
    rmse = float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
    r2 = float(r2_score(y_test_reg, y_pred_reg))

    report = ModelEvaluationReport(
        accuracy=accuracy,
        classification_rep=clf_report,
        confusion_mat=conf_matrix,
        classes=CROWD_LEVEL_ORDER,
        mae=mae,
        rmse=rmse,
        r2=r2,
        y_test_reg_actual=y_test_reg.values,
        y_test_reg_pred=y_pred_reg,
        y_test_clf_actual=y_test_clf.values,
        y_test_clf_pred=y_pred_clf,
    )

    return clf_pipeline, reg_pipeline, report, X_test


def predict_single_trip(
    clf_pipeline: Pipeline,
    reg_pipeline: Pipeline,
    input_features: Union[Dict[str, Any], pd.DataFrame],
) -> Dict[str, Any]:
    """Generate both crowd classification and passenger regression for input features.

    Parameters
    ----------
    clf_pipeline : Pipeline
        Trained classifier pipeline.
    reg_pipeline : Pipeline
        Trained regressor pipeline.
    input_features : Dict or DataFrame
        Input features. Unseen categories or missing values are safely handled.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - crowd_level: str
        - crowd_probabilities: Dict[str, float]
        - estimated_passengers: float (rounded to integer)
        - uncertainty_std: float (inter-tree prediction spread)
    """
    if isinstance(input_features, dict):
        df_input = pd.DataFrame([input_features])
    else:
        df_input = input_features.copy()

    # Fill any missing required columns with safe defaults
    defaults = {
        "route_id": "Route_101",
        "day_type": "Weekday",
        "departure_hour": 8,
        "weather": "Clear",
        "nearby_event": 0,
        "college_term": 1,
        "trip_duration": 35.0,
    }
    for col, default_val in defaults.items():
        if col not in df_input.columns:
            df_input[col] = default_val

    # Ensure correct column ordering
    df_input = df_input[ALL_FEATURE_COLUMNS]

    # Predict crowd classification
    clf_pred = clf_pipeline.predict(df_input)[0]
    clf_probs_array = clf_pipeline.predict_proba(df_input)[0]
    clf_classes = list(clf_pipeline.classes_)

    # Align probabilities to standard CROWD_LEVEL_ORDER
    prob_dict = {}
    for level in CROWD_LEVEL_ORDER:
        if level in clf_classes:
            prob_dict[level] = float(clf_probs_array[clf_classes.index(level)])
        else:
            prob_dict[level] = 0.0

    # Predict passenger count regression
    reg_pred = float(reg_pipeline.predict(df_input)[0])
    reg_pred_rounded = max(1, int(round(reg_pred)))

    # Estimate ensemble uncertainty from individual trees
    try:
        reg_model: RandomForestRegressor = reg_pipeline.named_steps["regressor"]
        preprocessor = reg_pipeline.named_steps["preprocessor"]
        X_trans = preprocessor.transform(df_input)
        tree_predictions = [tree.predict(X_trans)[0] for tree in reg_model.estimators_]
        uncertainty_std = float(np.std(tree_predictions))
    except Exception:
        uncertainty_std = 3.5

    return {
        "crowd_level": clf_pred,
        "crowd_probabilities": prob_dict,
        "estimated_passengers": reg_pred_rounded,
        "uncertainty_std": round(uncertainty_std, 1),
    }
