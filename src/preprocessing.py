"""Preprocessing module for CrowdCast AI.

Defines feature definitions, crowd level threshold logic,
and scikit-learn preprocessing pipelines.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Consistent threshold cutoffs for public transport capacity
# Low: < 35 passengers (ample seating available)
# Moderate: 35 - 70 passengers (busy, standing room available)
# High: > 70 passengers (packed, near or at crush capacity)
LOW_THRESHOLD: int = 35
HIGH_THRESHOLD: int = 70

CROWD_LEVEL_ORDER: List[str] = ["Low", "Moderate", "High"]

CATEGORICAL_FEATURES: List[str] = [
    "route_id",
    "day_type",
    "weather",
]

NUMERICAL_FEATURES: List[str] = [
    "departure_hour",
    "nearby_event",
    "college_term",
    "trip_duration",
]

ALL_FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_REGRESSION: str = "passenger_count"
TARGET_CLASSIFICATION: str = "crowd_level"


def passenger_to_crowd_level(passenger_count: float) -> str:
    """Map a continuous passenger count to categorical crowd level.

    Parameters
    ----------
    passenger_count : float or int
        Observed or estimated number of passengers.

    Returns
    -------
    str
        'Low', 'Moderate', or 'High'.
    """
    if pd.isna(passenger_count):
        return "Low"
    if passenger_count < LOW_THRESHOLD:
        return "Low"
    elif passenger_count <= HIGH_THRESHOLD:
        return "Moderate"
    else:
        return "High"


def vectorize_crowd_levels(passenger_counts: np.ndarray) -> np.ndarray:
    """Vectorized conversion from passenger counts to crowd levels."""
    conditions = [
        passenger_counts < LOW_THRESHOLD,
        (passenger_counts >= LOW_THRESHOLD) & (passenger_counts <= HIGH_THRESHOLD),
        passenger_counts > HIGH_THRESHOLD,
    ]
    choices = ["Low", "Moderate", "High"]
    return np.select(conditions, choices, default="Low")


def validate_dataframe(df: pd.DataFrame, is_training: bool = False) -> Tuple[bool, str]:
    """Validate that required feature (and optional target) columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to inspect.
    is_training : bool, optional
        Whether the DataFrame must also contain target columns, by default False.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input data is not a valid pandas DataFrame."

    missing_features = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        return False, f"Missing required feature columns: {missing_features}"

    if is_training:
        missing_targets = [
            col
            for col in [TARGET_REGRESSION, TARGET_CLASSIFICATION]
            if col not in df.columns
        ]
        if missing_targets:
            return False, f"Missing training target columns: {missing_targets}"

    return True, "Dataset structure is valid."


def build_preprocessor() -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for numerical and categorical features.

    Numerical pipeline:
    - Median imputation for missing values
    - StandardScaler normalization

    Categorical pipeline:
    - Most frequent imputation for missing categories
    - OneHotEncoder with handle_unknown='ignore' to handle unseen route/weather gracefully
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
