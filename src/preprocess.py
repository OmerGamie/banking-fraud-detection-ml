"""
preprocess.py

Preprocessing utilities for the Banking Fraud Detection Project.

This module handles:
--------------------
1. Feature engineering integration
2. Train/test splitting
3. Feature preprocessing
4. Numerical scaling
5. Categorical encoding
6. Pipeline construction

The preprocessing pipeline guarantees that all models receive
consistent and production-ready input data.
"""
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler
)

from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

from src.features import engineer_features

# Build Preprocessing Pipeline
def build_preprocessor(
        numerical_features: list,
        categorical_features: list
) ->ColumnTransformer:
    """
    Create preprocessing pipeline.

    Parameters:
    numerical_features : list
        Numerical feature column names.
    categorical_features : list
        Categorical feature column names.

    Returns:
    ColumnTransformer
        Configured preprocessing pipeline.
    """
    # Construct Numerical and Categorical Pipelines
    numerical_pipline = Pipeline([
        ("scaler", RobustScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="if_binary"))
    ])

    # Construct Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ("num", numerical_pipline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

# Main Preprocessing Function
def preprocess_dataset(df: pd.DataFrame) -> tuple:
    """
Complete preprocessing workflow.

    Steps:
    ------
    1. Feature engineering
    2. Split features/target
    3. Detect numerical/categorical columns
    4. Train-test split
    5. Build preprocessing pipeline
    6. Transform datasets

    Parameters
    ----------
    df : pd.DataFrame
        Raw banking dataset.

    Returns
    -------
    tuple
        (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            preprocessor
        )
"""

    # Feature Engineering
    df = engineer_features(df)
    
    # Split features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Remove transaction ID if present
    if "transaction_id" in X.columns:
        X = X.drop(columns=["transaction_id"])

    # Detect numerical and categorical columns
    numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.to_list()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Build Preprocessor
    preprocesor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    # Transform Data
    X_train_processed = preprocesor.fit_transform(X_train)
    X_test_processed = preprocesor.transform(X_test)

    print("\nPreprocessing Completed Successfully.")

    print(f"Training Data Shape: {X_train_processed.shape}")
    print(f"Testing Data Shape: {X_test_processed.shape}")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocesor
    )

# Independent Testing
if __name__ == "__main__":
    """
    Run module independently for testing.

    Example:
    python -m src.preprocess
    """
    from src.data_loader import load_raw_data

    df = load_raw_data()

    preprocess_dataset(df)
