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
        categorical_features: list,
        passthrough_features: list
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
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="if_binary"
        ))
    ])

    # Construct Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ("num", numerical_pipline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
        ("pass", "passthrough", passthrough_features)
    ], remainder="drop")

    if hasattr(preprocessor, "set_output"):
        preprocessor.set_output(transform="pandas")
    
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
    categorical_features = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.to_list()

    # Keep binary flags unscaled
    passthrough_features = []

    binary_flags = [
        "night_transaction_flag",
        "suspicious_ip_flag",
        "international_transaction_flag",
        "card_present_flag",
    ]

    for flag in binary_flags:
        if flag in numerical_features:
            numerical_features.remove(flag)
            passthrough_features.append(flag)
    
    print("\nPreprocessing Summary")
    print("-" * 40)

    print(f"Numerical Features   : {len(numerical_features)}")
    print(f"Categorical Features : {len(categorical_features)}")
    print(f"Binary Flags         : {len(passthrough_features)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Build Preprocessor
    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        passthrough_features=passthrough_features
    )

    # Transform Data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing Completed Successfully.")

    print(f"Training Data Shape: {X_train_processed.shape}")
    print(f"Testing Data Shape: {X_test_processed.shape}")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
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

    X_train, X_test, y_train, y_test, preprocessor = (
        preprocess_dataset(df)
    )
    print("\nPreview:")
    print(X_train.head())
