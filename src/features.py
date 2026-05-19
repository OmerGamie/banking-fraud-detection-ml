"""
features.py

Feature Engineering Functions for the Banking Fraud Detection Project.
"""
import pandas as pd

# Interaction Features
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Interaction based fraud risk features."""
    # Device risk amplified by anomaly behavior
    df['device_anomaly_interaction'] = (
        df["anomaly_score"] * df["device_risk_score"]
    )
    # Login intensity against risky devices
    df['login_device_interaction'] = (
        df["login_attempts"] * df["device_risk_score"]
    )
    return df

# Velocity Features
def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Velocity based fraud risk features."""
    # Operational transaction velocity risk
    df['velocity_risk_index'] = (
        df["transfer_frequency"] * (df["failed_transactions_last_30d"] + 1)
    )
    
    # Transaction size × frequency
    df['transaction_velocity_risk'] = (
        df["transaction_amount"] * df["transfer_frequency"]
    )
    
    return df

# Financial Ratios Features
def add_financial_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Financial Ratio based fraud risk features."""
    # Transaction amount relative to customer balance
    df['transaction_balance_ratio'] = (
        df["transaction_amount"] / (df["avg_monthly_balance"] + 1)
    )

    # Transaction amount relative to account age
    df['amount_to_account_age_ratio'] = (
        df["transaction_amount"] / (df["account_age_days"] + 1)
    )
    
    # Failed transactions relative to transfer activity
    df['failed_transaction_ratio'] = (
        df["failed_transactions_last_30d"] / (df["transfer_frequency"] + 1)
    )
    
    return df

# Behavioral Features
def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Behavioral based fraud risk features."""
    # Authentication risk score
    df['authentication_risk_score'] = (
        df["suspicious_ip_flag"] * df["login_attempts"]
    )
    return df

# Temporal Features
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Time based fraud indicators."""
    # Flag transactions during unusual hours
    df['night_transaction_flag'] = (
        (df["transaction_time_hour"] <= 5) |
        (df["transaction_time_hour"] >= 23)
    ).astype(int)
    
    return df

# Composite Risk Features
def add_composite_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates Composite Risk based fraud risk features."""
    df['composite_risk_score'] = (
        df["anomaly_score"] * 0.4 +
        df["device_risk_score"] * 0.3 +
        df["login_attempts"] * 0.2 +
        df["failed_transactions_last_30d"] * 0.1
    )
    return df

# Master Feature Engineering Pipeline
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    # Copy dataframe
    df_out = df.copy()
    
    # Chain modular Generation Layers
    df_out = add_interaction_features(df_out)
    df_out = add_velocity_features(df_out)
    df_out = add_financial_ratio_features(df_out)
    df_out = add_behavioral_features(df_out)
    df_out = add_temporal_features(df_out)
    df_out = add_composite_risk_features(df_out)

    return df_out

if __name__ == "__main__":
    """
    Run module independently for testing.

    Example:
    python -m src.features
    """
    from src.data_loader import load_raw_data

    df = load_raw_data()
    
    df_engineered = engineer_features(df)

    print(f"Feature Engineering Complete. Feature Shape: {df_engineered.shape}")