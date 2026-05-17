"""
data_loader.py

Utilities for loading banking fraud detection datasets.

This module is responsible for:

1. Loading raw CSV datasets

2. Validating dataset existence

3. Performing basic dataset checks

4. Returning pandas DataFrames for downstream pipelines

This module should ONLY handle data ingestion logic.
"""

from pathlib import Path
import pandas as pd 

from src.config import RAW_DATA_DIR, DATASET_NAME

# Load Raw Dataset
def load_raw_data (filename: str= DATASET_NAME) -> pd.DataFrame:
    """
    Load raw banking transaction dataset.

    Parameters
    ----------
    filename : str, optional

    Name of CSV file inside data/raw directory.
        
    Returns
    -------
    pd.DataFrame
     Loaded dataset as pandas DataFrame.
    
    Raises
    ------
    FileNotFoundError
    If dataset file does not exist.
    """
    filepath = RAW_DATA_DIR / filename
    
    # Check file existance
    if not filepath.exists():
        
        raise FileNotFoundError(
            f"File not found at: {filepath}"
        )
    
    print("\nLoading Dataset...")
    print(f"Path: {filepath}")
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    print("\nDataset Loaded Successfully")
    print(f"Shape: {df.shape}")
    
    return df

# Dataset Summary
def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Print basic dataset summary information.

    Parameters
    ----------
    df : pd.DataFrame
    Input dataset.
    """
    print("\nDataset Summary")
    print("-" * 40)
    
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    
    print("\nColumns Names")
    print("-" * 40)
    
    print(df.columns.to_list())
    
    print("\n Missing Values")
    print("-" * 40)
    
    print(df.isnull().sum())
    
    print("\n Data Types")
    print("-" * 40)
    
    print(df.dtypes)
    
# Duplicates Check
def check_duplicates(df: pd.DataFrame) -> int:
    """
    Check duplicate rows in dataset.
     
    Parameters
    ----------
    df : pd.DataFrame

    Input dataset.

    Returns
    -------
    int
    Number of duplicate rows.
    """
    duplicate_count = df.duplicated().sum()
    
    print("\nDuplicated Rows")
    print("-" * 40)
    
    print(f"Duplicates Found: {duplicate_count}")
    
    return duplicate_count

# Main test
if __name__ == "__main__":
    """
    Run module independently for testing.

    Example:
    python -m src.data_loader
    """
    df = load_raw_data()
    
    summarize_dataset(df)
    
    check_duplicates(df)
    
    print("\nFirst 5 Rows: ")
    print("-" * 40)
    
    print(df.head())
    
