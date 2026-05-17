# Setup
import sys
sys.path.append(".")

# Imports
import subprocess

from src.config import RAW_DATA_DIR

def download_kaggle_dataset(dataset: str) -> None:
    """
    Download and unzip a kaggle dataset.
    
    Parameters:
    -----------
    dataset: str
        Kaggle dataset identifier 
    
    Example:
    --------
    deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset
    """
    # Create Raw data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading dataset from kaggle...")
    print(f"Dataset: {dataset}")
    
    # Build kaggle CLI command
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(RAW_DATA_DIR),
        "--unzip"
    ]
   
    # Run command
    result = subprocess.run(cmd)
    
    # Check result
    if result.returncode == 0:
        print("\nDataset downloaded successfully.")
        print(f"Saved to : {RAW_DATA_DIR}")
    
    else:
        print("\nDataset download Failed")
        
if __name__ == "__main__":
    
    DATASET_ID = ("deepeshkansotia/banking-fraud-detection-and-risk-analytics-dataset")
    
    download_kaggle_dataset(DATASET_ID)
    