from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models" / "trained"
REPORT_DIR = BASE_DIR / "reports"

# Dataset
DATASET_NAME = "banking_transactions.csv"

TARGET_COLUMN = "fraud_flag"

RANDOM_STATE = 42
TEST_SIZE = 0.2
