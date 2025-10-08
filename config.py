import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Raw Dataset
RAW_DATA_DIR_PATH = os.path.join(DATA_DIR,"raw")
PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR,"preprocessed")
BANK_MARKETING_DATASET = os.path.join(RAW_DATA_DIR_PATH,"bank-additional-full.csv")
RAW_DATASET_PATH = os.path.join(RAW_DATA_DIR_PATH,"raw-bank-marketing.csv")
PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR_PATH,"preprocessed-bank-marketing.csv")


# TRAIN TEST SETS
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "X_train.parquet")
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "X_test.parquet")

Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "y_train.parquet")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "y_test.parquet")

X_TRAIN_OHE = os.path.join(ARTIFACTS_DIR, "X_train_ohe.parquet")
X_TEST_OHE = os.path.join(ARTIFACTS_DIR, "X_test_ohe.parquet")

X_TRAIN_LE = os.path.join(ARTIFACTS_DIR, "X_train_le.parquet")
X_TEST_LE = os.path.join(ARTIFACTS_DIR, "X_test_le.parquet")


# Baseline Paths
BASELINE_MODELS = os.path.join(ARTIFACTS_DIR, "baseline_models")
LOGISTIC_BASELINE = os.path.join(BASELINE_MODELS, "logistic_model.pkl")

