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
RANDOM_FOREST_BASLEINE = os.path.join(BASELINE_MODELS, "random_forrest_model.pkl")
XGBOOST_BASELINE = os.path.join(BASELINE_MODELS, "xgboost_model.pkl")
ANN_BASELINE = os.path.join(BASELINE_MODELS, "neural_network_baseline.pt")


# Class Imbalance Handled Model Paths
CLASS_IMBALANCE_MODELS = os.path.join(ARTIFACTS_DIR, "class_imbalance_models")
IMBALANCE_LOGISTIC_BASELINE = os.path.join(CLASS_IMBALANCE_MODELS, "imbalance_logistic_model.pkl")
IMBALANCE_RANDOM_FOREST_BASELINE = os.path.join(CLASS_IMBALANCE_MODELS, "imbalance_random_forrest_model.pkl")
IMBALANCE_XGBOOST_BASELINE = os.path.join(CLASS_IMBALANCE_MODELS, "imbalance_xgboost_model.pkl")
IMBALANCE_ANN_BASELINE = os.path.join(CLASS_IMBALANCE_MODELS, "imbalance_neural_network.pt")


#Hyperparamter tuning
LOGREG_PARAMETERS = os.path.join(ARTIFACTS_DIR, "best_logreg_params.pkl")
LOGREG2_PARAMETERS = os.path.join(ARTIFACTS_DIR, "best_logreg_params_with_resampling.pkl")
LOGREG2_MODEL = os.path.join(ARTIFACTS_DIR, "best_logreg_model.pkl")
RANDOM_FOREST_PARAMETERS = os.path.join(ARTIFACTS_DIR, "best_rf_params.pkl")
XGBOOST_PARAMETERS = os.path.join(ARTIFACTS_DIR, "best_xgb_params.pkl")
NEURAL_NET_PARAMETERS = os.path.join(ARTIFACTS_DIR, "best_nn_params.pkl")


# Improved Models paths
MODEL_IMPROVEMENT_DIR = os.path.join(ARTIFACTS_DIR, "model_improvement")
X_TRAIN_STRAT_PATH = os.path.join(MODEL_IMPROVEMENT_DIR, "X_train_strat.parquet")
X_TEST_STRAT_PATH = os.path.join(MODEL_IMPROVEMENT_DIR, "X_test_strat.parquet")

Y_TRAIN_STRAT_PATH = os.path.join(MODEL_IMPROVEMENT_DIR, "y_train_strat.parquet")
Y_TEST_STRAT_PATH = os.path.join(MODEL_IMPROVEMENT_DIR, "y_test_strat.parquet")

X_TRAIN_LR_NN = os.path.join(MODEL_IMPROVEMENT_DIR, "X_train_lr_nn.parquet")
X_TEST_LR_NN = os.path.join(MODEL_IMPROVEMENT_DIR, "X_test_lr_nn.parquet")

X_TRAIN_RF_XG = os.path.join(MODEL_IMPROVEMENT_DIR, "X_train_rf_xg.parquet")
X_TEST_RF_XG = os.path.join(MODEL_IMPROVEMENT_DIR, "X_test_rf_xg.parquet")