import pandas as pd

from src.preprocessing.data_preprocessing import preprocessing_pipeling
from src.baseline_models.logistic_regression_baseline import baseline_logistic_regression_model
from src.baseline_models.random_forrest_baseline import baseline_random_forrest
from src.baseline_models.xgboost_baseline import baseline_xgboost_model


def main():
    baseline_xgboost_model()



if __name__ == "__main__":
    main()