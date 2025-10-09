from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import wandb
from config import (RAW_DATASET_PATH,
                    X_TRAIN_STRAT_PATH, X_TEST_STRAT_PATH, Y_TRAIN_STRAT_PATH, Y_TEST_STRAT_PATH,
                    X_TRAIN_LR_NN, X_TEST_LR_NN, X_TRAIN_RF_XG, X_TEST_RF_XG)


def feature_engineering(dataset):
    # From pdays variable
    dataset['contact_recency'] = dataset['pdays'].apply(lambda x: 'never' if x == 999 else 'recent' if x < 90 else 'long')

    # From previous variable
    dataset['has_previous_contact'] = np.where(dataset['previous'] > 0, 1, 0)

    # Interaction count
    dataset['interaction_count'] = dataset['campaign'] + dataset['previous']

    # Socioeconomic index 
    macro_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(dataset[macro_features])
    pca = PCA(n_components=1, random_state=42)
    dataset['socioeconomic_index'] = pca.fit_transform(macro_scaled)

    return dataset



def outlier_handling(dataset):
    # Right tail capping for campaign column
    cap_value = dataset['campaign'].quantile(0.99)
    dataset['campaign'] = np.clip(dataset['campaign'], None, cap_value)

    # Right tail capping for age column
    age_cap = dataset['age'].quantile(0.99)
    dataset['age'] = np.clip(dataset['age'], None, age_cap)

    return dataset



def drop_columns(dataset):
    # Drop leakage and redundant columns
    dataset = dataset.drop(columns=['duration'], errors='ignore')
    dataset = dataset.drop(columns=['pdays'], errors='ignore')

    return dataset



def stratified_split(dataset):
    X = dataset.drop("y", axis=1)
    y = dataset["y"]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols




def one_hot_encoding(X_train, X_test, categorical_cols, numeric_cols):

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    X_train_lin = X_train.copy()
    X_test_lin = X_test.copy()

    # One-hot encode categorical features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_ohe = pd.DataFrame(
        ohe.fit_transform(X_train_lin[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=X_train_lin.index
    )
    X_test_ohe = pd.DataFrame(
        ohe.transform(X_test_lin[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=X_test_lin.index
    )

    # Replace categorical with encoded
    X_train_LR = pd.concat([X_train_lin.drop(columns=categorical_cols), X_train_ohe], axis=1)
    X_test_LR = pd.concat([X_test_lin.drop(columns=categorical_cols), X_test_ohe], axis=1)

    return X_train_LR, X_test_LR
    


def label_encoding(X_train, X_test, categorical_cols):
    
    X_train_tree = X_train.copy()
    X_test_tree = X_test.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_train_tree[col] = le.fit_transform(X_train_tree[col])
        # Handle unseen labels
        X_test_tree[col] = X_test_tree[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_test_tree[col] = le.transform(X_test_tree[col])

    return X_train_tree, X_test_tree



def improved_preprocessing_pipeling():

    run = wandb.init(project="Bank_Marketing_Classifier", name="improved_preprocessing")

    # Get raw dataset
    raw_dataset = pd.read_csv(RAW_DATASET_PATH)
    cleaned_dataset = outlier_handling(raw_dataset)
    processed_dataset = feature_engineering(cleaned_dataset)
    final_dataset = drop_columns(processed_dataset)

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = stratified_split(final_dataset)

    X_train_lr_nn, X_test_lr_nn = one_hot_encoding(X_train, X_test, categorical_cols, numeric_cols)
    X_train_rf_xg, X_test_rf_xg = label_encoding(X_train, X_test, categorical_cols)

    X_train_lr_nn.to_parquet(X_TRAIN_LR_NN)
    X_test_lr_nn.to_parquet(X_TEST_LR_NN)
    X_train_rf_xg.to_parquet(X_TRAIN_RF_XG)
    X_test_rf_xg.to_parquet(X_TEST_RF_XG)

    # Save all datasets
    X_train.to_parquet(X_TRAIN_STRAT_PATH)
    X_test.to_parquet(X_TEST_STRAT_PATH)
    y_train.to_frame(name="target").to_parquet(Y_TRAIN_STRAT_PATH)
    y_test.to_frame(name="target").to_parquet(Y_TEST_STRAT_PATH)

    # Upload raw dataset to W&B artifact
    split_artifact = wandb.Artifact(
        name="bank_marketing_dataset", 
        type="dataset",
        description="Datasets after new improvements"
    )
    split_artifact.add_file(X_TRAIN_STRAT_PATH)
    split_artifact.add_file(X_TEST_STRAT_PATH)
    split_artifact.add_file(Y_TRAIN_STRAT_PATH)
    split_artifact.add_file(Y_TEST_STRAT_PATH)
    split_artifact.add_file(X_TRAIN_LR_NN)
    split_artifact.add_file(X_TEST_LR_NN)
    split_artifact.add_file(X_TRAIN_RF_XG)
    split_artifact.add_file(X_TEST_RF_XG)


    run.log_artifact(split_artifact)
    wandb.finish()


