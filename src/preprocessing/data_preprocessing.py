import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import wandb

from config import (RAW_DATASET_PATH, BANK_MARKETING_DATASET, PROCESSED_DATASET_PATH,
                    X_TEST_PATH, X_TRAIN_PATH, Y_TEST_PATH, Y_TRAIN_PATH,
                    X_TRAIN_OHE, X_TEST_OHE, X_TRAIN_LE, X_TEST_LE)


def load_dataset():

    run = wandb.init(project="Bank_Marketing_Classifier", name="loading_raw_dataset")

    raw_dataset = pd.read_csv(BANK_MARKETING_DATASET, sep=';', quotechar='"')
    raw_dataset.to_csv(RAW_DATASET_PATH)

     # Upload raw dataset to W&B artifact
    raw_dataset_artifact = wandb.Artifact(
        name="bank_marketing_dataset", 
        type="dataset",
        description="Raw bank marketing dataset"
    )
    raw_dataset_artifact.add_file(RAW_DATASET_PATH)
    run.log_artifact(raw_dataset_artifact)
    wandb.finish()
    
    return raw_dataset



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
    dataset = dataset.drop(columns=['previous'], errors='ignore')

    macro_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    dataset = dataset.drop(columns=macro_features, errors='ignore')

    return dataset



def split_datasets(dataset, target_col='y'):
    
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col].apply(lambda x: 1 if x == 'yes' else 0)

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def data_scaling(X_train, X_test,numeric_cols):
    # Scale numerical columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test


def one_hot_encoding(X_train, X_test, categorical_cols):

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




def preprocessing_pipeling():

    run = wandb.init(project="Bank_Marketing_Classifier", name="splitting_data")

    # Get raw dataset
    # raw_dataset = pd.read_csv(RAW_DATASET_PATH)
    # cleaned_dataset = outlier_handling(raw_dataset)
    # processed_dataset = feature_engineering(cleaned_dataset)
    # final_dataset = drop_columns(processed_dataset)
    # final_dataset.to_csv(PROCESSED_DATASET_PATH)

    final_dataset = pd.read_csv(PROCESSED_DATASET_PATH)

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = split_datasets(final_dataset)
    X_train_scaled, X_test_scaled = data_scaling(X_train, X_test,numeric_cols)

    # Save all datasets
    X_train_scaled.to_parquet(X_TRAIN_PATH)
    X_test_scaled.to_parquet(X_TEST_PATH)
    y_train.to_frame(name="target").to_parquet(Y_TRAIN_PATH)
    y_test.to_frame(name="target").to_parquet(Y_TEST_PATH)

    X_train_ohe, X_test_ohe = one_hot_encoding(X_train_scaled, X_test_scaled, categorical_cols)
    X_train_le, X_test_le = label_encoding(X_train_scaled, X_test_scaled, categorical_cols)

    X_train_ohe.to_parquet(X_TRAIN_OHE)
    X_test_ohe.to_parquet(X_TEST_OHE)
    X_train_le.to_parquet(X_TRAIN_LE)
    X_test_le.to_parquet(X_TEST_LE)

    

    # Upload raw dataset to W&B artifact
    split_artifact = wandb.Artifact(
        name="bank_marketing_dataset", 
        type="dataset",
        description="Datasets after train test split"
    )
    split_artifact.add_file(X_TRAIN_PATH)
    split_artifact.add_file(X_TEST_PATH)
    split_artifact.add_file(Y_TRAIN_PATH)
    split_artifact.add_file(Y_TEST_PATH)
    split_artifact.add_file(X_TRAIN_OHE)
    split_artifact.add_file(X_TEST_OHE)
    split_artifact.add_file(X_TRAIN_LE)
    split_artifact.add_file(X_TEST_LE)


    run.log_artifact(split_artifact)
    wandb.finish()

    #print("Preprocessing Completed")


# datasets = {
#         "linear": (X_train_lin, X_test_lin, y_train, y_test), -> ohe
#         "neural": (X_train_lin, X_test_lin, y_train, y_test),
#         "tree": (X_train_tree, X_test_tree, y_train, y_test), -> le
#         "boost": (X_train_tree, X_test_tree, y_train, y_test)
#     }
