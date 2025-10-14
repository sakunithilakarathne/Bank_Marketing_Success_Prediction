import wandb
import pandas as pd

def load_data_ohe(run, artifact_name: str):
    
    #Load X_train and y_train parquet files from a W&B artifact.
    artifact = run.use_artifact(artifact_name, type="dataset")
    artifact_dir = artifact.download()

    X_train = pd.read_parquet(f"{artifact_dir}/X_train_ohe.parquet")
    y_train = pd.read_parquet(f"{artifact_dir}/y_train.parquet")

    return X_train, y_train.squeeze()  # ensure y_train is a Series


def load_data_le(run, artifact_name: str):
    
    #Load X_train and y_train parquet files from a W&B artifact.
    artifact = run.use_artifact(artifact_name, type="dataset")
    artifact_dir = artifact.download()

    X_train = pd.read_parquet(f"{artifact_dir}/X_train_le.parquet")
    y_train = pd.read_parquet(f"{artifact_dir}/y_train.parquet")

    return X_train, y_train.squeeze()  # ensure y_train is a Series


def load_data_lr_nn(run, artifact_name: str):
    
    #Load X_train and y_train parquet files from a W&B artifact.
    artifact = run.use_artifact(artifact_name, type="dataset")
    artifact_dir = artifact.download()

    X_train = pd.read_parquet(f"{artifact_dir}/X_train_lr_nn.parquet")
    y_train = pd.read_parquet(f"{artifact_dir}/y_train_strat.parquet")

    return X_train, y_train.squeeze()  # ensure y_train is a Series


def load_data_rf_xg(run, artifact_name: str):
    
    #Load X_train and y_train parquet files from a W&B artifact.
    artifact = run.use_artifact(artifact_name, type="dataset")
    artifact_dir = artifact.download()

    X_train = pd.read_parquet(f"{artifact_dir}/X_train_rf_xg.parquet")
    y_train = pd.read_parquet(f"{artifact_dir}/y_train_strat.parquet")

    return X_train, y_train.squeeze()  # ensure y_train is a Series

