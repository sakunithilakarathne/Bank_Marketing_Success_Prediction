import wandb
import pandas as pd

def load_data(run, artifact_name: str):
    
    #Load X_train and y_train parquet files from a W&B artifact.
    artifact = run.use_artifact(artifact_name, type="dataset")
    artifact_dir = artifact.download()

    X_train = pd.read_parquet(f"{artifact_dir}/X_train_ohe.parquet")
    y_train = pd.read_parquet(f"{artifact_dir}/y_train.parquet")

    return X_train, y_train.squeeze()  # ensure y_train is a Series
