import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.utils.data_loader import load_data_ohe
from sklearn.metrics import average_precision_score
import joblib
from config import NEURAL_NET_PARAMETERS


def neural_network_tuning():

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="hp_tuning", name="nn_tuning")

    X, y = load_data_ohe(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    param_grid = {
        "hidden_size": [32, 64],
        "dropout": [0.2, 0.3],
        "lr": [1e-3, 1e-4],
        "epochs": [20],
        "batch_size": [64]
    }

    def build_model(input_dim, hidden_size, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score, best_params = 0, None

    for hidden_size in param_grid["hidden_size"]:
        for dropout in param_grid["dropout"]:
            for lr in param_grid["lr"]:
                fold_scores = []
                for train_idx, val_idx in cv.split(X, y):
                    model = build_model(X.shape[1], hidden_size, dropout)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.BCELoss()

                    X_train = X_t[train_idx]
                    y_train = y_t[train_idx]
                    X_val = X_t[val_idx]
                    y_val = y_t[val_idx]

                    train_ds = TensorDataset(X_train, y_train)
                    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

                    # Train
                    for epoch in range(20):
                        model.train()
                        for xb, yb in train_dl:
                            optimizer.zero_grad()
                            loss = criterion(model(xb), yb)
                            loss.backward()
                            optimizer.step()

                    # Validate
                    model.eval()
                    with torch.no_grad():
                        y_prob = model(X_val).numpy().flatten()
                    score = average_precision_score(y_val, y_prob)
                    fold_scores.append(score)

                avg_score = np.mean(fold_scores)
                wandb.log({"hidden_size": hidden_size, "dropout": dropout, "lr": lr, "pr_auc": avg_score})
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {"hidden_size": hidden_size, "dropout": dropout, "lr": lr}

    joblib.dump(best_params, NEURAL_NET_PARAMETERS)

    wandb.log({"best_params": best_params, "best_score": best_score})

    model_artifact = wandb.Artifact("ann_model", type="hyperparamters", description="Nueral Network HP")
    model_artifact.add_file(NEURAL_NET_PARAMETERS)
    run.log_artifact(model_artifact)
    

    run.finish()
