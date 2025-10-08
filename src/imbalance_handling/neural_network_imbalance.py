import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.data_loader import load_data_ohe
from src.utils.metrics import evaluate_and_log
from config import IMBALANCE_ANN_BASELINE, CLASS_IMBALANCE_MODELS


def imbalance_neural_network():
    os.makedirs(CLASS_IMBALANCE_MODELS, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="imbalance", name="neural_network_imbalance")

    X, y = load_data_ohe(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ---- Compute pos_weight ----
    neg, pos = y_train.value_counts()
    pos_weight = torch.tensor([neg / pos])
    print(f"pos_weight = {pos_weight.item():.2f}")

    # ---- Weighted sampler ----
    class_weights = 1. / y_train.value_counts()
    sample_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # ---- Prepare data ----
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler)

    input_dim = X_train_t.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    wandb.watch(model, log="all")

    for epoch in range(15):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        wandb.log({"epoch": epoch + 1, "loss": loss.item()})

    torch.save(model.state_dict(), IMBALANCE_ANN_BASELINE)

    model.eval()
    with torch.no_grad():
        y_logits = model(X_val_t).numpy().flatten()
        y_prob = 1 / (1 + np.exp(-y_logits))  # Sigmoid manually
        y_pred = (y_prob > 0.5).astype(int)

    metrics = evaluate_and_log("Neural Network (Imbalance)", y_val, y_pred, y_prob, class_labels=["no", "yes"])

    model_artifact = wandb.Artifact("ann_model", type="model", description="Neural Network with BCEWithLogitsLoss and Weighted Sampling")
    model_artifact.add_file(IMBALANCE_ANN_BASELINE)
    run.log_artifact(model_artifact)
    run.finish()
