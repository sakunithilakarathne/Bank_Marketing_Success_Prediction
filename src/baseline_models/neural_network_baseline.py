import wandb
from src.utils.data_loader import load_data
from src.utils.metrics import evaluate_and_log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
from config import ANN_BASELINE, BASELINE_MODELS


def baseline_neural_network():
    os.makedirs(BASELINE_MODELS, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="baseline", name="neural_network_baseline")

    X, y = load_data(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Define model
    input_dim = X_train_t.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    wandb.watch(model, log="all")

    # Training loop
    for epoch in range(10):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        wandb.log({"epoch": epoch+1, "loss": loss.item()})

    torch.save(model.state_dict(), ANN_BASELINE)
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_prob = model(X_val_t).numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)

    metrics = evaluate_and_log("Neural Network", y_val, y_pred, y_prob, class_labels=["no", "yes"])
    
    model_artifact = wandb.Artifact("ann_model", type="model", description="Baseline Neural Network Model")
    model_artifact.add_file(ANN_BASELINE)
    run.log_artifact(model_artifact)

    run.finish()