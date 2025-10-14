import os
import wandb
from src.utils.data_loader import load_data_lr_nn, load_data_rf_xg
from src.utils.metrics import evaluate_and_log
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from config import MODEL_IMPROVEMENT_DIR, IMP_BASELINE_LR_MODEL, IMP_BASELINE_RF_MODEL, IMP_BASELINE_XG_MODEL, IMP_BASELINE_NN_MODEL


# New Baseline Logistic Regression Model

def improved_lr_baseline():
    os.makedirs(MODEL_IMPROVEMENT_DIR, exist_ok=True)

    # Initialize W&B run
    run = wandb.init(project="Bank_Marketing_Classifier", job_type="improved_baseline", name="improved_logistic_regression_baseline")

    # Load data
    X, y = load_data_lr_nn(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v3')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Model
    params = {"solver": "liblinear", "class_weight": "balanced", "max_iter": 1000}
    model = LogisticRegression(**params)
    run.config.update(params)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Evaluate
    metrics = evaluate_and_log("Logistic Regression", y_val, y_pred, y_prob, class_labels=["no", "yes"])

    joblib.dump(model, IMP_BASELINE_LR_MODEL)

    model_artifact = wandb.Artifact("logistic_regression_model", type="model", description="Improved Baseline Logistic Regression Model")
    model_artifact.add_file(IMP_BASELINE_LR_MODEL)
    run.log_artifact(model_artifact)

    run.finish()


# New Basline Random Forest Model
def baseline_random_forest():
    os.makedirs(MODEL_IMPROVEMENT_DIR, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="improved_baseline", name="improved_random_forest_baseline")

    X, y = load_data_rf_xg(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v3')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    params = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced", "n_jobs": -1}
    run.config.update(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_and_log("Random Forest", y_val, y_pred, y_prob, class_labels=["no", "yes"])
    
    joblib.dump(model, IMP_BASELINE_RF_MODEL)

    model_artifact = wandb.Artifact("random_forest_model", type="model", description="Improved Baseline Random Forest Model")
    model_artifact.add_file(IMP_BASELINE_RF_MODEL)
    run.log_artifact(model_artifact)

    run.finish()



# New Baseline XGBoost Model
def baseline_xgboost_model():
    os.makedirs(MODEL_IMPROVEMENT_DIR, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="improved_baseline", name="improved_xgboost_baseline")

    X, y = load_data_rf_xg(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v3')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    params = {
        "learning_rate": 0.1,
        "n_estimators": 200,
        "max_depth": 5,
        "random_state": 42,
        "scale_pos_weight": (y_train.value_counts()[0] / y_train.value_counts()[1]),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    run.config.update(params)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_and_log("Improved XGBoost", y_val, y_pred, y_prob, class_labels=["no", "yes"])
    joblib.dump(model, IMP_BASELINE_XG_MODEL)

    model_artifact = wandb.Artifact("xgboost_model", type="model", description="Improved Baseline XGBoost Model")
    model_artifact.add_file(IMP_BASELINE_XG_MODEL)
    run.log_artifact(model_artifact)

    run.finish()



# New Basline Neural Network Model
def baseline_neural_network():
    os.makedirs(MODEL_IMPROVEMENT_DIR, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="ímproved_baseline", name="improved_nn_baseline")

    X, y = load_data_lr_nn(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v3')
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

    torch.save(model.state_dict(), IMP_BASELINE_NN_MODEL)
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_prob = model(X_val_t).numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)

    metrics = evaluate_and_log("Improved Baseline Neural Network", y_val, y_pred, y_prob, class_labels=["no", "yes"])
    
    model_artifact = wandb.Artifact("ann_model", type="model", description="Improved Baseline Neural Network Model")
    model_artifact.add_file(IMP_BASELINE_NN_MODEL)
    run.log_artifact(model_artifact)

    run.finish()