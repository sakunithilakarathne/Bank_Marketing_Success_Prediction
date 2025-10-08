import wandb
from src.utils.data_loader import load_data_ohe
from src.utils.metrics import evaluate_and_log
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from config import (LOGISTIC_BASELINE)

def baseline_logistic_regression_model():

    # Initialize W&B run
    run = wandb.init(project="Bank_Marketing_Classifier", job_type="baseline", name="logistic_regression_baseline")

    # Load data
    X, y = load_data_ohe(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
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

    joblib.dump(model, LOGISTIC_BASELINE)

    model_artifact = wandb.Artifact("logistic_regression_model", type="model", description="Baseline Logistic Regression Model")
    model_artifact.add_file(LOGISTIC_BASELINE)
    run.log_artifact(model_artifact)

    run.finish()
