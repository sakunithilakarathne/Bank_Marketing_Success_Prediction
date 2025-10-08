import os
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
from src.utils.data_loader import load_data_ohe
from src.utils.metrics import evaluate_and_log
from config import IMBALANCE_LOGISTIC_BASELINE, CLASS_IMBALANCE_MODELS

def imbalance_logistic_regression_model():
    os.makedirs(CLASS_IMBALANCE_MODELS, exist_ok=True)
    
    run = wandb.init(project="Bank_Marketing_Classifier", job_type="imbalance", name="logistic_regression_imbalance")

    # Load data
    X, y = load_data_ohe(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Including SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"Before SMOTE: {Counter(y_train)} | After SMOTE: {Counter(y_res)}")

    params = {"solver": "liblinear", "class_weight": "balanced", "max_iter": 1000}
    run.config.update(params)

    model = LogisticRegression(**params)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_and_log("Logistic Regression (SMOTE)", y_val, y_pred, y_prob, class_labels=["no", "yes"])

    joblib.dump(model, IMBALANCE_LOGISTIC_BASELINE)
    model_artifact = wandb.Artifact("logistic_regression_model", type="model", description="Logistic Regression with SMOTE + Balanced Class Weights")
    model_artifact.add_file(IMBALANCE_LOGISTIC_BASELINE)
    run.log_artifact(model_artifact)
    run.finish()
