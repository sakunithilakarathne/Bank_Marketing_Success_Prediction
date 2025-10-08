import os
import wandb
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from src.utils.data_loader import load_data_le
from src.utils.metrics import evaluate_and_log
from config import IMBALANCE_XGBOOST_BASELINE, CLASS_IMBALANCE_MODELS


def imbalance_xgboost_model():

    os.makedirs(CLASS_IMBALANCE_MODELS, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="imbalance", name="xgboost_imbalance")

    X, y = load_data_le(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    X_res, y_res = SMOTE().fit_resample(X_train, y_train)
    print(f"Before SMOTE: {Counter(y_train)} | After SMOTE: {Counter(y_res)}")

    # ---- Compute imbalance ratio ----
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    params = {
        "learning_rate": 0.1,
        "n_estimators": 300,
        "max_depth": 6,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    run.config.update(params)

    model = XGBClassifier(**params)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_and_log("XGBoost (Imbalance)", y_val, y_pred, y_prob, class_labels=["no", "yes"])

    joblib.dump(model, IMBALANCE_XGBOOST_BASELINE)
    model_artifact = wandb.Artifact("xgboost_model", type="model", description="XGBoost with scale_pos_weight")
    model_artifact.add_file(IMBALANCE_XGBOOST_BASELINE)
    run.log_artifact(model_artifact)
    run.finish()
