import wandb
from src.utils.data_loader import load_data_le
from src.utils.metrics import evaluate_and_log
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from config import XGBOOST_BASELINE


def baseline_xgboost_model():

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="baseline", name="xgboost_baseline")

    X, y = load_data_le(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
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

    metrics = evaluate_and_log("XGBoost", y_val, y_pred, y_prob, class_labels=["no", "yes"])
    joblib.dump(model, XGBOOST_BASELINE)

    model_artifact = wandb.Artifact("xgboost_model", type="model", description="Baseline XGBoost Model")
    model_artifact.add_file(XGBOOST_BASELINE)
    run.log_artifact(model_artifact)

    run.finish()
