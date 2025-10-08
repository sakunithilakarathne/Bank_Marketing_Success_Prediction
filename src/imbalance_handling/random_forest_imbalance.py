import os
import wandb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from src.utils.data_loader import load_data_le
from src.utils.metrics import evaluate_and_log
from config import IMBALANCE_RANDOM_FOREST_BASELINE, CLASS_IMBALANCE_MODELS



def imbalance_random_forest():

    os.makedirs(CLASS_IMBALANCE_MODELS, exist_ok=True)

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="imbalance", name="random_forest_imbalance")

    X, y = load_data_le(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    X_res, y_res = SMOTE().fit_resample(X_train, y_train)
    print(f"Before SMOTE: {Counter(y_train)} | After SMOTE: {Counter(y_res)}")

    params = {
        "n_estimators": 200,
        "random_state": 42,
        "class_weight": "balanced_subsample",
        "max_depth": None,
        "n_jobs": -1
    }
    run.config.update(params)

    model = RandomForestClassifier(**params)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_and_log("Random Forest (Balanced Subsample)", y_val, y_pred, y_prob, class_labels=["no", "yes"])

    joblib.dump(model, IMBALANCE_RANDOM_FOREST_BASELINE)
    model_artifact = wandb.Artifact("random_forest_model", type="model", description="Random Forest with balanced_subsample class weight")
    model_artifact.add_file(IMBALANCE_RANDOM_FOREST_BASELINE)
    run.log_artifact(model_artifact)
    run.finish()