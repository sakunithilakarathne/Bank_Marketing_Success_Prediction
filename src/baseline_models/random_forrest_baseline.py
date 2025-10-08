import wandb
from utils.data_loader import load_data
from utils.metrics import evaluate_and_log
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def baseline_random_forrest():
    run = wandb.init(project="Bank_Marketing_Classifier", job_type="baseline", name="random_forest_baseline")

    X, y = load_data('scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    params = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced", "n_jobs": -1}
    wandb.config.update(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = evaluate_and_log("Random Forest", y_val, y_pred, y_prob, class_labels=["no", "yes"])
    wandb.sklearn.log_model(model, "random_forest", X_train, X_val, y_train, y_val)

    run.finish()
