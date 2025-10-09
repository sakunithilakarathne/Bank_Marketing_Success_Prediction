from config import LOGREG2_PARAMETERS, LOGREG2_MODEL
import wandb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, average_precision_score, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from src.utils.data_loader import load_data_ohe



def logistic_regression_tuning():

    run = wandb.init(project="Bank_Marketing_Classifier", 
                     job_type="hp_tuning", 
                     name="logreg_tuning_resampled")

    # Load preprocessed data
    X, y = load_data_ohe(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')

    # Expanded hyperparameter grid
    param_grid = {
        "logreg__C": [0.001, 0.01, 0.1, 1, 10],
        "logreg__penalty": ["l1", "l2"],
        "logreg__solver": ["liblinear", "saga"],
        "logreg__class_weight": ["balanced"]
    }

    # Define pipeline with SMOTE + LogisticRegression
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("logreg", LogisticRegression(max_iter=2000))
    ])

    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use Average Precision as primary scoring
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='average_precision',
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )

    # Fit grid
    grid.fit(X, y)

    # Log best hyperparameters
    best_params = grid.best_params_
    wandb.log({"best_params": best_params, "best_score": grid.best_score_})
    joblib.dump(best_params, LOGREG2_PARAMETERS)

    # Log as W&B artifacts
    params_artifact = wandb.Artifact("logreg_hyperparameters", type="hyperparameters",
                                     description="Best Logistic Regression hyperparameters")
    params_artifact.add_file(LOGREG2_PARAMETERS)
    run.log_artifact(params_artifact)


    run.finish()
