import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.utils.data_loader import load_data_ohe
from sklearn.metrics import make_scorer, average_precision_score, f1_score
import joblib
from config import LOGREG_PARAMETERS

def logsitic_regression_tuning():

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="hp_tuning", name="logreg_tuning")

    X, y = load_data_ohe(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear","saga"],
        "class_weight": ["balanced", None]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(max_iter=1000)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='average_precision', verbose=2, n_jobs=-1)
    grid.fit(X, y)

    run.log({"best_params": grid.best_params_, "best_score": grid.best_score_})
    joblib.dump(grid.best_params_, LOGREG_PARAMETERS)
    
    model_artifact = wandb.Artifact("logistic_regression_parameters", type="hyperparamters", description="Logistic Regression HP")
    model_artifact.add_file(LOGREG_PARAMETERS)
    run.log_artifact(model_artifact)


    run.finish()