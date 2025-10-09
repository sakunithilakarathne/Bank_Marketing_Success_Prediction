import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.utils.data_loader import load_data_le
from sklearn.metrics import make_scorer, average_precision_score
import joblib
from config import RANDOM_FOREST_PARAMETERS


def random_forest_tuning():

    run = wandb.init(project="Bank_Marketing_Classifier", job_type="hp_tuning", name="random_forest_tuning")

    X, y = load_data_le(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="average_precision", n_jobs=-1, verbose=2)

    grid.fit(X, y)
    run.log({"best_params": grid.best_params_, "best_score": grid.best_score_})
    joblib.dump(grid.best_params_, RANDOM_FOREST_PARAMETERS)

    model_artifact = wandb.Artifact("random_forest_model", type="hyperparamters", description="Random Forest HP")
    model_artifact.add_file(RANDOM_FOREST_PARAMETERS)
    run.log_artifact(model_artifact)

    run.finish()