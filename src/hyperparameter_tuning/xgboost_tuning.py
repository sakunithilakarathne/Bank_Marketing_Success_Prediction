import wandb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.utils.data_loader import load_data_le
from sklearn.metrics import make_scorer, average_precision_score
import joblib
from config import XGBOOST_PARAMETERS



def xgboost_tuning():


    run = wandb.init(project="Bank_Marketing_Classifier", job_type="hp_tuning", name="xgboost_tuning")

    X, y = load_data_le(run, 'scsthilakarathne-nibm/Bank_Marketing_Classifier/bank_marketing_dataset:v2')

    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
        "scale_pos_weight": [scale_pos_weight],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")

    grid = GridSearchCV(model, param_grid, cv=cv, scoring="average_precision", n_jobs=-1, verbose=2)
    grid.fit(X, y)

    run.log({"best_params": grid.best_params_, "best_score": grid.best_score_})
    joblib.dump(grid.best_params_, XGBOOST_PARAMETERS)

    model_artifact = wandb.Artifact("xgboost_parameters", type="hyperparamters", description="XGBoost HP")
    model_artifact.add_file(XGBOOST_PARAMETERS)
    run.log_artifact(model_artifact)


    run.finish()
