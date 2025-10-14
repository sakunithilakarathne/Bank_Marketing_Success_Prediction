import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

def evaluate_and_log(model_name, y_true, y_pred, y_prob, class_labels):
    # ---- Classification report and metrics ----
    if y_true.dtype == 'O':
        y_true = y_true.map({'no': 0, 'yes': 1})
        y_pred = pd.Series(y_pred).map({'no': 0, 'yes': 1}) if isinstance(y_pred[0], str) else y_pred

    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # ---- Class-level metrics ----
    class_metrics = {}
    metrics_for_plot = {"Class": [], "Metric": [], "Score": []}

    for cls in class_labels:
        precision = report[cls]["precision"]
        recall = report[cls]["recall"]
        f1 = report[cls]["f1-score"]
        class_metrics[f"{cls}_precision"] = precision
        class_metrics[f"{cls}_recall"] = recall
        class_metrics[f"{cls}_f1"] = f1

        # Add for plotting
        metrics_for_plot["Class"].extend([cls, cls, cls])
        metrics_for_plot["Metric"].extend(["Precision", "Recall", "F1-Score"])
        metrics_for_plot["Score"].extend([precision, recall, f1])

    # ---- Macro metrics ----
    macro_metrics = {
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    # ---- Log metrics ----
    wandb.log({**class_metrics, **macro_metrics})

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curve")
    wandb.log({"roc_curve": wandb.Image(fig)})
    plt.close(fig)

    # ---- Precision-Recall Curve ----
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} Precision-Recall Curve")
    wandb.log({"pr_curve": wandb.Image(fig)})
    plt.close(fig)

    # ---- Class-level metric comparison bar plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x="Class", y="Score", hue="Metric", data=metrics_for_plot, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} - Class-level Precision, Recall & F1")
    ax.legend(title="Metric")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2, fontsize=9)
    plt.tight_layout()
    wandb.log({"class_level_metrics_barplot": wandb.Image(fig)})
    plt.close(fig)

    return {
        "class_metrics": class_metrics,
        "macro_metrics": macro_metrics
    }
