import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# -------------------------
# Binary classification
# -------------------------

def binary_classification_metrics(
    y_true,
    y_prob,
    threshold=0.5
):
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        Ground-truth labels (0 or 1)
    y_prob : array-like, shape (N,)
        Predicted probabilities for the positive class
    threshold : float
        Threshold to convert probabilities to class labels

    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # probability-based metrics
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # recall
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    })

    return metrics


# -------------------------
# Multi-class classification
# -------------------------

def multiclass_classification_metrics(
    y_true,
    y_prob,
    average="macro"
):
    """
    Compute metrics for multi-class classification.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        Ground-truth class labels
    y_prob : array-like, shape (N, C)
        Predicted class probabilities
    average : str
        Averaging strategy for precision/recall/F1

    Returns
    -------
    metrics : dict
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics