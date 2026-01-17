"""
Evaluation metrics for anomaly detection.

Includes AUROC, AUPRC, F1, and threshold-dependent metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from typing import Dict, Tuple


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Threshold-independent metrics
    metrics['auroc'] = compute_auroc(y_true, y_prob)
    metrics['auprc'] = compute_auprc(y_true, y_prob)

    # Find optimal threshold for F1
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, y_prob)
    metrics['f1_optimal'] = optimal_f1
    metrics['optimal_threshold'] = optimal_threshold

    # Threshold-dependent metrics at optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

    # Precision at fixed recall levels
    metrics['precision_at_recall_90'] = precision_at_recall(y_true, y_prob, 0.90)
    metrics['precision_at_recall_80'] = precision_at_recall(y_true, y_prob, 0.80)

    # Recall at fixed precision levels
    metrics['recall_at_precision_90'] = recall_at_precision(y_true, y_prob, 0.90)
    metrics['recall_at_precision_80'] = recall_at_precision(y_true, y_prob, 0.80)

    return metrics


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    if len(np.unique(y_true)) < 2:
        return 0.5  # Undefined for single-class
    return roc_auc_score(y_true, y_prob)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    if len(np.unique(y_true)) < 2:
        return np.mean(y_true)  # Baseline for single-class
    return average_precision_score(y_true, y_prob)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        (optimal_threshold, optimal_metric_value)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Compute F1 at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float
) -> float:
    """
    Find precision at a given recall level.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_recall: Target recall level (e.g., 0.90)

    Returns:
        Precision at the target recall
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # Find the highest precision where recall >= target
    valid_indices = recall >= target_recall
    if not np.any(valid_indices):
        return 0.0

    return float(np.max(precision[valid_indices]))


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float
) -> float:
    """
    Find recall at a given precision level.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        target_precision: Target precision level (e.g., 0.90)

    Returns:
        Recall at the target precision
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # Find the highest recall where precision >= target
    valid_indices = precision >= target_precision
    if not np.any(valid_indices):
        return 0.0

    return float(np.max(recall[valid_indices]))


def get_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Get confusion matrix at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def get_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def get_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Precision-Recall curve data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds
