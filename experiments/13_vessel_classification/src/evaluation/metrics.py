"""
Evaluation metrics for vessel classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Optional, List
import json


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        labels: True labels
        predictions: Predicted labels
        num_classes: Number of classes
        probs: Predicted probabilities (optional)

    Returns:
        metrics: Dictionary of metric values
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
    }

    return metrics


def compute_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names

    Returns:
        per_class: Dictionary mapping class name to metrics
    """
    per_class = {}

    for idx, name in enumerate(class_names):
        # Binary classification: class vs rest
        binary_labels = (labels == idx).astype(int)
        binary_preds = (predictions == idx).astype(int)

        per_class[name] = {
            'precision': precision_score(binary_labels, binary_preds, zero_division=0),
            'recall': recall_score(binary_labels, binary_preds, zero_division=0),
            'f1': f1_score(binary_labels, binary_preds, zero_division=0),
            'support': int((labels == idx).sum()),
        }

    return per_class


def compute_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute confusion matrix.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names

    Returns:
        result: Dictionary with confusion matrix and class names
    """
    cm = confusion_matrix(labels, predictions)

    return {
        'matrix': cm.tolist(),
        'class_names': class_names,
        'normalized': (cm / cm.sum(axis=1, keepdims=True)).tolist(),
    }


def format_results(
    metrics: Dict[str, float],
    per_class: Dict[str, Dict[str, float]],
    confusion: Dict,
    condition: str,
) -> str:
    """Format results as a readable string."""
    lines = [
        f"\n{'='*60}",
        f"Results for {condition}",
        f"{'='*60}",
        "",
        "Overall Metrics:",
        f"  Accuracy:        {metrics['accuracy']:.4f}",
        f"  F1 (macro):      {metrics['f1_macro']:.4f}",
        f"  F1 (weighted):   {metrics['f1_weighted']:.4f}",
        f"  Precision:       {metrics['precision_macro']:.4f}",
        f"  Recall:          {metrics['recall_macro']:.4f}",
        "",
        "Per-Class Metrics:",
    ]

    for name, class_metrics in per_class.items():
        lines.append(f"  {name}:")
        lines.append(f"    Precision: {class_metrics['precision']:.4f}")
        lines.append(f"    Recall:    {class_metrics['recall']:.4f}")
        lines.append(f"    F1:        {class_metrics['f1']:.4f}")
        lines.append(f"    Support:   {class_metrics['support']}")

    lines.extend([
        "",
        "Confusion Matrix:",
        "  " + "  ".join(f"{n:>10}" for n in confusion['class_names']),
    ])

    for i, row in enumerate(confusion['matrix']):
        name = confusion['class_names'][i]
        lines.append(f"  {name:>10}: " + "  ".join(f"{v:>10}" for v in row))

    return "\n".join(lines)


def save_results(
    results: Dict,
    output_path: str,
):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
