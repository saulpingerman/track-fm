"""
Cross-validation split generation for nested CV.

Uses stratified splitting to maintain class balance across folds.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List
import json
from pathlib import Path


def create_nested_cv_splits(
    labels: np.ndarray,
    outer_folds: int = 5,
    inner_folds: int = 4,
    random_state: int = 42
) -> Dict:
    """
    Create nested cross-validation splits.

    Outer loop: Test set selection
    Inner loop: Validation set selection (for hyperparameter tuning)

    Args:
        labels: Binary labels (0=normal, 1=anomaly)
        outer_folds: Number of outer folds
        inner_folds: Number of inner folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with split indices
    """
    n_samples = len(labels)
    indices = np.arange(n_samples)

    splits = {}

    # Outer CV
    outer_cv = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_state
    )

    for outer_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(indices, labels)):
        splits[f'outer_fold_{outer_idx}'] = {
            'test_indices': test_indices.tolist(),
            'train_val_indices': train_val_indices.tolist()
        }

        # Inner CV on train_val set
        train_val_labels = labels[train_val_indices]

        if inner_folds == 1:
            # No inner CV - use simple 80/20 stratified split for train/val
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_indices)),
                test_size=0.2,
                stratify=train_val_labels,
                random_state=random_state + outer_idx + 1
            )
            train_indices = train_val_indices[train_idx]
            val_indices = train_val_indices[val_idx]

            splits[f'outer_fold_{outer_idx}']['inner_fold_0'] = {
                'train_indices': train_indices.tolist(),
                'val_indices': val_indices.tolist()
            }
        else:
            inner_cv = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=random_state + outer_idx + 1
            )

            for inner_idx, (train_idx, val_idx) in enumerate(inner_cv.split(train_val_indices, train_val_labels)):
                # Map back to original indices
                train_indices = train_val_indices[train_idx]
                val_indices = train_val_indices[val_idx]

                splits[f'outer_fold_{outer_idx}'][f'inner_fold_{inner_idx}'] = {
                    'train_indices': train_indices.tolist(),
                    'val_indices': val_indices.tolist()
                }

    # Add metadata
    splits['metadata'] = {
        'n_samples': n_samples,
        'n_positive': int(labels.sum()),
        'n_negative': int(len(labels) - labels.sum()),
        'outer_folds': outer_folds,
        'inner_folds': inner_folds,
        'random_state': random_state
    }

    return splits


def save_splits(splits: Dict, output_path: str):
    """Save splits to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits to {output_path}")


def load_splits(splits_path: str) -> Dict:
    """Load splits from JSON file."""
    with open(splits_path, 'r') as f:
        return json.load(f)


def get_fold_data(
    splits: Dict,
    outer_fold: int,
    inner_fold: int = None
) -> Dict[str, np.ndarray]:
    """
    Get indices for a specific fold.

    Args:
        splits: Splits dictionary
        outer_fold: Outer fold index
        inner_fold: Inner fold index (None for just test split)

    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    outer_key = f'outer_fold_{outer_fold}'

    test_indices = np.array(splits[outer_key]['test_indices'])

    if inner_fold is not None:
        inner_key = f'inner_fold_{inner_fold}'
        train_indices = np.array(splits[outer_key][inner_key]['train_indices'])
        val_indices = np.array(splits[outer_key][inner_key]['val_indices'])
    else:
        # Use all train_val as training (no validation)
        train_indices = np.array(splits[outer_key]['train_val_indices'])
        val_indices = np.array([])

    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }


def print_split_summary(splits: Dict):
    """Print summary of splits."""
    meta = splits['metadata']

    print(f"\nDataset: {meta['n_samples']} samples")
    print(f"  Positive (anomaly): {meta['n_positive']} ({100*meta['n_positive']/meta['n_samples']:.1f}%)")
    print(f"  Negative (normal): {meta['n_negative']} ({100*meta['n_negative']/meta['n_samples']:.1f}%)")
    print(f"\nNested CV: {meta['outer_folds']} outer × {meta['inner_folds']} inner folds")

    # Show first outer fold as example
    outer_key = 'outer_fold_0'
    test_size = len(splits[outer_key]['test_indices'])
    train_val_size = len(splits[outer_key]['train_val_indices'])

    inner_key = 'inner_fold_0'
    train_size = len(splits[outer_key][inner_key]['train_indices'])
    val_size = len(splits[outer_key][inner_key]['val_indices'])

    print(f"\nExample split (outer=0, inner=0):")
    print(f"  Train: {train_size} ({100*train_size/meta['n_samples']:.1f}%)")
    print(f"  Val: {val_size} ({100*val_size/meta['n_samples']:.1f}%)")
    print(f"  Test: {test_size} ({100*test_size/meta['n_samples']:.1f}%)")
