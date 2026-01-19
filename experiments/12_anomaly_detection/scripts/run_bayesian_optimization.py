#!/usr/bin/env python3
"""
Bayesian Optimization for Anomaly Detection Hyperparameters.

Uses Facebook's Ax library to optimize:
- learning_rate
- weight_decay
- beta1, beta2 (AdamW)
- pooling (mean vs last)

For both random_init and pretrained conditions.

Note: This dataset is small (521 trajectories, 25 anomalies), so results
may have higher variance. We use 5-fold CV within each trial for more
robust estimates.
"""

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.logger import get_logger

# Suppress warnings
warnings.filterwarnings('ignore')
get_logger("ax").setLevel("WARNING")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dtu_dataset import DTUDataset
from data.preprocessing import load_processed_data
from data.cv_splits import load_splits
from data.augmentation import TrajectoryAugmentor
from models.model_factory import create_model as factory_create_model


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    features = torch.stack([item['features'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'features': features,
        'length': lengths,
        'label': labels
    }


def create_model_with_pooling(condition: str, config: dict, pooling: str, device: str = "cuda"):
    """Create model with specified pooling strategy."""
    trial_config = copy.deepcopy(config)
    trial_config['model']['classifier']['pooling'] = pooling
    return factory_create_model(condition, trial_config, device)


def train_single_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
    positive_weight: float = 10.0,
) -> dict:
    """Train model on a single fold and return validation AUPRC."""

    model = model.to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    # Scheduler
    total_steps = len(train_loader) * max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss with class imbalance handling
    pos_weight = torch.tensor([positive_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # AMP
    scaler = GradScaler('cuda')

    # Training loop
    best_auprc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device).float()
            lengths = batch['length'].to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(features, lengths).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validate
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                lengths = batch['length'].to(device)

                with autocast('cuda'):
                    logits = model(features, lengths).squeeze(-1)
                    probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute AUPRC
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        if len(np.unique(all_labels)) > 1:
            auprc = average_precision_score(all_labels, all_probs)
        else:
            auprc = 0.0

        # Early stopping
        if auprc > best_auprc:
            best_auprc = auprc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return {'auprc': best_auprc}


def train_and_evaluate_cv(
    config: dict,
    condition: str,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    pooling: str,
    device: torch.device,
    trajectories: list,
    labels: np.ndarray,
    splits: dict,
    max_epochs: int = 50,
    patience: int = 10,
) -> dict:
    """Train model with 5-fold CV and return mean AUPRC."""

    fold_auprcs = []

    # Create augmentor for training
    augmentor = TrajectoryAugmentor(config.get('augmentation', {}))

    # Use outer folds for CV (5 folds)
    for outer_fold in range(5):
        fold_key = f"outer_fold_{outer_fold}"
        if fold_key not in splits:
            continue

        fold_splits = splits[fold_key]

        # Get train/val indices (use inner fold 0's train as train, val as val)
        inner_key = "inner_fold_0"
        if inner_key in fold_splits:
            train_idx = np.array(fold_splits[inner_key]['train_indices'])
            val_idx = np.array(fold_splits[inner_key]['val_indices'])
        else:
            # Fallback: use train_val as train, test as val
            train_idx = np.array(fold_splits.get('train_val_indices', []))
            val_idx = np.array(fold_splits.get('test_indices', []))

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        # Create datasets using DTUDataset interface
        train_dataset = DTUDataset(
            trajectories=trajectories,
            labels=labels,
            indices=train_idx,
            config=config,
            augmentor=augmentor
        )
        val_dataset = DTUDataset(
            trajectories=trajectories,
            labels=labels,
            indices=val_idx,
            config=config,
            augmentor=None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Create fresh model for each fold
        model = create_model_with_pooling(condition, config, pooling, str(device))

        # Train
        result = train_single_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )

        fold_auprcs.append(result['auprc'])

        # Clean up
        del model
        torch.cuda.empty_cache()

    if not fold_auprcs:
        return {'auprc': 0.0, 'auprc_std': 0.0}

    return {
        'auprc': np.mean(fold_auprcs),
        'auprc_std': np.std(fold_auprcs),
    }


def run_optimization(
    condition: str,
    config: dict,
    trajectories: list,
    labels: np.ndarray,
    splits: dict,
    n_trials: int = 30,
    output_dir: Path = None,
):
    """Run Bayesian optimization for a given condition."""

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization for {condition.upper()}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {len(trajectories)} trajectories, {int(labels.sum())} anomalies")

    # Initialize Ax client
    ax_client = AxClient(verbose_logging=False)

    # Define search space with sensible ranges
    ax_client.create_experiment(
        name=f"anomaly_detection_{condition}",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-6, 1e-2],  # Wide range
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [0.0, 0.3],  # Include 0, wider upper
                "value_type": "float",
            },
            {
                "name": "beta1",
                "type": "range",
                "bounds": [0.8, 0.99],  # Wider upper bound
                "value_type": "float",
            },
            {
                "name": "beta2",
                "type": "range",
                "bounds": [0.99, 0.9999],  # Higher lower bound
                "value_type": "float",
            },
            {
                "name": "pooling",
                "type": "choice",
                "values": ["mean", "last"],
                "value_type": "str",
            },
        ],
        objectives={"auprc": ObjectiveProperties(minimize=False)},
    )

    # Run optimization
    print(f"\nRunning {n_trials} trials (5-fold CV per trial)...")

    trial_results = []

    for i in range(n_trials):
        # Get next hyperparameters from Ax
        parameters, trial_index = ax_client.get_next_trial()

        print(f"\nTrial {i+1}/{n_trials}:")
        print(f"  lr={parameters['learning_rate']:.2e}, "
              f"wd={parameters['weight_decay']:.4f}, "
              f"β1={parameters['beta1']:.3f}, "
              f"β2={parameters['beta2']:.5f}, "
              f"pool={parameters['pooling']}")

        try:
            # Train and evaluate with 5-fold CV
            results = train_and_evaluate_cv(
                config=config,
                condition=condition,
                learning_rate=parameters['learning_rate'],
                weight_decay=parameters['weight_decay'],
                beta1=parameters['beta1'],
                beta2=parameters['beta2'],
                pooling=parameters['pooling'],
                device=device,
                trajectories=trajectories,
                labels=labels,
                splits=splits,
            )

            print(f"  Result: AUPRC={results['auprc']:.4f} ± {results['auprc_std']:.4f}")

            # Report to Ax
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"auprc": results['auprc']},
            )

            # Store results
            trial_results.append({
                'trial': i + 1,
                'parameters': parameters,
                'auprc': results['auprc'],
                'auprc_std': results['auprc_std'],
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            ax_client.log_trial_failure(trial_index=trial_index)

    # Get ACTUAL best observed (not model predicted)
    if trial_results:
        best_trial = max(trial_results, key=lambda x: x['auprc'])
        best_parameters = best_trial['parameters']
        best_auprc = best_trial['auprc']
        best_auprc_std = best_trial['auprc_std']
        best_trial_num = best_trial['trial']
    else:
        best_parameters = {}
        best_auprc = 0.0
        best_auprc_std = 0.0
        best_trial_num = 0

    print(f"\n{'='*60}")
    print(f"BEST OBSERVED PARAMETERS for {condition} (Trial {best_trial_num}):")
    print(f"{'='*60}")
    print(f"  learning_rate: {best_parameters.get('learning_rate', 0):.2e}")
    print(f"  weight_decay:  {best_parameters.get('weight_decay', 0):.4f}")
    print(f"  beta1:         {best_parameters.get('beta1', 0):.4f}")
    print(f"  beta2:         {best_parameters.get('beta2', 0):.5f}")
    print(f"  pooling:       {best_parameters.get('pooling', 'N/A')}")
    print(f"  Best AUPRC:    {best_auprc:.4f} ± {best_auprc_std:.4f}")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'condition': condition,
            'n_trials': n_trials,
            'best_trial': best_trial_num,
            'best_parameters': best_parameters,
            'best_auprc': best_auprc,
            'best_auprc_std': best_auprc_std,
            'all_trials': trial_results,
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_dir / f'{condition}_optimization.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_dir / f'{condition}_optimization.json'}")

    return best_parameters, best_auprc, trial_results


def main():
    parser = argparse.ArgumentParser(description='Bayesian optimization for anomaly detection')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['random_init', 'pretrained', 'both'],
                        help='Condition to optimize')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of optimization trials')
    parser.add_argument('--exp-name', type=str, default='bayesian_opt_v1',
                        help='Experiment name for output directory')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load processed data (returns tuple: trajectories, labels)
    processed_path = Path(__file__).parent.parent / config['data']['processed_path']
    print(f"Loading processed data from {processed_path}")
    trajectories, labels = load_processed_data(processed_path)

    # Load CV splits
    splits_path = Path(__file__).parent.parent / config['data']['splits_path'] / 'splits.json'
    print(f"Loading CV splits from {splits_path}")
    splits = load_splits(splits_path)

    output_dir = Path(__file__).parent.parent / 'experiments' / args.exp_name

    results = {}

    if args.condition in ['random_init', 'both']:
        best_params, best_auprc, trials = run_optimization(
            condition='random_init',
            config=config,
            trajectories=trajectories,
            labels=labels,
            splits=splits,
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
        results['random_init'] = {
            'best_parameters': best_params,
            'best_auprc': best_auprc,
        }

    if args.condition in ['pretrained', 'both']:
        best_params, best_auprc, trials = run_optimization(
            condition='pretrained',
            config=config,
            trajectories=trajectories,
            labels=labels,
            splits=splits,
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
        results['pretrained'] = {
            'best_parameters': best_params,
            'best_auprc': best_auprc,
        }

    # Print summary
    if args.condition == 'both':
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for cond, res in results.items():
            print(f"\n{cond}:")
            print(f"  Best AUPRC: {res['best_auprc']:.4f}")
            print(f"  LR: {res['best_parameters']['learning_rate']:.2e}")
            print(f"  WD: {res['best_parameters']['weight_decay']:.4f}")
            print(f"  Pooling: {res['best_parameters']['pooling']}")

        # Determine winner
        random_auprc = results['random_init']['best_auprc']
        pretrained_auprc = results['pretrained']['best_auprc']
        diff = pretrained_auprc - random_auprc
        if diff > 0:
            print(f"\nWINNER: pretrained (+{diff:.4f} AUPRC)")
        else:
            print(f"\nWINNER: random_init (+{-diff:.4f} AUPRC)")

        # Save comparison
        with open(output_dir / 'comparison.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
