#!/usr/bin/env python3
"""
Bayesian Optimization for Vessel Classification Hyperparameters.

Uses Facebook's Ax library to optimize:
- learning_rate
- weight_decay
- beta1, beta2 (AdamW)
- pooling (mean vs last)

For both random_init and pretrained conditions.
"""

import argparse
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

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.logger import get_logger

# Suppress warnings
warnings.filterwarnings('ignore')
get_logger("ax").setLevel("WARNING")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import VesselClassificationDataset, collate_fn
from src.data.extract_trajectories import load_processed_data
from src.models.factory import create_model
from src.evaluation.metrics import compute_metrics


def load_data(config: dict, batch_size: int = 64):
    """Load train/val datasets."""
    data_dir = Path(__file__).parent.parent / config['data']['processed_path']

    # Load processed data
    trajectories, labels, splits, stats = load_processed_data(data_dir)

    # Create datasets
    train_trajs = [trajectories[i] for i in splits['train']]
    train_labels = [labels[i] for i in splits['train']]
    val_trajs = [trajectories[i] for i in splits['val']]
    val_labels = [labels[i] for i in splits['val']]

    train_dataset = VesselClassificationDataset(
        train_trajs, train_labels,
        max_length=config['model']['encoder']['max_seq_length'],
        augment=True,
    )
    val_dataset = VesselClassificationDataset(
        val_trajs, val_labels,
        max_length=config['model']['encoder']['max_seq_length'],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def train_and_evaluate(
    config: dict,
    condition: str,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    pooling: str,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 30,
    patience: int = 7,
) -> dict:
    """Train model with given hyperparameters and return validation metrics."""

    # Create a copy of config with the pooling override
    import copy
    trial_config = copy.deepcopy(config)
    trial_config['model']['classifier']['pooling'] = pooling

    # Create model using factory (handles pretrained loading etc.)
    model = create_model(trial_config, condition)
    model = model.to(device)

    # Create optimizer with specified hyperparameters
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    # Scheduler
    total_steps = len(train_loader) * max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss and AMP
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')

    # Training loop with early stopping
    best_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(features, lengths)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths'].to(device)

                with autocast('cuda'):
                    logits = model(features, lengths)

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            num_classes=config['data']['num_classes'],
        )

        current_f1 = metrics['f1_macro']

        # Early stopping check
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_acc = metrics['accuracy']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        'f1_macro': best_f1,
        'accuracy': best_acc,
    }


def run_optimization(
    condition: str,
    config: dict,
    n_trials: int = 30,
    output_dir: Path = None,
):
    """Run Bayesian optimization for a given condition."""

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization for {condition.upper()}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data once
    print("Loading data...")
    train_loader, val_loader = load_data(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize Ax client
    ax_client = AxClient(verbose_logging=False)

    # Define search space
    ax_client.create_experiment(
        name=f"vessel_classification_{condition}",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-5, 1e-3],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [1e-4, 0.1],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "beta1",
                "type": "range",
                "bounds": [0.8, 0.95],
                "value_type": "float",
            },
            {
                "name": "beta2",
                "type": "range",
                "bounds": [0.95, 0.999],
                "value_type": "float",
            },
            {
                "name": "pooling",
                "type": "choice",
                "values": ["mean", "last"],
                "value_type": "str",
            },
        ],
        objectives={"f1_macro": ObjectiveProperties(minimize=False)},
    )

    # Run optimization
    print(f"\nRunning {n_trials} trials...")

    trial_results = []

    for i in range(n_trials):
        # Get next hyperparameters from Ax
        parameters, trial_index = ax_client.get_next_trial()

        print(f"\nTrial {i+1}/{n_trials}:")
        print(f"  lr={parameters['learning_rate']:.2e}, "
              f"wd={parameters['weight_decay']:.2e}, "
              f"β1={parameters['beta1']:.3f}, "
              f"β2={parameters['beta2']:.4f}, "
              f"pool={parameters['pooling']}")

        try:
            # Train and evaluate
            results = train_and_evaluate(
                config=config,
                condition=condition,
                learning_rate=parameters['learning_rate'],
                weight_decay=parameters['weight_decay'],
                beta1=parameters['beta1'],
                beta2=parameters['beta2'],
                pooling=parameters['pooling'],
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
            )

            print(f"  Result: F1={results['f1_macro']:.4f}, Acc={results['accuracy']:.4f}")

            # Report to Ax
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"f1_macro": results['f1_macro']},
            )

            # Store results
            trial_results.append({
                'trial': i + 1,
                'parameters': parameters,
                'f1_macro': results['f1_macro'],
                'accuracy': results['accuracy'],
            })

        except Exception as e:
            print(f"  Error: {e}")
            ax_client.log_trial_failure(trial_index=trial_index)

    # Get ACTUAL best observed (not model predicted)
    if trial_results:
        best_trial = max(trial_results, key=lambda x: x['f1_macro'])
        best_parameters = best_trial['parameters']
        best_f1 = best_trial['f1_macro']
        best_acc = best_trial['accuracy']
        best_trial_num = best_trial['trial']
    else:
        best_parameters = {}
        best_f1 = 0.0
        best_acc = 0.0
        best_trial_num = 0

    print(f"\n{'='*60}")
    print(f"BEST OBSERVED PARAMETERS for {condition} (Trial {best_trial_num}):")
    print(f"{'='*60}")
    print(f"  learning_rate: {best_parameters.get('learning_rate', 0):.2e}")
    print(f"  weight_decay:  {best_parameters.get('weight_decay', 0):.2e}")
    print(f"  beta1:         {best_parameters.get('beta1', 0):.4f}")
    print(f"  beta2:         {best_parameters.get('beta2', 0):.5f}")
    print(f"  pooling:       {best_parameters.get('pooling', 'N/A')}")
    print(f"  Best F1:       {best_f1:.4f}")
    print(f"  Best Accuracy: {best_acc:.4f}")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'condition': condition,
            'n_trials': n_trials,
            'best_trial': best_trial_num,
            'best_parameters': best_parameters,
            'best_f1_macro': best_f1,
            'best_accuracy': best_acc,
            'all_trials': trial_results,
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_dir / f'{condition}_optimization.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_dir / f'{condition}_optimization.json'}")

    return best_parameters, best_f1, trial_results


def main():
    parser = argparse.ArgumentParser(description='Bayesian optimization for vessel classification')
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

    output_dir = Path(__file__).parent.parent / 'experiments' / args.exp_name

    results = {}

    if args.condition in ['random_init', 'both']:
        best_params, best_f1, trials = run_optimization(
            condition='random_init',
            config=config,
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
        results['random_init'] = {
            'best_parameters': best_params,
            'best_f1': best_f1,
        }

    if args.condition in ['pretrained', 'both']:
        best_params, best_f1, trials = run_optimization(
            condition='pretrained',
            config=config,
            n_trials=args.n_trials,
            output_dir=output_dir,
        )
        results['pretrained'] = {
            'best_parameters': best_params,
            'best_f1': best_f1,
        }

    # Print summary
    if args.condition == 'both':
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for cond, res in results.items():
            print(f"\n{cond}:")
            print(f"  Best F1: {res['best_f1']:.4f}")
            print(f"  LR: {res['best_parameters']['learning_rate']:.2e}")
            print(f"  WD: {res['best_parameters']['weight_decay']:.2e}")
            print(f"  Pooling: {res['best_parameters']['pooling']}")

        # Save comparison
        with open(output_dir / 'comparison.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
