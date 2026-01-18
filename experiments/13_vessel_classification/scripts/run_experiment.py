#!/usr/bin/env python3
"""
Run vessel classification experiment.

Compares pretrained, random_init, and frozen_pretrained conditions.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from typing import Dict

from src.data.extract_trajectories import load_processed_data
from src.data.dataset import VesselClassificationDataset, collate_fn
from src.models.factory import create_model, get_optimizer, get_optimizer_stage2
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix,
    format_results,
    save_results,
)


def run_condition(
    condition: str,
    config: Dict,
    train_dataset: VesselClassificationDataset,
    val_dataset: VesselClassificationDataset,
    test_dataset: VesselClassificationDataset,
    output_dir: Path,
    device: torch.device,
    data_fraction: float = 1.0,
) -> Dict:
    """
    Train and evaluate a single condition.

    Args:
        condition: One of 'pretrained', 'random_init', 'frozen_pretrained'
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        output_dir: Output directory
        device: Device to train on
        data_fraction: Fraction of training data to use

    Returns:
        results: Dictionary with all results
    """
    print(f"\n{'='*60}")
    print(f"Running condition: {condition} (data_fraction={data_fraction})")
    print(f"{'='*60}")

    training_cfg = config['training']
    condition_dir = output_dir / f"{condition}_frac{data_fraction}"
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Subsample training data if needed
    if data_fraction < 1.0:
        n_samples = int(len(train_dataset) * data_fraction)
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        train_subset = Subset(train_dataset, indices)
    else:
        train_subset = train_dataset

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=training_cfg.get('pin_memory', True),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=training_cfg.get('pin_memory', True),
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=training_cfg.get('pin_memory', True),
        collate_fn=collate_fn,
    )

    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    model = create_model(config, condition)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = get_optimizer(model, config, condition)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        output_dir=condition_dir,
        condition=condition,
    )

    # Two-stage training: first train head only, then unfreeze and fine-tune all
    if condition == 'two_stage':
        warmup_epochs = training_cfg.get('two_stage_warmup_epochs', 10)
        print(f"\n--- Stage 1: Training classifier head only ({warmup_epochs} epochs) ---")

        # Override max_epochs for stage 1
        original_max_epochs = trainer.max_epochs
        trainer.max_epochs = warmup_epochs
        trainer.patience = warmup_epochs + 1  # Don't early stop during warmup

        # Train stage 1 (head only)
        stage1_results = trainer.train(train_loader, val_loader)

        # Unfreeze encoder
        print(f"\n--- Stage 2: Unfreezing encoder and fine-tuning all parameters ---")
        for param in model.encoder.parameters():
            param.requires_grad = True

        # Create new optimizer with differential LR
        optimizer = get_optimizer_stage2(model, config)
        trainer.optimizer = optimizer

        # Reset trainer state for stage 2
        trainer.max_epochs = original_max_epochs
        trainer.patience = training_cfg['early_stopping_patience']
        trainer.best_metric = 0.0
        trainer.best_epoch = 0
        trainer.epochs_without_improvement = 0
        trainer.history = {'train': [], 'val': []}

        # Update trainable params count
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (after unfreeze): {trainable_params:,}")

        # Train stage 2 (all params)
        train_results = trainer.train(train_loader, val_loader)
        train_results['stage1'] = stage1_results
    else:
        # Standard training
        train_results = trainer.train(train_loader, val_loader)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Get predictions for detailed analysis
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['labels']
            lengths = batch['lengths'].to(device)

            logits = model(features, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class metrics
    class_names = config['data']['classes']
    per_class = compute_per_class_metrics(all_labels, all_preds, class_names)

    # Confusion matrix
    confusion = compute_confusion_matrix(all_labels, all_preds, class_names)

    # Format and print results
    results_str = format_results(test_metrics, per_class, confusion, condition)
    print(results_str)

    # Save results
    results = {
        'condition': condition,
        'data_fraction': data_fraction,
        'train_samples': len(train_subset),
        'test_metrics': test_metrics,
        'per_class_metrics': per_class,
        'confusion_matrix': confusion,
        'training': train_results,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }

    save_results(results, condition_dir / 'results.json')

    return results


def main():
    parser = argparse.ArgumentParser(description='Run vessel classification experiment')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    parser.add_argument('--exp-name', default=None, help='Experiment name')
    parser.add_argument('--condition', default=None, help='Run single condition')
    parser.add_argument('--data-fraction', type=float, default=1.0, help='Fraction of training data')
    parser.add_argument('--learning-curves', action='store_true', help='Run learning curve ablation')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create experiment directory
    exp_name = args.exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(__file__).parent.parent / 'experiments' / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load processed data
    data_dir = Path(__file__).parent.parent / config['data']['processed_path']
    trajectories, labels, splits, stats = load_processed_data(data_dir)

    print(f"\nLoaded {len(trajectories)} trajectories")
    print(f"Classes: {stats['classes']}")
    print(f"Distribution: {stats['class_distribution']}")

    # Create datasets
    train_trajectories = [trajectories[i] for i in splits['train']]
    train_labels = [labels[i] for i in splits['train']]

    val_trajectories = [trajectories[i] for i in splits['val']]
    val_labels = [labels[i] for i in splits['val']]

    test_trajectories = [trajectories[i] for i in splits['test']]
    test_labels = [labels[i] for i in splits['test']]

    train_dataset = VesselClassificationDataset(
        train_trajectories,
        train_labels,
        max_length=config['data']['max_trajectory_length'],
        augment=True,
    )

    val_dataset = VesselClassificationDataset(
        val_trajectories,
        val_labels,
        max_length=config['data']['max_trajectory_length'],
        augment=False,
    )

    test_dataset = VesselClassificationDataset(
        test_trajectories,
        test_labels,
        max_length=config['data']['max_trajectory_length'],
        augment=False,
    )

    # Determine conditions to run
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = config['training']['conditions']

    # Determine data fractions
    if args.learning_curves:
        data_fractions = config['training']['limited_data_fractions']
    else:
        data_fractions = [args.data_fraction]

    # Run experiments
    all_results = []

    for data_fraction in data_fractions:
        for condition in conditions:
            results = run_condition(
                condition=condition,
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                output_dir=exp_dir,
                device=device,
                data_fraction=data_fraction,
            )
            all_results.append(results)

    # Save summary
    summary = {
        'experiment_name': exp_name,
        'results': all_results,
    }

    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Condition':<20} {'Data %':<10} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Weighted':<12}")
    print("-"*80)

    for r in all_results:
        print(f"{r['condition']:<20} {r['data_fraction']*100:>6.1f}%    "
              f"{r['test_metrics']['accuracy']:<12.4f} "
              f"{r['test_metrics']['f1_macro']:<12.4f} "
              f"{r['test_metrics']['f1_weighted']:<12.4f}")

    print("\nResults saved to:", exp_dir)


if __name__ == '__main__':
    main()
