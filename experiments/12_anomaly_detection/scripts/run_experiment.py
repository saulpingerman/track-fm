#!/usr/bin/env python3
"""
Main experiment runner for Experiment 12: Anomaly Detection Fine-tuning.

Usage:
    python run_experiment.py --exp-name my_experiment
    python run_experiment.py --exp-name my_experiment --condition pretrained
    python run_experiment.py --exp-name test --condition pretrained --outer-fold 0 --num-seeds 1
"""

import argparse
import yaml
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dtu_dataset import DTUDataset, load_dtu_data
from data.preprocessing import preprocess_trajectories, save_processed_data, load_processed_data
from data.cv_splits import create_nested_cv_splits, save_splits, load_splits, print_split_summary
from data.augmentation import TrajectoryAugmentor
from models.model_factory import create_model
from training.trainer import Trainer, collate_fn
from evaluation.metrics import compute_all_metrics
from evaluation.statistics import aggregate_results, run_statistical_tests, compute_metric_cis


def set_seed(seed: int, benchmark: bool = True):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # benchmark=True for faster training (slight non-determinism)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = not benchmark


def run_single_fold(
    model,
    train_loader,
    val_loader,
    test_loader,
    config,
    condition,
    exp_dir,
    fold_info
):
    """Run training and evaluation for a single fold."""

    # Create fold-specific directory
    fold_dir = exp_dir / f"fold_{fold_info['outer_fold']}_{fold_info['inner_fold']}_seed{fold_info['seed']}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        condition=condition,
        exp_dir=fold_dir,
        fold_info=fold_info
    )

    # Train
    train_history = trainer.fit()

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)

    # Save fold results
    fold_results = {
        'train_history': {
            'train_loss': train_history['train_loss'],
            'val_loss': train_history['val_loss']
        },
        'test_metrics': test_metrics,
        **fold_info
    }

    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(fold_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    return fold_results


def main():
    parser = argparse.ArgumentParser(description='Run anomaly detection experiment')
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment name')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--condition', type=str, default=None,
                        help='Run single condition (pretrained/random_init/frozen_pretrained)')
    parser.add_argument('--outer-fold', type=int, default=None, help='Run single outer fold')
    parser.add_argument('--inner-fold', type=int, default=None, help='Run single inner fold')
    parser.add_argument('--num-seeds', type=int, default=None, help='Override number of seeds')
    parser.add_argument('--quick', action='store_true', help='Quick test with reduced iterations')
    args = parser.parse_args()

    # Change to experiment directory
    exp_root = Path(__file__).parent.parent
    os.chdir(exp_root)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Quick mode: reduce iterations for testing
    if args.quick:
        print("QUICK MODE: Reducing iterations for testing")
        config['experiment']['num_seeds'] = 1
        config['data']['outer_folds'] = 2
        config['data']['inner_folds'] = 2
        config['training']['max_epochs'] = 10
        config['training']['warmup_epochs'] = 2
        config['training']['early_stopping_patience'] = 3
        args.outer_fold = 0  # Only run first outer fold
        args.inner_fold = 0  # Only run first inner fold

    # Override num_seeds if specified
    if args.num_seeds is not None:
        config['experiment']['num_seeds'] = args.num_seeds

    # Create experiment directory
    exp_dir = Path(f'experiments/{args.exp_name}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print("=" * 60)
    print("Experiment 12: Anomaly Detection Fine-tuning")
    print(f"Name: {args.exp_name}")
    print(f"Device: {config['experiment']['device']}")
    print("=" * 60)

    # Check for processed data
    processed_path = Path(config['data']['processed_path'])
    if (processed_path / 'trajectories.pkl').exists():
        print("\nLoading preprocessed data...")
        trajectories, labels = load_processed_data(processed_path)
    else:
        # Load and preprocess data
        print("\nLoading DTU dataset...")
        raw_path = Path(config['data']['raw_path'])

        # Check for data files (pickle, csv, or parquet)
        has_pkl = any(raw_path.glob('data_*.pkl'))
        has_csv = any(raw_path.glob('**/*.csv'))
        has_parquet = any(raw_path.glob('**/*.parquet'))

        if not has_pkl and not has_csv and not has_parquet:
            print(f"ERROR: No data found in {raw_path}")
            print("Please download the DTU dataset first:")
            print("  python scripts/download_data.py --output data/raw/")
            return

        trajectories, labels = load_dtu_data(str(raw_path))
        print("\nPreprocessing trajectories...")
        trajectories = preprocess_trajectories(trajectories, config)
        save_processed_data(trajectories, labels, str(processed_path))

    print(f"\nLoaded {len(trajectories)} trajectories, {sum(labels)} anomalies ({100*sum(labels)/len(labels):.1f}%)")

    # Create or load CV splits
    splits_path = Path(config['data']['splits_path']) / 'splits.json'
    if splits_path.exists():
        print("\nLoading existing CV splits...")
        splits = load_splits(str(splits_path))
    else:
        print("\nCreating cross-validation splits...")
        splits = create_nested_cv_splits(
            labels=np.array(labels),
            outer_folds=config['data']['outer_folds'],
            inner_folds=config['data']['inner_folds'],
            random_state=config['experiment']['seed']
        )
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        save_splits(splits, str(splits_path))

    print_split_summary(splits)

    # Determine conditions to run
    conditions = [args.condition] if args.condition else config['training']['conditions']

    # Results storage
    all_results = {cond: [] for cond in conditions}

    # Main experiment loop
    device = config['experiment']['device']

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        # Determine folds to run
        if args.outer_fold is not None:
            outer_folds = [args.outer_fold]
        else:
            outer_folds = range(config['data']['outer_folds'])

        for outer_fold in outer_folds:
            test_indices = np.array(splits[f'outer_fold_{outer_fold}']['test_indices'])

            # Determine inner folds
            if args.inner_fold is not None:
                inner_folds = [args.inner_fold]
            else:
                inner_folds = range(config['data']['inner_folds'])

            for inner_fold in inner_folds:
                train_indices = np.array(splits[f'outer_fold_{outer_fold}'][f'inner_fold_{inner_fold}']['train_indices'])
                val_indices = np.array(splits[f'outer_fold_{outer_fold}'][f'inner_fold_{inner_fold}']['val_indices'])

                for seed in range(config['experiment']['num_seeds']):
                    print(f"\n--- Outer {outer_fold}, Inner {inner_fold}, Seed {seed} ---")

                    set_seed(config['experiment']['seed'] + seed)

                    # Create augmentor (training only)
                    augmentor = TrajectoryAugmentor(config['augmentation'])

                    # Create datasets
                    train_dataset = DTUDataset(
                        trajectories, np.array(labels), train_indices,
                        augmentor=augmentor, config=config
                    )
                    val_dataset = DTUDataset(
                        trajectories, np.array(labels), val_indices,
                        config=config
                    )
                    test_dataset = DTUDataset(
                        trajectories, np.array(labels), test_indices,
                        config=config
                    )

                    # Create data loaders with performance settings
                    num_workers = config['training'].get('num_workers', 0)
                    pin_memory = config['training'].get('pin_memory', False) and device == 'cuda'

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn
                    )

                    # Create model
                    model = create_model(condition, config, device)

                    # Run fold
                    fold_result = run_single_fold(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        config=config,
                        condition=condition,
                        exp_dir=exp_dir / condition,
                        fold_info={
                            'outer_fold': outer_fold,
                            'inner_fold': inner_fold,
                            'seed': seed
                        }
                    )

                    all_results[condition].append(fold_result)

                    # Log progress
                    metrics = fold_result['test_metrics']
                    print(f"  AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}, "
                          f"F1: {metrics['f1_optimal']:.4f}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregating results...")
    print("=" * 60)

    summary = aggregate_results(all_results)
    summary.to_csv(exp_dir / 'results' / 'summary_metrics.csv', index=False)
    print("\n" + summary.to_string())

    # Compute bootstrap CIs
    print("\nComputing bootstrap confidence intervals...")
    cis = compute_metric_cis(all_results, n_bootstrap=config['evaluation']['bootstrap_samples'])
    cis.to_csv(exp_dir / 'results' / 'confidence_intervals.csv', index=False)

    # Statistical tests
    if len(conditions) > 1:
        print("\nRunning statistical tests...")
        stat_results = run_statistical_tests(all_results, config)
        stat_results.to_csv(exp_dir / 'results' / 'statistical_tests.csv', index=False)
        print("\n" + stat_results.to_string())

    # Save raw results
    with open(exp_dir / 'results' / 'all_results.json', 'w') as f:
        # Convert to serializable format
        serializable = {}
        for cond, results in all_results.items():
            serializable[cond] = []
            for r in results:
                result_dict = {
                    'outer_fold': r['outer_fold'],
                    'inner_fold': r['inner_fold'],
                    'seed': r['seed'],
                    'test_metrics': {k: float(v) for k, v in r['test_metrics'].items()}
                }
                serializable[cond].append(result_dict)
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {exp_dir / 'results'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
