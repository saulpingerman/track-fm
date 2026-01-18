#!/usr/bin/env python3
"""
Run ablation study on different pooling strategies and LoRA.

Compares:
1. Pooling strategies: mean (baseline), attention, mha, hybrid, hybrid_attention
2. LoRA adaptation vs full fine-tuning
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
from torch.utils.data import DataLoader
from datetime import datetime

from src.data.extract_trajectories import load_processed_data
from src.data.dataset import VesselClassificationDataset, collate_fn
from src.models.factory import create_model, get_optimizer
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_per_class_metrics,
    compute_confusion_matrix,
    format_results,
    save_results,
)


def run_single_experiment(
    condition: str,
    config: dict,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run a single experimental condition."""
    print(f"\n{'='*60}")
    print(f"Running condition: {condition}")
    print(f"{'='*60}")

    training_cfg = config['training']
    condition_dir = output_dir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Train samples: {len(train_dataset)}")
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

    # Train
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
    confusion = compute_confusion_matrix(all_labels, all_preds, class_names)

    # Print results
    results_str = format_results(test_metrics, per_class, confusion, condition)
    print(results_str)

    # Save results
    results = {
        'condition': condition,
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
    parser = argparse.ArgumentParser(description='Run pooling ablation study')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    parser.add_argument('--exp-name', default=None, help='Experiment name')
    parser.add_argument('--conditions', nargs='+', default=None, help='Specific conditions to run')
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
    exp_name = args.exp_name or f"pooling_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(__file__).parent.parent / 'experiments' / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load data
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
        train_trajectories, train_labels,
        max_length=config['data']['max_trajectory_length'],
        augment=True,
    )
    val_dataset = VesselClassificationDataset(
        val_trajectories, val_labels,
        max_length=config['data']['max_trajectory_length'],
        augment=False,
    )
    test_dataset = VesselClassificationDataset(
        test_trajectories, test_labels,
        max_length=config['data']['max_trajectory_length'],
        augment=False,
    )

    # Define conditions to test
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = [
            # Baseline (already tested)
            # 'pretrained',
            # 'random_init',

            # Pooling ablations (with pretrained encoder)
            'pretrained_attention',      # Attention pooling
            'pretrained_mha',            # Multi-head attention pooling
            'pretrained_hybrid',         # Mean + Max
            'pretrained_hybrid_attention',  # Mean + Max + Attention

            # LoRA variants
            'pretrained_lora',           # LoRA with mean pooling
            'pretrained_attention_lora', # LoRA with attention pooling
        ]

    # Run experiments
    all_results = []
    for condition in conditions:
        try:
            results = run_single_experiment(
                condition=condition,
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                output_dir=exp_dir,
                device=device,
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR running {condition}: {e}")
            import traceback
            traceback.print_exc()

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
    print(f"{'Condition':<30} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Weighted':<12} {'Trainable':<15}")
    print("-"*80)

    for r in all_results:
        print(f"{r['condition']:<30} "
              f"{r['test_metrics']['accuracy']:<12.4f} "
              f"{r['test_metrics']['f1_macro']:<12.4f} "
              f"{r['test_metrics']['f1_weighted']:<12.4f} "
              f"{r['trainable_params']:>12,}")

    print(f"\nResults saved to: {exp_dir}")


if __name__ == '__main__':
    main()
