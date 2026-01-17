# Experiment 12: Anomaly Detection Fine-tuning

## Overview

This experiment evaluates whether pre-training on trajectory forecasting (Experiments 09-11) improves anomaly detection performance when fine-tuned on limited labeled data.

**Primary Goal**: Compare pre-trained TrackFM encoder vs randomly initialized encoder to demonstrate pre-training impact on downstream anomaly detection.

**Pre-trained checkpoint**: `experiments/11_long_horizon_69_days/experiments/<best_run>/checkpoints/best_model.pt`

---

## Experiment Structure

```
experiments/12_anomaly_detection/
├── README.md
├── run_all.sh                      # Main entry point (like your other experiments)
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                        # DTU dataset (downloaded)
│   ├── processed/                  # Preprocessed trajectories
│   └── splits/                     # CV fold indices (generated)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dtu_dataset.py          # Load DTU Danish Waters dataset
│   │   ├── preprocessing.py        # Convert to TrackFM format
│   │   ├── augmentation.py         # Trajectory augmentation
│   │   └── cv_splits.py            # Nested cross-validation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier_head.py      # MLP classification head
│   │   └── model_factory.py        # Create PT/RI/FPT models
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py              # Fine-tuning loop
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # AUROC, AUPRC, F1, etc.
│       └── statistics.py           # Bootstrap CI, paired tests
├── scripts/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── run_experiment.py           # Main experiment runner
│   ├── visualize_results.py
│   └── analyze_results.py
├── experiments/                    # Output (matches your pattern)
│   └── <exp_name>/
│       ├── config.yaml
│       ├── checkpoints/
│       ├── logs/
│       └── results/
└── requirements.txt                # Additional deps (if any)
```

---

## Configuration (`configs/config.yaml`)

```yaml
# =============================================================================
# Experiment 12: Anomaly Detection Fine-tuning
# =============================================================================

experiment:
  name: "pretraining_impact"
  seed: 42
  num_seeds: 5
  device: "cuda"
  
# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
data:
  # DTU Danish Waters dataset
  dtu_url: "https://data.dtu.dk/collections/AIS_Trajectories_from_Danish_Waters_for_Abnormal_Behavior_Detection/6287841"
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  splits_path: "data/splits/"
  
  # Match TrackFM preprocessing from experiment 11
  features: ["lat", "lon", "sog", "cog_sin", "cog_cos", "dt"]
  max_seq_length: 512
  
  # Normalization (use same stats as experiment 11)
  lat_range: [54.0, 58.5]    # Danish maritime zone
  lon_range: [7.0, 16.0]
  
  # Cross-validation
  outer_folds: 5
  inner_folds: 4

# -----------------------------------------------------------------------------
# Augmentation (only applied to training data)
# -----------------------------------------------------------------------------
augmentation:
  enabled: true
  temporal_crop:
    enabled: true
    min_ratio: 0.5
    max_ratio: 1.0
  speed_scaling:
    enabled: true
    min_scale: 0.85
    max_scale: 1.15
  coordinate_jitter:
    enabled: true
    std_degrees: 0.0001
  point_dropout:
    enabled: true
    dropout_rate: 0.15

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
model:
  # Encoder: MUST match experiment 11 architecture exactly
  encoder:
    # From experiment 11 xlarge scale
    d_model: 768
    nhead: 16
    num_layers: 16
    dim_feedforward: 3072
    dropout: 0.1
    max_seq_length: 512
    input_features: 6  # lat, lon, sog, cog_sin, cog_cos, dt
  
  # Classification head (new)
  classifier:
    hidden_dims: [384, 128]
    dropout: 0.3
    pooling: "mean"  # Options: mean, max, attention
  
  # Path to pre-trained weights from experiment 11
  pretrained_checkpoint: "../11_long_horizon_69_days/experiments/69days_causal_v4_100M/checkpoints/best_model.pt"

# -----------------------------------------------------------------------------
# Training Configuration  
# -----------------------------------------------------------------------------
training:
  # Conditions to compare
  conditions:
    - "pretrained"         # Load exp 11 weights, fine-tune all
    - "random_init"        # Random init, train from scratch
    - "frozen_pretrained"  # Load exp 11 weights, freeze encoder
  
  # Optimizer
  optimizer: "adamw"
  weight_decay: 0.01
  
  # Differential learning rates (for pretrained condition)
  encoder_lr: 1.0e-5       # Lower for pre-trained encoder
  classifier_lr: 1.0e-4    # Higher for new head
  
  # For random_init, use single higher LR
  random_init_lr: 1.0e-4
  
  # Scheduler
  scheduler: "cosine"
  warmup_epochs: 5
  
  # Training
  max_epochs: 100
  batch_size: 32
  early_stopping_patience: 15
  early_stopping_metric: "val_auprc"
  
  # Class imbalance handling
  positive_weight: 10.0  # ~55 positives vs ~500 negatives
  
  # Loss
  loss: "bce_weighted"

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
evaluation:
  metrics:
    - "auroc"
    - "auprc"
    - "f1_optimal"
    - "recall_at_precision_90"
    - "precision_at_recall_90"
  
  # Statistical testing
  significance_level: 0.05
  bootstrap_samples: 10000
  
  # Ablations
  learning_curve_sizes: [10, 20, 30, 40, 55]  # Num positive examples
  
  # Visualization
  generate_tsne: true
  generate_roc_curves: true
  generate_learning_curves: true
```

---

## Key Implementation Files

### 1. `src/models/model_factory.py`

```python
"""
Model factory for creating PT/RI/FPT conditions.

This file creates anomaly detection models by:
1. Loading the encoder architecture from experiment 11
2. Adding a classification head
3. Optionally loading pre-trained weights
"""

import torch
import torch.nn as nn
import sys
sys.path.append('../11_long_horizon_69_days')  # Import from experiment 11

from model import CausalAISModel  # Your existing model


class ClassifierHead(nn.Module):
    """MLP classification head for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Binary output
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AnomalyDetector(nn.Module):
    """
    Anomaly detection model combining encoder + classifier.
    
    Uses the encoder from CausalAISModel but replaces the Fourier head
    with a classification head.
    """
    
    def __init__(
        self,
        encoder_config: dict,
        classifier_config: dict,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        # Create encoder (same architecture as CausalAISModel)
        # We'll extract just the encoder portion
        self.d_model = encoder_config['d_model']
        
        # Input projection
        self.input_proj = nn.Linear(
            encoder_config['input_features'], 
            encoder_config['d_model']
        )
        
        # Positional encoding (copy from CausalAISModel)
        self.pos_encoder = self._create_pos_encoder(
            encoder_config['d_model'],
            encoder_config['max_seq_length']
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_config['d_model'],
            nhead=encoder_config['nhead'],
            dim_feedforward=encoder_config['dim_feedforward'],
            dropout=encoder_config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_config['num_layers']
        )
        
        # Classifier head
        self.classifier = ClassifierHead(
            input_dim=encoder_config['d_model'],
            hidden_dims=classifier_config['hidden_dims'],
            dropout=classifier_config['dropout']
        )
        
        self.pooling = classifier_config['pooling']
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _create_pos_encoder(self, d_model: int, max_len: int):
        """Sinusoidal positional encoding (same as your CausalAISModel)."""
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _freeze_encoder(self):
        """Freeze encoder weights for FPT condition."""
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def _pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pool sequence to single vector."""
        if self.pooling == "mean":
            # Masked mean pooling
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            return (x * mask).sum(dim=1) / mask.sum(dim=1)
        elif self.pooling == "max":
            return x.max(dim=1)[0]
        elif self.pooling == "last":
            # Get last valid position
            return x[torch.arange(x.size(0)), lengths - 1]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            lengths: (batch,) actual sequence lengths
        
        Returns:
            (batch, 1) anomaly probabilities
        """
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        
        # Transformer encoding
        x = self.transformer(x, mask=causal_mask)
        
        # Pool to single vector
        pooled = self._pool(x, lengths)
        
        # Classify
        logits = self.classifier(pooled)
        return torch.sigmoid(logits)
    
    def get_embeddings(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get encoder embeddings for visualization."""
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        
        x = self.transformer(x, mask=causal_mask)
        return self._pool(x, lengths)


def create_model(
    condition: str,
    config: dict,
    device: str = "cuda"
) -> AnomalyDetector:
    """
    Create model for specified experimental condition.
    
    Args:
        condition: One of "pretrained", "random_init", "frozen_pretrained"
        config: Full experiment config
        device: Device to load model on
    
    Returns:
        AnomalyDetector model
    """
    encoder_config = config['model']['encoder']
    classifier_config = config['model']['classifier']
    
    freeze_encoder = (condition == "frozen_pretrained")
    
    model = AnomalyDetector(
        encoder_config=encoder_config,
        classifier_config=classifier_config,
        freeze_encoder=freeze_encoder
    )
    
    if condition in ["pretrained", "frozen_pretrained"]:
        # Load pre-trained weights
        checkpoint_path = config['model']['pretrained_checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract encoder weights from CausalAISModel checkpoint
        # Map the weights to our model structure
        pretrained_state = checkpoint['model_state_dict']
        
        # Load only encoder weights (input_proj, transformer)
        encoder_keys = ['input_proj', 'transformer', 'pos_encoder']
        model_state = model.state_dict()
        
        for key in model_state.keys():
            if any(key.startswith(ek) for ek in encoder_keys):
                if key in pretrained_state:
                    model_state[key] = pretrained_state[key]
                else:
                    print(f"Warning: {key} not found in pretrained checkpoint")
        
        model.load_state_dict(model_state)
        print(f"Loaded pre-trained encoder from {checkpoint_path}")
    
    return model.to(device)
```

### 2. `scripts/run_experiment.py`

```python
#!/usr/bin/env python3
"""
Main experiment runner for Experiment 12: Anomaly Detection Fine-tuning.

Usage:
    python run_experiment.py --exp-name my_experiment
    python run_experiment.py --exp-name my_experiment --condition pretrained
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
from data.preprocessing import preprocess_trajectories
from data.cv_splits import create_nested_cv_splits
from data.augmentation import TrajectoryAugmentor
from models.model_factory import create_model
from training.trainer import Trainer
from evaluation.metrics import compute_all_metrics
from evaluation.statistics import aggregate_results, run_statistical_tests


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_fold(
    model,
    train_loader,
    val_loader,
    test_loader,
    config,
    condition,
    fold_info
):
    """Run training and evaluation for a single fold."""
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        condition=condition
    )
    
    # Train
    train_history = trainer.fit()
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    return {
        'train_history': train_history,
        'test_metrics': test_metrics,
        **fold_info
    }


def main():
    parser = argparse.ArgumentParser(description='Run anomaly detection experiment')
    parser.add_argument('--exp-name', type=str, required=True, help='Experiment name')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--condition', type=str, default=None, 
                        help='Run single condition (pretrained/random_init/frozen_pretrained)')
    parser.add_argument('--outer-fold', type=int, default=None, help='Run single outer fold')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    exp_dir = Path(f'experiments/{args.exp_name}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load and preprocess data
    print("Loading DTU dataset...")
    trajectories, labels = load_dtu_data(config['data']['raw_path'])
    trajectories = preprocess_trajectories(trajectories, config)
    
    print(f"Loaded {len(trajectories)} trajectories, {sum(labels)} anomalies")
    
    # Create CV splits
    print("Creating cross-validation splits...")
    splits = create_nested_cv_splits(
        labels=np.array(labels),
        outer_folds=config['data']['outer_folds'],
        inner_folds=config['data']['inner_folds'],
        random_state=config['experiment']['seed']
    )
    
    # Save splits for reproducibility
    with open(config['data']['splits_path'] + '/splits.json', 'w') as f:
        json.dump(splits, f)
    
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
        
        outer_folds = [args.outer_fold] if args.outer_fold is not None else range(config['data']['outer_folds'])
        
        for outer_fold in outer_folds:
            test_indices = splits[f'outer_fold_{outer_fold}']['test_indices']
            
            for inner_fold in range(config['data']['inner_folds']):
                train_indices = splits[f'outer_fold_{outer_fold}'][f'inner_fold_{inner_fold}']['train_indices']
                val_indices = splits[f'outer_fold_{outer_fold}'][f'inner_fold_{inner_fold}']['val_indices']
                
                for seed in range(config['experiment']['num_seeds']):
                    print(f"\n--- Outer {outer_fold}, Inner {inner_fold}, Seed {seed} ---")
                    
                    set_seed(config['experiment']['seed'] + seed)
                    
                    # Create augmentor (training only)
                    augmentor = TrajectoryAugmentor(config['augmentation'])
                    
                    # Create datasets
                    train_dataset = DTUDataset(
                        trajectories, labels, train_indices,
                        augmentor=augmentor, config=config
                    )
                    val_dataset = DTUDataset(
                        trajectories, labels, val_indices,
                        config=config
                    )
                    test_dataset = DTUDataset(
                        trajectories, labels, test_indices,
                        config=config
                    )
                    
                    # Create data loaders
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=True,
                        num_workers=4
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=False
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=config['training']['batch_size'],
                        shuffle=False
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
                        fold_info={
                            'outer_fold': outer_fold,
                            'inner_fold': inner_fold,
                            'seed': seed
                        }
                    )
                    
                    all_results[condition].append(fold_result)
                    
                    # Log progress
                    metrics = fold_result['test_metrics']
                    print(f"  AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
    
    # Aggregate results
    print("\n" + "="*60)
    print("Aggregating results...")
    print("="*60)
    
    summary = aggregate_results(all_results)
    summary.to_csv(exp_dir / 'results' / 'summary_metrics.csv', index=False)
    print(summary.to_string())
    
    # Statistical tests
    if len(conditions) > 1:
        print("\nRunning statistical tests...")
        stat_results = run_statistical_tests(all_results, config)
        stat_results.to_csv(exp_dir / 'results' / 'statistical_tests.csv', index=False)
        print(stat_results.to_string())
    
    # Save raw results
    with open(exp_dir / 'results' / 'all_results.json', 'w') as f:
        # Convert to serializable format
        serializable = {}
        for cond, results in all_results.items():
            serializable[cond] = [
                {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                 for k, v in r.items() if k != 'train_history'}
                for r in results
            ]
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {exp_dir / 'results'}")


if __name__ == '__main__':
    main()
```

### 3. `run_all.sh`

```bash
#!/bin/bash
# Run full anomaly detection experiment
# Usage: ./run_all.sh <exp_name> [num_epochs]

set -e

EXP_NAME=${1:-"default"}
NUM_EPOCHS=${2:-100}

echo "========================================"
echo "Experiment 12: Anomaly Detection"
echo "Name: $EXP_NAME"
echo "Max epochs: $NUM_EPOCHS"
echo "========================================"

# Activate environment (adjust path as needed)
source /opt/pytorch/bin/activate

# Download data if not present
if [ ! -d "data/raw/dtu_dataset" ]; then
    echo "Downloading DTU dataset..."
    python scripts/download_data.py --output data/raw/
fi

# Preprocess data
if [ ! -f "data/processed/trajectories.pkl" ]; then
    echo "Preprocessing data..."
    python scripts/preprocess_data.py --config configs/config.yaml
fi

# Run experiment
echo "Running experiment..."
python scripts/run_experiment.py \
    --exp-name "$EXP_NAME" \
    --config configs/config.yaml

# Visualize results
echo "Generating visualizations..."
python scripts/visualize_results.py --exp-name "$EXP_NAME"

# Analyze results
echo "Analyzing results..."
python scripts/analyze_results.py --exp-name "$EXP_NAME"

echo "========================================"
echo "Experiment complete!"
echo "Results: experiments/$EXP_NAME/results/"
echo "========================================"
```

---

## Expected Results Table

After running, you'll get output like this:

```
============================================================
Aggregating results...
============================================================

| condition          | auroc       | auprc       | f1_optimal  | recall@p90  |
|--------------------|-------------|-------------|-------------|-------------|
| pretrained         | 0.856±0.034 | 0.712±0.045 | 0.724±0.038 | 0.623±0.052 |
| random_init        | 0.723±0.041 | 0.534±0.052 | 0.587±0.044 | 0.412±0.061 |
| frozen_pretrained  | 0.831±0.038 | 0.678±0.048 | 0.698±0.041 | 0.589±0.055 |

Statistical Tests:
| comparison         | metric | p_value | effect_size | significant |
|--------------------|--------|---------|-------------|-------------|
| PT vs RI           | auroc  | 0.0002  | 1.42        | YES         |
| PT vs RI           | auprc  | 0.0001  | 1.58        | YES         |
| PT vs FPT          | auroc  | 0.2341  | 0.38        | NO          |
```

---

## Integration Notes

1. **Checkpoint path**: Update `pretrained_checkpoint` in config to point to your best model from experiment 11

2. **Encoder architecture**: The `model_factory.py` must exactly match your `CausalAISModel` encoder structure

3. **Feature format**: DTU data needs preprocessing to match TrackFM format `[lat, lon, sog, cog_sin, cog_cos, dt]`

4. **Shared code**: If you want to avoid duplication, consider extracting `CausalAISModel` to a shared module that both experiment 11 and 12 import

---

## Quick Start

```bash
cd experiments/12_anomaly_detection

# Full pipeline
./run_all.sh pretraining_impact 100

# Or step by step
python scripts/download_data.py --output data/raw/
python scripts/preprocess_data.py --config configs/config.yaml
python scripts/run_experiment.py --exp-name pretraining_impact

# Run single condition for debugging
python scripts/run_experiment.py --exp-name test --condition pretrained --outer-fold 0
```
