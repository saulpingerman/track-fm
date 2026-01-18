# Experiment 13: Vessel Type Classification (Behavior Recognition)

## Overview

This experiment evaluates pre-training impact on **vessel type classification** — a behavior recognition task with **thousands of labeled examples** directly from your DMA pre-training data.

**Why this is better than anomaly detection (Exp 12):**
- Your DMA data already has `Ship type` labels (Fishing=30, Passenger=60, Cargo=70, Tanker=80, etc.)
- Thousands of vessels per class vs 25-55 anomalies
- Same domain as pre-training (Danish waters) — no distribution shift
- Well-studied benchmark task with known baselines

**Task**: Classify vessel trajectories into types based on movement patterns, ignoring the self-reported type field.

---

## Available Datasets

### Option 1: DMA Danish Maritime Authority (RECOMMENDED)

**Your pre-training data already has labels!**

From the DMA CSV format:
```
Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,ROT,SOG,COG,Heading,IMO,Callsign,Name,Ship type,...
```

The `Ship type` field contains AIS type codes:
- **30** = Fishing
- **60-69** = Passenger
- **70-79** = Cargo
- **80-89** = Tanker
- **31-32** = Towing
- **36-37** = Sailing/Pleasure
- **50-59** = Special craft (SAR, tugs, pilots, etc.)

**Expected counts** (based on literature using DMA data):
- Cargo: ~2M+ records, thousands of vessels
- Tanker: ~2.8M+ records
- Fishing: ~2M+ records  
- Passenger: ~8M+ records

This gives you **orders of magnitude more labeled data** than the anomaly detection task.

### Option 2: Global Fishing Watch Training Data

- GitHub: https://github.com/GlobalFishingWatch/training-data
- Point-level fishing/not-fishing labels
- ~1000+ labeled vessel tracks
- Good for fishing behavior detection specifically

### Option 3: MarineCadastre (US Waters)

- https://marinecadastre.gov/ais/
- Similar ship type codes
- Good for testing cross-domain transfer (train on Danish, test on US)

---

## Experimental Design

### Task Formulation

**Input**: Trajectory sequence `[lat, lon, sog, cog_sin, cog_cos, dt]` (same as pre-training)

**Output**: Vessel type class (4-class or 5-class)

**Key insight**: We withhold the `Ship type` field from the model — it must infer type purely from movement patterns.

### Classes

**4-class version** (cleaner separation):
1. Fishing (code 30)
2. Cargo (codes 70-79)
3. Tanker (codes 80-89)
4. Passenger (codes 60-69)

**5-class version** (adds challenge):
5. Other/Special (codes 31-59, 90+)

### Why This Tests Pre-training

Different vessel types have **distinct movement signatures**:
- **Fishing**: Irregular patterns, circling, speed changes, concentrated in fishing grounds
- **Cargo**: Direct routes between ports, consistent speed, follows shipping lanes
- **Tanker**: Similar to cargo but different speed profiles, wider turns
- **Passenger**: Regular schedules, ferry routes, may have multiple stops

A model pre-trained on trajectory forecasting has learned these motion patterns. The hypothesis is that these representations transfer to classification.

---

## Data Splits Strategy

Since you have abundant data, use proper train/val/test splits:

```
Total vessels per class: ~1000+ each (after filtering)

Split by MMSI (no vessel appears in multiple splits):
- Train: 70% of vessels (~700 per class minimum)
- Val: 15% of vessels (~150 per class)
- Test: 15% of vessels (~150 per class)
```

**Stratified sampling** ensures class balance.

**No cross-validation needed** — you have enough data for held-out test set.

---

## Experiment Structure

```
experiments/13_vessel_classification/
├── README.md
├── run_all.sh
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                        # Symlink to /mnt/fsx/data/
│   ├── processed/
│   │   ├── trajectories.pkl        # Extracted trajectories with labels
│   │   └── splits.json             # Train/val/test MMSI lists
│   └── stats/
│       └── class_distribution.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extract_labeled_trajectories.py  # Extract from DMA CSVs
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py           # Multi-class classification head
│   │   └── factory.py              # PT/RI/FPT conditions
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # Accuracy, F1, confusion matrix
│       └── analysis.py             # Per-class analysis
├── scripts/
│   ├── extract_data.py
│   ├── run_experiment.py
│   ├── visualize_results.py
│   └── compare_conditions.py
├── experiments/
│   └── <exp_name>/
│       ├── config.yaml
│       ├── checkpoints/
│       ├── logs/
│       └── results/
└── requirements.txt
```

---

## Configuration

```yaml
# =============================================================================
# Experiment 13: Vessel Type Classification
# =============================================================================

experiment:
  name: "vessel_classification"
  seed: 42
  device: "cuda"

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
data:
  # Path to DMA data (same as pre-training)
  raw_path: "/mnt/fsx/data/"
  processed_path: "data/processed/"
  
  # Date range (use subset of your 69-day pre-training data)
  start_date: "2025-01-01"
  end_date: "2025-01-15"  # 2 weeks should give plenty of vessels
  
  # Class mapping
  class_mapping:
    30: "fishing"        # Fishing
    60: "passenger"      # Passenger (60-69)
    61: "passenger"
    62: "passenger"
    63: "passenger"
    64: "passenger"
    65: "passenger"
    66: "passenger"
    67: "passenger"
    68: "passenger"
    69: "passenger"
    70: "cargo"          # Cargo (70-79)
    71: "cargo"
    72: "cargo"
    73: "cargo"
    74: "cargo"
    75: "cargo"
    76: "cargo"
    77: "cargo"
    78: "cargo"
    79: "cargo"
    80: "tanker"         # Tanker (80-89)
    81: "tanker"
    82: "tanker"
    83: "tanker"
    84: "tanker"
    85: "tanker"
    86: "tanker"
    87: "tanker"
    88: "tanker"
    89: "tanker"
  
  # Classes to use
  classes: ["fishing", "cargo", "tanker", "passenger"]
  num_classes: 4
  
  # Filtering
  min_trajectory_length: 50       # Minimum points per trajectory
  max_trajectory_length: 512      # Match pre-training
  min_vessels_per_class: 200      # Ensure enough data
  
  # Features (same as pre-training)
  features: ["lat", "lon", "sog", "cog_sin", "cog_cos", "dt"]
  
  # Splits
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  
  # Sample size per class (for balanced training)
  max_samples_per_class: 2000     # Limit to prevent class imbalance issues
  balance_classes: true

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
model:
  # Encoder: MUST match experiment 11 architecture
  encoder:
    d_model: 768
    nhead: 16
    num_layers: 16
    dim_feedforward: 3072
    dropout: 0.1
    max_seq_length: 512
    input_features: 6
  
  # Classification head
  classifier:
    hidden_dims: [384, 128]
    dropout: 0.3
    num_classes: 4
    pooling: "mean"  # mean, max, last, attention
  
  # Pre-trained weights
  pretrained_checkpoint: "../11_long_horizon_69_days/experiments/69days_causal_v4_100M/checkpoints/best_model.pt"

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
training:
  # Conditions to compare
  conditions:
    - "pretrained"           # Load exp 11 weights, fine-tune all
    - "random_init"          # Random init, train from scratch
    - "frozen_pretrained"    # Load exp 11 weights, freeze encoder
  
  # Also test with limited data (ablation)
  limited_data_fractions: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]  # % of training data
  
  # Optimizer
  optimizer: "adamw"
  weight_decay: 0.01
  
  # Learning rates
  encoder_lr: 1.0e-5
  classifier_lr: 1.0e-3
  random_init_lr: 1.0e-4
  
  # Scheduler
  scheduler: "cosine"
  warmup_epochs: 3
  
  # Training
  max_epochs: 50
  batch_size: 64
  early_stopping_patience: 10
  early_stopping_metric: "val_f1_macro"
  
  # Loss
  loss: "cross_entropy"
  label_smoothing: 0.1

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
evaluation:
  metrics:
    - "accuracy"
    - "f1_macro"
    - "f1_weighted"
    - "precision_macro"
    - "recall_macro"
  
  # Per-class metrics
  per_class_metrics: true
  
  # Confusion matrix
  generate_confusion_matrix: true
  
  # Learning curves (performance vs data size)
  generate_learning_curves: true
  
  # t-SNE visualization
  generate_embeddings_plot: true
```

---

## Key Implementation: Data Extraction

```python
# src/data/extract_labeled_trajectories.py
"""
Extract labeled trajectories from DMA CSV files.

The key insight: DMA data has Ship type field which we use as labels,
but we withhold this field from the model input.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

def extract_trajectories_with_labels(
    data_dir: str,
    start_date: str,
    end_date: str,
    class_mapping: dict,
    config: dict
) -> tuple:
    """
    Extract trajectories from DMA CSVs with vessel type labels.
    
    Returns:
        trajectories: List of trajectory arrays (N, 6) for [lat, lon, sog, cog_sin, cog_cos, dt]
        labels: List of class indices
        mmsis: List of MMSI identifiers
    """
    
    data_path = Path(data_dir)
    
    # Find all CSV files in date range
    csv_files = sorted(data_path.glob("aisdk-*.csv"))
    
    # Filter by date range
    # Filename format: aisdk-YYYY-MM-DD.csv
    selected_files = []
    for f in csv_files:
        date_str = f.stem.replace("aisdk-", "")
        if start_date <= date_str <= end_date:
            selected_files.append(f)
    
    print(f"Processing {len(selected_files)} files...")
    
    # Collect data by MMSI
    vessel_data = defaultdict(list)
    vessel_types = {}
    
    for csv_file in selected_files:
        print(f"  Loading {csv_file.name}...")
        
        # Read CSV
        df = pd.read_csv(csv_file, usecols=[
            'Timestamp', 'MMSI', 'Latitude', 'Longitude', 
            'SOG', 'COG', 'Ship type'
        ])
        
        # Filter to valid ship types
        df = df[df['Ship type'].isin(class_mapping.keys())]
        
        # Drop invalid coordinates
        df = df[(df['Latitude'].between(-90, 90)) & 
                (df['Longitude'].between(-180, 180))]
        
        # Drop invalid SOG/COG
        df = df[(df['SOG'] >= 0) & (df['SOG'] < 50)]  # Max 50 knots
        df = df[(df['COG'] >= 0) & (df['COG'] <= 360)]
        
        # Parse timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Group by MMSI and collect
        for mmsi, group in df.groupby('MMSI'):
            vessel_data[mmsi].append(group)
            
            # Store ship type (take mode in case of conflicts)
            ship_type = group['Ship type'].mode().iloc[0]
            vessel_types[mmsi] = ship_type
    
    # Combine data for each vessel
    print("Combining vessel trajectories...")
    
    trajectories = []
    labels = []
    mmsis = []
    
    class_to_idx = {cls: idx for idx, cls in enumerate(config['classes'])}
    
    for mmsi, data_chunks in vessel_data.items():
        # Concatenate all data for this vessel
        vessel_df = pd.concat(data_chunks).sort_values('Timestamp')
        
        # Skip if too short
        if len(vessel_df) < config['min_trajectory_length']:
            continue
        
        # Get ship type
        ship_type = vessel_types[mmsi]
        class_name = class_mapping.get(ship_type)
        
        if class_name not in class_to_idx:
            continue
        
        # Extract features
        lat = vessel_df['Latitude'].values
        lon = vessel_df['Longitude'].values
        sog = vessel_df['SOG'].values
        cog = vessel_df['COG'].values
        timestamps = vessel_df['Timestamp'].values
        
        # Compute dt (time since previous point, in seconds)
        dt = np.zeros(len(timestamps))
        dt[1:] = np.diff(timestamps).astype('timedelta64[s]').astype(float)
        dt[0] = dt[1] if len(dt) > 1 else 0
        
        # Convert COG to sin/cos
        cog_rad = np.deg2rad(cog)
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)
        
        # Normalize lat/lon (relative to trajectory center)
        lat_center = lat.mean()
        lon_center = lon.mean()
        lat_norm = lat - lat_center
        lon_norm = lon - lon_center
        
        # Normalize SOG (0-50 knots -> 0-1)
        sog_norm = sog / 50.0
        
        # Normalize dt (clip and scale)
        dt_clipped = np.clip(dt, 0, 600)  # Max 10 minutes between points
        dt_norm = dt_clipped / 600.0
        
        # Stack features
        features = np.stack([
            lat_norm, lon_norm, sog_norm, cog_sin, cog_cos, dt_norm
        ], axis=1)
        
        # Truncate or split if too long
        max_len = config['max_trajectory_length']
        if len(features) > max_len:
            # Take multiple non-overlapping windows
            n_windows = len(features) // max_len
            for i in range(n_windows):
                start = i * max_len
                end = start + max_len
                trajectories.append(features[start:end])
                labels.append(class_to_idx[class_name])
                mmsis.append(mmsi)
        else:
            trajectories.append(features)
            labels.append(class_to_idx[class_name])
            mmsis.append(mmsi)
    
    return trajectories, labels, mmsis


def create_balanced_splits(
    trajectories: list,
    labels: list,
    mmsis: list,
    config: dict
) -> dict:
    """
    Create train/val/test splits, balanced by class.
    Split by MMSI to prevent data leakage.
    """
    
    # Group by class and MMSI
    class_vessels = defaultdict(set)
    for label, mmsi in zip(labels, mmsis):
        class_vessels[label].add(mmsi)
    
    # Split vessels within each class
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_idx, vessels in class_vessels.items():
        vessels = list(vessels)
        np.random.shuffle(vessels)
        
        n = len(vessels)
        n_train = int(n * config['train_ratio'])
        n_val = int(n * config['val_ratio'])
        
        train_vessels = set(vessels[:n_train])
        val_vessels = set(vessels[n_train:n_train + n_val])
        test_vessels = set(vessels[n_train + n_val:])
        
        # Assign trajectories to splits
        for i, (traj, label, mmsi) in enumerate(zip(trajectories, labels, mmsis)):
            if label != class_idx:
                continue
            if mmsi in train_vessels:
                splits['train'].append(i)
            elif mmsi in val_vessels:
                splits['val'].append(i)
            else:
                splits['test'].append(i)
    
    return splits
```

---

## Expected Results

With thousands of examples per class, you should see:

### Baseline Performance (Random Init)

Based on literature:
- **Accuracy**: 82-87%
- **F1 Macro**: 0.80-0.85
- Cargo/Tanker confusion is common (similar movement patterns)
- Fishing is usually well-separated

### Pre-trained Performance

Expected improvement:
- **Accuracy**: 88-93% (+5-8%)
- **F1 Macro**: 0.86-0.91
- Better separation of Cargo/Tanker
- Faster convergence

### Key Ablations

1. **Learning Curves**: Plot accuracy vs % training data
   - Pre-trained should dominate at low data (1%, 5%, 10%)
   - Gap should narrow at 100% data

2. **Convergence Speed**: Pre-trained should reach 90% of final accuracy in ~5 epochs vs ~15 epochs for random init

3. **Per-Class Analysis**: Which classes benefit most from pre-training?

---

## Results Table Template

```
=============================================================
Vessel Type Classification Results
=============================================================

| Condition          | Accuracy | F1 Macro | F1 Weighted |
|--------------------|----------|----------|-------------|
| Pre-trained (PT)   | 0.XX     | 0.XX     | 0.XX        |
| Random Init (RI)   | 0.XX     | 0.XX     | 0.XX        |
| Frozen PT (FPT)    | 0.XX     | 0.XX     | 0.XX        |
| Δ (PT - RI)        | +0.XX    | +0.XX    | +0.XX       |

Per-Class F1 Scores:
| Class     | PT   | RI   | Δ     |
|-----------|------|------|-------|
| Fishing   | 0.XX | 0.XX | +0.XX |
| Cargo     | 0.XX | 0.XX | +0.XX |
| Tanker    | 0.XX | 0.XX | +0.XX |
| Passenger | 0.XX | 0.XX | +0.XX |

Learning Curve (Accuracy at different training data %):
| Data %  | PT   | RI   | Δ     |
|---------|------|------|-------|
| 1%      | 0.XX | 0.XX | +0.XX |
| 5%      | 0.XX | 0.XX | +0.XX |
| 10%     | 0.XX | 0.XX | +0.XX |
| 25%     | 0.XX | 0.XX | +0.XX |
| 50%     | 0.XX | 0.XX | +0.XX |
| 100%    | 0.XX | 0.XX | +0.XX |
```

---

## Quick Start

```bash
cd experiments/13_vessel_classification

# Extract labeled trajectories from DMA data
python scripts/extract_data.py --config configs/config.yaml

# Check class distribution
python scripts/analyze_data.py --config configs/config.yaml

# Run full experiment
./run_all.sh vessel_class_exp 50

# Or step by step
python scripts/run_experiment.py --exp-name test --condition pretrained
python scripts/run_experiment.py --exp-name test --condition random_init
python scripts/compare_conditions.py --exp-name test
```

---

## Why This Will Show Significance

1. **Statistical power**: 1000+ samples per class vs 25 anomalies
2. **Same domain**: Danish waters for both pre-training and fine-tuning
3. **Clear class separation**: Fishing vessels move very differently than cargo
4. **Well-studied task**: Can compare to published baselines (82-87% accuracy)
5. **Learning curves**: Can demonstrate pre-training helps most with limited data

---

## Comparison to Experiment 12

| Aspect | Exp 12 (Anomaly) | Exp 13 (Classification) |
|--------|------------------|-------------------------|
| Positive examples | ~25-55 | 1000+ per class |
| Classes | 2 (binary) | 4 (multi-class) |
| Domain | DTU (different) | DMA (same as pre-train) |
| Expected Δ | Small (not significant) | Large (significant) |
| Statistical power | Low | High |
