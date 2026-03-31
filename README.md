# TrackFM: A Foundation Model for Vessel Trajectory Understanding

A decoder-only causal transformer with 2D Fourier density heads, pre-trained via next-position prediction on AIS data. Pre-trained representations transfer to downstream maritime tasks.

**Paper**: [`paper/trackfm.tex`](paper/trackfm.tex)

## Key Results

| Task | Metric | Result |
|------|--------|--------|
| Trajectory forecasting (116M model) | vs Dead Reckoning | **2.9x lower loss** |
| Trajectory forecasting (116M model) | vs Last Position | **4.2x lower loss** |
| Anomaly detection (pretrained) | AUPRC | **0.872** (vs 0.753 from scratch, +15.8%) |
| Vessel classification (pretrained) | Accuracy | **73.8%** (vs 70.8% from scratch, +3%) |

## Repository Structure

```
trackfm/
├── README.md                          # This file
├── HYPERPARAMETERS.md                 # All hyperparameters for reproducing results
├── MATERIALIZED_DATA.md               # Pre-shuffled training data format docs
├── TRAINING_HANDOFF.md                # Training data loading guide
├── paper/
│   └── trackfm.tex                    # Journal paper (LaTeX)
└── experiments/
    ├── 11_long_horizon_69_days/       # Pre-training: 69 days, 116M model
    ├── 12_anomaly_detection/          # Downstream: anomaly detection fine-tuning
    ├── 13_vessel_classification/      # Downstream: vessel classification fine-tuning
    └── 14_800_horizon_1_year/         # Extended: 1-year data, 800-step horizon
```

## Paper → Code Mapping

| Paper Section | Experiment | Key Files |
|---|---|---|
| Architecture (Sec 3) | `11_long_horizon_69_days` | `run_experiment.py` (model classes) |
| Pre-training (Sec 4-5) | `11_long_horizon_69_days` | `run_experiment.py`, `config.json` in `experiments/69days_causal_v4_100M/` |
| Scaling study (Table 1) | `11_long_horizon_69_days` | `run_scaling_study.sh` |
| Anomaly detection (Sec 6.1) | `12_anomaly_detection` | `scripts/run_bayesian_optimization.py`, `configs/config.yaml` |
| Vessel classification (Sec 6.2) | `13_vessel_classification` | `scripts/run_bayesian_optimization.py`, `configs/config.yaml` |
| Extended horizon (not in paper) | `14_800_horizon_1_year` | `run_experiment.py`, configs in `experiments/` |

## Experiments Overview

### Experiment 11: Pre-training (Paper Backbone)

The foundation model. 116M parameter causal transformer trained on 69 days of Danish maritime AIS data (13K tracks, 140M positions). This is the backbone used for all downstream tasks.

- **Code**: `experiments/11_long_horizon_69_days/run_experiment.py`
- **Config**: `experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/config.json`
- **Checkpoint**: `best_model.pt` (478MB, in download bundle)
- **Result**: 2.9x lower loss than dead reckoning, 4.2x lower than last position

### Experiment 12: Anomaly Detection (Downstream)

Binary classification of anomalous vessel behavior. Fine-tunes the Exp 11 encoder with an MLP head.

- **Code**: `experiments/12_anomaly_detection/src/` (modular: data, models, training, evaluation)
- **Config**: `experiments/12_anomaly_detection/configs/config.yaml`
- **Data**: DTU Danish Waters (521 trajectories, 25 anomalies) — included in `data/` directory
- **Bayesian opt results**: `experiments/bayesian_opt_v1/comparison.json`
- **Result**: Pretrained 0.872 AUPRC vs random 0.753 AUPRC (+15.8%)

### Experiment 13: Vessel Classification (Downstream)

4-class vessel type classification (Fishing, Cargo, Tanker, Passenger).

- **Code**: `experiments/13_vessel_classification/src/` (modular: data, models, training, evaluation)
- **Config**: `experiments/13_vessel_classification/configs/config.yaml`
- **Data**: 2,257 trajectories with labels — included in `data/processed/` directory
- **Bayesian opt results**: `experiments/bayesian_opt_v1/comparison.json`
- **Result**: Pretrained 73.8% accuracy vs random 70.8% (+3%)

### Experiment 14: Extended Horizon (Not in Paper)

Trains a separate 18M model from scratch on 1 year of AIS data with 800-step (2-hour) horizons and 128x128 grids. Does NOT use the Exp 11 backbone.

- **Code**: `experiments/14_800_horizon_1_year/run_experiment.py`
- **Configs**: `experiments/14_800_horizon_1_year/experiments/run_*/config.json`
- **Data**: 109M pre-shuffled samples from S3 (see `MATERIALIZED_DATA.md`)

## Model Architecture

```
CausalAISModel (decoder-only transformer)
├── Input Projection: Linear(6 → d_model)
│   Features: [lat, lon, sog, sin(cog), cos(cog), dt]
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: L layers, causal masking
│   Each layer: MaskedMHA → LayerNorm → FFN → LayerNorm
├── Cumulative Time Encoding: Sinusoidal time → concat with hidden state
└── Fourier Head 2D: Linear → DCT coefficients → softmax → G×G probability grid
```

### Model Scales

| Scale | d_model | Heads | Layers | FFN Dim | Params | Grid |
|-------|---------|-------|--------|---------|--------|------|
| Small | 128 | 8 | 4 | 512 | ~1M | 64×64 |
| Medium | 256 | 8 | 6 | 1024 | ~5M | 64×64 |
| Large | 384 | 16 | 8 | 2048 | ~18M | 64×64 |
| **XLarge** | **768** | **16** | **16** | **3072** | **~116M** | **64×64** |

### Feature Normalization (Exp 11)

| Feature | Transform |
|---------|-----------|
| lat | `(lat - 56.25) / 1.0` |
| lon | `(lon - 11.5) / 2.0` |
| sog | `sog / 30.0` |
| cog | `sin(cog × π/180)`, `cos(cog × π/180)` |
| dt | `dt_seconds / 300.0` |

See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for complete hyperparameter reference.

## Data

### Pre-training Data (Exp 11)

AIS data from the Danish maritime zone (54–58.5°N, 7–16°E), 69 days from Jan–Feb 2025. After filtering (SOG ≥ 0.5 knots, gap segmentation at 30min): 13,000 tracks, 140M positions. Split: 80% train, 10% val, 10% test.

Data lives on S3/FSx, not in this repo. See [`MATERIALIZED_DATA.md`](MATERIALIZED_DATA.md) and [`TRAINING_HANDOFF.md`](TRAINING_HANDOFF.md).

### Fine-tuning Datasets (Included in Repo)

**Anomaly Detection (Exp 12)**:
- Location: `experiments/12_anomaly_detection/data/`
- Source: [DTU Danish Waters AIS dataset](https://data.dtu.dk/articles/dataset/19446300)
- Format: Pickle (raw) + NumPy (processed labels)
- Size: 521 trajectories, ~10:1 class imbalance

**Vessel Classification (Exp 13)**:
- Location: `experiments/13_vessel_classification/data/processed/`
- Source: DMA Danish Maritime Authority
- Format: NumPy arrays (`trajectories.npy`, `labels.npy`)
- Size: 2,257 trajectories across 4 classes (Fishing: 717, Cargo: 624, Passenger: 645, Tanker: 271)

## Critical Finding: Hyperparameters Differ Between Pretrained and Random Init

A key finding of this work: **optimal hyperparameters are substantially different** for pretrained vs randomly-initialized models. Using the same hyperparameters for both conditions will produce misleading results (random init may appear to match or beat pretrained).

| Setting | Pretrained | Random Init |
|---------|-----------|-------------|
| Pooling | mean | last |
| Weight decay | low/zero | moderate |
| Learning rate | higher | lower |

Always tune hyperparameters separately for each condition. See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for the exact values from Bayesian optimization.

## Hardware

- NVIDIA L40S GPU (48GB)
- ~43GB GPU memory for XLarge model
- FSx for Lustre for AIS data storage
- AWS EC2 instance
