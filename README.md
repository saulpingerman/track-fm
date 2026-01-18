# TrackFM: Vessel Trajectory Prediction with Transformers

![Vessel Trajectory Prediction](experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/results/horizon_video_track100.gif)

*116M parameter model predicting vessel position up to 1 hour ahead. Yellow/blue heatmap shows predicted probability distribution, green dot is actual position, red X is dead reckoning baseline.*

This repository contains experiments on vessel trajectory prediction using causal transformers with 2D Fourier density heads. Starting with synthetic data experiments (01-08) and progressing to real AIS vessel tracking data (09-11).

## Key Results

- **Experiment 01.5**: Synthetic data validation (linear tracks)
  - **54% improvement** over dead reckoning with velocity features
  - **34% improvement** without velocity features (model infers velocity from positions)
  - Validates architecture before real data testing

- **Experiment 10**: Long-horizon prediction (up to 1 hour ahead) on real AIS data
  - **44% improvement** over dead reckoning baseline (large model, 18M params)
  - **63% improvement** over last position baseline
  - Predicts probability distributions over 64x64 grids covering +/-33km
  - Trained on 3,754 vessel tracks (23M positions) from Danish maritime zone

- **Experiment 11**: Extended training with 69 days of data + 116M parameter model
  - **65% improvement** over dead reckoning (116M param model)
  - **76% improvement** over last position baseline
  - 10x more data than Experiment 10 (13K tracks, 140M positions after filtering)
  - Causal subwindow training: ~30x more training signal per batch
  - Leak test verified: model makes genuine predictions, no data leakage

- **Experiment 12**: Anomaly detection transfer learning
  - Fine-tuned 116M encoder on DTU Danish Waters anomaly dataset (521 trajectories, 25 anomalies)
  - **Pretrained encoder achieves 0.498 AUPRC** (~10x better than random baseline of 0.05)
  - Fine-tuning significantly outperforms frozen encoder (p=0.036)
  - Pre-training vs random initialization: inconclusive due to small dataset

- **Experiment 13**: Vessel type classification
  - 4-class classification: Fishing, Cargo, Tanker, Passenger (2,257 trajectories)
  - **Random initialization (68.6%) outperforms pretrained (66.3%)**
  - Tested multiple transfer techniques: attention pooling, MHA pooling, LoRA, two-stage fine-tuning
  - **Finding**: Pre-training on trajectory forecasting provides no benefit for classification

## Repository Structure

```
track-fm/
├── README.md                    # This file
├── .gitignore
└── experiments/
    ├── 01_fixed_dt_baseline/    # Synthetic: Fixed time horizon
    ├── 01.5_dm_data/            # Synthetic: Linear track validation
    ├── 02_variable_dt_methods/  # Synthetic: Variable dt conditioning
    ├── 03_multi_horizon/        # Synthetic: Multi-horizon prediction
    ├── 04_pointwise_nll/        # Synthetic: Loss function comparison
    ├── 05_variable_dt_loss/     # Synthetic: Loss with variable dt
    ├── 06_relative_displacement/# Synthetic: Relative displacement grids
    ├── 07_decoder_transformer/  # Synthetic: Causal vs bidirectional
    ├── 08_causal_multihorizon/  # Synthetic: Causal + multi-horizon
    ├── 09_ais_real_data/        # Real AIS: Short-horizon (20 steps)
    ├── 10_long_horizon/         # Real AIS: Long-horizon (400 steps, ~1 hour)
    ├── 11_long_horizon_69_days/ # Real AIS: 69 days + causal training
    ├── 12_anomaly_detection/    # Transfer learning: Anomaly detection fine-tuning
    └── 13_vessel_classification/# Transfer learning: Vessel type classification
```

## Experiments

### Synthetic Data (01-08, 01.5)

These experiments use generated trajectories to validate the transformer + Fourier head approach.

| Exp | Question | Key Finding |
|-----|----------|-------------|
| 01 | Can transformer + Fourier head learn dead reckoning? | YES with curriculum learning |
| **01.5** | **Does architecture work on linear tracks?** | **YES, +54% vs DR with velocity, +34% without** |
| 02 | How to incorporate variable dt? | Concat to input is best |
| 03 | Can we predict multiple horizons efficiently? | YES, 4.7x faster than separate |
| 04 | What loss function works best? | Soft targets (sigma=0.5) |
| 05 | Does loss choice hold with variable dt? | Soft targets remain best |
| 06 | Does relative displacement change results? | sigma must scale with grid range |
| 07 | Causal vs bidirectional attention? | Causal works for trajectories |
| 08 | Causal + multi-horizon together? | 99.4% MSE reduction vs baselines |

#### Experiment 01.5: Synthetic Linear Track Validation

Validates the architecture on constant-velocity linear tracks before testing on real data.

| Variant | Features | vs Dead Reckoning |
|---------|----------|-------------------|
| With velocity | `[lat, lon, sog, cog, dt]` | **+53.9%** |
| No velocity (dt) | `[lat, lon, dt]` | **+33.6%** |
| No velocity (raw t) | `[lat, lon, t]` | **+29.3%** |

Key insight: Model genuinely learns trajectory patterns - it can beat dead reckoning even without explicit velocity features.

See `experiments/01.5_dm_data/README.md` for full details.

### Real AIS Data (09-11)

These experiments use real vessel tracking data from the Danish maritime zone.

#### Experiment 9: Real AIS Data (Short Horizon)
- **Data**: ~73M positions, ~7,775 valid tracks
- **Horizon**: 20 steps (~3-10 minutes)
- **Result**: 83.4% improvement over dead reckoning

#### Experiment 10: Long Horizon Prediction (Up to 1 Hour)
- **Data**: 3,754 tracks, 23M positions (filtered for moving vessels)
- **Horizon**: 400 steps (~1 hour ahead)
- **Grid**: +/-33km prediction range
- **Model sizes**: Small (1M), Medium (5M), Large (18M params)
- **Features**: Early stopping, fair baseline comparison, random horizon sampling, auto batch size
- **Result**: 44% improvement over DR, 63% over LP (large model)

See `experiments/10_long_horizon/README.md` for full details.

#### Experiment 11: 69 Days + 116M Parameter Model
- **Data**: 69 days (13K tracks, 140M positions after filtering)
- **Model**: 116M parameters (xlarge scale)
- **Result**: **65% improvement** over DR, **76% over LP**
- **Training**: Causal subwindow training - ~30x more training signal per batch
- **Verification**: Leak test confirms model makes genuine predictions

See `experiments/11_long_horizon_69_days/README.md` for full details.

#### Experiment 12: Anomaly Detection Transfer Learning
- **Task**: Binary classification (normal vs anomalous vessel behavior)
- **Data**: DTU Danish Waters dataset (521 trajectories, 25 labeled anomalies)
- **Model**: 116M encoder from Exp 11 + MLP classifier head
- **Conditions**: pretrained (fine-tune all), random_init (from scratch), frozen_pretrained (freeze encoder)
- **Result**: Pretrained achieves **0.498 AUPRC** (10x better than random baseline)
- **Finding**: Fine-tuning significantly outperforms frozen encoder (p=0.036); pre-training vs random init inconclusive

See `experiments/12_anomaly_detection/README.md` for full details.

#### Experiment 13: Vessel Type Classification
- **Task**: 4-class classification (Fishing, Cargo, Tanker, Passenger)
- **Data**: DMA AIS data (2,257 trajectories from same domain as pre-training)
- **Model**: 116M encoder from Exp 11 + MLP classifier head
- **Conditions**: pretrained, random_init, frozen_pretrained, two_stage, + pooling ablations + LoRA
- **Result**: Random init achieves **68.6% accuracy**, outperforming best pretrained (65.7%)
- **Finding**: Pre-training on trajectory forecasting provides no benefit for vessel classification

**Key Results:**

| Condition | Accuracy | F1 Macro |
|-----------|----------|----------|
| **random_init** | **68.6%** | **0.56** |
| pretrained_attention | 65.7% | 0.57 |
| pretrained | 66.3% | 0.55 |
| pretrained_lora | 63.4% | 0.52 |

**Why No Transfer Benefit:**
1. Sequence length mismatch (128 vs 512 positions)
2. Task mismatch (local prediction vs global classification)
3. Class similarity (Cargo/Tanker nearly indistinguishable)

See `experiments/13_vessel_classification/README.md` for full details.

## Model Architecture

```
CausalAISModel
├── Input Projection: 6 features -> d_model
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: num_layers layers, nhead heads, causal masking
├── Cumulative Time Encoding: For arbitrary horizon prediction
└── Fourier Head 2D: Outputs 64x64 probability grid
```

### Model Scales (Experiment 10/11)

| Scale | d_model | nhead | layers | dim_ff | Params |
|-------|---------|-------|--------|--------|--------|
| small | 128 | 8 | 4 | 512 | ~1M |
| medium | 256 | 8 | 6 | 1024 | ~5M |
| large | 384 | 16 | 8 | 2048 | ~18M |
| **xlarge** | **768** | **16** | **16** | **3072** | **~116M** |

### Input Features

| Feature | Description |
|---------|-------------|
| lat, lon | Position (normalized) |
| sog | Speed over ground |
| cog_sin, cog_cos | Course over ground |
| dt | Time since previous position |

## Quick Start

```bash
# Activate environment
source /opt/pytorch/bin/activate

# Run experiment 01.5 (synthetic data validation)
cd experiments/01.5_dm_data
./run_all.sh my_exp 5  # 5 epochs

# Run experiment 10 with large model (recommended)
cd experiments/10_long_horizon
python run_experiment.py --exp-name my_exp --model-scale large --batch-size 0 --num-epochs 100
python visualize_predictions.py --exp-name my_exp
python make_horizon_videos.py --exp-name my_exp --max-horizon 400

# Run experiment 11 with causal training (69 days of data)
cd experiments/11_long_horizon_69_days
python run_experiment.py --exp-name my_exp --model-scale large --batch-size 0 --num-epochs 100 --num-horizons 8

# Or use run_all.sh for full pipeline:
./run_all.sh my_experiment 100  # 100 epochs with early stopping
```

## Key Learnings

1. **Model learns genuine patterns** - Not just memorizing velocity; beats DR even without explicit velocity features
2. **Random horizon sampling works** - Train on random samples from 400 horizons per batch
3. **Larger models help significantly** - 116M params achieves 65% vs 44% improvement over 18M params
4. **Auto batch size** - Binary search for ~90% GPU utilization maximizes training efficiency
5. **Cumulative time encoding** - Enables prediction at any horizon, including beyond training
6. **Fair baseline comparison matters** - DR baseline needs appropriate sigma (~0.05 deg vs 0.003 deg)
7. **Early stopping essential** - Validation loss plateaus quickly (4-10 checks)
8. **Causal subwindow training** - Uses all positions in sequence for ~30x more training signal
9. **Leak testing is crucial** - Verified model doesn't cheat by seeing future positions

## Hardware

- NVIDIA L40S GPU (48GB)
- ~43GB GPU memory used for large model (90% utilization)
- FSx for Lustre for AIS data storage

## Data

Real AIS data from Danish maritime zone (not included in repo):
- **Source**: FSx for Lustre linked to S3
- **Path**: `/mnt/fsx/data/`
- **Region**: 54-58.5N, 7-16E
- **Time**: Jan-Feb 2025
