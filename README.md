# TrackFM: Vessel Trajectory Prediction with Transformers

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
  - Prediction error: 0.9 km at 17 min, 12 km at 1 hour

- **Experiment 11**: Extended training with 69 days of data + causal subwindow training
  - 10x more data than Experiment 10 (~580M positions, ~185K tracks)
  - Causal subwindow training: predicts from all positions, not just the last one
  - ~30x more training signal per batch

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
    └── 11_long_horizon_69_days/ # Real AIS: 69 days + causal training
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

#### Experiment 11: 69 Days + Causal Subwindow Training
- **Data**: 69 days (~580M positions, ~185K tracks)
- **Training**: Causal subwindow training - predicts from all positions, not just last
- **Benefits**: ~30x more training signal per batch, learns with varying context lengths
- **Features**: Lazy loading option for memory efficiency

See `experiments/11_long_horizon_69_days/README.md` for full details.

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
3. **Larger models help** - 18M param model significantly outperforms 1M param model
4. **Auto batch size** - Binary search for ~90% GPU utilization maximizes training efficiency
5. **Cumulative time encoding** - Enables prediction at any horizon, including beyond training
6. **Fair baseline comparison matters** - DR baseline needs appropriate sigma (~0.05 deg vs 0.003 deg)
7. **Early stopping essential** - Validation loss plateaus quickly (4-10 checks)
8. **Causal subwindow training** - Uses all positions in sequence for ~30x more training signal

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
