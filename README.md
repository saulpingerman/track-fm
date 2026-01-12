# TrackFM: Vessel Trajectory Prediction with Transformers

This repository contains experiments on vessel trajectory prediction using causal transformers with 2D Fourier density heads. Starting with synthetic data experiments (01-08) and progressing to real AIS vessel tracking data (09-10).

## Key Results

- **Experiment 10**: Long-horizon prediction (up to 1 hour ahead) on real AIS data
  - **~40% improvement** over dead reckoning baseline (with fair sigma comparison)
  - Predicts probability distributions over 64x64 grids covering ±33km
  - Trained on 3,754 vessel tracks (23M positions) from Danish maritime zone

## Repository Structure

```
track-fm/
├── README.md                    # This file
├── .gitignore
└── experiments/
    ├── 01_fixed_dt_baseline/    # Synthetic: Fixed time horizon
    ├── 02_variable_dt_methods/  # Synthetic: Variable Δt conditioning
    ├── 03_multi_horizon/        # Synthetic: Multi-horizon prediction
    ├── 04_pointwise_nll/        # Synthetic: Loss function comparison
    ├── 05_variable_dt_loss/     # Synthetic: Loss with variable Δt
    ├── 06_relative_displacement/# Synthetic: Relative displacement grids
    ├── 07_decoder_transformer/  # Synthetic: Causal vs bidirectional
    ├── 08_causal_multihorizon/  # Synthetic: Causal + multi-horizon
    ├── 09_ais_real_data/        # Real AIS: Short-horizon (20 steps)
    └── 10_long_horizon/         # Real AIS: Long-horizon (400 steps, ~1 hour)
```

## Experiments

### Synthetic Data (01-08)

These experiments use generated trajectories to validate the transformer + Fourier head approach.

| Exp | Question | Key Finding |
|-----|----------|-------------|
| 01 | Can transformer + Fourier head learn dead reckoning? | YES with curriculum learning |
| 02 | How to incorporate variable Δt? | Concat to input is best |
| 03 | Can we predict multiple horizons efficiently? | YES, 4.7x faster than separate |
| 04 | What loss function works best? | Soft targets (σ=0.5) |
| 05 | Does loss choice hold with variable Δt? | Soft targets remain best |
| 06 | Does relative displacement change results? | σ must scale with grid range |
| 07 | Causal vs bidirectional attention? | Causal works for trajectories |
| 08 | Causal + multi-horizon together? | 99.4% MSE reduction vs baselines |

### Real AIS Data (09-10)

These experiments use real vessel tracking data from the Danish maritime zone.

#### Experiment 9: Real AIS Data (Short Horizon)
- **Data**: ~73M positions, ~7,775 valid tracks
- **Horizon**: 20 steps (~3-10 minutes)
- **Result**: 83.4% improvement over dead reckoning

#### Experiment 10: Long Horizon Prediction (Up to 1 Hour)
- **Data**: 3,754 tracks, 23M positions (filtered for moving vessels)
- **Horizon**: 400 steps (~1 hour ahead)
- **Grid**: ±33km prediction range
- **Features**: Early stopping, fair baseline comparison, random horizon sampling
- **Result**: ~40% improvement over dead reckoning (with optimized baseline sigma)

See `experiments/10_long_horizon/README.md` for full details.

## Model Architecture

```
CausalAISModel
├── Input Projection: 6 features → 128 dim
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: 4 layers, 8 heads, causal masking
├── Cumulative Time Encoding: For arbitrary horizon prediction
└── Fourier Head 2D: Outputs 64x64 probability grid
```

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

# Run experiment 10 (long horizon on real AIS data)
cd experiments/10_long_horizon
./run_all.sh my_experiment 100  # 100 epochs with early stopping

# Or run individual steps:
python run_experiment.py --exp-name my_exp --num-epochs 100
python visualize_predictions.py --exp-name my_exp
python make_horizon_videos.py --exp-name my_exp --max-horizon 400
```

## Key Learnings

1. **Random horizon sampling works** - Train on 40 random samples from 400 horizons per batch
2. **Cumulative time encoding** - Enables prediction at any horizon, including beyond training
3. **Fair baseline comparison matters** - DR baseline needs appropriate sigma (~0.05° vs 0.003°)
4. **Early stopping essential** - Validation loss plateaus quickly (4-10 checks)
5. **Large batch + chunked loss** - 8000 batch size with 512-chunk loss computation for GPU efficiency

## Hardware

- NVIDIA L40S GPU (48GB)
- ~44GB GPU memory used for experiment 10
- FSx for Lustre for AIS data storage

## Data

Real AIS data from Danish maritime zone (not included in repo):
- **Source**: FSx for Lustre linked to S3
- **Path**: `/mnt/fsx/data/`
- **Region**: 54-58.5°N, 7-16°E
- **Time**: Jan-Feb 2025
