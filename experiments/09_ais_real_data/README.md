# Experiment 9: Real AIS Data - Causal Multi-Horizon Prediction

## Overview

This experiment implements a causal multi-horizon transformer model for vessel trajectory prediction using real AIS (Automatic Identification System) data from the Danish maritime zone. The model outputs probability distributions over future positions using a 2D Fourier density head.

**Key Result**: The model achieves **83.4% improvement** over dead reckoning baseline, with prediction errors of ~78 meters across all horizons (1-20 steps ahead).

## Data

- **Source**: FSx for Lustre linked to S3 bucket
- **Path**: `/mnt/fsx/data/`
- **Catalog**: `/mnt/fsx/data/track_catalog.parquet`
- **Size**: ~73 million positions, ~7,775 valid tracks
- **Region**: Danish EEZ (54-58.5°N, 7-16°E)
- **Time range**: Jan 1 - Feb 7, 2025
- **Time intervals**: NOT interpolated - real variable intervals (median ~10s, range 0s-14,359s, 2,576 unique dt values)

### Input Features (6 dimensions)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| lat | Latitude (degrees) | (x - 56.25) / 1.0 |
| lon | Longitude (degrees) | (x - 11.5) / 2.0 |
| sog | Speed over ground (knots) | x / 30.0 |
| cog_sin | sin(course over ground) | [-1, 1] |
| cog_cos | cos(course over ground) | [-1, 1] |
| dt | Time since previous position (seconds) | x / 300.0 |

### Train/Validation Split

- **Split method**: By TRACK (not by position) - ensures no data leakage
- **Ratio**: 90% train, 10% validation
- **Train tracks**: 1,383 tracks
- **Val tracks**: 154 tracks
- **No overlap**: A track is entirely in train OR validation, never both

## Model Architecture

```
CausalAISModel
├── Input Projection: 6 → 128 (Linear)
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: 4 layers, 8 heads, 512 FFN dim
│   └── Causal mask: Each position only attends to itself and past
├── Horizon Embeddings: 20 learnable embeddings
└── Fourier Head 2D: Outputs 64x64 probability grid per horizon
```

### Configuration

```python
d_model: 128
nhead: 8
num_layers: 4
dim_feedforward: 512
dropout: 0.1
grid_size: 64
grid_range: 0.02  # ±0.02° = ±2.2km prediction range
max_horizon: 20
max_seq_len: 128
batch_size: 64
learning_rate: 0.0001
```

## Training Approach: Causal Multi-Horizon

The model uses **causal training** where it predicts from EVERY position in the sequence, not just the last one.

### How It Works

For a sequence of length `seq_len` with `max_horizon` predictions:

1. **Within-sequence predictions**: For each horizon h (1 to max_horizon):
   - Position 0 predicts position h
   - Position 1 predicts position h+1
   - ...
   - Position (seq_len - h - 1) predicts position (seq_len - 1)

2. **Future predictions**: From the last input position (seq_len - 1):
   - Predict positions seq_len, seq_len+1, ..., seq_len+max_horizon-1

This generates many prediction pairs per sample, enabling efficient training.

### Loss Function

- **Type**: KL Divergence between predicted density and soft Gaussian target
- **Target**: Gaussian centered on actual future position (sigma=0.001)
- **Grid**: 64x64 over ±0.02° (±2.2km)

## Baselines

### 1. Dead Reckoning (DR)
Predicts using constant velocity extrapolation:
```python
velocity = (position[t] - position[t-1]) / dt[t]
prediction = velocity * cumulative_time_to_target
```
**Result**: Loss = 0.917

### 2. Last Position (LP)
Predicts zero displacement (vessel stays in place):
**Result**: Loss = 8.90

### 3. Random Model
Untrained model with random weights:
**Result**: Loss = 7.63

### 4. Trained Model
**Result**: Loss = 0.153 (83.4% better than DR)

## Results

### Final Metrics (Horizon 20 Run)

| Horizon | Error (km) | Time Ahead* |
|---------|------------|-------------|
| 1 | 0.078 | ~10-30s |
| 5 | 0.079 | ~50-150s |
| 10 | 0.081 | ~100-300s |
| 15 | 0.078 | ~150-450s |
| 20 | 0.078 | ~200-600s |

*Time varies per track based on actual dt values

### Key Findings

1. **Model significantly beats dead reckoning**: 83% improvement in KL divergence loss
2. **Consistent accuracy across horizons**: ~78m error from horizon 1-20
3. **Handles course changes well**: Model excels when vessels deviate from constant velocity
4. **DR wins on straight tracks**: When vessels maintain constant velocity, DR is competitive

## Files

```
09_ais_real_data/
├── run_experiment.py          # Main training script
├── visualize_predictions.py   # Generate static visualizations
├── make_horizon_videos.py     # Generate animated GIFs across horizons
├── README.md                  # This file
└── experiments/
    └── <exp-name>/            # One folder per experiment
        ├── config.json        # Experiment configuration
        ├── checkpoints/
        │   ├── best_model.pt
        │   └── checkpoint_step_*.pt
        └── results/
            ├── results.json           # Metrics and training history
            ├── training_history.png   # Loss curves (log-log scale)
            ├── viz_multiple_tracks.png
            ├── viz_single_track_horizons.png
            ├── viz_horizon_comparison.png
            ├── viz_9_tracks.png
            ├── viz_track_context_*.png
            ├── viz_max_horizon_16tracks.png
            └── horizon_video_track*.gif  # Animated predictions
```

## Running

### Training

```bash
source /opt/pytorch/bin/activate
cd "/home/ec2-user/trackfm-toy studies/experiments/09_ais_real_data"

# Basic run with default settings (creates timestamped experiment folder)
python run_experiment.py

# Named experiment with custom settings
python run_experiment.py --exp-name horizon30 --max-horizon 30 --num-epochs 10

# All available options
python run_experiment.py \
    --exp-name <name>           # Experiment name (default: timestamped)
    --max-horizon <int>         # Max prediction horizon (default: 20)
    --num-epochs <int>          # Training epochs (default: 5)
    --batch-size <int>          # Batch size (default: 64)
    --learning-rate <float>     # Learning rate (default: 0.0001)
```

### Static Visualizations

```bash
# Generate all static visualizations for an experiment
python visualize_predictions.py --exp-name horizon20
```

### Animated GIFs

```bash
# Generate horizon prediction videos (GIFs at 5fps)
python make_horizon_videos.py --exp-name horizon20

# Customize number of tracks and max horizon shown
python make_horizon_videos.py \
    --exp-name horizon20 \
    --max-horizon 40 \      # Can exceed training horizon (model is time-conditioned)
    --num-tracks 6
```

**Note**: The model can predict at arbitrary horizons beyond its training horizon because it is time-conditioned. A model trained with `max_horizon=20` can still generate predictions at horizon 40 by providing the appropriate cumulative time.

## Visualizations Explained

### viz_multiple_tracks.png
6 validation tracks showing model vs DR at horizon 3. Shows error in meters for each.

### viz_single_track_horizons.png
Single track with predictions at horizons 1, 5, 10, 15, 20. Shows how uncertainty grows with horizon.

### viz_track_context_*.png
Two-panel view: left shows full trajectory with prediction grid box, right shows zoomed prediction with heatmap. Explains why tracks appear "straight" in zoomed view (grid is 2.2km, tracks are 3-8km).

### viz_horizon_comparison.png
Model predictions vs DR across all 20 horizons for a single track.

### viz_9_tracks.png
9 random validation tracks with Model vs DR error comparison.

### viz_max_horizon_16tracks.png
16 tracks showing predictions at the maximum training horizon. Shows model performance at the furthest prediction point.

### horizon_video_track*.gif
Animated GIFs showing how predictions evolve across horizons (1 to 40). Each frame shows:
- Heatmap of predicted probability distribution
- Ground truth position (green star)
- Dead reckoning prediction (red X)
- Model's expected position (cyan triangle)
- Horizon number and forecast time in title
- Error comparison (Model vs DR in meters)

The color scale is fixed across all frames for consistent comparison.

## Important Implementation Details

### 1. Validation Loss Calculation
Training and validation MUST compute loss identically - on ALL prediction pairs (within-sequence + future), not just future predictions. Mismatch causes validation to appear artificially better than training.

### 2. Dead Reckoning Baseline
Must use proper physics:
```python
velocity = position_diff / time_diff  # degrees per second
prediction = velocity * cumulative_time_to_horizon
```
NOT: `position_diff * horizon_number` (wrong - ignores variable time intervals)

### 3. Fourier Head Output
The density output has shape `[batch, num_predictions, grid_size, grid_size]` where axes are `(delta_lat, delta_lon)`. No transpose needed for imshow with `origin='lower'`.

### 4. Time Labels in Visualization
Use actual cumulative time from data, not hardcoded values:
```python
dt_seconds = features[:, 5] * config.dt_max  # Denormalize
cumulative_time = np.cumsum(dt_seconds[seq_len:])
```

## Lessons Learned

1. **Variable time intervals matter**: Real AIS data has highly variable dt (0s to hours). Must use actual time for velocity calculations and predictions.

2. **Train/val split by track, not position**: Splitting by position causes data leakage - model sees parts of same track in both sets.

3. **Causal training is data-efficient**: Predicting from every position generates many training pairs per sequence.

4. **Grid range vs track extent**: 2.2km grid << 3-8km track extent. Predictions look "local" but model sees full trajectory context.

5. **Batch size tradeoff**: max_horizon=20 requires 4x more output than max_horizon=5. Reduced batch from 256 to 64 to avoid OOM.

## Future Work

- [x] ~~Add timestamped result directories for each run~~ (Done: experiments/<exp-name>/ structure)
- [x] ~~Add animated visualizations~~ (Done: make_horizon_videos.py generates GIFs)
- [ ] Increase grid_range for longer-range predictions
- [ ] Experiment with longer sequences (currently 128)
- [ ] Add uncertainty calibration analysis
- [ ] Test on different vessel types (cargo vs tanker vs passenger)
- [ ] Add track selection by curvature (find tracks with interesting maneuvers)
