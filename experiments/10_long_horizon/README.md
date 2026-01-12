# Experiment 10: Long Horizon Prediction (Up to 1 Hour)

## Overview

This experiment extends Experiment 9's causal multi-horizon transformer to predict vessel trajectories up to **1 hour into the future** (400 horizons vs 20). To make training with 400 horizons computationally feasible, we use **random horizon sampling** - selecting 40 horizons uniformly from 1-400 per batch rather than computing all horizons.

**Key Result**: The model achieves **68.6% improvement** over dead reckoning baseline at 5 epochs, with prediction errors of ~12.75 km at horizon 400 (~1 hour ahead).

## Key Differences from Experiment 9

| Aspect | Experiment 9 | Experiment 10 |
|--------|-------------|---------------|
| Max horizon | 20 | **400** |
| Horizon sampling | All horizons | **40 random samples per batch** |
| Grid range | ±0.02° (~2.2km) | **±0.3° (~33km)** |
| Batch size | 64 | **8000** |
| GPU memory | ~8GB | **~44GB (91% of 48GB L40S)** |
| Prediction range | ~3-10 minutes | **~1 hour** |

## Data

Same as Experiment 9:
- **Source**: FSx for Lustre linked to S3 bucket
- **Path**: `/mnt/fsx/data/`
- **Catalog**: `/mnt/fsx/data/track_catalog.parquet`
- **Size**: ~73 million positions, ~7,775 valid tracks
- **Region**: Danish EEZ (54-58.5°N, 7-16°E)
- **Time range**: Jan 1 - Feb 7, 2025
- **Time intervals**: Variable (median ~10s)

### Input Features (6 dimensions)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| lat | Latitude (degrees) | (x - 56.25) / 1.0 |
| lon | Longitude (degrees) | (x - 11.5) / 2.0 |
| sog | Speed over ground (knots) | x / 30.0 |
| cog_sin | sin(course over ground) | [-1, 1] |
| cog_cos | cos(course over ground) | [-1, 1] |
| dt | Time since previous position (seconds) | x / 300.0 |

## Model Architecture

```
CausalAISModel
├── Input Projection: 6 → 128 (Linear)
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: 4 layers, 8 heads, 512 FFN dim
│   └── Causal mask: Each position only attends to itself and past
├── Cumulative Time Encoding: Sinusoidal (enables arbitrary horizon prediction)
└── Fourier Head 2D: Outputs 64x64 probability grid per sampled horizon
```

### Configuration

```python
d_model: 128
nhead: 8
num_layers: 4
dim_feedforward: 512
dropout: 0.1
grid_size: 64
grid_range: 0.3           # ±0.3° = ±33km prediction range (15x larger than exp 9)
max_horizon: 400          # Up to 1 hour ahead
num_horizon_samples: 40   # Sample 40 horizons per batch
max_seq_len: 128
batch_size: 8000          # Large batch for GPU efficiency
learning_rate: 0.0003
use_amp: True             # Mixed precision for memory efficiency
sigma: 0.003              # Soft target sigma for model
dr_sigma: 0.05            # Baseline sigma (optimized on training data)
early_stop_patience: 4    # Stop after 4 val checks with no improvement
```

## Training Approach: Random Horizon Sampling

### Why Random Sampling?

Computing predictions for all 400 horizons would require 20x more memory than experiment 9. Instead, we:

1. **Sample uniformly**: Each batch samples 40 horizons uniformly from [1, 400]
2. **Different horizons per batch**: Each batch sees different random horizons
3. **Cumulative time encoding**: Model learns to predict at ANY horizon via time conditioning

### How It Works

```python
def forward_train(self, x, horizon_indices):
    """
    Args:
        x: Input features [batch, seq_len, 6]
        horizon_indices: Tensor of horizon values to predict [num_horizon_samples]

    Returns:
        log_densities: [batch * (seq_len - max_sampled_h), num_horizon_samples, 64, 64]
        targets: [batch * (seq_len - max_sampled_h), num_horizon_samples, 2]
        horizon_indices: The input horizon indices (for loss computation)
    """
    # For each sampled horizon h, predict from positions 0...(seq_len-h-1)
    # Uses cumulative time to horizon for time-conditioned prediction
```

### Chunked Loss Computation

To avoid OOM when computing soft target KL divergence:

```python
chunk_size = 512  # Process predictions in chunks
for start_idx in range(0, num_predictions, chunk_size):
    end_idx = min(start_idx + chunk_size, num_predictions)
    chunk_targets = targets[start_idx:end_idx]
    chunk_log_densities = log_densities[start_idx:end_idx]
    # Compute loss for this chunk only
```

## Validation Schedule

Uses **exponential validation schedule** for better early training visibility:

```python
# Validate at batches: 1, 2, 4, 8, 16, 32, 64, 128, ...
if batch_idx & (batch_idx - 1) == 0:  # Power of 2 check
    run_validation()
```

This provides frequent early feedback (every batch initially) while avoiding excessive validation overhead later.

## Early Stopping

Training uses early stopping to prevent overfitting and save time:

```python
early_stop_patience: int = 4      # Stop after 4 validation checks with no improvement
early_stop_min_delta: float = 0.01  # Require 1% relative improvement
```

Training stops when validation loss hasn't improved by at least 1% for 4 consecutive validation checks.

## Baselines

### 1. Dead Reckoning (DR)
Constant velocity extrapolation using cumulative time:
```python
velocity = (position[t] - position[t-1]) / dt[t]
prediction = velocity * cumulative_time_to_target
```

### 2. Last Position (LP)
Predicts zero displacement (vessel stays in place).

### Fair Baseline Comparison

Baselines are point predictions wrapped in Gaussians for KL divergence comparison. We use **optimized sigma values** for fair comparison:

| Component | Sigma | Description |
|-----------|-------|-------------|
| Model target | 0.003° (~333m) | Tight target for model's distributional output |
| DR/LP baseline | 0.05° (~5.5km) | Optimized on training data to match DR error distribution |

Without this optimization, DR would be unfairly penalized (sigma=0.003 is too tight for DR's ~10km errors).

## Results

### Training Run: long_horizon_5ep (5 epochs)

| Metric | Value |
|--------|-------|
| Final Train Loss | 0.161 |
| Final Val Loss | 0.158 |
| Dead Reckoning Loss | 0.503 |
| Improvement vs DR | **68.6%** |
| GPU Memory Used | 43.56 GB (91%) |

### Error by Horizon (5 epochs)

| Horizon | Approx Time | Error (km) |
|---------|-------------|------------|
| 1 | ~10s | ~0.08 |
| 100 | ~15 min | ~3.5 |
| 200 | ~30 min | ~7.0 |
| 300 | ~45 min | ~10.0 |
| 400 | ~1 hour | ~12.75 |

Note: Actual time varies per track based on dt values.

## Files

```
10_long_horizon/
├── run_experiment.py          # Main training script with random horizon sampling
├── visualize_predictions.py   # Generate training history plot
├── make_horizon_videos.py     # Generate animated GIFs (50fps, batched)
├── run_all.sh                 # Shell script to run full pipeline
├── README.md                  # This file
└── experiments/
    └── <exp-name>/
        ├── config.json
        ├── checkpoints/
        │   ├── best_model.pt
        │   └── checkpoint_step_*.pt
        └── results/
            ├── training.log              # Full training output
            ├── training_history.png      # Loss curves (model vs baselines)
            └── horizon_video_track*.gif  # 50fps animated predictions
```

## Running

### Full Pipeline (Recommended)

```bash
source /opt/pytorch/bin/activate
cd "/home/ec2-user/trackfm-toy studies/experiments/10_long_horizon"

# Run training + visualizations + videos
./run_all.sh <exp-name> [epochs] [max-horizon-video] [num-tracks]

# Examples:
./run_all.sh my_experiment           # 5 epochs, 400 horizon video, 6 tracks
./run_all.sh my_experiment 10        # 10 epochs
./run_all.sh my_experiment 5 200     # 5 epochs, 200 horizon video
./run_all.sh my_experiment 5 400 10  # 5 epochs, 400 horizon, 10 track videos
```

### Individual Scripts

```bash
# Training only
python run_experiment.py --exp-name my_exp --num-epochs 5

# Available training options:
python run_experiment.py \
    --exp-name <name>              # Experiment name (required)
    --max-horizon <int>            # Max horizon (default: 400)
    --num-horizon-samples <int>    # Samples per batch (default: 40)
    --num-epochs <int>             # Training epochs (default: 100)
    --batch-size <int>             # Batch size (default: 8000)
    --learning-rate <float>        # Learning rate (default: 0.0003)
    --grid-range <float>           # Grid range in degrees (default: 0.3)
    --max-tracks <int>             # Max tracks to load (default: all)
    --early-stop-patience <int>    # Early stopping patience (default: 4)
    --no-early-stop                # Disable early stopping

# Training history plot
python visualize_predictions.py --exp-name my_exp

# Animated GIFs (50fps)
python make_horizon_videos.py --exp-name my_exp --max-horizon 400 --num-tracks 6
```

## Video Generation Details

### Batched Prediction for Speed

Video generation pre-computes all horizon predictions in batches of 50 for GPU efficiency:

```python
batch_size = 50  # Process 50 horizons at a time
for start_h in range(1, max_horizon + 1, batch_size):
    end_h = min(start_h + batch_size, max_horizon + 1)
    horizon_batch = torch.arange(start_h, end_h, device=DEVICE)
    log_densities, _, _ = model.forward_train(features, horizon_batch)
```

### Output Format

- **Frame rate**: 50 fps (20ms per frame)
- **Color scale**: Auto-scaling per frame (not fixed)
- **Resolution**: Reduced figure size for faster rendering
- **Format**: GIF with imageio

## Important Implementation Details

### 1. forward_train Returns 3 Values

The training forward pass returns a tuple of 3 values:
```python
log_densities, targets, horizon_indices = model.forward_train(features, horizon_indices)
```

All callers must unpack correctly or use `_` placeholder.

### 2. Chunked Loss Computation

Soft target creation (`create_soft_target_batch_optimized`) creates large tensors. Must chunk:
```python
chunk_size = 512
for i in range(0, len(predictions), chunk_size):
    # Process chunk
```

### 3. Validation Batch Size

Validation uses smaller batch to avoid OOM:
```python
val_batch_size = min(config.batch_size, 1024)
```

### 4. Time-Conditioned Prediction

The model uses cumulative time encoding, so it can predict at ANY horizon, including horizons beyond training. A model trained with random samples from [1, 400] can predict at horizon 500+ by providing the appropriate cumulative time.

### 5. Grid Range for Long Horizons

At horizon 400 (~1 hour), vessels can travel 15-30 km. Grid range of ±0.3° (±33km) covers typical movements.

## Lessons Learned (Experiment 10)

1. **Random horizon sampling works**: Model learns to predict at all horizons despite only seeing 40 random samples per batch.

2. **Batch size matters for GPU efficiency**: Large batch (8000) with chunked loss achieves 91% GPU utilization.

3. **Exponential validation schedule**: Early frequent validation + later sparse validation balances feedback vs speed.

4. **Batched video generation**: Computing 400 frames serially is slow. Batching horizon predictions (50 at a time) dramatically speeds up GIF generation.

5. **Auto color scale for videos**: Fixed color scale doesn't work across 400 horizons where probability distributions vary widely.

6. **50fps for smooth videos**: With 400 frames, 50fps gives 8-second videos showing smooth prediction evolution.

## Known Issues

1. **Error scaling with horizon**: Error grows roughly linearly with horizon (expected for chaotic trajectories).

2. **Dead reckoning still competitive on straight tracks**: DR excels when vessels maintain constant velocity.

3. **Video generation takes several minutes**: Even with batching, generating 400-frame GIFs for multiple tracks takes time.

## Future Work

- [ ] Adaptive horizon sampling (sample more near decision-critical horizons)
- [ ] Longer training (10+ epochs) to see if accuracy continues improving
- [ ] Curriculum learning (start with short horizons, gradually increase)
- [ ] Track selection by curvature (focus on tracks with maneuvers)
- [ ] Uncertainty calibration analysis at long horizons
- [ ] Compare with RNN/LSTM baselines for long-horizon prediction
