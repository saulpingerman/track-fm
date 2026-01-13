# Experiment 11: Long Horizon Prediction (69 Days of Data)

## Overview

This experiment extends Experiment 10 to use **69 days of AIS data** (10x more than Exp 10's 7 days) and adds **causal subwindow training** for more efficient learning.

**Key changes from Experiment 10:**
- 69 days of data instead of 7 days (~10x more positions)
- Added **lazy loading** mode for memory-efficient training
- Added **causal subwindow training** - uses all positions in the sequence, not just the last one
- Same model architecture

## Causal Subwindow Training

Traditional training only predicts from the **last position** of each input sequence. Causal subwindow training predicts from **all positions**, leveraging the causal mask that's already applied in the transformer.

### How It Works

For each sampled horizon `h`:
- **Within-sequence** (if h < seq_len): Positions 0 to (seq_len-h-1) predict positions h to (seq_len-1)
- **Future**: Last position predicts h steps into the future

This provides **much more training signal** per batch:
- Without causal: 8 horizons × 1 prediction each = 8 predictions/sample
- With causal: 8 horizons × ~30 predictions each = ~240 predictions/sample

### Benefits

1. **More training signal** - Each sequence contributes many more gradients
2. **Learns with varying context** - Model learns to predict with 10 steps of context, 20 steps, etc.
3. **Better generalization** - Sees more diverse (position, horizon) combinations

### Note on Train vs Val Loss

With causal training, validation loss is typically **lower** than training loss because:
- Validation uses fixed horizons including h=1 (produces many "easy" 1-step predictions)
- Training uses random horizons that may not include such easy cases

This is expected behavior, not a bug.

## Data

| Aspect | Experiment 10 | Experiment 11 |
|--------|---------------|---------------|
| Time range | 7 days | **69 days** |
| Estimated positions | ~23M | **~580M** |
| Estimated tracks | ~3,750 | **~185,000** |
| RAM required (eager) | ~1 GB | **~6-8 GB** |
| RAM required (lazy) | ~1 GB | **~500 MB** |

### Data Source
- **Path**: `/mnt/fsx/data/`
- **Catalog**: `/mnt/fsx/data/track_catalog.parquet`
- **Region**: Danish EEZ (54-58.5N, 7-16E)

## Data Loading Modes

### 1. Eager Loading (Recommended for this dataset)
Loads all track data into memory upfront. Fast training.

```bash
python run_experiment.py --exp-name my_exp  # Default: eager loading
```

### 2. Lazy Loading
Builds a lightweight index upfront, loads tracks on-demand with LRU caching.

```bash
python run_experiment.py --exp-name my_exp --lazy-load --cache-size 1000
```

**Note**: Lazy loading has significant I/O overhead with causal training. Eager loading is recommended if you have sufficient RAM.

## Running

### Quick Start

```bash
source /opt/pytorch/bin/activate
cd "/home/ec2-user/trackfm-toy studies/experiments/11_long_horizon_69_days"

# Recommended: Eager loading with causal training
python run_experiment.py \
    --exp-name 69days_causal \
    --model-scale large \
    --batch-size 0 \
    --num-epochs 100 \
    --num-horizons 8 \
    --max-val-samples 40000
```

### Full Pipeline

```bash
./run_all.sh my_experiment 100  # 100 epochs with early stopping
```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--lazy-load` | Use memory-efficient lazy loading |
| `--cache-size N` | LRU cache size for lazy loading (default: 1000) |
| `--model-scale` | small/medium/large model preset |
| `--batch-size 0` | Auto-find optimal batch size |
| `--num-epochs N` | Number of training epochs |
| `--num-horizons N` | Number of horizon samples per batch (default: 8) |
| `--max-val-samples N` | Limit validation samples (default: 40000) |
| `--log-interval N` | Log training loss every N batches (default: 10) |
| `--max-tracks N` | Limit number of tracks (for testing) |

## Model Architecture

Same as Experiment 10:

```
CausalAISModel
├── Input Projection: 6 → d_model
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: num_layers, nhead heads, causal mask
├── Cumulative Time Encoding: For arbitrary horizon prediction
└── Fourier Head 2D: 64x64 probability grid
```

### Model Scales

| Scale | d_model | nhead | layers | dim_ff | Params |
|-------|---------|-------|--------|--------|--------|
| small | 128 | 8 | 4 | 512 | ~1M |
| medium | 256 | 8 | 6 | 1024 | ~5M |
| large | 384 | 16 | 8 | 2048 | ~18M |

## Expected Results

With 10x more data and causal training, we expect:
- Better generalization (more diverse tracks)
- More efficient training (more signal per batch)
- Potentially higher accuracy

## Files

```
11_long_horizon_69_days/
├── run_experiment.py          # Training script with causal training
├── visualize_predictions.py   # Training history plot
├── make_horizon_videos.py     # Animated GIFs
├── run_all.sh                 # Full pipeline
├── README.md                  # This file
└── experiments/
    └── <exp-name>/
        ├── config.json
        ├── checkpoints/
        └── results/
```

## Comparison with Previous Experiments

| Aspect | Exp 10 | Exp 11 |
|--------|--------|--------|
| Data | 7 days | 69 days |
| Training | Last position only | **Causal subwindows** |
| Predictions/batch | ~8 per sample | **~240 per sample** |
| Loading | Eager only | Eager + Lazy |
| RAM (eager) | ~1 GB | ~6-8 GB |
