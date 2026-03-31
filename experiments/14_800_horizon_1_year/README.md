# Experiment 14: 800 Horizon Prediction with 1 Year of Data

## Overview

Pre-trains an **18M parameter** model on **1 year of AIS data** (~3.5B positions) to predict vessel positions up to **2 hours** into the future.

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model Size | 18M params | d_model=384, nhead=16, num_layers=8 |
| Max Horizon | 800 | ~2 hours at ~10s intervals |
| Grid Size | 128x128 | Doubled from 64 to maintain cell resolution |
| Grid Range | 0.6 degrees | ~66km coverage for 2-hour predictions |
| Min Track Length | 1000 | Need seq_len(128) + horizon(800) |
| Data | 1 year | ~3.5B positions |

## Changes from Experiment 11

- `max_horizon`: 400 → **800** (2 hours instead of 1 hour)
- `grid_range`: 0.3 → **0.6** degrees (doubled for longer horizon)
- `grid_size`: 64 → **128** (doubled to maintain ~520m cell resolution)
- `min_track_length`: 600 → **1000** (need longer tracks)
- Data: 69 days → **1 year** (~3.5B positions)

## Running

```bash
source /opt/pytorch/bin/activate
cd /home/ec2-user/trackfm/experiments/14_800_horizon_1_year

# Train with S3 sharded data (recommended)
python run_experiment.py \
    --s3-shards \
    --exp-name s3_train \
    --batch-size 0 \
    --num-epochs 1

# Generate videos after training
python make_horizon_videos.py --exp-name s3_train --max-horizon 800
python make_interesting_videos.py --exp-name s3_train --max-horizon 800 --num-tracks 30
```

## Data Loading Modes

### S3 Sharded (Recommended)
Uses pre-sharded data from S3 with double-buffered loading:
- **Data**: `s3://ais-pipeline-data-10179bbf-us-east-1/sharded/` (256 shards, 651K tracks)
- **Double-buffer**: Trains on buffer A while prefetching buffer B from S3
- **Memory efficient**: Only 10 shards in memory at a time (~13GB)

```bash
python run_experiment.py --s3-shards --shards-per-buffer 10
```

### Chunked (FSx)
Loads from FSx day-partitioned files in chunks:
```bash
python run_experiment.py --chunked --num-chunks 10
```

## Model Architecture

```
CausalAISModel (18M parameters)
├── Input Projection: 6 → 384
├── Positional Encoding: Sinusoidal
├── Transformer Encoder: 8 layers, 16 heads, causal mask
├── Cumulative Time Encoding: For arbitrary horizon prediction
└── Fourier Head 2D: 128x128 probability grid over 0.6° range (~520m/cell)
```

## Expected Results

Based on scaling from Experiment 11:
- Training on 1 year of data should provide better generalization
- 800 horizon predictions (~2 hours ahead) will have higher uncertainty
- Model should learn seasonal patterns with full year of data
