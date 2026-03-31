# Materialized Training Data

This document describes the pre-shuffled, materialized training data format optimized for ML training on AIS vessel trajectories.

## Overview

The materialized data contains **109 million training samples** pre-shuffled globally using the [Jane Street 2-pass algorithm](https://blog.janestreet.com/how-to-shuffle-a-big-dataset/). Each sample is a fixed-size window of 928 positions × 5 features, ready for direct use in sequence models.

**Key benefit:** Sequential reading through any shard gives you random samples from the ENTIRE dataset. No complex shuffling needed during training.

## Data Formats

### MosaicML MDS Format [RECOMMENDED]
```
s3://ais-pipeline-data-10179bbf-us-east-1/mds/
├── worker_00/           # MDS shards from parallel conversion
│   ├── index.json
│   ├── shard.00000.mds.zstd   # ~22 MB each (64 MB uncompressed)
│   ├── shard.00001.mds.zstd
│   └── ...
├── worker_01/
├── ...
├── worker_15/
└── manifest.json        # Lists all worker directories
```

**Benefits:**
- Memory-efficient streaming (~22 MB shards)
- Built-in shuffling across entire dataset
- Optimized for distributed training
- Works great on 32GB GPU instances

### Original Parquet Format (Large Shards - 2.7 GB each)
```
s3://ais-pipeline-data-10179bbf-us-east-1/materialized/
├── samples_000.parquet   # ~426K samples, 2.7 GB compressed
├── ...
├── samples_255.parquet
└── validation.parquet
```

**WARNING:** These shards expand to 25-30 GB peak memory! Not suitable for most GPU instances.

## Statistics

| Metric | Original Parquet | MDS Format |
|--------|------------------|------------|
| Training samples | 109M | 108.7M |
| Total files | 256 | ~30,000 |
| Shard size (compressed) | 2.7 GB | ~22 MB |
| Total size | 645 GB | 632 GB |
| **Peak memory per shard** | **25-30 GB** | **~200 MB** |
| Safe for 32GB instance | NO | YES |

## Sample Format

Each parquet file has a single column `features` containing fixed-size arrays:

```
Schema:
  features: FixedSizeList[4640, float32]
```

Each row is a flattened window: `(928 positions × 5 features) = 4640 floats`

### Feature Order (per position)
| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | lat | -90 to 90 | Latitude in degrees |
| 1 | lon | -180 to 180 | Longitude in degrees |
| 2 | sog | 0 to ~50 | Speed over ground (knots) |
| 3 | cog | 0 to 360 | Course over ground (degrees) |
| 4 | dt_seconds | 0 to ~14400 | Time delta from previous position |

### Window Structure
```
Positions 0-127:   Input sequence (SEQ_LEN = 128)
Positions 128-927: Prediction horizon (MAX_HORIZON = 800)

Total: 928 positions per sample
```

## Loading Data

### MosaicML StreamingDataset (Recommended)

```python
from streaming import StreamingDataset
from torch.utils.data import DataLoader
import numpy as np

# Create dataset - reads from all worker directories automatically
dataset = StreamingDataset(
    remote='s3://ais-pipeline-data-10179bbf-us-east-1/mds',
    local='/tmp/mds-cache',
    shuffle=True,
    batch_size=64,
)

# Wrap in DataLoader
loader = DataLoader(dataset, batch_size=64)

for batch in loader:
    # Each sample is bytes - decode to numpy
    features = np.frombuffer(batch['features'], dtype=np.float32)
    features = features.reshape(-1, 928, 5)  # (batch_size, 928, 5)

    input_seq = features[:, :128, :]   # (64, 128, 5)
    target_seq = features[:, 128:, :]  # (64, 800, 5)
    # Train on this batch
```

### Using the Provided Wrapper

```python
from ais_pipeline.io.streaming_loader import AISMDSDataset, normalize_batch, split_input_target

# Create dataset
dataset = AISMDSDataset(
    remote='s3://ais-pipeline-data-10179bbf-us-east-1/mds',
    local='/tmp/mds-cache',
    batch_size=64,
    shuffle=True,
)

for batch in dataset:
    # batch: (64, 928, 5) torch.Tensor
    batch = normalize_batch(batch)
    input_seq, target_seq = split_input_target(batch)
    # Train...
```

### Multi-GPU / Distributed Training

MosaicML StreamingDataset handles distributed training automatically:

```python
from streaming import StreamingDataset
import torch.distributed as dist

# Each rank gets different samples automatically
dataset = StreamingDataset(
    remote='s3://ais-pipeline-data-10179bbf-us-east-1/mds',
    local='/tmp/mds-cache',
    shuffle=True,
)

# Works with DistributedDataParallel out of the box
```

## Validation Data

Validation data remains in parquet format:

```python
import pyarrow.parquet as pq
import numpy as np

def stream_validation(batch_size: int = 64):
    """Stream validation batches."""
    path = "s3://ais-pipeline-data-10179bbf-us-east-1/materialized/validation.parquet"
    pf = pq.ParquetFile(path)

    for batch in pf.iter_batches(batch_size=batch_size):
        features = np.array(batch['features'].to_pylist(), dtype=np.float32)
        features = features.reshape(-1, 928, 5)
        yield features

# ~313K validation samples
```

## Feature Engineering

### Normalization
```python
def normalize_features(batch):
    """Normalize features to reasonable ranges."""
    # batch shape: (B, 928, 5)
    normalized = batch.clone()

    # lat: [-90, 90] -> [-1, 1]
    normalized[:, :, 0] = batch[:, :, 0] / 90.0

    # lon: [-180, 180] -> [-1, 1]
    normalized[:, :, 1] = batch[:, :, 1] / 180.0

    # sog: [0, ~50] -> [0, 1] (clip outliers)
    normalized[:, :, 2] = torch.clamp(batch[:, :, 2], 0, 30) / 30.0

    # cog: [0, 360] -> sin/cos encoding (handles wraparound)
    cog_rad = batch[:, :, 3] * (np.pi / 180.0)
    # Replace cog with sin and cos (increases feature dim by 1)

    # dt_seconds: log transform
    normalized[:, :, 4] = torch.log1p(batch[:, :, 4]) / 10.0  # log(14400) ≈ 9.6

    return normalized
```

### Course Encoding (Recommended)
```python
def encode_course(cog_degrees):
    """Encode course as sin/cos to handle 0°/360° wraparound."""
    radians = cog_degrees * (np.pi / 180.0)
    return np.sin(radians), np.cos(radians)
```

## Training Tasks

### Next Position Prediction
```python
# Input: positions 0-127
# Output: position 128 (lat, lon)
input_seq = batch[:, :128, :]      # (B, 128, 5)
target_pos = batch[:, 128, :2]     # (B, 2) - lat, lon
```

### Multi-Step Forecasting
```python
# Input: positions 0-127
# Output: positions 128-227 (next 100 positions)
input_seq = batch[:, :128, :]
target_seq = batch[:, 128:228, :2]  # (B, 100, 2)
```

### Full Horizon Prediction
```python
# Input: positions 0-127
# Output: positions 128-927 (full 800 position horizon)
input_seq = batch[:, :128, :]
target_seq = batch[:, 128:, :2]     # (B, 800, 2)
```

## Memory Budgeting

For a 32GB GPU instance:

| Component | Memory |
|-----------|--------|
| OS + Framework | ~4 GB |
| Model (typical) | 1-4 GB |
| Batch (64 samples) | 0.5 GB |
| Gradients + Optimizer | 2-8 GB |
| **Available headroom** | ~16-24 GB |

**Recommended batch sizes by instance:**
- 16 GB RAM: batch_size = 32
- 32 GB RAM: batch_size = 64-128
- 64 GB RAM: batch_size = 256

## Epochs and Iterations

```python
# Per epoch:
samples_per_epoch = 109_039_946
batch_size = 64
iterations_per_epoch = samples_per_epoch // batch_size  # ~1.7M iterations

# With 256 shards, each shard has ~426K samples
# At batch_size=64, that's ~6,650 batches per shard
```

## Why Pre-Shuffled?

Traditional approach:
1. Load data into buffer
2. Shuffle buffer
3. Train on buffer
4. Reload new data → **Model overfits to buffer before seeing diverse data**

Pre-shuffled approach:
1. Data already globally shuffled
2. Stream sequentially through any shard
3. Every batch is random from full dataset
4. **No overfitting to subsets**

## Regenerating the Data

If you need to regenerate the materialized data (e.g., different window size):

### Step 1: Materialize to Parquet (128GB instance, 2.5TB disk)

```bash
cd ~/ais-analysis
source ~/ais-env/bin/activate

python scripts/materialize_samples.py \
    --bucket ais-pipeline-data-10179bbf-us-east-1 \
    --input-prefix sharded/ \
    --output-prefix materialized/ \
    --num-output-shards 256 \
    --temp-dir /tmp/materialize_temp \
    --seed 42
```

### Step 2: Convert to MDS Format

```bash
python scripts/parallel_mds_convert.py \
    --num-workers 16 \
    --bucket ais-pipeline-data-10179bbf-us-east-1 \
    --input-prefix materialized/ \
    --output-base s3://ais-pipeline-data-10179bbf-us-east-1/mds
```

This takes ~6 hours with 16 workers on a 128GB instance.

## Installation Requirements

```bash
pip install mosaicml-streaming pyarrow boto3 torch numpy
```
