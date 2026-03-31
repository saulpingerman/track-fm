# AIS Vessel Trajectory ML Training Data

## Data Location

```
s3://ais-pipeline-data-10179bbf-us-east-1/materialized/
├── samples_000.parquet    # ~426K pre-shuffled training samples
├── samples_001.parquet
├── ...
├── samples_255.parquet
└── validation.parquet     # 313K validation samples
```

## Key Statistics

| Metric | Value |
|--------|-------|
| Training samples | ~109 million |
| Validation samples | 313,000 |
| Training shards | 256 |
| Total size | 645 GB (compressed) |
| Samples per shard | ~426,000 |
| Window size | 928 positions |
| Features per position | 5 |
| Floats per sample | 4,640 (928 × 5) |

## Data Format

Each parquet file contains a single column `features` where each row is a **FixedSizeList** of 4,640 float32 values representing a flattened training window.

### Window Structure
```
Shape: (928, 5) flattened to (4640,)

Position features (in order):
  [0] lat        - Latitude (-90 to 90)
  [1] lon        - Longitude (-180 to 180)
  [2] sog        - Speed over ground (knots)
  [3] cog        - Course over ground (0-360 degrees)
  [4] dt_seconds - Time delta from previous position (seconds)

Window layout:
  positions 0-127:   Input sequence (SEQ_LEN=128)
  positions 128-927: Prediction horizon (MAX_HORIZON=800)
```

### Reshaping
```python
# After loading flat array of 4640 floats:
window = features.reshape(928, 5)

# Split into input and target:
SEQ_LEN = 128
input_seq = window[:SEQ_LEN]      # (128, 5)
target_seq = window[SEQ_LEN:]     # (800, 5)
```

## Why This Format Works

The data is **pre-shuffled globally** using the Jane Street 2-pass algorithm:
1. Windows were extracted from 175K tracks across 255 input shards
2. Each window was randomly assigned to one of 256 output piles
3. Each pile was shuffled internally before writing

**Result:** Reading sequentially through ANY shard gives you random samples from the ENTIRE dataset. No need for complex shuffling during training.

## Loading Data

### Option 1: PyArrow (Recommended for streaming)
```python
import pyarrow.parquet as pq
import numpy as np

def load_shard(shard_id: int, batch_size: int = 64):
    """Stream batches from a shard."""
    path = f"s3://ais-pipeline-data-10179bbf-us-east-1/materialized/samples_{shard_id:03d}.parquet"
    pf = pq.ParquetFile(path)

    for batch in pf.iter_batches(batch_size=batch_size):
        # Convert to numpy and reshape
        features = np.array(batch['features'].to_pylist(), dtype=np.float32)
        features = features.reshape(-1, 928, 5)
        yield features

# Usage
for batch in load_shard(0, batch_size=64):
    # batch shape: (64, 928, 5)
    input_seq = batch[:, :128, :]   # (64, 128, 5)
    target_seq = batch[:, 128:, :]  # (64, 800, 5)
```

### Option 2: PyTorch IterableDataset
```python
import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import numpy as np

class AISDataset(IterableDataset):
    def __init__(self, shard_ids: list[int], batch_size: int = 64):
        self.shard_ids = shard_ids
        self.batch_size = batch_size
        self.bucket = "ais-pipeline-data-10179bbf-us-east-1"

    def __iter__(self):
        # Shuffle shard order each epoch
        shard_order = np.random.permutation(self.shard_ids)

        for shard_id in shard_order:
            path = f"s3://{self.bucket}/materialized/samples_{shard_id:03d}.parquet"
            pf = pq.ParquetFile(path)

            for batch in pf.iter_batches(batch_size=self.batch_size):
                features = np.array(batch['features'].to_pylist(), dtype=np.float32)
                features = features.reshape(-1, 928, 5)
                yield torch.from_numpy(features)

# Usage
train_shards = list(range(256))
dataset = AISDataset(train_shards, batch_size=64)
loader = DataLoader(dataset, batch_size=None)  # Batching handled internally

for batch in loader:
    # batch: torch.Tensor of shape (64, 928, 5)
    pass
```

### Option 3: Multi-worker DataLoader
```python
class AISDatasetWorker(IterableDataset):
    """Dataset that distributes shards across DataLoader workers."""

    def __init__(self, shard_ids: list[int], batch_size: int = 64):
        self.shard_ids = shard_ids
        self.batch_size = batch_size
        self.bucket = "ais-pipeline-data-10179bbf-us-east-1"

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker
            my_shards = self.shard_ids
        else:
            # Distribute shards across workers
            my_shards = self.shard_ids[worker_info.id::worker_info.num_workers]

        # Shuffle this worker's shards
        np.random.shuffle(my_shards)

        for shard_id in my_shards:
            path = f"s3://{self.bucket}/materialized/samples_{shard_id:03d}.parquet"
            pf = pq.ParquetFile(path)

            for batch in pf.iter_batches(batch_size=self.batch_size):
                features = np.array(batch['features'].to_pylist(), dtype=np.float32)
                features = features.reshape(-1, 928, 5)
                yield torch.from_numpy(features)

# Usage with multiple workers
dataset = AISDatasetWorker(list(range(256)), batch_size=64)
loader = DataLoader(dataset, batch_size=None, num_workers=4)
```

## Validation Data

```python
def load_validation(batch_size: int = 64):
    path = "s3://ais-pipeline-data-10179bbf-us-east-1/materialized/validation.parquet"
    pf = pq.ParquetFile(path)

    for batch in pf.iter_batches(batch_size=batch_size):
        features = np.array(batch['features'].to_pylist(), dtype=np.float32)
        features = features.reshape(-1, 928, 5)
        yield features
```

## Feature Engineering Notes

### Normalization
```python
# Suggested normalization (compute on training data):
# lat: divide by 90 -> [-1, 1]
# lon: divide by 180 -> [-1, 1]
# sog: divide by 30 (clip outliers first) -> [0, 1]
# cog: encode as sin(cog*pi/180), cos(cog*pi/180) -> [-1, 1]
# dt_seconds: log1p transform, then normalize
```

### Course Encoding (COG)
```python
import numpy as np

def encode_cog(cog_degrees):
    """Encode course as sin/cos to handle wraparound."""
    radians = cog_degrees * np.pi / 180
    return np.sin(radians), np.cos(radians)
```

### Handling dt_seconds
The `dt_seconds` feature varies widely (seconds to hours). Consider:
```python
# Log transform to compress range
dt_log = np.log1p(dt_seconds)

# Or bucket into categories
# Or normalize per-sample
```

## Typical Training Tasks

### 1. Next Position Prediction
```python
# Input: first 128 positions
# Output: position 129
input_seq = batch[:, :128, :]    # (B, 128, 5)
target = batch[:, 128, :2]       # (B, 2) - lat, lon only
```

### 2. Trajectory Forecasting
```python
# Input: first 128 positions
# Output: next N positions
input_seq = batch[:, :128, :]
target_seq = batch[:, 128:128+N, :2]  # (B, N, 2)
```

### 3. Full Sequence Prediction
```python
# Input: first 128 positions
# Output: remaining 800 positions
input_seq = batch[:, :128, :]
target_seq = batch[:, 128:, :]  # (B, 800, 5)
```

## AWS Setup for Training

### Recommended Instance
- **p3.2xlarge** or **g4dn.xlarge** for single GPU
- **p3.8xlarge** for multi-GPU

### Required Packages
```bash
pip install torch pyarrow boto3 numpy
```

### S3 Access
Ensure your EC2 instance has an IAM role with S3 read access to the bucket.

## Quick Verification

```python
import pyarrow.parquet as pq
import numpy as np

# Test loading
pf = pq.ParquetFile("s3://ais-pipeline-data-10179bbf-us-east-1/materialized/samples_000.parquet")
print(f"Rows in shard 0: {pf.metadata.num_rows:,}")

# Load one batch
batch = next(pf.iter_batches(batch_size=10))
features = np.array(batch['features'].to_pylist(), dtype=np.float32)
features = features.reshape(-1, 928, 5)

print(f"Batch shape: {features.shape}")  # (10, 928, 5)
print(f"Feature ranges:")
print(f"  lat: [{features[:,:,0].min():.2f}, {features[:,:,0].max():.2f}]")
print(f"  lon: [{features[:,:,1].min():.2f}, {features[:,:,1].max():.2f}]")
print(f"  sog: [{features[:,:,2].min():.2f}, {features[:,:,2].max():.2f}]")
print(f"  cog: [{features[:,:,3].min():.2f}, {features[:,:,3].max():.2f}]")
print(f"  dt:  [{features[:,:,4].min():.2f}, {features[:,:,4].max():.2f}]")
```

## Summary

- **109M training samples** pre-shuffled across 256 shards
- **313K validation samples** in separate file
- Sequential reading = random sampling from full dataset
- No complex data loading logic needed
- Just iterate through shards in random order each epoch
