# Guide: Materializing Pre-Shuffled Training Samples

## Overview

**Goal:** Convert track-based parquet shards into sample-based parquet files where each row is a complete training window (928 positions × 5 features), globally shuffled.

**Result:** Sequential reading = true random sampling across all 130K tracks.

---

## Data Structure

**Input (current):**
```
s3://ais-pipeline-data-10179bbf-us-east-1/sharded/
  shard=000/tracks.parquet  # ~2500 complete tracks
  shard=001/tracks.parquet
  ...
  shard=254/tracks.parquet  # (255 for validation, exclude)
```

**Output (materialized):**
```
s3://ais-pipeline-data-10179bbf-us-east-1/materialized/
  samples_000.parquet  # ~250K random samples from ALL tracks
  samples_001.parquet
  ...
  samples_255.parquet
```

Each output shard contains samples from tracks across the ENTIRE dataset, pre-shuffled.

---

## Algorithm (Jane Street 2-Pass)

Reference: [Jane Street Blog - How to Shuffle a Big Dataset](https://blog.janestreet.com/how-to-shuffle-a-big-dataset/)

### Pass 1: Generate & Distribute

```python
import numpy as np
import polars as pl
from collections import defaultdict

# Config
SEQ_LEN = 128
MAX_HORIZON = 800
WINDOW_SIZE = SEQ_LEN + MAX_HORIZON  # 928
STRIDE = 32  # 75% overlap
NUM_OUTPUT_SHARDS = 256
MIN_SOG = 3.0

# Accumulate samples into random output piles
piles = defaultdict(list)

for shard_id in range(255):  # Exclude shard 255 (validation)
    shard_path = f"s3://bucket/sharded/shard={shard_id:03d}/tracks.parquet"
    df = pl.read_parquet(shard_path)
    df = df.filter(pl.col("sog") >= MIN_SOG)

    for track_df in df.partition_by("track_id"):
        track_id = track_df["track_id"][0]

        # Extract features
        features = track_df.select([
            "lat", "lon", "sog", "cog", "dt_seconds"
        ]).to_numpy().astype(np.float32)

        features = np.nan_to_num(features, nan=0.0)

        if len(features) < WINDOW_SIZE:
            continue

        # Generate all valid windows
        for start in range(0, len(features) - WINDOW_SIZE + 1, STRIDE):
            window = features[start:start + WINDOW_SIZE]  # (928, 5)

            # Randomly assign to output pile
            pile_id = np.random.randint(0, NUM_OUTPUT_SHARDS)
            piles[pile_id].append(window)

    print(f"Processed shard {shard_id}/254")
```

### Pass 2: Shuffle & Write

```python
import pyarrow as pa
import pyarrow.parquet as pq

for pile_id in range(NUM_OUTPUT_SHARDS):
    samples = piles[pile_id]

    # Shuffle within pile
    np.random.shuffle(samples)

    # Stack into array
    features = np.stack(samples, axis=0)  # (N, 928, 5)

    # Flatten for storage
    features_flat = features.reshape(len(samples), -1)  # (N, 4640)

    # Create PyArrow table
    table = pa.table({
        'features': pa.FixedSizeListArray.from_arrays(
            pa.array(features_flat.ravel(), type=pa.float32()),
            WINDOW_SIZE * 5  # 4640
        )
    })

    # Write with compression
    output_path = f"s3://bucket/materialized/samples_{pile_id:03d}.parquet"
    pq.write_table(table, output_path, compression='zstd', compression_level=3)

    print(f"Written pile {pile_id}: {len(samples):,} samples")
```

---

## Expected Sizes

| Metric | Value |
|--------|-------|
| Input shards | 255 × ~170MB = 44GB |
| Tracks | ~130K |
| Samples per track | ~500 (with stride 32) |
| Total samples | ~65M |
| Bytes per sample (raw) | 928 × 5 × 4 = 18,560 bytes |
| Raw total | ~1.2 TB |
| Compressed (~4x) | **~300 GB** |
| Samples per output shard | ~250K |

---

## Memory Considerations

**Problem:** Can't hold all 65M samples in memory during Pass 1.

**Solution: Streaming approach with temp files**

Instead of accumulating in memory, write to temporary files:

```python
import tempfile
import os

# Create temp files for each pile
temp_dir = tempfile.mkdtemp()
pile_files = {}
for i in range(NUM_OUTPUT_SHARDS):
    pile_files[i] = open(f"{temp_dir}/pile_{i:03d}.bin", 'wb')

# Pass 1: Stream samples to temp files
for shard_id in range(255):
    # ... load shard ...
    for window in windows:
        pile_id = np.random.randint(0, NUM_OUTPUT_SHARDS)
        pile_files[pile_id].write(window.tobytes())

# Close all files
for f in pile_files.values():
    f.close()

# Pass 2: Read each temp file, shuffle, write parquet
for pile_id in range(NUM_OUTPUT_SHARDS):
    # Memory-map the temp file for shuffling
    data = np.memmap(f"{temp_dir}/pile_{pile_id:03d}.bin",
                     dtype=np.float32, mode='r')
    data = data.reshape(-1, WINDOW_SIZE, 5)

    # Shuffle indices (not data itself)
    indices = np.random.permutation(len(data))

    # Write in shuffled order
    # ... write to parquet in chunks ...
```

---

## Recommended Instance

For data processing:
- **r6i.4xlarge** or similar (128GB RAM, 16 vCPUs)
- Large EBS volume (500GB+) for temp files
- Good network bandwidth for S3

---

## Training Code Changes

After materialization, training becomes simple:

```python
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, DataLoader

class MaterializedDataset(IterableDataset):
    def __init__(self, shard_paths, batch_size=64):
        self.shard_paths = shard_paths
        self.batch_size = batch_size

    def __iter__(self):
        # Shuffle shard order each epoch
        shard_order = np.random.permutation(self.shard_paths)

        for shard_path in shard_order:
            pf = pq.ParquetFile(shard_path)

            for batch in pf.iter_batches(batch_size=self.batch_size):
                features = np.array(batch['features'].to_pylist(),
                                   dtype=np.float32)
                features = features.reshape(-1, 928, 5)
                yield torch.from_numpy(features)

# Usage
shard_paths = [f"s3://bucket/materialized/samples_{i:03d}.parquet"
               for i in range(256)]
dataset = MaterializedDataset(shard_paths)
loader = DataLoader(dataset, batch_size=None)  # Batching handled internally

for batch in loader:
    # batch is (batch_size, 928, 5) - already random from full dataset!
    train_step(batch)
```

---

## Validation Data

Keep shard 255 separate, materialize it the same way:

```python
# Materialize validation shard separately
val_path = "s3://bucket/materialized/validation.parquet"
# Use same process but only for shard 255
```

---

## Testing First

Before full materialization, test with ONE shard:

```bash
python materialize_samples.py --shard-id 0 --output-dir ./test_materialized
```

Check:
1. File size (expect ~1-2GB compressed per input shard worth of samples)
2. Loading speed
3. Data integrity (spot check a few samples)

---

## Full Pipeline Script

```bash
#!/bin/bash
# materialize_all.sh

# Process all training shards in parallel (adjust based on memory)
for shard_id in $(seq 0 254); do
    python materialize_samples.py \
        --shard-id $shard_id \
        --output-dir s3://bucket/materialized/ &

    # Limit parallelism to avoid OOM
    if (( $(jobs -r | wc -l) >= 8 )); then
        wait -n
    fi
done
wait

echo "All shards materialized!"
```

---

## Why This Works

### The Problem
- Current: Load 10 shards (5K tracks), train 27K batches on same tracks, then swap
- Model overfits to current buffer before seeing diverse data

### The Solution
- Materialized: Each output shard has random samples from ALL 130K tracks
- Sequential reading through any shard = random sampling from full dataset
- No buffer-switching, no overfitting to subsets

### The Math (Jane Street proof)
Randomly assigning each sample to a pile, then shuffling within piles, is mathematically equivalent to a global shuffle. The algorithm produces unbiased permutations.

---

## Summary

1. **Instance:** Spin up r6i.4xlarge or similar
2. **Test:** Materialize shard 0, verify size/speed
3. **Full run:** Materialize all 255 training shards with 2-pass algorithm
4. **Upload:** Ensure all output in S3
5. **Train:** Use simple sequential DataLoader on pre-shuffled files

The storage cost (~300GB vs 44GB) buys you true random sampling with sequential I/O - no more overfitting to buffer contents.
