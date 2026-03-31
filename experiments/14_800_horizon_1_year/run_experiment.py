#!/usr/bin/env python3
"""
Experiment 14: 800 Horizon Prediction with 1 Year of Data

Pre-trains an 18M parameter model on 1 year of AIS data (~3.5B positions).
Predicts up to 2 hours into the future (horizon 800).

Key features:
- 18M parameter model (d_model=384, nhead=16, num_layers=8)
- 800 horizon steps (~2 hours at ~10s intervals)
- Larger grid_range (0.6 degrees = ~66km) for 2-hour predictions
- 1 year of training data
- Causal subwindow training for efficient learning
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
import time
import threading
from dataclasses import dataclass

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Data paths
    data_path: str = "/mnt/fsx/data"
    catalog_path: str = "/mnt/fsx/data/track_catalog.parquet"
    s3_bucket: str = "s3://ais-pipeline-data-10179bbf-us-east-1/sharded"  # S3 sharded data

    # Data filtering
    min_track_length: int = 1000  # Need longer tracks for horizon 800
    max_seq_len: int = 128  # Sequence length for training
    min_sog: float = 3.0  # Minimum speed over ground (knots) - filter stationary vessels

    # Model architecture - 18M parameter model
    d_model: int = 384
    nhead: int = 16
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Model scale presets (small=1x, medium=~5x, large=~17x params)
    # small:  d_model=128, nhead=8,  num_layers=4, dim_ff=512   (~1M params)
    # medium: d_model=256, nhead=8,  num_layers=6, dim_ff=1024  (~5M params)
    # large:  d_model=384, nhead=16, num_layers=8, dim_ff=2048  (~18M params) <- DEFAULT

    # Fourier head
    grid_size: int = 128  # Doubled from 64 to maintain cell resolution with larger range
    num_freqs: int = 12
    grid_range: float = 0.6  # Degrees - covers ~66km for 2-hour predictions

    # Multi-horizon prediction
    max_horizon: int = 800  # Maximum horizon (2 hours at ~10s intervals)
    num_horizon_samples: int = 40  # Number of random horizons to sample per batch

    # Training
    batch_size: int = 8000  # Large batch with chunked loss computation (~91% GPU utilization)
    learning_rate: float = 3e-4  # Higher LR for large batch (linear scaling)
    weight_decay: float = 1e-5
    num_epochs: int = 1  # Single epoch for initial testing
    warmup_steps: int = 500  # Warmup steps (increased for cosine schedule)
    min_lr_ratio: float = 0.033  # Min LR = base_lr * ratio (3e-4 * 0.033 = 1e-5)
    total_steps_estimate: int = 3000  # Estimated total steps per epoch for cosine decay
    val_every_n_batches: int = 20  # Validate ~3x per epoch (59 batches total)
    val_max_batches: int = 200  # Limit validation batches

    # Loss
    sigma: float = 0.003  # Sigma for model's soft target (~333m)
    dr_sigma: float = 0.05  # Sigma for dead reckoning baseline (optimized on training data)

    # Early stopping
    early_stop_patience: int = 4  # Stop if no improvement for N validation checks
    early_stop_min_delta: float = 0.01  # Minimum relative improvement required (1%)

    # Optimization
    use_amp: bool = True  # Automatic mixed precision
    num_workers: int = 0  # Start with 0 for stability, increase if no NaN
    pin_memory: bool = True
    gradient_accumulation: int = 1

    # Normalization parameters (computed from data)
    lat_mean: float = 56.25  # Center of Danish EEZ
    lat_std: float = 1.0
    lon_mean: float = 11.5
    lon_std: float = 2.0
    sog_max: float = 30.0  # Normalize SOG to [0, 1]
    dt_max: float = 300.0  # Cap dt_seconds at 5 minutes


# ============================================================================
# Data Loading
# ============================================================================

class AISTrackDataset(Dataset):
    """
    Efficient dataset for AIS track sequences.
    Pre-processes all tracks into fixed-length windows.
    """

    def __init__(self, track_data: Dict[str, np.ndarray],
                 seq_len: int, max_horizon: int, config: Config):
        """
        Args:
            track_data: Dict mapping track_id -> numpy array of shape (N, 5)
                        Features: [lat, lon, sog, cog, dt_seconds]
            seq_len: Sequence length for each sample
            max_horizon: Maximum prediction horizon
            config: Configuration object
        """
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        self.config = config

        # Build index of valid (track_id, start_idx) pairs
        self.samples = []
        for track_id, data in track_data.items():
            # Need seq_len + max_horizon positions
            min_len = seq_len + max_horizon
            if len(data) >= min_len:
                # Create overlapping windows with stride
                stride = max(1, seq_len // 4)  # 75% overlap
                for start in range(0, len(data) - min_len + 1, stride):
                    self.samples.append((track_id, start))

        self.track_data = track_data
        print(f"  Created {len(self.samples):,} training samples from {len(track_data)} tracks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_id, start = self.samples[idx]
        data = self.track_data[track_id]

        # Extract sequence + horizon
        end = start + self.seq_len + self.max_horizon
        seq = data[start:end].copy()

        # Handle NaN values FIRST
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        lat_norm = (seq[:, 0] - self.config.lat_mean) / self.config.lat_std
        lon_norm = (seq[:, 1] - self.config.lon_mean) / self.config.lon_std
        sog_norm = np.clip(seq[:, 2], 0, self.config.sog_max) / self.config.sog_max

        # cog: convert to sin/cos (handle edge cases)
        cog_rad = np.radians(seq[:, 3])
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)

        dt_norm = np.clip(seq[:, 4], 0, self.config.dt_max) / self.config.dt_max

        # Build feature array: [lat, lon, sog, cog_sin, cog_cos, dt]
        features = np.zeros((len(seq), 6), dtype=np.float32)
        features[:, 0] = lat_norm
        features[:, 1] = lon_norm
        features[:, 2] = sog_norm
        features[:, 3] = cog_sin
        features[:, 4] = cog_cos
        features[:, 5] = dt_norm

        # Final NaN check
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features)


def load_ais_data(config: Config, max_tracks: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load and preprocess AIS data from FSx - memory efficient version."""

    print("Loading track catalog...", flush=True)
    catalog = pl.read_parquet(config.catalog_path)

    # Filter tracks by minimum length and sample
    valid_tracks = catalog.filter(
        (pl.col("num_positions") >= config.min_track_length) &
        (pl.col("num_positions") <= 50000)  # Cap very long tracks
    ).sort("num_positions", descending=True)  # Prioritize longer tracks

    # Sample tracks if max_tracks is specified
    if max_tracks is not None and len(valid_tracks) > max_tracks:
        valid_tracks = valid_tracks.head(max_tracks)

    track_ids = valid_tracks.select("track_id").to_series().to_list()
    print(f"  Selected {len(track_ids)} tracks for training", flush=True)

    # Load data file by file to reduce memory
    print("Loading track data from FSx (streaming)...", flush=True)
    start = time.time()

    track_data = {}
    track_id_set = set(track_ids)

    # Process each parquet file
    import glob
    parquet_files = sorted(glob.glob(f"{config.data_path}/year=2025/**/*.parquet", recursive=True))

    for pf in parquet_files:
        df = pl.read_parquet(pf)
        df = df.filter(
            (pl.col("track_id").is_in(track_id_set)) &
            (pl.col("sog") >= config.min_sog)  # Filter for moving vessels only
        ).sort(["track_id", "timestamp"])

        if df.height == 0:
            continue

        df = df.select(["track_id", "lat", "lon", "sog", "cog", "dt_seconds"])

        for track_df in df.partition_by("track_id", maintain_order=True):
            track_id = track_df["track_id"][0]
            features = track_df.select(["lat", "lon", "sog", "cog", "dt_seconds"]).to_numpy().astype(np.float32)
            features[:, 3] = np.nan_to_num(features[:, 3], nan=0.0)

            if track_id in track_data:
                # Append to existing track (spans multiple files)
                track_data[track_id] = np.vstack([track_data[track_id], features])
            else:
                track_data[track_id] = features

        del df
        print(f"    Processed {pf.split('/')[-1]}, tracks so far: {len(track_data)}", flush=True)

    print(f"  Loaded {len(track_data)} tracks in {time.time()-start:.1f}s", flush=True)

    # Filter tracks that are too short after processing
    track_data = {k: v for k, v in track_data.items()
                  if len(v) >= config.min_track_length + config.max_horizon}

    total_positions = sum(len(v) for v in track_data.values())
    print(f"  Total positions: {total_positions:,}", flush=True)

    # Split into train/val (90/10 by track)
    track_ids = sorted(track_data.keys())  # Sort for deterministic order
    np.random.seed(42)  # Fixed seed for reproducible train/val split
    np.random.shuffle(track_ids)
    split_idx = int(0.9 * len(track_ids))

    train_ids = set(track_ids[:split_idx])
    val_ids = set(track_ids[split_idx:])

    train_data = {k: v for k, v in track_data.items() if k in train_ids}
    val_data = {k: v for k, v in track_data.items() if k in val_ids}

    print(f"  Train tracks: {len(train_data)}, Val tracks: {len(val_data)}", flush=True)

    return train_data, val_data


# ============================================================================
# Lazy Loading (Memory-Efficient for Large Datasets)
# ============================================================================

class LazyAISTrackDataset(Dataset):
    """
    Memory-efficient dataset that loads track data on-demand.

    Instead of loading all tracks into RAM upfront, this dataset:
    1. Builds a lightweight index of (track_id, start_idx, file_path) tuples
    2. Loads track data lazily when __getitem__ is called
    3. Uses an LRU cache to avoid re-reading frequently accessed tracks

    This enables training on datasets that don't fit in memory.
    """

    def __init__(self, track_index: List[Tuple[str, int, str]],
                 track_file_map: Dict[str, List[str]],
                 seq_len: int, max_horizon: int, config: Config,
                 cache_size: int = 1000):
        """
        Args:
            track_index: List of (track_id, start_idx, primary_file) tuples
            track_file_map: Dict mapping track_id -> list of parquet files containing it
            seq_len: Sequence length for each sample
            max_horizon: Maximum prediction horizon
            config: Configuration object
            cache_size: Number of tracks to keep in LRU cache
        """
        self.samples = track_index
        self.track_file_map = track_file_map
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        self.config = config
        self.cache_size = cache_size

        # LRU cache for loaded tracks
        from functools import lru_cache
        self._load_track = lru_cache(maxsize=cache_size)(self._load_track_impl)

        print(f"  Created {len(self.samples):,} training samples (lazy loading, cache={cache_size})")

    def _load_track_impl(self, track_id: str) -> np.ndarray:
        """Load a single track's data from parquet files."""
        files = self.track_file_map.get(track_id, [])
        if not files:
            raise ValueError(f"No files found for track_id: {track_id}")

        all_data = []
        for pf in files:
            df = pl.read_parquet(pf)
            df = df.filter(
                (pl.col("track_id") == track_id) &
                (pl.col("sog") >= self.config.min_sog)
            ).sort("timestamp")

            if df.height > 0:
                features = df.select(["lat", "lon", "sog", "cog", "dt_seconds"]).to_numpy().astype(np.float32)
                features[:, 3] = np.nan_to_num(features[:, 3], nan=0.0)
                all_data.append(features)

        if not all_data:
            # Return minimal valid array if no data found
            return np.zeros((self.seq_len + self.max_horizon, 5), dtype=np.float32)

        return np.vstack(all_data) if len(all_data) > 1 else all_data[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        track_id, start, _ = self.samples[idx]
        data = self._load_track(track_id)

        # Extract sequence + horizon
        end = start + self.seq_len + self.max_horizon
        seq = data[start:end].copy()

        # Pad if necessary (shouldn't happen with proper indexing)
        if len(seq) < self.seq_len + self.max_horizon:
            padded = np.zeros((self.seq_len + self.max_horizon, 5), dtype=np.float32)
            padded[:len(seq)] = seq
            seq = padded

        # Handle NaN values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        lat_norm = (seq[:, 0] - self.config.lat_mean) / self.config.lat_std
        lon_norm = (seq[:, 1] - self.config.lon_mean) / self.config.lon_std
        sog_norm = np.clip(seq[:, 2], 0, self.config.sog_max) / self.config.sog_max

        cog_rad = np.radians(seq[:, 3])
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)

        dt_norm = np.clip(seq[:, 4], 0, self.config.dt_max) / self.config.dt_max

        # Build feature array: [lat, lon, sog, cog_sin, cog_cos, dt]
        features = np.zeros((len(seq), 6), dtype=np.float32)
        features[:, 0] = lat_norm
        features[:, 1] = lon_norm
        features[:, 2] = sog_norm
        features[:, 3] = cog_sin
        features[:, 4] = cog_cos
        features[:, 5] = dt_norm

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features)

    def clear_cache(self):
        """Clear the track cache to free memory."""
        self._load_track.cache_clear()


def load_ais_data_lazy(config: Config, max_tracks: int = None) -> Tuple[List, List, Dict]:
    """
    Build lightweight index for lazy loading without loading actual track data.

    Returns:
        train_index: List of (track_id, start_idx, primary_file) for training
        val_index: List of (track_id, start_idx, primary_file) for validation
        track_file_map: Dict mapping track_id -> list of files containing it
    """
    import glob

    print("Building lazy-load index...", flush=True)
    start_time = time.time()

    # Load catalog to get track lengths
    print("  Loading track catalog...", flush=True)
    catalog = pl.read_parquet(config.catalog_path)

    # Filter tracks by minimum length
    min_len = config.min_track_length + config.max_horizon
    valid_tracks = catalog.filter(
        (pl.col("num_positions") >= min_len) &
        (pl.col("num_positions") <= 50000)
    ).sort("num_positions", descending=True)

    if max_tracks is not None and len(valid_tracks) > max_tracks:
        valid_tracks = valid_tracks.head(max_tracks)

    track_info = {row["track_id"]: row["num_positions"]
                  for row in valid_tracks.iter_rows(named=True)}

    print(f"  Found {len(track_info):,} valid tracks in catalog", flush=True)

    # Scan parquet files to build track -> file mapping
    print("  Scanning parquet files for track locations...", flush=True)
    parquet_files = sorted(glob.glob(f"{config.data_path}/year=2025/**/*.parquet", recursive=True))

    track_file_map = {}  # track_id -> [file1, file2, ...]
    track_lengths = {}   # track_id -> actual length after filtering

    for pf_idx, pf in enumerate(parquet_files):
        # Read just track_id column to find which tracks are in this file
        df = pl.read_parquet(pf, columns=["track_id", "sog"])

        # Filter for valid positions
        df = df.filter(
            (pl.col("track_id").is_in(list(track_info.keys()))) &
            (pl.col("sog") >= config.min_sog)
        )

        # Count positions per track in this file
        track_counts = df.group_by("track_id").agg(pl.len().alias("count"))

        for row in track_counts.iter_rows(named=True):
            tid = row["track_id"]
            count = row["count"]

            if tid not in track_file_map:
                track_file_map[tid] = []
                track_lengths[tid] = 0

            track_file_map[tid].append(pf)
            track_lengths[tid] += count

        if (pf_idx + 1) % 10 == 0 or pf_idx == len(parquet_files) - 1:
            print(f"    Scanned {pf_idx + 1}/{len(parquet_files)} files, "
                  f"found {len(track_file_map):,} tracks", flush=True)

        del df

    # Filter tracks that are too short after sog filtering
    valid_track_ids = [tid for tid, length in track_lengths.items()
                       if length >= min_len]

    print(f"  {len(valid_track_ids):,} tracks have sufficient length after filtering", flush=True)

    # Split into train/val (90/10)
    valid_track_ids = sorted(valid_track_ids)  # Sort for deterministic order
    np.random.seed(42)  # Fixed seed for reproducible train/val split
    np.random.shuffle(valid_track_ids)
    split_idx = int(0.9 * len(valid_track_ids))
    train_ids = set(valid_track_ids[:split_idx])
    val_ids = set(valid_track_ids[split_idx:])

    # Build sample indices
    stride = max(1, config.max_seq_len // 4)

    train_index = []
    val_index = []

    for tid in valid_track_ids:
        length = track_lengths[tid]
        primary_file = track_file_map[tid][0]  # Use first file as reference

        # Generate sample start indices
        for start in range(0, length - min_len + 1, stride):
            sample = (tid, start, primary_file)
            if tid in train_ids:
                train_index.append(sample)
            else:
                val_index.append(sample)

    print(f"  Train samples: {len(train_index):,}", flush=True)
    print(f"  Val samples: {len(val_index):,}", flush=True)
    print(f"  Index built in {time.time() - start_time:.1f}s", flush=True)
    print(f"  Memory: ~{(len(train_index) + len(val_index)) * 100 / 1e6:.1f} MB for index", flush=True)

    return train_index, val_index, track_file_map


# ============================================================================
# Chunked Loading (Conveyor Belt) for Very Large Datasets
# ============================================================================

class ChunkedAISDataset:
    """
    Memory-efficient dataset using chunked/conveyor-belt loading.

    Instead of loading all data or doing per-sample I/O:
    1. Splits tracks into N chunks (e.g., 10)
    2. Loads one chunk into RAM at a time
    3. Pre-loads next chunk in background thread while training
    4. Swaps buffers when current chunk is exhausted

    This ensures training is never waiting for I/O.
    """

    def __init__(self, track_ids: List[str], track_file_map: Dict[str, List[str]],
                 seq_len: int, max_horizon: int, config: Config,
                 num_chunks: int = 10, prefetch: bool = True):
        """
        Args:
            track_ids: List of track IDs to include
            track_file_map: Dict mapping track_id -> list of parquet files
            seq_len: Sequence length for each sample
            max_horizon: Maximum prediction horizon
            config: Configuration object
            num_chunks: Number of chunks to split data into (default: 10)
            prefetch: Whether to prefetch next chunk in background (default: True)
        """
        self.track_file_map = track_file_map
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        self.config = config
        self.num_chunks = num_chunks
        self.prefetch = prefetch

        # Split track IDs into chunks
        np.random.seed(42)
        shuffled_ids = list(track_ids)
        np.random.shuffle(shuffled_ids)

        chunk_size = len(shuffled_ids) // num_chunks
        self.track_chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(shuffled_ids)
            self.track_chunks.append(shuffled_ids[start:end])

        print(f"  Split {len(track_ids):,} tracks into {num_chunks} chunks "
              f"(~{chunk_size:,} tracks each)", flush=True)

        # Current state
        self.current_chunk_idx = -1
        self.current_data = None  # Dict[track_id -> np.ndarray]
        self.current_samples = []  # List of (track_id, start_idx)
        self.sample_idx = 0

        # Background loading
        self.next_data = None
        self.next_samples = None
        self.loader_thread = None
        self.loading_chunk_idx = -1

        import threading
        self.lock = threading.Lock()
        self.load_complete = threading.Event()

        # Estimated samples per chunk (for progress reporting)
        self._estimate_samples_per_chunk()

    def _estimate_samples_per_chunk(self):
        """Estimate total samples based on first chunk."""
        # We'll update this after loading first chunk
        self.estimated_total_samples = 0

    def _load_chunk_data(self, chunk_idx: int) -> Tuple[Dict[str, np.ndarray], List]:
        """Load all track data for a chunk into memory."""
        track_ids = self.track_chunks[chunk_idx]
        track_data = {}
        samples = []

        min_len = self.seq_len + self.max_horizon
        stride = max(1, self.seq_len // 4)

        # Group tracks by file for efficient loading
        file_to_tracks = {}
        for tid in track_ids:
            files = self.track_file_map.get(tid, [])
            for f in files:
                if f not in file_to_tracks:
                    file_to_tracks[f] = []
                file_to_tracks[f].append(tid)

        # Load data from each file
        for pf, tids in file_to_tracks.items():
            try:
                df = pl.read_parquet(pf)
                df = df.filter(
                    (pl.col("track_id").is_in(tids)) &
                    (pl.col("sog") >= self.config.min_sog)
                ).sort(["track_id", "timestamp"])

                if df.height == 0:
                    continue

                df = df.select(["track_id", "lat", "lon", "sog", "cog", "dt_seconds"])

                for track_df in df.partition_by("track_id", maintain_order=True):
                    tid = track_df["track_id"][0]
                    features = track_df.select(["lat", "lon", "sog", "cog", "dt_seconds"]).to_numpy().astype(np.float32)
                    features[:, 3] = np.nan_to_num(features[:, 3], nan=0.0)

                    if tid in track_data:
                        track_data[tid] = np.vstack([track_data[tid], features])
                    else:
                        track_data[tid] = features

                del df
            except Exception as e:
                print(f"    Warning: Error loading {pf}: {e}", flush=True)
                continue

        # Build sample indices for this chunk
        for tid, data in track_data.items():
            if len(data) >= min_len:
                for start in range(0, len(data) - min_len + 1, stride):
                    samples.append((tid, start))

        # Shuffle samples within chunk
        np.random.shuffle(samples)

        return track_data, samples

    def _background_loader(self, chunk_idx: int):
        """Background thread function to load next chunk."""
        data, samples = self._load_chunk_data(chunk_idx)
        with self.lock:
            self.next_data = data
            self.next_samples = samples
            self.loading_chunk_idx = chunk_idx
        self.load_complete.set()

    def load_next_chunk(self):
        """Load the next chunk, using prefetched data if available."""
        import threading

        self.current_chunk_idx = (self.current_chunk_idx + 1) % self.num_chunks

        # Check if we have prefetched data for this chunk
        with self.lock:
            if self.next_data is not None and self.loading_chunk_idx == self.current_chunk_idx:
                # Use prefetched data
                self.current_data = self.next_data
                self.current_samples = self.next_samples
                self.next_data = None
                self.next_samples = None
                print(f"  Using prefetched chunk {self.current_chunk_idx + 1}/{self.num_chunks} "
                      f"({len(self.current_samples):,} samples)", flush=True)
            else:
                # Need to load synchronously
                print(f"  Loading chunk {self.current_chunk_idx + 1}/{self.num_chunks}...", flush=True)
                self.current_data, self.current_samples = self._load_chunk_data(self.current_chunk_idx)
                print(f"  Loaded {len(self.current_samples):,} samples from "
                      f"{len(self.current_data):,} tracks", flush=True)

        self.sample_idx = 0

        # Update estimated total samples
        if self.estimated_total_samples == 0:
            self.estimated_total_samples = len(self.current_samples) * self.num_chunks

        # Start prefetching next chunk in background
        if self.prefetch:
            next_chunk_idx = (self.current_chunk_idx + 1) % self.num_chunks
            self.load_complete.clear()
            self.loader_thread = threading.Thread(
                target=self._background_loader, args=(next_chunk_idx,))
            self.loader_thread.daemon = True
            self.loader_thread.start()

    def __len__(self):
        """Return current chunk size (or estimated total)."""
        if self.current_samples:
            return len(self.current_samples)
        return self.estimated_total_samples // self.num_chunks

    def get_sample(self, idx: int) -> torch.Tensor:
        """Get a single sample from current chunk."""
        track_id, start = self.current_samples[idx]
        data = self.current_data[track_id]

        # Extract sequence + horizon
        end = start + self.seq_len + self.max_horizon
        seq = data[start:end].copy()

        # Pad if necessary
        if len(seq) < self.seq_len + self.max_horizon:
            padded = np.zeros((self.seq_len + self.max_horizon, 5), dtype=np.float32)
            padded[:len(seq)] = seq
            seq = padded

        # Handle NaN values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        lat_norm = (seq[:, 0] - self.config.lat_mean) / self.config.lat_std
        lon_norm = (seq[:, 1] - self.config.lon_mean) / self.config.lon_std
        sog_norm = np.clip(seq[:, 2], 0, self.config.sog_max) / self.config.sog_max

        cog_rad = np.radians(seq[:, 3])
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)

        dt_norm = np.clip(seq[:, 4], 0, self.config.dt_max) / self.config.dt_max

        # Build feature array: [lat, lon, sog, cog_sin, cog_cos, dt]
        features = np.zeros((len(seq), 6), dtype=np.float32)
        features[:, 0] = lat_norm
        features[:, 1] = lon_norm
        features[:, 2] = sog_norm
        features[:, 3] = cog_sin
        features[:, 4] = cog_cos
        features[:, 5] = dt_norm

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(features)

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Get a batch of samples, loading next chunk if needed."""
        if self.current_data is None or self.sample_idx >= len(self.current_samples):
            self.load_next_chunk()

        # Get batch indices
        end_idx = min(self.sample_idx + batch_size, len(self.current_samples))
        batch_indices = range(self.sample_idx, end_idx)
        self.sample_idx = end_idx

        # Build batch
        batch = torch.stack([self.get_sample(i) for i in batch_indices])
        return batch

    def has_more_samples(self) -> bool:
        """Check if there are more samples in current chunk."""
        return self.sample_idx < len(self.current_samples)

    def get_progress(self) -> Tuple[int, int, int]:
        """Return (current_chunk, samples_in_chunk, sample_idx)."""
        return self.current_chunk_idx + 1, len(self.current_samples), self.sample_idx

    def reset_epoch(self):
        """Reset for new epoch - reshuffles chunks."""
        self.current_chunk_idx = -1
        self.sample_idx = 0
        # Shuffle chunk order for new epoch
        np.random.shuffle(self.track_chunks)
        # Clear prefetched data
        with self.lock:
            self.next_data = None
            self.next_samples = None


class ChunkedDataLoader:
    """
    Wrapper that makes ChunkedAISDataset behave like a DataLoader.

    Provides an iterator interface for seamless integration with existing training loops.
    """

    def __init__(self, chunked_dataset: ChunkedAISDataset, batch_size: int):
        self.dataset = chunked_dataset
        self.batch_size = batch_size
        self._len = None

    def __iter__(self):
        """Iterate through all chunks, yielding batches."""
        self.dataset.reset_epoch()

        for chunk_idx in range(self.dataset.num_chunks):
            self.dataset.load_next_chunk()

            while self.dataset.has_more_samples():
                batch = self.dataset.get_batch(self.batch_size)
                yield batch

    def __len__(self):
        """Estimate total batches across all chunks."""
        if self._len is None:
            # Load first chunk to estimate
            if self.dataset.current_data is None:
                self.dataset.load_next_chunk()
            samples_per_chunk = len(self.dataset.current_samples)
            batches_per_chunk = (samples_per_chunk + self.batch_size - 1) // self.batch_size
            self._len = batches_per_chunk * self.dataset.num_chunks
        return self._len


def build_chunked_dataset(config: Config, num_chunks: int = 10,
                          max_tracks: int = None) -> Tuple['ChunkedAISDataset', 'ChunkedAISDataset']:
    """
    Build chunked datasets for training and validation.

    Returns:
        train_dataset: ChunkedAISDataset for training
        val_dataset: ChunkedAISDataset for validation (smaller, eager-loaded)
    """
    import glob

    print("Building chunked dataset index...", flush=True)
    start_time = time.time()

    # Load catalog
    print("  Loading track catalog...", flush=True)
    catalog = pl.read_parquet(config.catalog_path)

    # Filter tracks by minimum length
    min_len = config.min_track_length + config.max_horizon
    valid_tracks = catalog.filter(
        (pl.col("num_positions") >= min_len) &
        (pl.col("num_positions") <= 50000)
    ).sort("num_positions", descending=True)

    if max_tracks is not None and len(valid_tracks) > max_tracks:
        valid_tracks = valid_tracks.head(max_tracks)

    track_ids = valid_tracks.select("track_id").to_series().to_list()
    print(f"  Found {len(track_ids):,} valid tracks", flush=True)

    # Build track -> file mapping
    print("  Scanning parquet files...", flush=True)
    parquet_files = sorted(glob.glob(f"{config.data_path}/year=*/**/*.parquet", recursive=True))

    track_file_map = {}
    track_id_set = set(track_ids)

    for pf_idx, pf in enumerate(parquet_files):
        try:
            df = pl.read_parquet(pf, columns=["track_id"])
            tracks_in_file = set(df["track_id"].unique().to_list()) & track_id_set

            for tid in tracks_in_file:
                if tid not in track_file_map:
                    track_file_map[tid] = []
                track_file_map[tid].append(pf)

            del df
        except Exception as e:
            print(f"    Warning: Error scanning {pf}: {e}", flush=True)
            continue

        if (pf_idx + 1) % 20 == 0 or pf_idx == len(parquet_files) - 1:
            print(f"    Scanned {pf_idx + 1}/{len(parquet_files)} files", flush=True)

    # Filter to tracks we found in files
    valid_track_ids = [tid for tid in track_ids if tid in track_file_map]
    print(f"  {len(valid_track_ids):,} tracks found in parquet files", flush=True)

    # Split into train/val (90/10)
    np.random.seed(42)
    np.random.shuffle(valid_track_ids)
    split_idx = int(0.9 * len(valid_track_ids))
    train_ids = valid_track_ids[:split_idx]
    val_ids = valid_track_ids[split_idx:]

    print(f"  Train tracks: {len(train_ids):,}", flush=True)
    print(f"  Val tracks: {len(val_ids):,}", flush=True)

    # Create chunked dataset for training
    print(f"\nCreating chunked training dataset ({num_chunks} chunks)...", flush=True)
    train_dataset = ChunkedAISDataset(
        train_ids, track_file_map,
        config.max_seq_len, config.max_horizon, config,
        num_chunks=num_chunks, prefetch=True
    )

    # Load validation data EAGERLY (not chunked) - val set is small enough to fit in memory
    print(f"\nLoading validation data eagerly ({len(val_ids):,} tracks)...", flush=True)
    val_track_data = {}

    # Group tracks by file for efficient loading
    file_to_tracks = {}
    for tid in val_ids:
        files = track_file_map.get(tid, [])
        for f in files:
            if f not in file_to_tracks:
                file_to_tracks[f] = []
            file_to_tracks[f].append(tid)

    # Load data from each file
    files_processed = 0
    total_files = len(file_to_tracks)
    for pf, tids in file_to_tracks.items():
        try:
            df = pl.read_parquet(pf)
            df = df.filter(
                (pl.col("track_id").is_in(tids)) &
                (pl.col("sog") >= config.min_sog)
            ).sort(["track_id", "timestamp"])

            if df.height == 0:
                continue

            df = df.select(["track_id", "lat", "lon", "sog", "cog", "dt_seconds"])

            for track_df in df.partition_by("track_id", maintain_order=True):
                tid = track_df["track_id"][0]
                features = track_df.select(["lat", "lon", "sog", "cog", "dt_seconds"]).to_numpy().astype(np.float32)
                features[:, 3] = np.nan_to_num(features[:, 3], nan=0.0)

                if tid in val_track_data:
                    val_track_data[tid] = np.vstack([val_track_data[tid], features])
                else:
                    val_track_data[tid] = features

            del df
        except Exception as e:
            print(f"    Warning: Error loading {pf}: {e}", flush=True)
            continue

        files_processed += 1
        if files_processed % 50 == 0 or files_processed == total_files:
            print(f"    Processed {files_processed}/{total_files} files for validation", flush=True)

    print(f"  Loaded {len(val_track_data):,} validation tracks", flush=True)

    # Create regular AISTrackDataset for validation (not chunked)
    val_dataset = AISTrackDataset(
        val_track_data, config.max_seq_len, config.max_horizon, config
    )

    print(f"\nDataset setup complete in {time.time() - start_time:.1f}s", flush=True)

    # Return train_chunked (ChunkedAISDataset), val_dataset (AISTrackDataset), val_track_data (for memory)
    return train_dataset, val_dataset, val_track_data


# ============================================================================
# S3 Shard Loading with Double Buffering
# ============================================================================

class ShardBuffer:
    """Holds loaded shard data and sample indices."""

    def __init__(self):
        self.track_data: Dict[str, np.ndarray] = {}
        self.samples: List[Tuple[str, int]] = []
        self.shard_ids: List[int] = []
        self.ready = threading.Event()
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.track_data.clear()
            self.samples.clear()
            self.shard_ids.clear()
            self.ready.clear()

    def set_data(self, track_data: Dict[str, np.ndarray],
                 samples: List[Tuple[str, int]], shard_ids: List[int]):
        with self.lock:
            self.track_data = track_data
            self.samples = samples
            self.shard_ids = shard_ids
            self.ready.set()

    def wait_ready(self, timeout: float = None) -> bool:
        return self.ready.wait(timeout)

    def is_ready(self) -> bool:
        return self.ready.is_set()


class DoubleBufferShardLoader:
    """
    Double-buffered S3 shard loader.
    Trains on buffer A while prefetching into buffer B.
    """

    def __init__(self, config: Config, shard_ids: List[int],
                 shards_per_buffer: int = 10):
        self.config = config
        self.all_shard_ids = shard_ids.copy()
        self.shards_per_buffer = shards_per_buffer

        # Double buffers
        self.buffer_a = ShardBuffer()
        self.buffer_b = ShardBuffer()
        self.active_buffer = self.buffer_a
        self.loading_buffer = self.buffer_b

        # Track which shards we've used
        self.next_shard_idx = 0
        self.epoch = 0

        # Background loading thread
        self.prefetch_thread: Optional[threading.Thread] = None
        self.stop_prefetch = threading.Event()

        # Stats
        self.load_times: List[float] = []

    def _get_next_shard_batch(self) -> List[int]:
        """Get next batch of shard IDs to load."""
        batch = []
        for _ in range(self.shards_per_buffer):
            if self.next_shard_idx >= len(self.all_shard_ids):
                self.next_shard_idx = 0
                self.epoch += 1
                np.random.shuffle(self.all_shard_ids)
            batch.append(self.all_shard_ids[self.next_shard_idx])
            self.next_shard_idx += 1
        return batch

    def _load_shards(self, shard_ids: List[int], buffer: ShardBuffer):
        """Load shards into buffer."""
        start_time = time.time()
        track_data = {}
        samples = []

        min_len = self.config.max_seq_len + self.config.max_horizon
        stride = max(1, self.config.max_seq_len // 4)

        for shard_id in shard_ids:
            if self.stop_prefetch.is_set():
                return

            shard_path = f"{self.config.s3_bucket}/shard={shard_id:03d}/tracks.parquet"
            try:
                df = pl.read_parquet(shard_path)

                # Filter for moving vessels
                df = df.filter(pl.col("sog") >= self.config.min_sog)

                # Process each track
                for track_df in df.partition_by("track_id", maintain_order=True):
                    track_id = track_df["track_id"][0]

                    # Extract features
                    features = track_df.select([
                        "lat", "lon", "sog", "cog", "dt_seconds"
                    ]).to_numpy().astype(np.float32)

                    # Handle NaN in cog
                    features[:, 3] = np.nan_to_num(features[:, 3], nan=0.0)

                    # Only keep tracks long enough
                    if len(features) >= min_len:
                        track_data[track_id] = features

                        # Create sample indices
                        for start in range(0, len(features) - min_len + 1, stride):
                            samples.append((track_id, start))

                del df

            except Exception as e:
                print(f"Warning: Error loading shard {shard_id}: {e}", flush=True)
                continue

        # Shuffle samples
        np.random.shuffle(samples)

        # Set buffer data
        buffer.set_data(track_data, samples, shard_ids)

        elapsed = time.time() - start_time
        self.load_times.append(elapsed)
        print(f"  Loaded {len(shard_ids)} shards ({len(track_data)} tracks, "
              f"{len(samples):,} samples) in {elapsed:.1f}s", flush=True)

    def start_prefetch(self):
        """Start prefetching next batch into loading buffer."""
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            return  # Already prefetching

        shard_ids = self._get_next_shard_batch()
        self.loading_buffer.clear()

        self.prefetch_thread = threading.Thread(
            target=self._load_shards,
            args=(shard_ids, self.loading_buffer),
            daemon=True
        )
        self.prefetch_thread.start()

    def load_initial(self):
        """Load initial buffer (blocking)."""
        print("Loading initial buffer...", flush=True)
        shard_ids = self._get_next_shard_batch()
        self._load_shards(shard_ids, self.active_buffer)

        # Start prefetching next batch
        self.start_prefetch()

    def swap_buffers(self):
        """Swap active and loading buffers."""
        # Wait for prefetch to complete
        if not self.loading_buffer.wait_ready(timeout=60):
            print("Warning: Prefetch not ready, waiting...", flush=True)
            self.loading_buffer.wait_ready()

        # Swap
        self.active_buffer, self.loading_buffer = self.loading_buffer, self.active_buffer

        # Start prefetching into the now-free buffer
        self.start_prefetch()

        print(f"  Swapped buffers. Now training on shards {self.active_buffer.shard_ids}",
              flush=True)

    def stop(self):
        """Stop prefetching."""
        self.stop_prefetch.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)

    def get_active_data(self) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, int]]]:
        """Get current active buffer data."""
        return self.active_buffer.track_data, self.active_buffer.samples


class S3ShardDataset(Dataset):
    """Dataset that wraps a ShardBuffer for PyTorch DataLoader."""

    def __init__(self, buffer: ShardBuffer, config: Config):
        self.buffer = buffer
        self.config = config

    def __len__(self):
        return len(self.buffer.samples)

    def __getitem__(self, idx):
        track_id, start = self.buffer.samples[idx]
        data = self.buffer.track_data[track_id]

        # Extract sequence + horizon
        end = start + self.config.max_seq_len + self.config.max_horizon
        seq = data[start:end].copy()

        # Handle NaN values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        lat_norm = (seq[:, 0] - self.config.lat_mean) / self.config.lat_std
        lon_norm = (seq[:, 1] - self.config.lon_mean) / self.config.lon_std
        sog_norm = np.clip(seq[:, 2], 0, self.config.sog_max) / self.config.sog_max
        cog_rad = np.radians(seq[:, 3])
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)
        dt_norm = np.clip(seq[:, 4], 0, self.config.dt_max) / self.config.dt_max

        # Stack features: [lat, lon, sog, cog_sin, cog_cos, dt]
        features = np.stack([lat_norm, lon_norm, sog_norm, cog_sin, cog_cos, dt_norm], axis=1)

        return torch.from_numpy(features.astype(np.float32))


class S3ShardDataLoader:
    """DataLoader-like wrapper for S3 shard double-buffered training."""

    def __init__(self, shard_loader: DoubleBufferShardLoader, config: Config, batch_size: int):
        self.shard_loader = shard_loader
        self.config = config
        self.batch_size = batch_size
        self._len = None

    def __iter__(self):
        """Iterate through batches, swapping buffers when exhausted."""
        while True:
            # Create dataset from current active buffer
            dataset = S3ShardDataset(self.shard_loader.active_buffer, self.config)

            if len(dataset) == 0:
                print("Warning: Empty buffer, swapping...", flush=True)
                self.shard_loader.swap_buffers()
                continue

            # Create DataLoader for this buffer
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            # Yield batches from this buffer
            for batch in loader:
                yield batch

            # Buffer exhausted, swap
            self.shard_loader.swap_buffers()

    def __len__(self):
        """Estimate total batches based on initial buffer."""
        if self._len is None:
            samples = len(self.shard_loader.active_buffer.samples)
            batches_per_buffer = (samples + self.batch_size - 1) // self.batch_size
            # Estimate based on 256 total shards
            num_buffers = 256 // self.shard_loader.shards_per_buffer
            self._len = batches_per_buffer * num_buffers
        return self._len


# ============================================================================
# Materialized Data Loading (MosaicML Streaming)
# ============================================================================

try:
    from streaming import StreamingDataset, Stream
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False
    print("Warning: mosaicml-streaming not installed. Install with: pip install mosaicml-streaming")


class MDSCollator:
    """Collate function for MDS streaming data - normalizes on the fly."""

    def __init__(self, config: Config):
        self.config = config
        self.window_size = config.max_seq_len + config.max_horizon  # 928

    def __call__(self, batch):
        """Convert batch of raw samples to normalized tensor."""
        features_list = []
        for sample in batch:
            # Decode bytes to numpy
            raw = np.frombuffer(sample['features'], dtype=np.float32).reshape(self.window_size, 5)

            # Normalize: raw is [lat, lon, sog, cog, dt_seconds]
            lat_norm = (raw[:, 0] - self.config.lat_mean) / self.config.lat_std
            lon_norm = (raw[:, 1] - self.config.lon_mean) / self.config.lon_std
            sog_norm = np.clip(raw[:, 2], 0, self.config.sog_max) / self.config.sog_max
            cog_rad = np.radians(raw[:, 3])
            cog_sin = np.sin(cog_rad)
            cog_cos = np.cos(cog_rad)
            dt_norm = np.clip(raw[:, 4], 0, self.config.dt_max) / self.config.dt_max

            # Stack: [lat, lon, sog, cog_sin, cog_cos, dt] - 6 features
            normalized = np.stack([lat_norm, lon_norm, sog_norm, cog_sin, cog_cos, dt_norm], axis=1)
            features_list.append(normalized)

        return torch.from_numpy(np.stack(features_list).astype(np.float32))


class LocalValidationStreamer:
    """
    Downloads validation parquet ONCE to local disk,
    then pre-loads the required batches into pinned memory for instant GPU access.

    Since we only use max_batches (e.g., 200) for validation, we pre-load
    those batches into pinned memory at init. Iteration is instant - zero I/O.
    """

    def __init__(self, config: Config, s3_path: str, cache_dir: str = '/home/ec2-user/mds-cache',
                 batch_size: int = 64, max_batches: int = 200):
        self.config = config
        self.window_size = config.max_seq_len + config.max_horizon
        self.batch_size = batch_size
        self.max_batches = max_batches

        # Local cache path
        os.makedirs(cache_dir, exist_ok=True)
        self.local_path = f"{cache_dir}/validation.parquet"

        # Download if not cached
        if not os.path.exists(self.local_path):
            print(f"  Downloading validation parquet to local disk...", flush=True)
            self._download_from_s3(s3_path)
        else:
            size_gb = os.path.getsize(self.local_path) / 1e9
            print(f"  Using cached validation: {self.local_path} ({size_gb:.2f} GB)", flush=True)

        # Count samples and determine batches to load
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.local_path)
        self.num_samples = pf.metadata.num_rows
        total_batches = (self.num_samples + batch_size - 1) // batch_size
        self.num_batches_to_load = min(max_batches, total_batches)

        print(f"  Validation: {self.num_samples:,} samples, {total_batches} total batches", flush=True)
        print(f"  Pre-loading {self.num_batches_to_load} batches into pinned memory...", flush=True)

        # Pre-load batches into pinned memory (one-time cost at startup)
        start = time.time()
        self.cached_batches = self._preload_batches()
        mem_mb = sum(b.numel() * 4 for b in self.cached_batches) / 1e6
        print(f"  Loaded {len(self.cached_batches)} batches ({mem_mb:.1f} MB) in {time.time()-start:.1f}s", flush=True)

    def _download_from_s3(self, s3_path: str):
        """Download parquet from S3 to local disk."""
        import subprocess
        start = time.time()
        result = subprocess.run(
            ['aws', 's3', 'cp', s3_path, self.local_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download validation: {result.stderr}")
        size_gb = os.path.getsize(self.local_path) / 1e9
        print(f"  Downloaded {size_gb:.2f} GB in {time.time() - start:.1f}s", flush=True)

    def _normalize_batch_vectorized(self, features_list):
        """Normalize entire batch vectorized and return pinned tensor."""
        raw = np.array(features_list, dtype=np.float32).reshape(-1, self.window_size, 5)

        lat_norm = (raw[:, :, 0] - self.config.lat_mean) / self.config.lat_std
        lon_norm = (raw[:, :, 1] - self.config.lon_mean) / self.config.lon_std
        sog_norm = np.clip(raw[:, :, 2], 0, self.config.sog_max) / self.config.sog_max
        cog_rad = np.radians(raw[:, :, 3])
        cog_sin = np.sin(cog_rad)
        cog_cos = np.cos(cog_rad)
        dt_norm = np.clip(raw[:, :, 4], 0, self.config.dt_max) / self.config.dt_max

        normalized = np.stack([lat_norm, lon_norm, sog_norm, cog_sin, cog_cos, dt_norm], axis=2)

        tensor = torch.from_numpy(normalized.astype(np.float32))
        if torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor

    def _preload_batches(self):
        """Load first max_batches into pinned memory."""
        import pyarrow.parquet as pq

        batches = []
        pf = pq.ParquetFile(self.local_path)

        for i, batch in enumerate(pf.iter_batches(batch_size=self.batch_size)):
            if i >= self.num_batches_to_load:
                break
            features_list = batch['features'].to_pylist()
            tensor = self._normalize_batch_vectorized(features_list)
            batches.append(tensor)

        return batches

    def __iter__(self):
        """Iterate over pre-loaded batches - instant, zero I/O."""
        for batch in self.cached_batches:
            yield batch

    def __len__(self):
        """Number of cached batches."""
        return len(self.cached_batches)


def build_mds_dataset(config: Config, batch_size: int, cache_dir: str = '/home/ec2-user/mds-cache'):
    """
    Build MosaicML streaming dataset for training.

    Args:
        config: Configuration object
        batch_size: Batch size for training
        cache_dir: Local cache directory for MDS shards

    Returns:
        train_dataset: StreamingDataset for training
        collate_fn: Collate function for DataLoader
    """
    if not HAS_STREAMING:
        raise ImportError("mosaicml-streaming required. Install with: pip install mosaicml-streaming")

    print("Building MDS STREAMING dataset...", flush=True)
    start_time = time.time()

    # Create streams for all 16 workers
    streams = []
    for i in range(16):
        streams.append(Stream(
            remote=f's3://ais-pipeline-data-10179bbf-us-east-1/mds/worker_{i:02d}',
            local=f'{cache_dir}/worker_{i:02d}',
        ))

    # Create StreamingDataset
    dataset = StreamingDataset(
        streams=streams,
        shuffle=False,               # Data is pre-shuffled, sequential for cache efficiency
        batch_size=batch_size,       # Required for streaming
        predownload=batch_size * 4,  # Prefetch more shards ahead for no gaps
        cache_limit='50gb',          # Cap local cache size
    )

    print(f"  Training samples: {len(dataset):,}", flush=True)
    print(f"  Workers: 16 (MDS format)", flush=True)
    print(f"  Cache: {cache_dir} (limit 50GB)", flush=True)
    print(f"  Setup complete in {time.time() - start_time:.1f}s", flush=True)

    # Create collate function
    collate_fn = MDSCollator(config)

    return dataset, collate_fn


def build_materialized_dataset(config: Config, batch_size: int, cache_dir: str = '/home/ec2-user/mds-cache',
                                skip_validation: bool = False):
    """
    Build MDS streaming training dataset and local validation streamer.

    Args:
        config: Configuration object
        batch_size: Batch size
        cache_dir: Local cache directory for MDS and validation
        skip_validation: If True, skip loading validation (used before batch size finding)

    Returns:
        train_dataset: StreamingDataset
        train_collate: Collate function for training
        val_streamer: LocalValidationStreamer or None if skip_validation=True
    """
    # Build MDS training dataset
    train_dataset, train_collate = build_mds_dataset(config, batch_size, cache_dir)

    if skip_validation:
        return train_dataset, train_collate, None

    # Build validation streamer - pre-load batches into pinned memory for instant GPU access
    # Use fixed smaller batch size for validation (48) to leave memory headroom after training
    val_batch_size = 48  # Fixed, not tied to training batch size
    val_path = "s3://ais-pipeline-data-10179bbf-us-east-1/materialized/validation.parquet"
    val_streamer = LocalValidationStreamer(
        config, val_path, cache_dir,
        batch_size=val_batch_size,
        max_batches=config.val_max_batches  # Only load what we'll use (200 batches)
    )

    return train_dataset, train_collate, val_streamer


def build_s3_shard_dataset(config: Config, shards_per_buffer: int = 10,
                           max_val_tracks: int = 500) -> Tuple[DoubleBufferShardLoader, 'AISTrackDataset']:
    """
    Build S3 shard datasets for training and validation.

    Args:
        config: Configuration object
        shards_per_buffer: Number of shards per double-buffer
        max_val_tracks: Maximum number of tracks for validation (default: 500)

    Returns:
        shard_loader: DoubleBufferShardLoader for training
        val_dataset: AISTrackDataset for validation (eagerly loaded)
    """
    print("Building S3 shard dataset...", flush=True)
    start_time = time.time()

    # Reserve shard 255 exclusively for validation - never train on it!
    val_shard_id = 255
    train_shard_ids = list(range(255))  # 0-254 only, excludes 255
    print(f"  Train shards: {len(train_shard_ids)} (0-254, excluding val shard {val_shard_id})", flush=True)
    print(f"  Shards per buffer: {shards_per_buffer}", flush=True)

    # Shuffle training shards
    np.random.seed(42)
    np.random.shuffle(train_shard_ids)

    # Create double-buffer loader for training
    shard_loader = DoubleBufferShardLoader(config, train_shard_ids, shards_per_buffer)

    # Load initial training buffer
    shard_loader.load_initial()

    # Load ALL tracks from validation shard (shard 255)
    print(f"\nLoading validation data (all valid tracks from shard {val_shard_id})...", flush=True)
    val_track_data = {}
    min_len = config.max_seq_len + config.max_horizon

    shard_path = f"{config.s3_bucket}/shard={val_shard_id:03d}/tracks.parquet"
    try:
        df = pl.read_parquet(shard_path)
        df = df.filter(pl.col("sog") >= config.min_sog)

        for track_df in df.partition_by("track_id", maintain_order=True):
            track_id = track_df["track_id"][0]
            features = track_df.select([
                "lat", "lon", "sog", "cog", "dt_seconds"
            ]).to_numpy().astype(np.float32)
            features[:, 3] = np.nan_to_num(features[:, 3], nan=0.0)

            if len(features) >= min_len:
                val_track_data[track_id] = features

        del df
        print(f"  Shard {val_shard_id}: loaded {len(val_track_data)} valid tracks", flush=True)

    except Exception as e:
        print(f"ERROR: Failed to load validation shard {val_shard_id}: {e}", flush=True)
        raise

    # Create validation dataset
    val_dataset = AISTrackDataset(
        val_track_data, config.max_seq_len, config.max_horizon, config
    )

    print(f"\nS3 shard setup complete in {time.time() - start_time:.1f}s", flush=True)
    print(f"  Training: {len(train_shard_ids)} shards with double-buffering", flush=True)
    print(f"  Validation: {len(val_track_data)} tracks, {len(val_dataset)} samples", flush=True)

    return shard_loader, val_dataset


# ============================================================================
# Model Architecture
# ============================================================================

class SinusoidalEncoding(nn.Module):
    """Sinusoidal encoding for continuous values."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class FourierHead2D(nn.Module):
    """2D Fourier density head."""

    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 12, grid_range: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        num_freq_pairs = (2 * num_freqs + 1) ** 2
        self.coeff_predictor = nn.Linear(d_model, 2 * num_freq_pairs)

        # Precompute grid and basis
        x = torch.linspace(-grid_range, grid_range, grid_size)
        y = torch.linspace(-grid_range, grid_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        # Clone to ensure contiguous memory (needed for state_dict loading)
        self.register_buffer('grid_x', xx.clone())
        self.register_buffer('grid_y', yy.clone())

        freqs = torch.arange(-num_freqs, num_freqs + 1, dtype=torch.float)
        freq_x, freq_y = torch.meshgrid(freqs, freqs, indexing='ij')
        self.register_buffer('freq_x', freq_x.flatten())
        self.register_buffer('freq_y', freq_y.flatten())

        # Precompute phase matrix
        L = 2 * grid_range
        phase = (2 * np.pi / L) * (
            self.freq_x.view(1, 1, -1) * xx.flatten().view(1, -1, 1) +
            self.freq_y.view(1, 1, -1) * yy.flatten().view(1, -1, 1)
        )
        self.register_buffer('cos_basis', torch.cos(phase).squeeze(0))
        self.register_buffer('sin_basis', torch.sin(phase).squeeze(0))

        nn.init.normal_(self.coeff_predictor.weight, std=0.01)
        nn.init.zeros_(self.coeff_predictor.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        coeffs = self.coeff_predictor(z)

        num_freq_pairs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_freq_pairs]
        sin_coeffs = coeffs[:, num_freq_pairs:]

        logits = cos_coeffs @ self.cos_basis.T + sin_coeffs @ self.sin_basis.T
        log_density = F.log_softmax(logits, dim=-1)

        return log_density.view(batch_size, self.grid_size, self.grid_size)


class CausalAISModel(nn.Module):
    """Causal transformer for AIS trajectory prediction."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Input projection: 6 features -> d_model
        self.input_proj = nn.Linear(6, config.d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalEncoding(config.d_model)

        # Time encoding for horizon conditioning
        self.time_proj = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Horizon conditioning
        self.horizon_proj = nn.Linear(config.d_model * 2, config.d_model)

        # Fourier head
        self.fourier_head = FourierHead2D(
            config.d_model, config.grid_size, config.num_freqs, config.grid_range
        )

        # Causal mask cache
        self._causal_masks = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device):
        if seq_len not in self._causal_masks:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self._causal_masks[seq_len] = mask
        return self._causal_masks[seq_len].to(device)

    def forward_train(self, features: torch.Tensor, horizon_indices: torch.Tensor = None,
                      causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with random horizon sampling.

        Args:
            features: (batch, seq_len + max_horizon, 6)
                      [lat, lon, sog, cog_sin, cog_cos, dt]
            horizon_indices: (num_horizon_samples,) - which horizons to predict
                            If None, randomly samples num_horizon_samples horizons
            causal: If True, use all valid positions for each horizon (causal subwindow training)
                   If False, only use last position (for validation/baseline comparison)

        Returns:
            log_densities: (batch, num_pairs, grid_size, grid_size)
            targets: (batch, num_pairs, 2) - relative displacement in lat/lon
            sampled_horizons: (num_horizon_samples,) - the horizon indices used
            future_mask: (num_pairs,) - boolean mask, True for future predictions (from last position)
        """
        batch_size = features.shape[0]
        seq_len = self.config.max_seq_len
        max_horizon = self.config.max_horizon
        num_samples = self.config.num_horizon_samples
        device = features.device

        # Sample random horizons if not provided
        if horizon_indices is None:
            # Uniform random sampling from 1 to max_horizon
            horizon_indices = torch.randint(1, max_horizon + 1, (num_samples,), device=device)
            horizon_indices = torch.sort(horizon_indices)[0]  # Sort for easier debugging

        num_samples = len(horizon_indices)

        # Split into input sequence and future positions
        input_seq = features[:, :seq_len, :]  # (batch, seq_len, 6)

        # Project and encode
        x = self.input_proj(input_seq)
        x = self.pos_encoder(x)

        # Causal transformer
        mask = self._get_causal_mask(seq_len, device)
        embeddings = self.transformer(x, mask=mask)

        # Compute cumulative dt for horizon conditioning
        dt = features[:, :, 5] * self.config.dt_max  # Denormalize all dt
        cumsum_dt = torch.cumsum(dt, dim=1)

        # Get all positions (denormalized) for causal training
        positions = torch.stack([
            features[:, :seq_len, 0] * self.config.lat_std + self.config.lat_mean,
            features[:, :seq_len, 1] * self.config.lon_std + self.config.lon_mean
        ], dim=-1)  # (batch, seq_len, 2)

        # Future positions (beyond input sequence)
        future_pos = torch.stack([
            features[:, seq_len:seq_len+max_horizon, 0] * self.config.lat_std + self.config.lat_mean,
            features[:, seq_len:seq_len+max_horizon, 1] * self.config.lon_std + self.config.lon_mean
        ], dim=-1)  # (batch, max_horizon, 2)

        if causal:
            # Causal subwindow training: predict from ALL positions for each horizon
            # This ensures equal training signal across all horizons (seq_len predictions each)

            # Combine input and future positions for easier indexing
            all_positions = torch.cat([positions, future_pos], dim=1)  # (batch, seq_len + max_horizon, 2)

            all_embeddings = []
            all_cum_dt = []
            all_targets = []
            future_mask_list = []

            for h in horizon_indices:
                h_val = h.item()

                # ALL seq_len positions predict h steps ahead
                # Source: positions 0 to seq_len-1
                # Target: positions h to seq_len-1+h (may be within-sequence or future)
                src_emb = embeddings  # (batch, seq_len, d_model)
                src_cumsum = cumsum_dt[:, :seq_len]  # (batch, seq_len)
                tgt_cumsum = cumsum_dt[:, h_val:seq_len+h_val]  # (batch, seq_len)
                cum_dt_h = tgt_cumsum - src_cumsum  # (batch, seq_len)

                src_pos = positions  # (batch, seq_len, 2)
                tgt_pos = all_positions[:, h_val:seq_len+h_val, :]  # (batch, seq_len, 2)
                tgt_rel = tgt_pos - src_pos  # (batch, seq_len, 2)

                all_embeddings.append(src_emb)
                all_cum_dt.append(cum_dt_h)
                all_targets.append(tgt_rel)

                # Mark which predictions are future (target index >= seq_len)
                # Position i predicts position i+h, which is future if i+h >= seq_len
                if h_val >= seq_len:
                    # All predictions are future
                    future_mask_list.extend([True] * seq_len)
                else:
                    # Positions 0 to (seq_len-h_val-1) predict within-sequence
                    # Positions (seq_len-h_val) to (seq_len-1) predict future
                    num_within = seq_len - h_val
                    num_future = h_val
                    future_mask_list.extend([False] * num_within + [True] * num_future)

            # Concatenate all predictions
            all_embeddings = torch.cat(all_embeddings, dim=1)  # (batch, num_horizons * seq_len, d_model)
            all_cum_dt = torch.cat(all_cum_dt, dim=1)  # (batch, num_horizons * seq_len)
            all_targets = torch.cat(all_targets, dim=1)  # (batch, num_horizons * seq_len, 2)
            future_mask = torch.tensor(future_mask_list, dtype=torch.bool, device=device)

            num_pairs = all_embeddings.shape[1]

            # Time encoding
            time_input = all_cum_dt.unsqueeze(-1) / 300.0  # Normalize by 5 minutes
            time_enc = self.time_proj(time_input)  # (batch, num_pairs, d_model)

            # Combine and project
            combined = torch.cat([all_embeddings, time_enc], dim=-1)
            conditioned = self.horizon_proj(combined)  # (batch, num_pairs, d_model)

            # Fourier head
            cond_flat = conditioned.view(batch_size * num_pairs, -1)
            log_dens_flat = self.fourier_head(cond_flat)
            log_densities = log_dens_flat.view(batch_size, num_pairs, self.config.grid_size, self.config.grid_size)

            return log_densities, all_targets, horizon_indices, future_mask

        else:
            # Non-causal: only use last position (for validation/baseline comparison)
            last_emb = embeddings[:, -1, :]  # (batch, d_model)
            last_cumsum = cumsum_dt[:, seq_len - 1]  # (batch,)
            src_pos = positions[:, -1, :]  # (batch, 2)

            # Get future positions for sampled horizons
            targets = []
            cum_times = []

            for h in horizon_indices:
                h_val = h.item()
                # Future prediction - use cumsum_dt directly (covers all positions)
                cum_time = cumsum_dt[:, seq_len + h_val - 1] - last_cumsum
                tgt_rel = future_pos[:, h_val-1, :] - src_pos

                targets.append(tgt_rel)
                cum_times.append(cum_time)

            targets = torch.stack(targets, dim=1)  # (batch, num_samples, 2)
            cum_times = torch.stack(cum_times, dim=1)  # (batch, num_samples)

            # Time encoding
            time_input = cum_times.unsqueeze(-1) / 300.0  # Normalize by 5 minutes
            time_enc = self.time_proj(time_input)  # (batch, num_samples, d_model)

            # Expand last embedding for all horizons
            last_emb_expanded = last_emb.unsqueeze(1).expand(-1, num_samples, -1)

            # Combine and project
            combined = torch.cat([last_emb_expanded, time_enc], dim=-1)
            conditioned = self.horizon_proj(combined)  # (batch, num_samples, d_model)

            # Fourier head
            cond_flat = conditioned.view(batch_size * num_samples, -1)
            log_dens_flat = self.fourier_head(cond_flat)
            log_densities = log_dens_flat.view(batch_size, num_samples, self.config.grid_size, self.config.grid_size)

            # All predictions are future predictions when causal=False
            future_mask = torch.ones(num_samples, dtype=torch.bool, device=device)

            return log_densities, targets, horizon_indices, future_mask


# ============================================================================
# Loss Function
# ============================================================================

def compute_soft_target_loss(log_density: torch.Tensor, target: torch.Tensor,
                             grid_range: float, grid_size: int, sigma: float,
                             chunk_size: int = 512) -> torch.Tensor:
    """Soft target KL divergence loss with chunked computation for memory efficiency."""
    batch_size, num_pairs, gs, _ = log_density.shape
    device = log_density.device

    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Clip targets to grid range
    target_clipped = torch.clamp(target, -grid_range * 0.99, grid_range * 0.99)

    # Process in chunks to avoid memory explosion
    total_loss = 0.0
    total_count = 0

    for i in range(0, batch_size, chunk_size):
        end_i = min(i + chunk_size, batch_size)
        chunk_target = target_clipped[i:end_i]
        chunk_log_density = log_density[i:end_i]

        target_x = chunk_target[:, :, 0:1, None]
        target_y = chunk_target[:, :, 1:2, None]

        dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
        soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
        soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

        chunk_loss = F.kl_div(chunk_log_density, soft_target, reduction='none').sum(dim=(-2, -1))
        total_loss = total_loss + chunk_loss.sum()
        total_count += chunk_loss.numel()

    return total_loss / total_count


# ============================================================================
# Training
# ============================================================================

def compute_dead_reckoning_loss(features: torch.Tensor, targets: torch.Tensor,
                                 horizon_indices: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Compute dead reckoning baseline loss for sampled horizons.

    Uses velocity at last input position to predict displacement to each sampled horizon.
    """
    device = features.device
    batch_size = features.shape[0]
    seq_len = config.max_seq_len
    num_samples = len(horizon_indices)

    # Get positions (denormalized to degrees)
    lat = features[:, :, 0] * config.lat_std + config.lat_mean
    lon = features[:, :, 1] * config.lon_std + config.lon_mean

    # Get time deltas (denormalized to seconds)
    dt = features[:, :, 5] * config.dt_max

    # Compute velocity at last input position
    last_dlat = lat[:, seq_len - 1] - lat[:, seq_len - 2]
    last_dlon = lon[:, seq_len - 1] - lon[:, seq_len - 2]
    last_dt = dt[:, seq_len - 1].clamp(min=1.0)
    vlat = last_dlat / last_dt  # (batch,)
    vlon = last_dlon / last_dt

    # Cumulative dt
    cumsum_dt = torch.cumsum(dt, dim=1)
    last_cumsum = cumsum_dt[:, seq_len - 1]

    # Compute DR predictions for each sampled horizon
    dr_preds = []
    for h in horizon_indices:
        h = h.item()
        tgt_idx = seq_len + h - 1
        cum_time = cumsum_dt[:, tgt_idx] - last_cumsum  # (batch,)

        dr_lat = vlat * cum_time
        dr_lon = vlon * cum_time
        dr_preds.append(torch.stack([dr_lat, dr_lon], dim=-1))

    dr_preds = torch.stack(dr_preds, dim=1)  # (batch, num_samples, 2)

    # Grid for computing loss
    grid_range = config.grid_range
    grid_size = config.grid_size
    dr_sigma = config.dr_sigma  # Use optimized sigma for DR
    target_sigma = config.sigma  # Keep tight sigma for target
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Clip to grid range
    dr_preds = torch.clamp(dr_preds, -grid_range * 0.99, grid_range * 0.99)

    # Reshape for grid computation
    dr_pred_lat = dr_preds[:, :, 0:1, None]  # (batch, num_samples, 1, 1)
    dr_pred_lon = dr_preds[:, :, 1:2, None]

    # Create Gaussian distribution centered at DR prediction (use dr_sigma)
    dist_sq = (xx - dr_pred_lat) ** 2 + (yy - dr_pred_lon) ** 2
    dr_density = torch.exp(-dist_sq / (2 * dr_sigma ** 2))
    dr_density = dr_density / (dr_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    dr_log_density = torch.log(dr_density + 1e-10)

    # Compute target soft distribution (clip targets to grid range)
    targets_clipped = torch.clamp(targets, -grid_range * 0.99, grid_range * 0.99)
    target_x = targets_clipped[:, :, 0:1, None]
    target_y = targets_clipped[:, :, 1:2, None]
    target_dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-target_dist_sq / (2 * target_sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    # KL divergence loss
    loss = F.kl_div(dr_log_density, soft_target, reduction='none').sum(dim=(-2, -1))
    return loss.mean()


def compute_last_position_loss(features: torch.Tensor, targets: torch.Tensor,
                                config: Config) -> torch.Tensor:
    """
    Compute last position baseline for ALL prediction pairs: predict zero displacement.
    This baseline predicts the target stays at the source position (no movement).
    """
    device = features.device
    batch_size = features.shape[0]
    num_pairs = targets.shape[1]  # Match the number of prediction pairs
    grid_range = config.grid_range
    grid_size = config.grid_size
    baseline_sigma = config.dr_sigma  # Use same sigma as DR for fair comparison
    target_sigma = config.sigma

    # Create grid
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Last position predicts ZERO displacement for ALL pairs
    lp_pred_lat = torch.zeros(batch_size, num_pairs, 1, 1, device=device)
    lp_pred_lon = torch.zeros(batch_size, num_pairs, 1, 1, device=device)

    dist_sq = (xx - lp_pred_lat) ** 2 + (yy - lp_pred_lon) ** 2
    lp_density = torch.exp(-dist_sq / (2 * baseline_sigma ** 2))
    lp_density = lp_density / (lp_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    lp_log_density = torch.log(lp_density + 1e-10)

    # Target distribution (clip to grid range)
    targets_clipped = torch.clamp(targets, -grid_range * 0.99, grid_range * 0.99)
    target_x = targets_clipped[:, :, 0:1, None]
    target_y = targets_clipped[:, :, 1:2, None]
    target_dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-target_dist_sq / (2 * target_sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    loss = F.kl_div(lp_log_density, soft_target, reduction='none').sum(dim=(-2, -1))
    return loss.mean()


def validate_with_baselines(model: nn.Module, val_loader: DataLoader, config: Config,
                            max_batches: int = None) -> dict:
    """
    Validate model AND compute ALL baselines on SAME data.
    Memory-optimized: no random model, compute baseline targets directly.

    Returns dict with:
    - model: trained model loss
    - dead_reckoning: velocity * horizon extrapolation
    - last_position: predict zero displacement
    """
    model.eval()

    results = {
        'model': [],
        'dead_reckoning': [],
        'last_position': []
    }

    max_batches = max_batches or len(val_loader)
    seq_len = config.max_seq_len

    # Use fixed horizons for validation (evenly spaced for coverage)
    fixed_horizons = torch.linspace(1, config.max_horizon, config.num_horizon_samples).long().to(DEVICE)

    with torch.no_grad():
        for batch_idx, features in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            features = features.to(DEVICE, non_blocking=True)

            with autocast('cuda', enabled=config.use_amp):
                # Model validation uses causal=True (same as training)
                log_densities, targets, _, _ = model.forward_train(features, fixed_horizons, causal=True)

                model_loss = compute_soft_target_loss(
                    log_densities, targets,
                    config.grid_range, config.grid_size, config.sigma
                )
                results['model'].append(model_loss.item())

                # Free memory immediately
                del log_densities, targets

                # Compute baseline targets directly from features (no forward pass needed)
                # Just need future_pos - src_pos for each horizon
                src_pos = torch.stack([
                    features[:, seq_len-1, 0] * config.lat_std + config.lat_mean,
                    features[:, seq_len-1, 1] * config.lon_std + config.lon_mean
                ], dim=-1)  # (batch, 2)

                baseline_targets = []
                for h in fixed_horizons:
                    h_val = h.item()
                    future_pos = torch.stack([
                        features[:, seq_len + h_val - 1, 0] * config.lat_std + config.lat_mean,
                        features[:, seq_len + h_val - 1, 1] * config.lon_std + config.lon_mean
                    ], dim=-1)  # (batch, 2)
                    baseline_targets.append(future_pos - src_pos)
                baseline_targets = torch.stack(baseline_targets, dim=1)  # (batch, num_horizons, 2)

                # Dead reckoning baseline
                dr_loss = compute_dead_reckoning_loss(features, baseline_targets, fixed_horizons, config)
                results['dead_reckoning'].append(dr_loss.item())

                # Last position baseline
                lp_loss = compute_last_position_loss(features, baseline_targets, config)
                results['last_position'].append(lp_loss.item())

                del baseline_targets

    return {k: np.mean(v) for k, v in results.items()}


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config,
                output_dir: Path, checkpoints_dir: Path = None,
                log_interval: int = 10, start_step: int = 0,
                optimizer_state: dict = None, early_stopping: bool = True) -> Dict:
    """Train the model with validation every N batches."""

    # Use checkpoints subfolder if provided, otherwise fall back to output_dir
    if checkpoints_dir is None:
        checkpoints_dir = output_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Load optimizer state if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print(f"  Loaded optimizer state from checkpoint", flush=True)

    # Learning rate scheduler with warmup + cosine annealing
    def lr_lambda(step):
        if step < config.warmup_steps:
            # Linear warmup
            return step / config.warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - config.warmup_steps) / max(1, config.total_steps_estimate - config.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to 1.0 if we exceed estimate
            # Cosine decay from 1.0 to min_lr_ratio
            return config.min_lr_ratio + 0.5 * (1.0 - config.min_lr_ratio) * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # If resuming, step scheduler to correct position
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resuming from step {start_step}, LR = {scheduler.get_last_lr()[0]:.2e}", flush=True)
    else:
        print(f"  LR schedule: warmup {config.warmup_steps} steps, then cosine decay to {config.learning_rate * config.min_lr_ratio:.1e}")

    # Mixed precision
    scaler = GradScaler('cuda') if config.use_amp else None

    history = {
        'train_loss': [],
        'val_loss': [],
        'dr_loss': [],
        'lp_loss': [],
        'lr': [],
        'step': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False

    # Initialize total_steps from start_step if resuming
    total_steps = start_step
    running_loss = 0
    running_count = 0

    # Exponential validation schedule: validate at batch 1, 2, 4, 8, 16, 32, 64, ...
    # Cap at min(2000 batches, once per epoch) for large datasets
    batches_per_epoch = len(train_loader)
    max_val_interval = min(2000, batches_per_epoch)  # Cap at 2000 batches for large datasets

    # Calculate next validation point based on start_step
    if start_step > 0:
        # Find where we are in the exponential schedule
        # Validation happens at batches: 1, 2, 4, 8, 16, 32, ... then every max_val_interval
        val_interval = 1
        next_val_batch = 1
        while next_val_batch <= start_step:
            val_interval = min(val_interval * 2, max_val_interval)
            next_val_batch += val_interval
        print(f"  Resuming: next validation at batch {next_val_batch} (interval={val_interval})", flush=True)
    else:
        next_val_batch = 1  # First validation after 1 batch
        val_interval = 1    # Current interval (doubles after each validation)

    print(f"  Validation schedule: exponential doubling, max interval = {max_val_interval} batches", flush=True)

    # Initial validation only if not resuming
    if start_step == 0:
        print("\nInitial validation (before training)...", flush=True)
        val_results = validate_with_baselines(model, val_loader, config, config.val_max_batches)
        print(f"  Model (untrained): {val_results['model']:.4f}", flush=True)
        print(f"  Dead Reckoning:    {val_results['dead_reckoning']:.4f}", flush=True)
        print(f"  Last Position:     {val_results['last_position']:.4f}", flush=True)

        history['val_loss'].append(val_results['model'])
        history['dr_loss'].append(val_results['dead_reckoning'])
        history['lp_loss'].append(val_results['last_position'])
        history['train_loss'].append(float('nan'))
        history['lr'].append(config.learning_rate)
        history['step'].append(0)
    else:
        print(f"\nResuming training from step {start_step}...", flush=True)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_start = time.time()

        for batch_idx, features in enumerate(train_loader):
            features = features.to(DEVICE, non_blocking=True)

            # Forward with mixed precision (random horizon sampling)
            with autocast('cuda', enabled=config.use_amp):
                log_densities, targets, _, _ = model.forward_train(features)
                loss = compute_soft_target_loss(
                    log_densities, targets,
                    config.grid_range, config.grid_size, config.sigma
                )
                loss = loss / config.gradient_accumulation

            # Check for NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at batch {batch_idx}", flush=True)
                optimizer.zero_grad()
                continue

            # Backward
            if config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step with accumulation
            if (batch_idx + 1) % config.gradient_accumulation == 0:
                if config.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                total_steps += 1

            running_loss += loss.item() * config.gradient_accumulation
            running_count += 1

            # Exponential validation schedule: 1, 2, 4, 8, 16, 32, ... batches
            # Offset by start_step when resuming so validation schedule continues correctly
            # NOTE: Check validation BEFORE log interval reset to preserve running_loss for avg_train
            global_batch = start_step + epoch * batches_per_epoch + (batch_idx + 1)
            if global_batch == next_val_batch:
                avg_train = running_loss / max(running_count, 1)
                # Free training memory before validation
                torch.cuda.empty_cache()
                val_results = validate_with_baselines(model, val_loader, config, config.val_max_batches)

                val_loss = val_results['model']
                dr_loss = val_results['dead_reckoning']
                lp_loss = val_results['last_position']

                history['train_loss'].append(avg_train)
                history['val_loss'].append(val_loss)
                history['dr_loss'].append(dr_loss)
                history['lp_loss'].append(lp_loss)
                history['lr'].append(scheduler.get_last_lr()[0])
                history['step'].append(total_steps)

                vs_dr = (dr_loss - val_loss) / dr_loss * 100
                vs_lp = (lp_loss - val_loss) / lp_loss * 100

                print(f"\n  >>> VALIDATION at step {total_steps} (batch {global_batch}):", flush=True)
                print(f"      Train Loss:     {avg_train:.4f}", flush=True)
                print(f"      ---", flush=True)
                print(f"      Model:          {val_loss:.4f}  (vs DR: {vs_dr:+.1f}%, vs LP: {vs_lp:+.1f}%)", flush=True)
                print(f"      Dead Reckoning: {dr_loss:.4f}", flush=True)
                print(f"      Last Position:  {lp_loss:.4f}\n", flush=True)

                # Save checkpoint at every validation
                checkpoint_path = checkpoints_dir / f'checkpoint_step_{total_steps}.pt'
                torch.save({
                    'step': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_results': val_results,
                }, checkpoint_path)
                print(f"      Saved checkpoint: {checkpoint_path.name}", flush=True)

                # Save best model and check early stopping
                improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss < float('inf') else 1.0
                if val_loss < best_val_loss and improvement >= config.early_stop_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), checkpoints_dir / 'best_model.pt')
                    print(f"      New best model!", flush=True)
                else:
                    patience_counter += 1
                    if early_stopping:
                        print(f"      No improvement ({patience_counter}/{config.early_stop_patience})", flush=True)

                        if patience_counter >= config.early_stop_patience:
                            print(f"\n  >>> EARLY STOPPING: No improvement for {config.early_stop_patience} validation checks", flush=True)
                            print(f"      Best validation loss: {best_val_loss:.4f}", flush=True)
                            early_stopped = True
                            break
                    else:
                        print(f"      No improvement (early stopping disabled)", flush=True)

                running_loss = 0
                running_count = 0
                model.train()

                # Double the interval for next validation, cap at max_val_interval
                val_interval = min(val_interval * 2, max_val_interval)
                next_val_batch = global_batch + val_interval

            # Progress every N batches (after validation check to preserve running_loss)
            elif (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / running_count
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                current_batch = start_step + epoch * batches_per_epoch + (batch_idx + 1)
                print(f"  Epoch {epoch+1}, Batch {current_batch}/{start_step + len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}, "
                      f"GPU: {gpu_mem:.1f}GB", flush=True)
                # Reset for next window
                running_loss = 0
                running_count = 0

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config.num_epochs} complete, Time={epoch_time:.1f}s", flush=True)

        # Check if we should stop
        if early_stopped:
            print(f"\nTraining stopped early at epoch {epoch+1}", flush=True)
            break

    # Final validation
    print("\nFinal validation...", flush=True)
    val_results = validate_with_baselines(model, val_loader, config, config.val_max_batches)

    val_loss = val_results['model']
    dr_loss = val_results['dead_reckoning']
    lp_loss = val_results['last_position']

    history['val_loss'].append(val_loss)
    history['dr_loss'].append(dr_loss)
    history['lp_loss'].append(lp_loss)
    history['train_loss'].append(running_loss / max(running_count, 1))
    history['lr'].append(scheduler.get_last_lr()[0])
    history['step'].append(total_steps)

    vs_dr = (dr_loss - val_loss) / dr_loss * 100
    vs_lp = (lp_loss - val_loss) / lp_loss * 100

    print(f"\n  FINAL RESULTS:", flush=True)
    print(f"  Model:          {val_loss:.4f}  (vs DR: {vs_dr:+.1f}%, vs LP: {vs_lp:+.1f}%)", flush=True)
    print(f"  Dead Reckoning: {dr_loss:.4f}", flush=True)
    print(f"  Last Position:  {lp_loss:.4f}", flush=True)

    return history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: Config) -> Dict:
    """Evaluate model prediction errors at specific horizon points."""
    model.eval()

    grid = torch.linspace(-config.grid_range, config.grid_range,
                          config.grid_size, device=DEVICE)

    # Evaluate at specific horizons: 1, 10, 50, 100, 200, 400
    eval_horizons = [1, 10, 50, 100, 200, min(400, config.max_horizon)]
    eval_horizons = [h for h in eval_horizons if h <= config.max_horizon]
    fixed_horizons = torch.tensor(eval_horizons, device=DEVICE)

    errors_by_horizon = {h: [] for h in eval_horizons}

    with torch.no_grad():
        for features in val_loader:
            features = features.to(DEVICE)
            log_densities, targets, horizons, _ = model.forward_train(features, fixed_horizons, causal=True)

            densities = torch.exp(log_densities)

            # Expected value
            marginal_x = densities.sum(dim=3)
            marginal_y = densities.sum(dim=2)
            exp_x = (marginal_x * grid).sum(dim=2)
            exp_y = (marginal_y * grid).sum(dim=2)

            # Error in degrees
            error = torch.sqrt((exp_x - targets[:, :, 0])**2 + (exp_y - targets[:, :, 1])**2)

            # Track errors by horizon
            for i, h in enumerate(horizons.cpu().numpy()):
                err = error[:, i].mean().item()
                errors_by_horizon[h].append(err)

    metrics = {}
    for h in eval_horizons:
        if errors_by_horizon[h]:
            mean_err = np.mean(errors_by_horizon[h])
            metrics[f'horizon_{h}_error_deg'] = mean_err
            metrics[f'horizon_{h}_error_km'] = mean_err * 111  # Approximate

    return metrics


# ============================================================================
# Main
# ============================================================================

def find_optimal_batch_size(model: nn.Module, config: Config, target_gpu_pct: float = 0.90) -> int:
    """
    Binary search to find the largest batch size that fits in target GPU memory.

    Args:
        model: The model to test
        config: Configuration object
        target_gpu_pct: Target GPU memory utilization (default 90%)

    Returns:
        Optimal batch size
    """
    import gc

    model = model.to(DEVICE)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    target_mem = total_mem * target_gpu_pct

    print(f"\nFinding optimal batch size for {target_gpu_pct*100:.0f}% GPU utilization...")
    print(f"  Total GPU memory: {total_mem / 1e9:.1f} GB")
    print(f"  Target memory: {target_mem / 1e9:.1f} GB")

    # Create a small synthetic batch for testing
    seq_len = config.max_seq_len + config.max_horizon

    # Binary search between min and max batch sizes
    min_bs, max_bs = 64, 16000
    best_bs = min_bs

    while min_bs <= max_bs:
        mid_bs = (min_bs + max_bs) // 2

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create synthetic batch
            features = torch.randn(mid_bs, seq_len, 6, device=DEVICE)

            # Create optimizer to test full training memory (optimizer state uses memory)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            scaler = GradScaler('cuda') if config.use_amp else None

            # Forward pass with gradient computation
            model.train()
            with autocast('cuda', enabled=config.use_amp):
                log_densities, targets, _, _ = model.forward_train(features)
                loss = compute_soft_target_loss(
                    log_densities, targets,
                    config.grid_range, config.grid_size, config.sigma
                )

            # Backward pass
            if config.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Check peak memory
            peak_mem = torch.cuda.max_memory_allocated()

            # Clean up
            del features, log_densities, targets, loss, optimizer, scaler
            model.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

            pct_used = peak_mem / total_mem * 100
            print(f"  Batch size {mid_bs}: {peak_mem / 1e9:.1f} GB ({pct_used:.1f}%)", end="")

            if peak_mem <= target_mem:
                best_bs = mid_bs
                min_bs = mid_bs + 1
                print(" ✓")
            else:
                max_bs = mid_bs - 1
                print(" (too high)")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {mid_bs}: OOM")
                max_bs = mid_bs - 1
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e

    # Round down to nearest 8 for GPU efficiency (not 100, that's too aggressive)
    best_bs = (best_bs // 8) * 8
    best_bs = max(best_bs, 64)  # Minimum 64

    print(f"\n  Optimal batch size: {best_bs}")
    return best_bs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train AIS trajectory prediction model')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (creates experiments/<exp-name>/ folder)')
    parser.add_argument('--max-horizon', type=int, default=None,
                        help='Maximum prediction horizon (overrides config)')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config, use 0 for auto-find)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Warmup steps for LR schedule (default: 500)')
    parser.add_argument('--total-steps', type=int, default=None,
                        help='Total steps estimate for cosine decay (default: 3000)')
    parser.add_argument('--early-stop-patience', type=int, default=None,
                        help='Early stopping patience (overrides config)')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--max-tracks', type=int, default=None,
                        help='Maximum number of tracks to load (default: all)')
    # Model size arguments
    parser.add_argument('--model-scale', type=str, choices=['small', 'medium', 'large'], default=None,
                        help='Model scale preset: small (~1M), medium (~5M), large (~17M params)')
    parser.add_argument('--d-model', type=int, default=None,
                        help='Model dimension (overrides preset)')
    parser.add_argument('--nhead', type=int, default=None,
                        help='Number of attention heads (overrides preset)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of transformer layers (overrides preset)')
    parser.add_argument('--dim-feedforward', type=int, default=None,
                        help='Feedforward dimension (overrides preset)')
    parser.add_argument('--target-gpu-pct', type=float, default=0.90,
                        help='Target GPU memory utilization for auto batch size (default: 0.90)')
    # Memory-efficient loading modes for large datasets
    parser.add_argument('--chunked', action='store_true', default=True,
                        help='Use chunked/conveyor-belt loading (DEFAULT for 1-year dataset)')
    parser.add_argument('--num-chunks', type=int, default=10,
                        help='Number of chunks for chunked loading (default: 10)')
    parser.add_argument('--lazy-load', action='store_true',
                        help='Use lazy loading (per-sample I/O with LRU cache)')
    parser.add_argument('--eager-load', action='store_true',
                        help='Force eager loading (loads ALL data into RAM - not recommended)')
    parser.add_argument('--cache-size', type=int, default=1000,
                        help='LRU cache size for lazy loading (default: 1000 tracks)')
    parser.add_argument('--max-val-samples', type=int, default=None,
                        help='Limit validation samples (e.g., 40000 to match Exp 10 speed)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log training loss every N batches (default: 10)')
    parser.add_argument('--num-horizons', type=int, default=8,
                        help='Number of horizon samples per batch (default: 8, lower for causal training)')
    # S3 shard loading mode
    parser.add_argument('--s3-shards', action='store_true',
                        help='Use S3 sharded data with double-buffer loading')
    parser.add_argument('--shards-per-buffer', type=int, default=10,
                        help='Number of shards per double-buffer (default: 10)')
    parser.add_argument('--max-val-tracks', type=int, default=500,
                        help='Max validation tracks for S3 mode (default: 500)')
    # Materialized (pre-shuffled) data loading mode
    parser.add_argument('--materialized', action='store_true',
                        help='Use pre-shuffled materialized samples from S3')
    parser.add_argument('--materialized-sog', type=float, default=None,
                        help='Min mean SOG filter for materialized samples (default: None)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--grid-range', type=float, default=None,
                        help='Grid range in degrees (default: 0.6, use 1.2 for ~133km)')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("EXPERIMENT 14: 800 HORIZON PREDICTION (1 YEAR DATA)", flush=True)
    print("=" * 70, flush=True)

    config = Config()

    # Apply model scale preset first
    if args.model_scale == 'small':
        config.d_model = 128
        config.nhead = 8
        config.num_layers = 4
        config.dim_feedforward = 512
    elif args.model_scale == 'medium':
        config.d_model = 256
        config.nhead = 8
        config.num_layers = 6
        config.dim_feedforward = 1024
    elif args.model_scale == 'large':
        config.d_model = 384
        config.nhead = 16
        config.num_layers = 8
        config.dim_feedforward = 2048

    # Override individual model params (takes precedence over preset)
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.nhead is not None:
        config.nhead = args.nhead
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.dim_feedforward is not None:
        config.dim_feedforward = args.dim_feedforward

    # Override other config with command line arguments
    if args.max_horizon is not None:
        config.max_horizon = args.max_horizon
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.total_steps is not None:
        config.total_steps_estimate = args.total_steps
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.no_early_stop:
        config.early_stop_patience = float('inf')  # Effectively disable
    if args.num_horizons is not None:
        config.num_horizon_samples = args.num_horizons
    if args.grid_range is not None:
        config.grid_range = args.grid_range

    # Handle batch size: if 0 or not specified with large model, use auto-find
    auto_batch_size = args.batch_size == 0 or (args.batch_size is None and args.model_scale in ['medium', 'large'])

    # If explicit batch size specified (not 0), use it
    if args.batch_size is not None and args.batch_size > 0:
        config.batch_size = args.batch_size

    # Handle resume: extract experiment directory from checkpoint path
    resume_checkpoint = None
    resume_start_step = 0
    resume_optimizer_state = None
    resume_mode = args.resume is not None

    if resume_mode:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}", flush=True)
            sys.exit(1)

        # Extract experiment directory from checkpoint path
        # Expected: .../experiments/<exp_name>/checkpoints/checkpoint_step_XXXX.pt
        output_dir = checkpoint_path.parent.parent
        checkpoints_dir = checkpoint_path.parent
        results_dir = output_dir / 'results'
        exp_name = output_dir.name

        print(f"RESUMING from checkpoint: {checkpoint_path}", flush=True)
        print(f"  Experiment: {exp_name}", flush=True)

        # Load checkpoint
        resume_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        resume_start_step = resume_checkpoint['step']
        resume_optimizer_state = resume_checkpoint['optimizer_state_dict']

        print(f"  Resuming from step: {resume_start_step}", flush=True)

    else:
        # Create new output directory
        from datetime import datetime
        if args.exp_name:
            exp_name = args.exp_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"run_{timestamp}_h{config.max_horizon}"

        output_dir = Path(__file__).parent / 'experiments' / exp_name
        checkpoints_dir = output_dir / 'checkpoints'
        results_dir = output_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file (tee-style: both stdout and file)
    import sys
    log_file = results_dir / 'training.log'

    # If resuming, trim log file to remove ending metrics (everything after last checkpoint save)
    if resume_mode and log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Find the last "Saved checkpoint:" line and trim everything after it
        trim_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if 'Saved checkpoint:' in lines[i]:
                # Keep up to and including the checkpoint line, plus one blank line
                trim_idx = i + 2  # +1 for the line itself, +1 for spacing
                break

        # Write trimmed log
        with open(log_file, 'w') as f:
            f.writelines(lines[:trim_idx])
            f.write(f"\n--- RESUMING FROM STEP {resume_start_step} ---\n\n")

    class TeeLogger:
        def __init__(self, filename, stream, append=False):
            mode = 'a' if append else 'w'
            self.file = open(filename, mode)
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stream.flush()
            self.file.flush()
    sys.stdout = TeeLogger(log_file, sys.stdout, append=resume_mode)

    print(f"\nExperiment: {exp_name}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"  Checkpoints: {checkpoints_dir}", flush=True)
    print(f"  Results: {results_dir}", flush=True)
    print(f"  Log file: {log_file}", flush=True)

    # Print config
    print("\nConfiguration:", flush=True)
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}", flush=True)

    # Save config for reproducibility
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Load data
    print("\n" + "=" * 70, flush=True)
    print("LOADING DATA", flush=True)
    print("=" * 70, flush=True)

    # Determine loading mode (materialized > S3 shards > chunked is default)
    use_materialized = args.materialized
    use_s3_shards = args.s3_shards and not use_materialized
    use_chunked = args.chunked and not args.eager_load and not args.lazy_load and not args.s3_shards and not use_materialized
    use_lazy = args.lazy_load and not args.eager_load and not args.s3_shards and not use_materialized
    use_eager = args.eager_load and not args.s3_shards and not use_materialized

    # Track mode for training loop
    chunked_mode = False
    s3_mode = False
    materialized_mode = False
    train_chunked = None
    val_chunked = None
    shard_loader = None

    if use_materialized:
        # MDS streaming (MosaicML format)
        print("Using MDS STREAMING mode (MosaicML format, memory-efficient)", flush=True)

        # Skip validation loading if we're doing auto batch size finding (will reload after)
        skip_val = auto_batch_size
        train_dataset, train_collate, val_dataset = build_materialized_dataset(
            config, config.batch_size, skip_validation=skip_val
        )
        materialized_mode = True

        # Create DataLoader wrapping StreamingDataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=train_collate,
            num_workers=4,           # Parallel batch preparation
            prefetch_factor=2,       # Each worker prefetches 2 batches
            pin_memory=True,
            persistent_workers=True,
        )

        # Validation: LocalValidationStreamer is already an iterator (no DataLoader needed)
        val_loader = val_dataset  # None if skip_val, else LocalValidationStreamer

        print(f"  Train loader ready (~{len(train_dataset) // config.batch_size:,} batches per epoch)", flush=True)
        if val_loader is not None:
            print(f"  Val loader ready ({len(val_loader)} batches, pinned memory)", flush=True)
        else:
            print(f"  Val loader: deferred until after batch size finding", flush=True)

    elif use_s3_shards:
        # S3 shard loading with double-buffering
        print("Using S3 SHARD LOADING mode (double-buffered, memory-efficient)", flush=True)
        print(f"  S3 bucket: {config.s3_bucket}", flush=True)
        print(f"  Shards per buffer: {args.shards_per_buffer}", flush=True)
        print(f"  Max val tracks: {args.max_val_tracks}", flush=True)

        shard_loader, val_dataset = build_s3_shard_dataset(
            config,
            shards_per_buffer=args.shards_per_buffer,
            max_val_tracks=args.max_val_tracks
        )
        s3_mode = True

        # Create S3ShardDataLoader for training
        train_loader = S3ShardDataLoader(shard_loader, config, config.batch_size)

        # Create regular DataLoader for validation
        val_loader = DataLoader(
            val_dataset, batch_size=min(config.batch_size, 1024),
            shuffle=False, num_workers=0, pin_memory=True
        )
        print(f"  Train loader ready (~{len(train_loader)} batches estimated)", flush=True)
        print(f"  Val loader ready ({len(val_loader)} batches)", flush=True)

    elif use_chunked:
        # Chunked/conveyor-belt loading: load 1/N of data at a time with prefetching
        print("Using CHUNKED LOADING mode (conveyor-belt, memory-efficient)", flush=True)
        print(f"  Number of chunks: {args.num_chunks}", flush=True)
        train_chunked, val_chunked = build_chunked_dataset(
            config, num_chunks=args.num_chunks, max_tracks=args.max_tracks
        )
        chunked_mode = True

        # Create ChunkedDataLoader wrappers for seamless integration with training loop
        train_loader = ChunkedDataLoader(train_chunked, config.batch_size)
        val_loader = ChunkedDataLoader(val_chunked, min(config.batch_size, 1024))
        print(f"  Train loader ready (~{len(train_loader)} batches per epoch)", flush=True)
        print(f"  Val loader ready", flush=True)

    elif use_lazy:
        # Lazy loading: build index only, load tracks on-demand
        print("Using LAZY LOADING mode (per-sample I/O with LRU cache)", flush=True)
        train_index, val_index, track_file_map = load_ais_data_lazy(config, max_tracks=args.max_tracks)

        # Create lazy datasets
        print("\nCreating lazy datasets...", flush=True)
        train_dataset = LazyAISTrackDataset(
            train_index, track_file_map,
            config.max_seq_len, config.max_horizon, config,
            cache_size=args.cache_size
        )

        # Limit validation samples if requested
        if args.max_val_samples and len(val_index) > args.max_val_samples:
            np.random.shuffle(val_index)
            val_index = val_index[:args.max_val_samples]
            print(f"  Limited validation to {len(val_index):,} samples", flush=True)

        val_dataset = LazyAISTrackDataset(
            val_index, track_file_map,
            config.max_seq_len, config.max_horizon, config,
            cache_size=args.cache_size
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=config.pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=min(config.batch_size, 1024),
            num_workers=0, pin_memory=config.pin_memory
        )
        print(f"  Train batches: {len(train_loader)}", flush=True)
        print(f"  Val batches: {len(val_loader)}", flush=True)

    else:
        # Traditional eager loading: load all data into memory upfront
        print("Using EAGER LOADING mode (all data in memory)", flush=True)
        print("  WARNING: This may fail with 1-year dataset due to memory constraints!", flush=True)
        train_data, val_data = load_ais_data(config, max_tracks=args.max_tracks)

        # Create datasets
        print("\nCreating datasets...", flush=True)
        train_dataset = AISTrackDataset(train_data, config.max_seq_len, config.max_horizon, config)
        val_dataset = AISTrackDataset(val_data, config.max_seq_len, config.max_horizon, config)

        # Limit validation samples if requested
        if args.max_val_samples and len(val_dataset) > args.max_val_samples:
            indices = np.random.permutation(len(val_dataset))[:args.max_val_samples]
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
            print(f"  Limited validation to {len(val_dataset):,} samples", flush=True)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            persistent_workers=config.num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=min(config.batch_size, 1024),
            num_workers=config.num_workers, pin_memory=config.pin_memory
        )
        print(f"  Train batches: {len(train_loader)}", flush=True)
        print(f"  Val batches: {len(val_loader)}", flush=True)

    # Create model
    print("\n" + "=" * 70, flush=True)
    print("CREATING MODEL", flush=True)
    print("=" * 70, flush=True)

    model = CausalAISModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}", flush=True)
    print(f"  Architecture: d_model={config.d_model}, nhead={config.nhead}, "
          f"num_layers={config.num_layers}, dim_ff={config.dim_feedforward}", flush=True)

    # Load model state if resuming
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        print(f"  Loaded model weights from checkpoint (step {resume_start_step})", flush=True)

    # Find optimal batch size if requested
    if auto_batch_size:
        print("\n" + "=" * 70, flush=True)
        print("FINDING OPTIMAL BATCH SIZE", flush=True)
        print("=" * 70, flush=True)
        config.batch_size = find_optimal_batch_size(model, config, args.target_gpu_pct)

        # Recreate data loaders with new batch size
        print(f"\nRecreating data loaders with batch_size={config.batch_size}...", flush=True)

        if materialized_mode:
            # Clean up old StreamingDataset before recreating (MosaicML requires this)
            del train_loader, train_dataset
            import gc
            gc.collect()
            try:
                from streaming.base.util import clean_stale_shared_memory
                clean_stale_shared_memory()
            except Exception:
                pass

            # Recreate MDS streaming dataset with new batch size
            train_dataset, train_collate, val_dataset = build_materialized_dataset(
                config, config.batch_size
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                collate_fn=train_collate,
                num_workers=4,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=True,
            )
            # val_dataset is LocalValidationStreamer - already an iterator
            val_loader = val_dataset
        elif s3_mode:
            # Recreate S3ShardDataLoader with new batch size
            train_loader = S3ShardDataLoader(shard_loader, config, config.batch_size)
            val_loader = DataLoader(
                val_dataset, batch_size=min(config.batch_size, 1024),
                shuffle=False, num_workers=0, pin_memory=True
            )
        elif chunked_mode:
            # Recreate ChunkedDataLoader wrappers with new batch size
            train_loader = ChunkedDataLoader(train_chunked, config.batch_size)
            val_loader = ChunkedDataLoader(val_chunked, min(config.batch_size, 1024))
        else:
            # Recreate standard DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.num_workers > 0
            )
            val_batch_size = min(config.batch_size, 1024)
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
        print(f"  Train batches: {len(train_loader)}", flush=True)
        print(f"  Val batches: {len(val_loader)}", flush=True)

        # Update config file with found batch size
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config.__dict__, f, indent=2)

    # Train
    print("\n" + "=" * 70, flush=True)
    print("TRAINING", flush=True)
    print("=" * 70, flush=True)

    history = train_model(model, train_loader, val_loader, config, output_dir, checkpoints_dir,
                          log_interval=args.log_interval,
                          start_step=resume_start_step,
                          optimizer_state=resume_optimizer_state,
                          early_stopping=not args.no_early_stop)

    # Evaluate
    print("\n" + "=" * 70, flush=True)
    print("EVALUATION", flush=True)
    print("=" * 70, flush=True)

    metrics = evaluate_model(model, val_loader, config)
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}", flush=True)

    # Save results
    print("\n" + "=" * 70, flush=True)
    print("SAVING RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {
        'config': config.__dict__,
        'metrics': metrics,
        'history': history
    }

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot training history
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = history['step']
    # Filter out step=0 for log scale
    valid_idx = [i for i, s in enumerate(steps) if s > 0]
    steps_log = [steps[i] for i in valid_idx]

    # Train loss (skip NaN values)
    valid_train = [(steps[i], history['train_loss'][i]) for i in valid_idx if not np.isnan(history['train_loss'][i])]
    if valid_train:
        train_steps, train_loss = zip(*valid_train)
        ax.plot(train_steps, train_loss, 'c-d', label='Train', markersize=4, alpha=0.7)

    # Val and baselines
    ax.plot(steps_log, [history['val_loss'][i] for i in valid_idx], 'b-o', label='Val', markersize=4)
    ax.plot(steps_log, [history['dr_loss'][i] for i in valid_idx], 'r--s', label='Dead Reckoning', markersize=4)
    ax.plot(steps_log, [history['lp_loss'][i] for i in valid_idx], 'g--^', label='Last Position', markersize=4)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (KL Divergence)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Training Progress vs Baselines (log-log)')

    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png', dpi=150)
    plt.close()

    print(f"\n  Results saved to: {output_dir}", flush=True)
    print(f"    - Checkpoints: {checkpoints_dir}", flush=True)
    print(f"    - Results: {results_dir}", flush=True)
    print("=" * 70, flush=True)
    print("DONE", flush=True)
    print("=" * 70, flush=True)

    return results


if __name__ == '__main__':
    main()
