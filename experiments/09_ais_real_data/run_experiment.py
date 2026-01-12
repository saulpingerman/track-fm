#!/usr/bin/env python3
"""
Experiment 9: Real AIS Data - Causal Multi-Horizon Trajectory Prediction

Adapts the causal multi-horizon model from Experiment 8 to real AIS vessel tracking data.
Optimized for high GPU utilization on L40S (46GB).

Key adaptations:
- Input: (lat, lon, sog, cog, dt_seconds) -> 5 features
- Relative displacement prediction in degrees
- Handles variable track lengths and time spacing
- Mixed precision training (bfloat16)
- Large batch sizes for GPU utilization
"""

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

    # Data filtering
    min_track_length: int = 100  # Minimum positions per track
    max_seq_len: int = 128  # Sequence length for training
    min_sog: float = 3.0  # Minimum speed over ground (knots) - filter stationary vessels

    # Model architecture
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Fourier head
    grid_size: int = 64
    num_freqs: int = 12
    grid_range: float = 0.02  # Degrees - covers ~2.2km (99th percentile is 670m)

    # Multi-horizon prediction
    max_horizon: int = 20

    # Training
    batch_size: int = 64  # Reduced for max_horizon=20 (4x more output)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    val_every_n_batches: int = 1000  # Validate every N batches
    val_max_batches: int = 50  # Limit validation batches for speed

    # Loss
    sigma: float = 0.001  # Soft target sigma in degrees (~111m, appropriate for moving vessels)

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


def load_ais_data(config: Config, max_tracks: int = 2000) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load and preprocess AIS data from FSx - memory efficient version."""

    print("Loading track catalog...", flush=True)
    catalog = pl.read_parquet(config.catalog_path)

    # Filter tracks by minimum length and sample
    valid_tracks = catalog.filter(
        (pl.col("num_positions") >= config.min_track_length) &
        (pl.col("num_positions") <= 50000)  # Cap very long tracks
    ).sort("num_positions", descending=True)  # Prioritize longer tracks

    # Sample tracks if too many
    if len(valid_tracks) > max_tracks:
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
            (pl.col("cluster_assignment").is_null()) &
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
    track_ids = list(track_data.keys())
    np.random.shuffle(track_ids)
    split_idx = int(0.9 * len(track_ids))

    train_ids = set(track_ids[:split_idx])
    val_ids = set(track_ids[split_idx:])

    train_data = {k: v for k, v in track_data.items() if k in train_ids}
    val_data = {k: v for k, v in track_data.items() if k in val_ids}

    print(f"  Train tracks: {len(train_data)}, Val tracks: {len(val_data)}", flush=True)

    return train_data, val_data


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
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)

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

    def forward_train(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with multi-horizon prediction.

        Args:
            features: (batch, seq_len + max_horizon, 6)
                      [lat, lon, sog, cog_sin, cog_cos, dt]

        Returns:
            log_densities: (batch, num_pairs, grid_size, grid_size)
            targets: (batch, num_pairs, 2) - relative displacement in lat/lon
        """
        batch_size = features.shape[0]
        seq_len = self.config.max_seq_len
        max_horizon = self.config.max_horizon
        device = features.device

        # Split into input sequence and future positions
        input_seq = features[:, :seq_len, :]  # (batch, seq_len, 6)

        # Project and encode
        x = self.input_proj(input_seq)
        x = self.pos_encoder(x)

        # Causal transformer
        mask = self._get_causal_mask(seq_len, device)
        embeddings = self.transformer(x, mask=mask)

        # Compute cumulative dt for horizon conditioning
        dt = input_seq[:, :, 5] * self.config.dt_max  # Denormalize
        cumsum_dt = torch.cumsum(dt, dim=1)

        # Get positions (denormalized)
        positions = torch.stack([
            input_seq[:, :, 0] * self.config.lat_std + self.config.lat_mean,
            input_seq[:, :, 1] * self.config.lon_std + self.config.lon_mean
        ], dim=-1)

        # Also get future positions
        future_pos = torch.stack([
            features[:, seq_len:seq_len+max_horizon, 0] * self.config.lat_std + self.config.lat_mean,
            features[:, seq_len:seq_len+max_horizon, 1] * self.config.lon_std + self.config.lon_mean
        ], dim=-1)
        future_dt = features[:, seq_len:seq_len+max_horizon, 5] * self.config.dt_max
        future_cumsum = cumsum_dt[:, -1:] + torch.cumsum(future_dt, dim=1)

        # Build all (position, horizon) pairs
        all_embeddings = []
        all_cum_dt = []
        all_targets = []

        # Predictions from within sequence
        for h in range(1, max_horizon + 1):
            valid_pos = seq_len - h
            if valid_pos <= 0:
                continue

            src_emb = embeddings[:, :valid_pos, :]
            src_cumsum = cumsum_dt[:, :valid_pos]
            tgt_cumsum = cumsum_dt[:, h:h+valid_pos]
            cum_dt = tgt_cumsum - src_cumsum

            src_pos = positions[:, :valid_pos, :]
            tgt_pos = positions[:, h:h+valid_pos, :]
            tgt_rel = tgt_pos - src_pos

            all_embeddings.append(src_emb)
            all_cum_dt.append(cum_dt)
            all_targets.append(tgt_rel)

        # Predictions from end of sequence to future
        last_emb = embeddings[:, -1:, :].expand(-1, max_horizon, -1)
        last_cumsum = cumsum_dt[:, -1:]

        for h in range(1, max_horizon + 1):
            cum_dt = future_cumsum[:, h-1:h] - last_cumsum
            tgt_rel = future_pos[:, h-1:h, :] - positions[:, -1:, :]

            all_embeddings.append(last_emb[:, h-1:h, :])
            all_cum_dt.append(cum_dt)
            all_targets.append(tgt_rel)

        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=1)
        all_cum_dt = torch.cat(all_cum_dt, dim=1)
        all_targets = torch.cat(all_targets, dim=1)

        num_pairs = all_embeddings.shape[1]

        # Time encoding
        time_input = all_cum_dt.unsqueeze(-1) / 300.0  # Normalize by 5 minutes
        time_enc = self.time_proj(time_input)

        # Combine and project
        combined = torch.cat([all_embeddings, time_enc], dim=-1)
        conditioned = self.horizon_proj(combined)

        # Fourier head
        cond_flat = conditioned.view(batch_size * num_pairs, -1)
        log_dens_flat = self.fourier_head(cond_flat)
        log_densities = log_dens_flat.view(batch_size, num_pairs, self.config.grid_size, self.config.grid_size)

        return log_densities, all_targets


# ============================================================================
# Loss Function
# ============================================================================

def compute_soft_target_loss(log_density: torch.Tensor, target: torch.Tensor,
                             grid_range: float, grid_size: int, sigma: float) -> torch.Tensor:
    """Soft target KL divergence loss."""
    batch_size, num_pairs, gs, _ = log_density.shape
    device = log_density.device

    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    target_x = target[:, :, 0:1, None]
    target_y = target[:, :, 1:2, None]

    dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    loss = F.kl_div(log_density, soft_target, reduction='none').sum(dim=(-2, -1))
    return loss.mean()


# ============================================================================
# Training
# ============================================================================

def compute_dead_reckoning_loss(features: torch.Tensor, targets: torch.Tensor,
                                 config: Config) -> torch.Tensor:
    """
    Compute dead reckoning baseline loss for ALL prediction pairs (matching training).

    For each (source_position, horizon) pair that the model predicts:
    - Compute velocity at source position: (pos[i] - pos[i-1]) / dt[i]
    - Predict displacement: velocity * cumulative_time_to_target
    - Compare against actual target displacement

    This matches the structure of forward_train() which generates predictions from
    every position, not just the last one.
    """
    device = features.device
    batch_size = features.shape[0]
    seq_len = config.max_seq_len
    max_horizon = config.max_horizon

    # Get positions (denormalized to degrees)
    lat = features[:, :, 0] * config.lat_std + config.lat_mean
    lon = features[:, :, 1] * config.lon_std + config.lon_mean

    # Get time deltas (denormalized to seconds)
    dt = features[:, :, 5] * config.dt_max  # (batch, seq_len + max_horizon)

    # Compute velocity at each position (backward difference)
    # velocity[i] = (pos[i] - pos[i-1]) / dt[i]
    dlat = lat[:, 1:] - lat[:, :-1]  # (batch, total_len - 1)
    dlon = lon[:, 1:] - lon[:, :-1]
    dt_safe = dt[:, 1:].clamp(min=1.0)  # avoid div by zero
    vlat = dlat / dt_safe  # (batch, total_len - 1)
    vlon = dlon / dt_safe

    # Pad velocity so vlat[i] is velocity AT position i (using backward diff)
    # Position 0 uses forward difference (velocity from 0 to 1)
    vlat = torch.cat([vlat[:, :1], vlat], dim=1)  # (batch, total_len)
    vlon = torch.cat([vlon[:, :1], vlon], dim=1)

    # Cumulative dt for computing time to targets
    cumsum_dt = torch.cumsum(dt, dim=1)

    # Grid for computing loss
    grid_range = config.grid_range
    grid_size = config.grid_size
    sigma = config.sigma
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Build DR predictions matching forward_train structure
    all_dr_preds = []

    # Within-sequence predictions
    for h in range(1, max_horizon + 1):
        valid_pos = seq_len - h
        if valid_pos <= 0:
            continue

        # Source positions: 0, 1, ..., valid_pos-1
        # Target positions: h, h+1, ..., seq_len-1
        src_vlat = vlat[:, :valid_pos]  # (batch, valid_pos)
        src_vlon = vlon[:, :valid_pos]

        # Cumulative time from source to target
        src_cumsum = cumsum_dt[:, :valid_pos]
        tgt_cumsum = cumsum_dt[:, h:h+valid_pos]
        cum_time = tgt_cumsum - src_cumsum  # (batch, valid_pos)

        # DR prediction: velocity * time
        dr_lat = src_vlat * cum_time
        dr_lon = src_vlon * cum_time
        all_dr_preds.append(torch.stack([dr_lat, dr_lon], dim=-1))

    # Future predictions (from last input position to future)
    last_vlat = vlat[:, seq_len-1:seq_len]  # (batch, 1)
    last_vlon = vlon[:, seq_len-1:seq_len]
    last_cumsum = cumsum_dt[:, seq_len-1:seq_len]

    for h in range(1, max_horizon + 1):
        future_cumsum = cumsum_dt[:, seq_len+h-1:seq_len+h]
        cum_time = future_cumsum - last_cumsum  # (batch, 1)

        dr_lat = last_vlat * cum_time
        dr_lon = last_vlon * cum_time
        all_dr_preds.append(torch.stack([dr_lat, dr_lon], dim=-1))

    # Concatenate all predictions
    all_dr_preds = torch.cat(all_dr_preds, dim=1)  # (batch, num_pairs, 2)

    # Clip to grid range
    all_dr_preds = torch.clamp(all_dr_preds, -grid_range * 0.99, grid_range * 0.99)

    # Reshape for grid computation
    dr_pred_lat = all_dr_preds[:, :, 0:1, None]  # (batch, num_pairs, 1, 1)
    dr_pred_lon = all_dr_preds[:, :, 1:2, None]

    # Create Gaussian distribution centered at DR prediction
    dist_sq = (xx - dr_pred_lat) ** 2 + (yy - dr_pred_lon) ** 2
    dr_density = torch.exp(-dist_sq / (2 * sigma ** 2))
    dr_density = dr_density / (dr_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    dr_log_density = torch.log(dr_density + 1e-10)

    # Compute target soft distribution
    target_x = targets[:, :, 0:1, None]
    target_y = targets[:, :, 1:2, None]
    target_dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-target_dist_sq / (2 * sigma ** 2))
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
    sigma = config.sigma

    # Create grid
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Last position predicts ZERO displacement for ALL pairs
    lp_pred_lat = torch.zeros(batch_size, num_pairs, 1, 1, device=device)
    lp_pred_lon = torch.zeros(batch_size, num_pairs, 1, 1, device=device)

    dist_sq = (xx - lp_pred_lat) ** 2 + (yy - lp_pred_lon) ** 2
    lp_density = torch.exp(-dist_sq / (2 * sigma ** 2))
    lp_density = lp_density / (lp_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    lp_log_density = torch.log(lp_density + 1e-10)

    # Target distribution (all pairs)
    target_x = targets[:, :, 0:1, None]
    target_y = targets[:, :, 1:2, None]
    target_dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-target_dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    loss = F.kl_div(lp_log_density, soft_target, reduction='none').sum(dim=(-2, -1))
    return loss.mean()


def validate_with_baselines(model: nn.Module, random_model: nn.Module,
                            val_loader: DataLoader, config: Config,
                            max_batches: int = None) -> dict:
    """
    Validate model AND compute ALL baselines on SAME data.

    Returns dict with:
    - model: trained model loss
    - random: untrained model loss
    - dead_reckoning: velocity * horizon extrapolation
    - last_position: predict zero displacement

    IMPORTANT: All losses computed on SAME subset of targets for fair comparison.
    """
    model.eval()
    random_model.eval()

    results = {
        'model': [],
        'random': [],
        'dead_reckoning': [],
        'last_position': []
    }

    max_batches = max_batches or len(val_loader)
    max_horizon = config.max_horizon

    with torch.no_grad():
        for batch_idx, features in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            features = features.to(DEVICE, non_blocking=True)

            with autocast('cuda', enabled=config.use_amp):
                # Trained model - use ALL predictions (matching training)
                log_densities, targets = model.forward_train(features)

                model_loss = compute_soft_target_loss(
                    log_densities, targets,
                    config.grid_range, config.grid_size, config.sigma
                )
                results['model'].append(model_loss.item())

                # Random (untrained) model - use ALL predictions
                random_log_densities, _ = random_model.forward_train(features)
                random_loss = compute_soft_target_loss(
                    random_log_densities, targets,
                    config.grid_range, config.grid_size, config.sigma
                )
                results['random'].append(random_loss.item())

                # Dead reckoning baseline - computed on ALL prediction pairs
                dr_loss = compute_dead_reckoning_loss(features, targets, config)
                results['dead_reckoning'].append(dr_loss.item())

                # Last position baseline - computed on ALL prediction pairs
                lp_loss = compute_last_position_loss(features, targets, config)
                results['last_position'].append(lp_loss.item())

    return {k: np.mean(v) for k, v in results.items()}


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, config: Config,
                output_dir: Path, checkpoints_dir: Path = None) -> Dict:
    """Train the model with validation every N batches."""

    # Use checkpoints subfolder if provided, otherwise fall back to output_dir
    if checkpoints_dir is None:
        checkpoints_dir = output_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model.to(DEVICE)

    # Create random model for baseline comparison (fixed throughout training)
    random_model = CausalAISModel(config).to(DEVICE)
    random_model.eval()
    for param in random_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler('cuda') if config.use_amp else None

    history = {
        'train_loss': [],
        'val_loss': [],
        'random_loss': [],
        'dr_loss': [],
        'lp_loss': [],
        'lr': [],
        'step': []
    }
    best_val_loss = float('inf')

    total_steps = 0
    running_loss = 0
    running_count = 0

    # Initial validation with ALL baselines
    print("\nInitial validation (before training)...", flush=True)
    val_results = validate_with_baselines(model, random_model, val_loader, config, config.val_max_batches)
    print(f"  Model (untrained): {val_results['model']:.4f}", flush=True)
    print(f"  Random Model:      {val_results['random']:.4f}", flush=True)
    print(f"  Dead Reckoning:    {val_results['dead_reckoning']:.4f}", flush=True)
    print(f"  Last Position:     {val_results['last_position']:.4f}", flush=True)

    history['val_loss'].append(val_results['model'])
    history['random_loss'].append(val_results['random'])
    history['dr_loss'].append(val_results['dead_reckoning'])
    history['lp_loss'].append(val_results['last_position'])
    history['train_loss'].append(float('nan'))
    history['lr'].append(config.learning_rate)
    history['step'].append(0)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_start = time.time()

        for batch_idx, features in enumerate(train_loader):
            features = features.to(DEVICE, non_blocking=True)

            # Forward with mixed precision
            with autocast('cuda', enabled=config.use_amp):
                log_densities, targets = model.forward_train(features)
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

            # Progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / running_count
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}, "
                      f"GPU: {gpu_mem:.1f}GB", flush=True)

            # Validation every N batches
            if (batch_idx + 1) % config.val_every_n_batches == 0:
                avg_train = running_loss / running_count
                val_results = validate_with_baselines(model, random_model, val_loader, config, config.val_max_batches)

                val_loss = val_results['model']
                dr_loss = val_results['dead_reckoning']
                lp_loss = val_results['last_position']
                random_loss = val_results['random']

                history['train_loss'].append(avg_train)
                history['val_loss'].append(val_loss)
                history['random_loss'].append(random_loss)
                history['dr_loss'].append(dr_loss)
                history['lp_loss'].append(lp_loss)
                history['lr'].append(scheduler.get_last_lr()[0])
                history['step'].append(total_steps)

                vs_dr = (dr_loss - val_loss) / dr_loss * 100
                vs_lp = (lp_loss - val_loss) / lp_loss * 100

                print(f"\n  >>> VALIDATION at step {total_steps}:", flush=True)
                print(f"      Train Loss:     {avg_train:.4f}", flush=True)
                print(f"      ---", flush=True)
                print(f"      Model:          {val_loss:.4f}  (vs DR: {vs_dr:+.1f}%, vs LP: {vs_lp:+.1f}%)", flush=True)
                print(f"      Dead Reckoning: {dr_loss:.4f}", flush=True)
                print(f"      Last Position:  {lp_loss:.4f}", flush=True)
                print(f"      Random Model:   {random_loss:.4f}\n", flush=True)

                # Save checkpoint at every validation
                checkpoint_path = checkpoints_dir / f'checkpoint_step_{total_steps}.pt'
                torch.save({
                    'step': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_results': val_results,
                }, checkpoint_path)
                print(f"      Saved checkpoint: {checkpoint_path.name}", flush=True)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), checkpoints_dir / 'best_model.pt')
                    print(f"      New best model!", flush=True)

                running_loss = 0
                running_count = 0
                model.train()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config.num_epochs} complete, Time={epoch_time:.1f}s", flush=True)

    # Final validation
    print("\nFinal validation...", flush=True)
    val_results = validate_with_baselines(model, random_model, val_loader, config, config.val_max_batches)

    val_loss = val_results['model']
    dr_loss = val_results['dead_reckoning']
    lp_loss = val_results['last_position']
    random_loss = val_results['random']

    history['val_loss'].append(val_loss)
    history['random_loss'].append(random_loss)
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
    print(f"  Random Model:   {random_loss:.4f}", flush=True)

    return history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: Config) -> Dict:
    """Evaluate model prediction errors."""
    model.eval()

    grid = torch.linspace(-config.grid_range, config.grid_range,
                          config.grid_size, device=DEVICE)

    errors_by_horizon = {h: [] for h in range(1, config.max_horizon + 1)}

    with torch.no_grad():
        for features in val_loader:
            features = features.to(DEVICE)
            log_densities, targets = model.forward_train(features)

            densities = torch.exp(log_densities)

            # Expected value
            marginal_x = densities.sum(dim=3)
            marginal_y = densities.sum(dim=2)
            exp_x = (marginal_x * grid).sum(dim=2)
            exp_y = (marginal_y * grid).sum(dim=2)

            # Error in degrees
            error = torch.sqrt((exp_x - targets[:, :, 0])**2 + (exp_y - targets[:, :, 1])**2)

            # Approximate conversion: distribute errors by horizon
            # (simplified - actual implementation should track horizon per sample)
            for i, err in enumerate(error.mean(dim=0).cpu().numpy()):
                h = (i % config.max_horizon) + 1
                errors_by_horizon[h].append(err)

    metrics = {}
    for h in range(1, config.max_horizon + 1):
        if errors_by_horizon[h]:
            mean_err = np.mean(errors_by_horizon[h])
            metrics[f'horizon_{h}_error_deg'] = mean_err
            metrics[f'horizon_{h}_error_km'] = mean_err * 111  # Approximate

    return metrics


# ============================================================================
# Main
# ============================================================================

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
                        help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("EXPERIMENT 9: REAL AIS DATA - CAUSAL MULTI-HORIZON PREDICTION", flush=True)
    print("=" * 70, flush=True)

    config = Config()

    # Override config with command line arguments
    if args.max_horizon is not None:
        config.max_horizon = args.max_horizon
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    # Create output directory
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
    print(f"\nExperiment: {exp_name}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"  Checkpoints: {checkpoints_dir}", flush=True)
    print(f"  Results: {results_dir}", flush=True)

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

    train_data, val_data = load_ais_data(config)

    # Create datasets
    print("\nCreating datasets...", flush=True)
    train_dataset = AISTrackDataset(train_data, config.max_seq_len, config.max_horizon, config)
    val_dataset = AISTrackDataset(val_data, config.max_seq_len, config.max_horizon, config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
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

    # Train
    print("\n" + "=" * 70, flush=True)
    print("TRAINING", flush=True)
    print("=" * 70, flush=True)

    history = train_model(model, train_loader, val_loader, config, output_dir, checkpoints_dir)

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
    ax.plot(steps_log, [history['random_loss'][i] for i in valid_idx], 'gray', linestyle=':', label='Random Model', alpha=0.7)

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
