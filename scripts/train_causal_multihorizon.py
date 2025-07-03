#!/usr/bin/env python
"""
Causal Multi-Horizon AIS Trajectory Forecasting
- Decoder-only transformer with causal attention
- Multi-horizon prediction (10 steps ahead)
- Low-rank Fourier head (64 frequencies, rank 4)
- S3 data loading with random sampling
"""

import math
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, List, Tuple

import boto3
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# Import components
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from four_head_2D_LR import FourierHead2DLR
from nano_gpt_trajectory import TrajectoryGPT, TrajectoryGPTConfig

# ─── Geometry helper ──────────────────────────────────────────────────────────
def latlon_to_local_uv(lat, lon, lat0, lon0, half_side_mi: float = 50.0):
    """Convert lat/lon to local UV coordinates in [-1,1]²"""
    R_mi = 69.0
    dx = R_mi * torch.cos(torch.deg2rad(lat0)) * (lon - lon0)
    dy = R_mi * (lat - lat0)
    u = dx / half_side_mi
    v = dy / half_side_mi
    logJ = (
        2.0 * math.log(half_side_mi)
        - 2.0 * math.log(R_mi)
        - torch.log(torch.cos(torch.deg2rad(lat0)).clamp_min(1e-6))
    )
    uv = torch.stack([u.clamp(-1, 1), v.clamp(-1, 1)], dim=-1)
    return uv, logJ

# ─── S3 Data Loading ──────────────────────────────────────────────────────────
def load_s3_data(bucket_name: str, prefix: str = "cleaned/", limit_rows: Optional[int] = None) -> pl.DataFrame:
    """Load cleaned AIS data from S3 bucket"""
    s3 = boto3.client('s3')
    
    print(f"🔍 Listing objects in s3://{bucket_name}/{prefix}")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        raise FileNotFoundError(f"No files found in s3://{bucket_name}/{prefix}")
    
    parquet_files = [
        obj['Key'] for obj in response['Contents'] 
        if obj['Key'].endswith('.parquet')
    ]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in s3://{bucket_name}/{prefix}")
    
    print(f"🟢 Found {len(parquet_files)} parquet files")
    
    # Download and load files with optional row limit
    dfs = []
    for file_key in tqdm(parquet_files, desc="Loading S3 files"):
        s3_uri = f"s3://{bucket_name}/{file_key}"
        
        # Use streaming to limit memory usage
        if limit_rows:
            # Read with row limit to control memory
            df = pl.scan_parquet(s3_uri).head(limit_rows).collect()
            print(f"📊 Limited to {limit_rows} rows for memory efficiency")
        else:
            df = pl.read_parquet(s3_uri)
        
        dfs.append(df)
    
    return pl.concat(dfs)

# ─── Streaming Multi-Horizon Dataset ─────────────────────────────────────────
class StreamingMultiHorizonAISDataset(IterableDataset):
    """
    Streaming multi-horizon causal prediction dataset that loads data on-demand.
    Does not load entire dataset into memory.
    """
    
    def __init__(
        self,
        bucket_name: str,
        prefix: str = "cleaned/",
        seq_len: int = 20,
        horizon: int = 10,
        half_side_mi: float = 50.0,
        min_track_len: int = 50,
    ):
        print(">>> STREAMING MULTI-HORIZON CAUSAL DATASET <<<", flush=True)
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.seq_len = seq_len
        self.horizon = horizon
        self.half_side_mi = half_side_mi
        self.min_track_len = min_track_len
        
        # Get list of S3 files
        s3 = boto3.client('s3')
        print(f"🔍 Listing objects in s3://{bucket_name}/{prefix}")
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            raise FileNotFoundError(f"No files found in s3://{bucket_name}/{prefix}")
        
        self.parquet_files = [
            obj['Key'] for obj in response['Contents'] 
            if obj['Key'].endswith('.parquet')
        ]
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in s3://{bucket_name}/{prefix}")
        
        print(f"🟢 Found {len(self.parquet_files)} parquet files for streaming")
    
    def _stream_file_windows(self, file_key: str, chunk_size: int = 200000):
        """Stream windows from one S3 file in chunks to avoid memory issues"""
        s3_uri = f"s3://{self.bucket_name}/{file_key}"
        
        # Read file in chunks using lazy loading
        df_scan = pl.scan_parquet(s3_uri)
        
        # Get total rows to calculate chunks
        total_rows = df_scan.select(pl.count()).collect().item()
        print(f"📊 File has {total_rows:,} rows, processing in chunks of {chunk_size:,}")
        
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            print(f"📦 Processing chunk {chunk_start:,}-{chunk_end:,}")
            
            # Load this chunk
            df_chunk = df_scan.slice(chunk_start, chunk_end - chunk_start).collect()
            
            # Sort by MMSI and timestamp, then group by track_id
            df_sorted = df_chunk.sort(["mmsi", "timestamp"])
            
            # Group by track_id (assuming track_id exists in cleaned data)
            if "track_id" in df_chunk.columns:
                grouped = df_sorted.group_by("track_id").agg([
                    pl.struct(["lat", "lon", "timestamp"]).alias("track_data")
                ])
            else:
                # Fallback: group by mmsi
                grouped = df_sorted.group_by("mmsi").agg([
                    pl.struct(["lat", "lon", "timestamp"]).alias("track_data")
                ])
            
            # Create windows for each track in this chunk
            chunk_windows = []
            for row in grouped.iter_rows(named=True):
                track_data = row["track_data"]
                if len(track_data) >= self.seq_len + self.horizon:
                    # Extract coordinates - handle both structured and list data
                    if hasattr(track_data[0], 'lat'):
                        # Structured data
                        coords = torch.tensor(
                            [[point.lat, point.lon] for point in track_data],
                            dtype=torch.float32
                        )
                    else:
                        # List data
                        coords = torch.tensor(
                            [[point['lat'], point['lon']] for point in track_data],
                            dtype=torch.float32
                        )
                    
                    # Create all valid windows for this track
                    for i in range(len(coords) - self.seq_len - self.horizon + 1):
                        chunk_windows.append(coords[i:i + self.seq_len + self.horizon])
            
            # Yield windows from this chunk
            random.shuffle(chunk_windows)
            for window in chunk_windows:
                yield window
            
            # Clear memory
            del df_chunk, chunk_windows
    
    def __iter__(self):
        # Shuffle file order for randomness
        shuffled_files = self.parquet_files.copy()
        random.shuffle(shuffled_files)
        
        for file_key in shuffled_files:
            print(f"📂 Processing {file_key}")
            
            # Stream windows from this file in chunks
            for window in self._stream_file_windows(file_key):
                # window shape: (seq_len + horizon, 2)
                input_positions = window[:self.seq_len]  # (20, 2)
                target_positions = window[self.seq_len:self.seq_len + self.horizon]  # (10, 2)
                
                # Use the last input position as the coordinate reference
                lat0, lon0 = input_positions[-1]
                
                # Convert to local coordinates
                input_uv, _ = latlon_to_local_uv(
                    input_positions[:, 0], input_positions[:, 1],
                    lat0, lon0, self.half_side_mi
                )
                
                target_uv, logJ = latlon_to_local_uv(
                    target_positions[:, 0], target_positions[:, 1],
                    lat0, lon0, self.half_side_mi
                )
                
                # Normalized center coordinates
                centre_ll_norm = torch.tensor(
                    [lat0 / 90.0, lon0 / 180.0],
                    dtype=torch.float32
                )
                
                # Create causal training examples
                for t in range(self.seq_len):
                    # Input: ALWAYS all seq_len positions (no padding!)
                    causal_input = input_uv  # (seq_len, 2) - always full window
                    
                    # Target: next horizon positions after position t
                    if t + 1 + self.horizon <= self.seq_len:
                        # Predict within input sequence
                        causal_target = input_uv[t+1:t+1+self.horizon]
                    elif t + 1 < self.seq_len:
                        # Predict partially in input, partially in future
                        remaining_input = input_uv[t+1:]
                        needed_future = self.horizon - len(remaining_input)
                        if needed_future <= len(target_uv):
                            future_part = target_uv[:needed_future]
                            causal_target = torch.cat([remaining_input, future_part], dim=0)
                        else:
                            # Not enough future positions - skip this sample
                            continue
                    else:
                        # Predict entirely in future (t = seq_len - 1)
                        if len(target_uv) >= self.horizon:
                            causal_target = target_uv[:self.horizon]
                        else:
                            # Not enough future positions - skip this sample
                            continue
                    
                    # Causal position indicator (which position's output to use for prediction)
                    causal_position = t
                    
                    yield causal_input, centre_ll_norm, causal_target, logJ, causal_position

# ─── Legacy Dataset (for compatibility) ───────────────────────────────────────
class MultiHorizonAISDataset(StreamingMultiHorizonAISDataset):
    """Legacy alias that now uses streaming"""
    def __init__(self, df: pl.DataFrame, seq_len: int = 20, horizon: int = 10, **kwargs):
        # This is a compatibility shim - in practice we'll use the streaming version
        # For now, fall back to the old behavior but print a warning
        print("⚠️  Using legacy dataset interface - recommend switching to streaming")
        super().__init__(
            bucket_name="ais-pipeline-data-10179bbf-us-east-1", 
            seq_len=seq_len, 
            horizon=horizon, 
            **kwargs
        )

# ─── NanoGPT-based Trajectory Model ───────────────────────────────────────────
class NanoGPTTrajectoryForecaster(nn.Module):
    """NanoGPT-based decoder-only transformer for trajectory forecasting"""
    
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_hidden: int,  # Not used in nanoGPT (fixed at 4*d_model)
        fourier_m: int,
        fourier_rank: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Create nanoGPT configuration
        self.gpt_config = TrajectoryGPTConfig(
            block_size=seq_len,
            n_layer=num_layers,
            n_head=nhead,
            n_embd=d_model,
            dropout=dropout,
            bias=False,
            input_dim=4,  # (x, y, context_lat, context_lon)
            fourier_m=fourier_m,
            fourier_rank=fourier_rank
        )
        
        # Create the nanoGPT backbone
        self.gpt = TrajectoryGPT(self.gpt_config)
        
        # Low-rank Fourier head for multi-horizon prediction
        self.fourier_head = FourierHead2DLR(
            dim_input=d_model,
            num_frequencies=fourier_m,
            rank=fourier_rank,
            device="cpu"  # Default to CPU, will be moved with .to(device)
        )
        
        # Output projection for Fourier head
        self.output_proj = nn.Linear(d_model, d_model)
    
    def to(self, device):
        """Override to method to handle Fourier head device"""
        super().to(device)
        # Update Fourier head device
        self.fourier_head.dev = device
        return self
        
    def forward(self, input_uv, centre_ll, target_uv, causal_position):
        """
        Args:
            input_uv: (B, seq_len, 2) - input positions (always full length)
            centre_ll: (B, 2) - normalized center coordinates
            target_uv: (B, horizon, 2) - target positions
            causal_position: (B,) - which position's output to use for prediction
        """
        B, S, _ = input_uv.shape
        H = target_uv.shape[1]  # horizon length
        
        # Prepare input: concatenate position with context
        context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
        input_sequence = torch.cat([input_uv, context_expanded], dim=-1)  # (B, S, 4)
        
        # Forward through nanoGPT backbone
        causal_hidden = self.gpt(input_sequence, causal_position)  # (B, d_model)
        
        # Project for output
        output_repr = self.output_proj(causal_hidden)  # (B, d_model)
        
        # Compute PDF for each target position
        log_probs = []
        for t in range(H):
            target_t = target_uv[:, t, :]  # (B, 2)
            pdf = self.fourier_head(output_repr, target_t)  # (B,)
            log_probs.append(pdf.log())
        
        return torch.stack(log_probs, dim=1)  # (B, H)

# ─── Legacy Model (for compatibility) ─────────────────────────────────────────
class CausalTransformerForecaster(NanoGPTTrajectoryForecaster):
    """Legacy alias for backward compatibility"""
    pass

# ─── Training Loop ────────────────────────────────────────────────────────────
def train(
    bucket_name: str,
    seq_len: int = 20,
    horizon: int = 10,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 6,
    ff_hidden: int = 512,
    fourier_m: int = 64,
    fourier_rank: int = 4,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 100,
    ckpt_dir: str = "checkpoints",
    model_tag: str = "causal_multihorizon",
    ckpt_every: int = 10000,
    device: Optional[str] = None,
    limit_rows: Optional[int] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    
    # Create streaming dataset (no need to load data into memory)
    dataset = StreamingMultiHorizonAISDataset(
        bucket_name=bucket_name, 
        seq_len=seq_len, 
        horizon=horizon
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset handles shuffling
        drop_last=True,
        num_workers=4,   # Use multiple workers for data loading
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    # Create model
    model = CausalTransformerForecaster(
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_hidden=ff_hidden,
        fourier_m=fourier_m,
        fourier_rank=fourier_rank
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Mixed precision training for better GPU utilization
    scaler = torch.cuda.amp.GradScaler()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create checkpoint directory
    ckpt_path = Path(ckpt_dir) / model_tag
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Checkpoints will be saved to: {ckpt_path}")
    
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start = time.time()
        
        print(f"🚀 Starting epoch {epoch}/{epochs}")
        
        for batch_idx, (input_uv, centre_ll, target_uv, logJ, causal_position) in enumerate(dataloader):
            batch_start_time = time.time()
            global_step += 1
            
            if batch_idx == 0:
                print(f"📦 Processing first batch of epoch {epoch}")
                print(f"  Input shape: {input_uv.shape}")
                print(f"  Target shape: {target_uv.shape}")
            
            # Move to device
            data_move_start = time.time()
            input_uv = input_uv.to(device)
            centre_ll = centre_ll.to(device)
            target_uv = target_uv.to(device)
            logJ = logJ.to(device)
            causal_position = causal_position.to(device)
            data_move_time = time.time() - data_move_start
            
            # Forward pass with mixed precision (but not for Fourier head)
            forward_start = time.time()
            with torch.cuda.amp.autocast():
                B, S, _ = input_uv.shape
                H = target_uv.shape[1]
                
                # Prepare input: concatenate position with context
                context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
                input_sequence = torch.cat([input_uv, context_expanded], dim=-1)
                
                # Forward through nanoGPT backbone (mixed precision)
                causal_hidden = model.gpt(input_sequence, causal_position)
                output_repr = model.output_proj(causal_hidden)
            forward_time = time.time() - forward_start
            
            # Fourier head computation (separate timing)
            fourier_start = time.time()
            log_probs = []
            for t in range(H):
                target_t = target_uv[:, t, :]
                pdf = model.fourier_head(output_repr, target_t)
                log_probs.append(pdf.log())
            log_probs = torch.stack(log_probs, dim=1)
            fourier_time = time.time() - fourier_start
            
            # Loss calculation (separate timing)
            loss_start = time.time()
            loss = -(log_probs + logJ.unsqueeze(1)).mean()
            loss_time = time.time() - loss_start
            
            # Backward pass with gradient scaling
            backward_start = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping with scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            backward_time = time.time() - backward_start
            
            # Update metrics
            total_loss += loss.item() * input_uv.size(0)
            total_samples += input_uv.size(0)
            
            batch_total_time = time.time() - batch_start_time
            
            # Print detailed timing every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Total={batch_total_time:.3f}s, "
                      f"GPT={forward_time:.3f}s, Fourier={fourier_time:.3f}s, "
                      f"Loss={loss_time:.3f}s, Backward={backward_time:.3f}s, "
                      f"LossVal={loss.item():.4f}")
            
            # Checkpoint
            if global_step % ckpt_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }
                ckpt_file = ckpt_path / f"step_{global_step:07d}.pt"
                torch.save(checkpoint, ckpt_file)
                print(f"💾 Checkpoint saved: {ckpt_file}")
        
        # End of epoch
        scheduler.step()
        avg_loss = total_loss / total_samples
        epoch_time = time.time() - epoch_start
        
        print(f"🟢 Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Time: {epoch_time/60:.1f}min | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ Training complete!")
    return model

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train with maximum batch size and full dataset streaming
    model = train(
        bucket_name="ais-pipeline-data-10179bbf-us-east-1",
        seq_len=20,
        horizon=10,
        d_model=128,           # Keep original model size
        nhead=8,               # Keep original attention heads
        num_layers=6,          # Keep original layers
        ff_hidden=512,         # Keep original feed-forward
        fourier_m=64,          # 64 frequencies as requested
        fourier_rank=4,        # Rank 4 as requested
        batch_size=2560,       # Push even higher
        lr=1e-4,
        epochs=50,
        model_tag="causal_multihorizon_maxbatch",
        limit_rows=None        # Use ALL data with streaming
    )