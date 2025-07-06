#!/usr/bin/env python3
"""
Time-aware multi-horizon causal trajectory prediction model
Incorporates time deltas between positions into the prediction
"""
import os
import sys
import time
import json
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pathlib import Path
from typing import Tuple, Iterator
import numpy as np
import polars as pl
import random
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nano_gpt_trajectory import TrajectoryGPT, TrajectoryGPTConfig
from four_head_2D_LR import FourierHead2DLR


def latlon_to_local_uv(lat, lon, lat0, lon0, half_side_mi=50.0):
    """Convert lat/lon to local UV coordinates."""
    R_EARTH_MI = 3959.0
    meters_per_degree_lat = (2 * np.pi * R_EARTH_MI / 360.0)
    meters_per_degree_lon = meters_per_degree_lat * torch.cos(torch.tensor(lat0) * np.pi / 180.0)
    
    u = (lon - lon0) * meters_per_degree_lon / half_side_mi
    v = (lat - lat0) * meters_per_degree_lat / half_side_mi
    
    # Jacobian for UV transformation
    J = meters_per_degree_lat * meters_per_degree_lon / (half_side_mi ** 2)
    logJ = torch.log(torch.tensor(J))
    
    return torch.stack([u, v], dim=-1).float(), logJ.float()


class TimeAwareNanoGPTTrajectoryForecaster(nn.Module):
    """Time-aware version of NanoGPT trajectory forecaster"""
    
    def __init__(self, seq_len=20, d_model=128, nhead=8, num_layers=6, 
                 ff_hidden=512, fourier_m=256, fourier_rank=4,
                 max_time_delta=3600, dropout=0.1):  # max 1 hour between positions
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.max_time_delta = max_time_delta
        
        # Create nanoGPT configuration
        # Position + context (lat/lon normalized) + time_delta = 5 features
        self.gpt_config = TrajectoryGPTConfig(
            block_size=seq_len,
            n_layer=num_layers,
            n_head=nhead,
            n_embd=d_model,
            dropout=dropout,
            bias=False,
            input_dim=5,  # (x, y, context_lat, context_lon, time_delta)
            fourier_m=fourier_m,
            fourier_rank=fourier_rank
        )
        
        # Create the nanoGPT backbone
        self.gpt = TrajectoryGPT(self.gpt_config)
        
        # Project to output representation
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Time encoding for each horizon
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
        # Combine causal representation with time encoding
        self.combine_proj = nn.Linear(d_model + d_model // 2, d_model)
        
        # Fourier head for PDF
        self.fourier_head = FourierHead2DLR(
            dim_input=d_model,
            num_frequencies=fourier_m,
            rank=fourier_rank,
            device="cpu"  # Will be moved with .to(device)
        )
    
    def to(self, device):
        """Override to method to handle Fourier head device"""
        super().to(device)
        # Update Fourier head device
        self.fourier_head.dev = device
        return self
    
    def encode_time_delta(self, time_delta_seconds):
        """Encode time delta in seconds to a feature vector"""
        # Normalize by max_time_delta
        normalized_time = time_delta_seconds / self.max_time_delta
        # Clamp to reasonable range
        normalized_time = torch.clamp(normalized_time, 0, 2.0)
        return self.time_encoder(normalized_time.unsqueeze(-1))
    
    def forward(self, input_uv, centre_ll, target_uv, causal_position, time_deltas):
        """
        Forward pass with time awareness
        
        Args:
            input_uv: (B, seq_len, 2) - input trajectory in UV coordinates
            centre_ll: (B, 2) - center lat/lon (normalized)
            target_uv: (B, horizon, 2) - target positions in UV
            causal_position: (B,) - which position to use for causal prediction
            time_deltas: (B, seq_len + horizon - 1) - time deltas between consecutive positions
        
        Returns:
            log_probs: (B, horizon) - log probabilities for each target
        """
        B, S, _ = input_uv.shape
        H = target_uv.shape[1]
        device = input_uv.device
        
        # Prepare input with time deltas
        # For each position, include the time delta from the previous position
        input_time_deltas = time_deltas[:, :S]  # (B, S)
        
        # Prepare input: position + context + time_delta
        context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
        time_expanded = input_time_deltas.unsqueeze(-1)  # (B, S, 1)
        
        input_sequence = torch.cat([
            input_uv,           # (B, S, 2)
            context_expanded,   # (B, S, 2)
            time_expanded      # (B, S, 1)
        ], dim=-1)  # (B, S, 5)
        
        # Forward through GPT backbone
        causal_hidden = self.gpt(input_sequence, causal_position)  # (B, d_model)
        base_repr = self.output_proj(causal_hidden)  # (B, d_model)
        
        # For each target horizon, combine base representation with time encoding
        log_probs = []
        
        for h in range(H):
            # Calculate cumulative time from causal position to this horizon
            # Need to sum time deltas from causal_position+1 to causal_position+h+1
            cumulative_time = torch.zeros(B, device=device)
            
            for b in range(B):
                start_idx = causal_position[b].item() + 1
                end_idx = causal_position[b].item() + h + 1
                
                # Handle different cases based on position
                if end_idx <= S:
                    # Target is within input sequence
                    cumulative_time[b] = time_deltas[b, start_idx:end_idx].sum()
                else:
                    # Target extends beyond input sequence
                    if start_idx < S:
                        # Part in input, part in future
                        input_part = time_deltas[b, start_idx:S].sum()
                        future_part = time_deltas[b, S:end_idx].sum()
                        cumulative_time[b] = input_part + future_part
                    else:
                        # Entirely in future
                        cumulative_time[b] = time_deltas[b, start_idx:end_idx].sum()
            
            # Encode time for this horizon
            time_repr = self.encode_time_delta(cumulative_time)  # (B, d_model//2)
            
            # Combine base representation with time encoding
            combined_repr = torch.cat([base_repr, time_repr], dim=-1)  # (B, d_model + d_model//2)
            horizon_repr = self.combine_proj(combined_repr)  # (B, d_model)
            
            # Get PDF for this target
            target_h = target_uv[:, h, :]  # (B, 2)
            pdf = self.fourier_head(horizon_repr, target_h)  # (B,)
            log_probs.append(pdf.log())
        
        log_probs = torch.stack(log_probs, dim=1)  # (B, H)
        return log_probs


class StreamingMultiHorizonAISDatasetTimeAware(torch.utils.data.IterableDataset):
    """Streaming dataset that includes time deltas"""
    
    def __init__(self, bucket_name, seq_len=20, horizon=10, 
                 chunk_size=500_000, half_side_mi=50.0):  # Increased chunk size
        self.bucket_name = bucket_name
        self.seq_len = seq_len
        self.horizon = horizon
        self.chunk_size = chunk_size
        self.half_side_mi = half_side_mi
        
        # List parquet files - use interpolated dataset
        self.s3 = boto3.client('s3')
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix='cleaned/interpolated/')
        # Filter for moving vessels file
        all_files = [obj['Key'] for obj in response.get('Contents', []) 
                     if obj['Key'].endswith('.parquet')]
        # Use only the moving vessels file
        self.parquet_files = [f for f in all_files if 'moving_vessels_5knots' in f]
        if not self.parquet_files:
            # Fallback to all interpolated files if moving vessels not found
            self.parquet_files = all_files
            print("⚠️ Warning: Moving vessels file not found, using all interpolated data")
        
        print(f">>> STREAMING TIME-AWARE MULTI-HORIZON CAUSAL DATASET (INTERPOLATED) <<<")
        print(f"🔍 Found {len(self.parquet_files)} interpolated parquet files")
    
    def _stream_file_windows(self, file_key):
        """Stream windows from a single parquet file with time information"""
        # Check if we have local data first
        import os
        local_path = "/home/ec2-user/ais_data/interpolated_ais_moving_vessels_5knots.parquet"
        if os.path.exists(local_path) and "moving_vessels" in file_key:
            print(f"📁 Using local file: {local_path}")
            df_scan = pl.scan_parquet(local_path)
        else:
            s3_path = f"s3://{self.bucket_name}/{file_key}"
            print(f"📡 Using S3 file: {s3_path}")
            df_scan = pl.scan_parquet(s3_path)
        
        total_rows = df_scan.select(pl.len()).collect().item()
        print(f"📊 File has {total_rows:,} rows, processing in chunks of {self.chunk_size:,}")
        
        for chunk_start in range(0, total_rows, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_rows)
            print(f"📦 Processing chunk {chunk_start:,}-{chunk_end:,}")
            
            df_chunk = df_scan.slice(chunk_start, chunk_end - chunk_start).collect()
            df_sorted = df_chunk.sort(["track_id", "timestamp"])
            
            # Group by track_id
            grouped = df_sorted.group_by("track_id").agg([
                pl.struct(["lat", "lon", "timestamp"]).alias("track_data")
            ])
            
            chunk_windows = []
            for row in grouped.iter_rows(named=True):
                track_data = row["track_data"]
                if len(track_data) >= self.seq_len + self.horizon:
                    # Extract positions and timestamps
                    positions = []
                    timestamps = []
                    
                    for point in track_data:
                        if hasattr(point, 'lat'):
                            positions.append([point.lat, point.lon])
                            timestamps.append(point.timestamp)
                        else:
                            positions.append([point['lat'], point['lon']])
                            timestamps.append(point['timestamp'])
                    
                    coords = torch.tensor(positions, dtype=torch.float32)
                    
                    # Calculate time deltas between consecutive positions
                    time_deltas = []
                    for i in range(1, len(timestamps)):
                        delta = timestamps[i] - timestamps[i-1]
                        if hasattr(delta, 'total_seconds'):
                            delta_seconds = delta.total_seconds()
                        else:
                            delta_seconds = delta / 1000  # Assume milliseconds
                        time_deltas.append(delta_seconds)
                    
                    time_deltas_tensor = torch.tensor(time_deltas, dtype=torch.float32)
                    
                    # Create windows with coordinates and time deltas
                    for i in range(len(coords) - self.seq_len - self.horizon + 1):
                        window_coords = coords[i:i + self.seq_len + self.horizon]
                        window_time_deltas = time_deltas_tensor[i:i + self.seq_len + self.horizon - 1]
                        chunk_windows.append((window_coords, window_time_deltas))
            
            random.shuffle(chunk_windows)
            for window in chunk_windows:
                yield window
            
            del df_chunk, chunk_windows
    
    def __iter__(self):
        shuffled_files = self.parquet_files.copy()
        random.shuffle(shuffled_files)
        
        for file_key in shuffled_files:
            print(f"📂 Processing {file_key}")
            
            for window_coords, window_time_deltas in self._stream_file_windows(file_key):
                input_positions = window_coords[:self.seq_len]
                target_positions = window_coords[self.seq_len:self.seq_len + self.horizon]
                
                lat0, lon0 = input_positions[-1]
                
                input_uv, _ = latlon_to_local_uv(
                    input_positions[:, 0], input_positions[:, 1],
                    lat0, lon0, self.half_side_mi
                )
                
                target_uv, logJ = latlon_to_local_uv(
                    target_positions[:, 0], target_positions[:, 1],
                    lat0, lon0, self.half_side_mi
                )
                
                centre_ll_norm = torch.tensor(
                    [lat0 / 90.0, lon0 / 180.0],
                    dtype=torch.float32
                )
                
                # For each causal position, yield training example
                for t in range(self.seq_len):
                    causal_input = input_uv
                    
                    # Determine target positions based on causal position
                    if t + 1 + self.horizon <= self.seq_len:
                        causal_target = input_uv[t+1:t+1+self.horizon]
                    elif t + 1 < self.seq_len:
                        remaining_input = input_uv[t+1:]
                        needed_future = self.horizon - len(remaining_input)
                        if needed_future <= len(target_uv):
                            future_part = target_uv[:needed_future]
                            causal_target = torch.cat([remaining_input, future_part], dim=0)
                        else:
                            continue
                    else:
                        if len(target_uv) >= self.horizon:
                            causal_target = target_uv[:self.horizon]
                        else:
                            continue
                    
                    causal_position = t
                    
                    # Prepare time deltas with a dummy first value (no previous position for first)
                    # Add a small time delta (10 seconds) for the first position
                    full_time_deltas = torch.cat([
                        torch.tensor([10.0]),  # Dummy first delta
                        window_time_deltas
                    ])
                    
                    yield (causal_input, centre_ll_norm, causal_target, logJ, 
                           causal_position, full_time_deltas)


# Simplified version for easier training
CausalTransformerForecasterTimeAware = TimeAwareNanoGPTTrajectoryForecaster


def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bucket_name = "ais-pipeline-data-10179bbf-us-east-1"
    
    # Model parameters
    seq_len = 20
    horizon = 10
    d_model = 128
    nhead = 8
    num_layers = 6
    ff_hidden = 512
    fourier_m = 128  # Reduced to 128 frequencies
    fourier_rank = 4
    
    # Training parameters
    batch_size = 1536  # Increased from 1024
    lr = 1e-4
    epochs = 10  # Reduced for initial testing
    save_every = 1000  # Save checkpoints every 1k steps
    log_every = 1  # Log every batch
    model_tag = f"time_aware_causal_multihorizon_moving_vessels_{fourier_m}freq_bs{batch_size}"
    ckpt_dir = "checkpoints"
    
    print(f"🚀 Training time-aware model on {device}")
    print(f"📊 Configuration:")
    print(f"  - Model: {fourier_m} frequencies, rank {fourier_rank}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Save every: {save_every} steps")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Prediction horizon: {horizon}")
    print()
    
    # Create dataset
    dataset = StreamingMultiHorizonAISDatasetTimeAware(
        bucket_name=bucket_name,
        seq_len=seq_len,
        horizon=horizon
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function to handle time deltas"""
        inputs, centres, targets, logJs, positions, time_deltas = zip(*batch)
        return (
            torch.stack(inputs),
            torch.stack(centres),
            torch.stack(targets),
            torch.stack(logJs),
            torch.tensor(positions),
            torch.stack(time_deltas)
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=2,  # Reduced to avoid CPU bottleneck
        prefetch_factor=2,
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=True  # Pin memory for faster GPU transfer
    )
    
    # Create model
    model = CausalTransformerForecasterTimeAware(
        seq_len=seq_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_hidden=ff_hidden,
        fourier_m=fourier_m,
        fourier_rank=fourier_rank
    ).to(device)
    
    print(f"📊 Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create checkpoint directory
    ckpt_path = Path(ckpt_dir) / model_tag
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        print(f"\n🚀 Starting epoch {epoch}/{epochs}")
        
        for batch_idx, (input_uv, centre_ll, target_uv, logJ, causal_position, time_deltas) in enumerate(dataloader):
            # Move to device
            input_uv = input_uv.to(device)
            centre_ll = centre_ll.to(device)
            target_uv = target_uv.to(device)
            logJ = logJ.to(device)
            causal_position = causal_position.to(device)
            time_deltas = time_deltas.to(device)
            
            # Forward pass
            if device.type == 'cuda' and scaler is not None:
                # Use mixed precision for main model, but not for Fourier head
                with torch.cuda.amp.autocast():
                    # Get representations from model
                    B, S, _ = input_uv.shape
                    H = target_uv.shape[1]
                    
                    # Prepare input with time deltas
                    input_time_deltas = time_deltas[:, :S]
                    context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
                    time_expanded = input_time_deltas.unsqueeze(-1)
                    input_sequence = torch.cat([input_uv, context_expanded, time_expanded], dim=-1)
                    
                    # Forward through GPT backbone
                    causal_hidden = model.gpt(input_sequence, causal_position)
                    base_repr = model.output_proj(causal_hidden)
                
                # Compute PDFs outside of autocast to avoid ComplexHalf issues
                log_probs = []
                for h in range(H):
                    # Calculate cumulative time
                    cumulative_time = torch.zeros(B, device=device)
                    for b in range(B):
                        start_idx = causal_position[b].item() + 1
                        end_idx = causal_position[b].item() + h + 1
                        if end_idx <= S:
                            cumulative_time[b] = time_deltas[b, start_idx:end_idx].sum()
                        else:
                            if start_idx < S:
                                input_part = time_deltas[b, start_idx:S].sum()
                                future_part = time_deltas[b, S:end_idx].sum()
                                cumulative_time[b] = input_part + future_part
                            else:
                                cumulative_time[b] = time_deltas[b, start_idx:end_idx].sum()
                    
                    # Encode time and combine with base representation
                    time_repr = model.encode_time_delta(cumulative_time)
                    combined_repr = torch.cat([base_repr, time_repr], dim=-1)
                    horizon_repr = model.combine_proj(combined_repr)
                    
                    # Get PDF for this target
                    target_h = target_uv[:, h, :]
                    pdf = model.fourier_head(horizon_repr, target_h)
                    log_probs.append(pdf.log())
                
                log_probs = torch.stack(log_probs, dim=1)
                loss = -(log_probs + logJ.unsqueeze(1)).mean()
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(input_uv, centre_ll, target_uv, causal_position, time_deltas)
                loss = -(log_probs + logJ.unsqueeze(1)).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            if batch_idx % log_every == 0:
                # Calculate detailed statistics
                mean_time_delta = time_deltas.mean().item()
                max_time_delta = time_deltas.max().item()
                min_time_delta = time_deltas.min().item()
                
                # Calculate per-horizon losses for first sample
                with torch.no_grad():
                    sample_log_probs = log_probs[0]  # First sample in batch
                    horizon_losses = (-(sample_log_probs + logJ[0])).tolist()
                
                # Get learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"  Batch {batch_idx} | Step {global_step} | Loss: {loss.item():.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Δt: {min_time_delta:.1f}-{mean_time_delta:.1f}-{max_time_delta:.1f}s | "
                      f"H-Loss: {[f'{hl:.2f}' for hl in horizon_losses[:3]]}...")
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'epoch': epoch,
                    'global_step': global_step
                }
                ckpt_file = ckpt_path / f"step_{global_step:08d}.pt"
                torch.save(checkpoint, ckpt_file)
                print(f"💾 Checkpoint saved: {ckpt_file}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"📊 Epoch {epoch} - Average loss: {avg_loss:.6f}")
        scheduler.step()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()