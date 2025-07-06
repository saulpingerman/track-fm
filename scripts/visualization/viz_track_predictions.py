#!/usr/bin/env python3
"""
Visualize time-aware model predictions using local interpolated AIS data
"""
import sys
sys.path.append('/home/ec2-user/repos/track-fm')
sys.path.append('/home/ec2-user/repos/track-fm/scripts')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import random

from train_causal_multihorizon_time_aware import TimeAwareNanoGPTTrajectoryForecaster, latlon_to_local_uv


def uv_to_miles(uv_coords):
    """Convert UV coordinates to miles"""
    return uv_coords * 50.0


def calculate_bounds(input_uv, target_point=None, padding=0.2):
    """Calculate bounds for visualization"""
    input_np = input_uv.cpu().numpy() if torch.is_tensor(input_uv) else input_uv
    
    x_min, x_max = input_np[:, 0].min(), input_np[:, 0].max()
    y_min, y_max = input_np[:, 1].min(), input_np[:, 1].max()
    
    if target_point is not None:
        x_min = min(x_min, target_point[0])
        x_max = max(x_max, target_point[0])
        y_min = min(y_min, target_point[1])
        y_max = max(y_max, target_point[1])
    
    x_range = max(x_max - x_min, 0.05)
    y_range = max(y_max - y_min, 0.05)
    
    return {
        'x_min': x_min - padding * x_range,
        'x_max': x_max + padding * x_range,
        'y_min': y_min - padding * y_range,
        'y_max': y_max + padding * y_range
    }


def load_tracks_from_local(data_path, n_tracks=5, min_length=35):
    """Load tracks from local parquet file"""
    print(f"Loading data from: {data_path}")
    
    # Load the full dataset
    df = pl.read_parquet(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Find tracks with sufficient length
    track_counts = df.group_by('track_id').agg(pl.len().alias('count'))
    valid_tracks = track_counts.filter(pl.col('count') >= min_length).sort('count', descending=True)
    
    print(f"Found {len(valid_tracks)} tracks with >= {min_length} points")
    
    # Select diverse tracks (some short, some long)
    if len(valid_tracks) > n_tracks:
        # Get a mix of track lengths
        n_short = n_tracks // 3
        n_medium = n_tracks // 3
        n_long = n_tracks - n_short - n_medium
        
        short_tracks = valid_tracks.filter(pl.col('count') < 50).head(n_short)['track_id'].to_list()
        medium_tracks = valid_tracks.filter((pl.col('count') >= 50) & (pl.col('count') < 100)).head(n_medium)['track_id'].to_list()
        long_tracks = valid_tracks.filter(pl.col('count') >= 100).head(n_long)['track_id'].to_list()
        
        selected_ids = short_tracks + medium_tracks + long_tracks
    else:
        selected_ids = valid_tracks['track_id'].to_list()
    
    tracks = []
    for track_id in selected_ids[:n_tracks]:
        track_df = df.filter(pl.col('track_id') == track_id).sort('timestamp')
        
        tracks.append({
            'track_id': track_id,
            'lats': track_df['lat'].to_numpy(),
            'lons': track_df['lon'].to_numpy(),
            'timestamps': track_df['timestamp'].to_numpy(),
            'length': len(track_df)
        })
        
        print(f"  Track {track_id}: {len(track_df)} points")
    
    return tracks


def evaluate_pdf_grid(model, input_uv, centre_ll, causal_position, time_deltas, 
                     horizon_idx, bounds, grid_size=100):
    """Evaluate PDF on a grid"""
    device = input_uv.device
    
    # Create grid
    x = torch.linspace(bounds['x_min'], bounds['x_max'], grid_size).to(device)
    y = torch.linspace(bounds['y_min'], bounds['y_max'], grid_size).to(device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten grid
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    
    # Evaluate in batches
    batch_size = 1000
    log_probs = []
    
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_size_actual = len(batch_points)
        
        # Create dummy targets
        dummy_targets = torch.zeros(batch_size_actual, horizon_idx + 1, 2).to(device)
        dummy_targets[:, horizon_idx, :] = batch_points
        
        # Expand inputs
        input_batch = input_uv.expand(batch_size_actual, -1, -1)
        centre_batch = centre_ll.expand(batch_size_actual, -1)
        causal_batch = causal_position.expand(batch_size_actual)
        time_batch = time_deltas.expand(batch_size_actual, -1)
        
        with torch.no_grad():
            batch_log_probs = model(input_batch, centre_batch, dummy_targets, 
                                  causal_batch, time_batch)
            log_probs.append(batch_log_probs[:, horizon_idx])
    
    log_probs = torch.cat(log_probs)
    probs = torch.exp(log_probs).reshape(grid_size, grid_size)
    
    return xx.cpu().numpy(), yy.cpu().numpy(), probs.cpu().numpy()


def visualize_track_predictions(model, track, device, output_path, seq_len=20):
    """Visualize predictions for a single track"""
    lats = track['lats']
    lons = track['lons']
    
    # Select a window from the middle of the track
    if len(lats) > seq_len + 10:
        start_idx = (len(lats) - seq_len - 10) // 2
    else:
        start_idx = 0
    
    input_lats = lats[start_idx:start_idx + seq_len]
    input_lons = lons[start_idx:start_idx + seq_len]
    future_lats = lats[start_idx + seq_len:start_idx + seq_len + 10]
    future_lons = lons[start_idx + seq_len:start_idx + seq_len + 10]
    
    # Reference point
    lat0, lon0 = float(input_lats[-1]), float(input_lons[-1])
    
    # Convert to UV
    input_uv, _ = latlon_to_local_uv(
        torch.tensor(input_lats, dtype=torch.float32),
        torch.tensor(input_lons, dtype=torch.float32),
        lat0, lon0
    )
    
    future_uv, logJ = latlon_to_local_uv(
        torch.tensor(future_lats, dtype=torch.float32),
        torch.tensor(future_lons, dtype=torch.float32),
        lat0, lon0
    )
    
    # Model inputs
    input_uv_batch = input_uv.unsqueeze(0).to(device)
    centre_ll = torch.tensor([[lat0/90.0, lon0/180.0]], dtype=torch.float32).to(device)
    causal_position = torch.tensor([19]).to(device)
    
    # Time deltas - all 600 seconds (10 minutes) for interpolated data
    time_deltas = torch.full((1, 30), 600.0).to(device)
    
    # Calculate NLL for all horizons
    future_uv_batch = future_uv.unsqueeze(0).to(device)
    logJ_batch = logJ.unsqueeze(0).to(device)
    
    with torch.no_grad():
        log_probs = model(input_uv_batch, centre_ll, future_uv_batch, 
                         causal_position, time_deltas)
        nll_per_horizon = -(log_probs + logJ_batch).squeeze(0).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    horizons = [0, 4, 9]  # 10, 50, 100 minutes
    
    for idx, (h, ax) in enumerate(zip(horizons, axes)):
        # Calculate bounds
        target_point = future_uv[h].numpy()
        bounds = calculate_bounds(input_uv.numpy(), target_point, padding=0.3)
        
        # Evaluate PDF
        xx, yy, probs = evaluate_pdf_grid(
            model, input_uv_batch, centre_ll, causal_position, 
            time_deltas, h, bounds, grid_size=100
        )
        
        # Convert to miles
        xx_miles = uv_to_miles(xx)
        yy_miles = uv_to_miles(yy)
        input_miles = uv_to_miles(input_uv.numpy())
        future_miles = uv_to_miles(future_uv.numpy())
        
        # Plot
        im = ax.contourf(xx_miles, yy_miles, probs, levels=20, cmap='viridis')
        ax.contour(xx_miles, yy_miles, probs, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        # Plot trajectories
        ax.plot(input_miles[:, 0], input_miles[:, 1], 'k.-', markersize=4, linewidth=1, 
                alpha=0.8, label='Input trajectory')
        ax.plot(input_miles[-1, 0], input_miles[-1, 1], 'ko', markersize=8, 
                label='Current position')
        
        # Plot actual future position
        ax.plot(future_miles[h, 0], future_miles[h, 1], 'ro', markersize=8, 
                label='Actual position')
        
        # Plot future trajectory faintly
        if h > 0:
            ax.plot(future_miles[:h, 0], future_miles[:h, 1], 'r--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('East-West (miles)')
        ax.set_ylabel('North-South (miles)')
        ax.set_title(f'{(h+1)*10} min ahead\nNLL: {nll_per_horizon[h]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        if idx == 2:
            plt.colorbar(im, ax=ax, label='Probability')
    
    # Calculate average NLL
    avg_nll = np.mean(nll_per_horizon)
    
    plt.suptitle(f'Real Track ID: {track["track_id"]} ({track["length"]} points) - Avg NLL: {avg_nll:.3f}\n'
                 f'Time-Aware Model Predictions on Interpolated Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    # Configuration
    data_path = "/home/ec2-user/ais_data/interpolated_ais_2025-07-06_01-34-27.parquet"
    checkpoint_path = "checkpoints/time_aware_causal_multihorizon_interpolated_128freq_bs512/step_00009000.pt"
    output_dir = "output/time_aware_predictions/real_data"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TimeAwareNanoGPTTrajectoryForecaster(
        seq_len=20,
        d_model=128,
        nhead=8,
        num_layers=6,
        ff_hidden=512,
        fourier_m=128,
        fourier_rank=4
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['global_step']}")
    
    # Load tracks
    print("\nLoading tracks from local data...")
    tracks = load_tracks_from_local(data_path, n_tracks=6)
    
    # Visualize each track
    print("\nGenerating visualizations...")
    for i, track in enumerate(tracks):
        output_path = Path(output_dir) / f"real_track_{track['track_id']}_predictions.png"
        visualize_track_predictions(model, track, device, output_path)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()