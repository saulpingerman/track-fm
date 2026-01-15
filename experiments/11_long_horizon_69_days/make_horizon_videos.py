#!/usr/bin/env python3
"""
Generate videos/GIFs showing model predictions across horizons.
Each frame shows the prediction for a different horizon (1, 2, ..., 40).
"""

import numpy as np
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
from pathlib import Path
import sys
import contextily as ctx


def create_density_colormap():
    """Create colormap: transparent (low) -> blue (medium) -> yellow (high)"""
    colors = [
        (0.0, 0.0, 0.0, 0.0),    # Transparent at 0
        (0.0, 0.0, 0.5, 0.3),    # Dark blue, slightly visible
        (0.0, 0.3, 0.8, 0.5),    # Blue
        (0.0, 0.6, 1.0, 0.7),    # Light blue
        (0.5, 0.8, 0.5, 0.8),    # Cyan/green transition
        (1.0, 1.0, 0.0, 0.9),    # Yellow (high density)
        (1.0, 0.8, 0.0, 1.0),    # Orange-yellow (very high)
    ]
    return LinearSegmentedColormap.from_list('density', colors, N=256)

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    Config, AISTrackDataset, CausalAISModel, load_ais_data, DEVICE
)


def find_latest_checkpoint(exp_dir: Path):
    """Find the latest checkpoint file in the experiment directory."""
    # Check in checkpoints subfolder first
    checkpoints_dir = exp_dir / 'checkpoints'
    if checkpoints_dir.exists():
        search_dir = checkpoints_dir
    else:
        search_dir = exp_dir  # Fall back to root for old experiments

    best_model = search_dir / 'best_model.pt'
    if best_model.exists():
        return best_model, "best_model"

    checkpoints = list(search_dir.glob('checkpoint_step_*.pt'))
    if not checkpoints:
        return None, None

    def get_step(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return 0

    latest = max(checkpoints, key=get_step)
    step = get_step(latest)
    return latest, f"checkpoint_step_{step}"


def generate_horizon_video(model, val_dataset, config, sample_idx, output_path, max_horizon_video=40):
    """
    Generate a video for a single track showing predictions across horizons.
    """
    model.eval()

    seq_len = config.max_seq_len
    max_horizon = config.max_horizon  # Model's max horizon (20)
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Get sample
    features = val_dataset[sample_idx].unsqueeze(0).to(DEVICE)

    # Denormalize positions
    feat_np = features[0].cpu().numpy()
    lat = feat_np[:, 0] * config.lat_std + config.lat_mean
    lon = feat_np[:, 1] * config.lon_std + config.lon_mean

    # Get time deltas
    dt_norm = feat_np[:, 5]
    dt_seconds = dt_norm * config.dt_max

    src_lat = lat[seq_len - 1]
    src_lon = lon[seq_len - 1]

    # Velocity for dead reckoning
    vlat = lat[seq_len - 1] - lat[seq_len - 2]
    vlon = lon[seq_len - 1] - lon[seq_len - 2]
    last_dt = max(dt_seconds[seq_len - 1], 1.0)
    velocity_lat = vlat / last_dt
    velocity_lon = vlon / last_dt

    # Cumulative time for future positions
    future_dt = dt_seconds[seq_len:seq_len + max_horizon]
    cumulative_time = np.cumsum(future_dt)

    # Grid for expected value
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Determine plot bounds (include all future positions)
    all_future_lat = lat[seq_len:seq_len + max_horizon_video]
    all_future_lon = lon[seq_len:seq_len + max_horizon_video]

    # Extend bounds a bit
    lat_min = min(src_lat - grid_range * 1.5, all_future_lat.min() - 0.02)
    lat_max = max(src_lat + grid_range * 1.5, all_future_lat.max() + 0.02)
    lon_min = min(src_lon - grid_range * 1.5, all_future_lon.min() - 0.02)
    lon_max = max(src_lon + grid_range * 1.5, all_future_lon.max() + 0.02)

    # Pre-compute all densities in batches for speed (GPU batching)
    print(f"    Pre-computing {max_horizon_video} horizon predictions (batched)...", flush=True)
    all_densities = []
    batch_size = 50  # Process 50 horizons at a time
    with torch.no_grad():
        for start_h in range(1, max_horizon_video + 1, batch_size):
            end_h = min(start_h + batch_size, max_horizon_video + 1)
            horizon_batch = torch.arange(start_h, end_h, device=DEVICE)
            with autocast('cuda', enabled=config.use_amp):
                log_densities, _, _, _ = model.forward_train(features, horizon_batch, causal=False)
            densities = torch.exp(log_densities[0]).cpu().numpy()  # (num_horizons, grid, grid)
            for i in range(densities.shape[0]):
                all_densities.append(densities[i])

    # Generate frames
    frames = []
    num_frames = max_horizon_video
    density_cmap = create_density_colormap()

    for h in range(1, num_frames + 1):
        fig, ax = plt.subplots(figsize=(8, 8))  # Smaller for faster rendering

        # Get density for this horizon
        density = all_densities[h - 1]

        # Compute cumulative time
        if h <= len(cumulative_time):
            cum_t = cumulative_time[h - 1]
        else:
            avg_dt = cumulative_time[-1] / len(cumulative_time)
            cum_t = cumulative_time[-1] + avg_dt * (h - len(cumulative_time))

        # Set axis limits first (needed for basemap)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add map background
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, zoom='auto')
        except Exception:
            pass  # Fall back to white background if map fails

        # Plot heatmap with custom yellow->blue->transparent colormap
        extent = [src_lon - grid_range, src_lon + grid_range,
                  src_lat - grid_range, src_lat + grid_range]
        im = ax.imshow(density, origin='lower', extent=extent,
                       cmap=density_cmap, aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Probability', shrink=0.8)

        # Plot full input trajectory (builds up and stays)
        ax.plot(lon[:seq_len], lat[:seq_len], 'b-', linewidth=2, alpha=0.8, label='History')

        # Plot only current target position (small visible point)
        actual_lat = lat[seq_len + h - 1]
        actual_lon = lon[seq_len + h - 1]
        ax.plot(actual_lon, actual_lat, 'go', markersize=6,
                markeredgecolor='white', markeredgewidth=1, label='Target')

        # Dead reckoning prediction (smaller X)
        dr_lat = src_lat + velocity_lat * cum_t
        dr_lon = src_lon + velocity_lon * cum_t
        ax.plot(dr_lon, dr_lat, 'rx', markersize=6, markeredgewidth=2, label='Dead Reckon')

        # Compute model expected value for error calculation (no marker plotted)
        exp_dlat = (density * xx).sum()
        exp_dlon = (density * yy).sum()

        # Compute errors
        model_error = np.sqrt((exp_dlat - (actual_lat - src_lat))**2 +
                              (exp_dlon - (actual_lon - src_lon))**2) * 111 * 1000
        dr_error = np.sqrt((dr_lat - actual_lat)**2 + (dr_lon - actual_lon)**2) * 111 * 1000
        error_text = f'Model: {model_error:.0f}m | DR: {dr_error:.0f}m'

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Horizon {h} | Forecast Time: {cum_t:.0f}s ({cum_t/60:.1f} min)\n{error_text}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)

        # Convert figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frames.append(img)
        plt.close(fig)

    # Save as GIF (50 fps = 20ms per frame for 400 horizons)
    output_gif = output_path.with_suffix('.gif')
    frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                   duration=20, loop=0)

    print(f"  Saved: {output_gif.name}")


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Generate horizon prediction videos/GIFs')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name (loads from experiments/<exp-name>/)')
    parser.add_argument('--max-horizon', type=int, default=400,
                        help='Maximum horizon to visualize (default: 400)')
    parser.add_argument('--num-tracks', type=int, default=6,
                        help='Number of tracks to generate videos for (default: 6)')
    parser.add_argument('--seed', type=int, default=789,
                        help='Random seed for track selection (default: 789)')
    parser.add_argument('--start-num', type=int, default=1,
                        help='Starting track number for filenames (default: 1)')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING HORIZON PREDICTION VIDEOS")
    print("=" * 70)

    # Set up paths
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    if not exp_dir.exists():
        print(f"  ERROR: Experiment directory not found: {exp_dir}")
        print(f"  Available experiments:")
        experiments_dir = Path(__file__).parent / 'experiments'
        if experiments_dir.exists():
            for d in sorted(experiments_dir.iterdir()):
                if d.is_dir():
                    print(f"    - {d.name}")
        return

    print(f"\nExperiment: {args.exp_name}")
    print(f"Directory: {exp_dir}")

    # Load config from experiment if available
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        print(f"  Loading config from {config_path.name}")
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        config = Config()
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        print("  Using default config")
        config = Config()

    # Load data
    print("\nLoading validation data...")
    train_data, val_data = load_ais_data(config, max_tracks=2000)
    val_dataset = AISTrackDataset(val_data, config.max_seq_len, config.max_horizon, config)
    print(f"  Validation samples: {len(val_dataset)}")

    # Load model
    print("\nLoading trained model...")
    model = CausalAISModel(config).to(DEVICE)

    checkpoint_path, checkpoint_name = find_latest_checkpoint(exp_dir)

    if checkpoint_path is None:
        print(f"  ERROR: No checkpoint found in {exp_dir}!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded: {checkpoint_name} (step {checkpoint.get('step', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded: {checkpoint_name}")

    model.eval()

    # Use results subfolder
    results_dir = exp_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = results_dir

    # Generate videos for several tracks
    print(f"\nGenerating horizon videos (max_horizon={args.max_horizon}, num_tracks={args.num_tracks})...")
    np.random.seed(args.seed)
    track_indices = np.random.choice(len(val_dataset), args.num_tracks, replace=False)

    for i, sample_idx in enumerate(track_indices):
        track_num = args.start_num + i
        print(f"\n  Track {track_num} (sample {sample_idx})...")
        output_path = output_dir / f'horizon_video_track{track_num}.gif'
        generate_horizon_video(model, val_dataset, config, sample_idx, output_path, max_horizon_video=args.max_horizon)

    print("\n" + "=" * 70)
    print("GIF GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir}")
    for i in range(args.num_tracks):
        print(f"  - horizon_video_track{args.start_num + i}.gif")


if __name__ == '__main__':
    main()
