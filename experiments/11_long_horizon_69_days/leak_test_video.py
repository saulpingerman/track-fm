#!/usr/bin/env python3
"""
LEAK TEST: Generate video using ONLY the 128 input positions.
No future/target positions are available to the model or visualization.
This verifies the model isn't cheating by seeing future data.
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

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    Config, AISTrackDataset, CausalAISModel, load_ais_data, DEVICE
)


def create_density_colormap():
    """Create colormap: transparent (low) -> blue (medium) -> yellow (high)"""
    colors = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.3),
        (0.0, 0.3, 0.8, 0.5),
        (0.0, 0.6, 1.0, 0.7),
        (0.5, 0.8, 0.5, 0.8),
        (1.0, 1.0, 0.0, 0.9),
        (1.0, 0.8, 0.0, 1.0),
    ]
    return LinearSegmentedColormap.from_list('density', colors, N=256)


def generate_leak_test_video(model, val_dataset, config, sample_idx, output_path, max_horizon_video=400):
    """
    LEAK TEST VERSION: Only uses input positions, no future data.
    """
    model.eval()

    seq_len = config.max_seq_len  # 128
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Get full sample (but we'll only use the first seq_len positions for actual data)
    full_features = val_dataset[sample_idx].clone()

    # CRITICAL: Zero out POSITION/VELOCITY data in future, but KEEP TIME (dt)
    # This simulates real inference: we know WHEN we want to predict, but not WHERE
    # Features: [lat, lon, sog_x, sog_y, cog, dt]
    #            0    1     2      3      4    5
    features_input_only = full_features.unsqueeze(0).to(DEVICE)

    # Zero out lat, lon, sog_x, sog_y, cog for future positions (keep dt for time encoding)
    features_input_only[0, seq_len:, 0] = 0  # lat
    features_input_only[0, seq_len:, 1] = 0  # lon
    features_input_only[0, seq_len:, 2] = 0  # sog_x
    features_input_only[0, seq_len:, 3] = 0  # sog_y
    features_input_only[0, seq_len:, 4] = 0  # cog
    # Keep column 5 (dt) - model needs this to know time horizons

    # Denormalize positions from INPUT ONLY (first seq_len positions)
    input_np = full_features[:seq_len].cpu().numpy()
    lat = input_np[:, 0] * config.lat_std + config.lat_mean
    lon = input_np[:, 1] * config.lon_std + config.lon_mean

    # Get time deltas from input
    dt_norm = input_np[:, 5]
    dt_seconds = dt_norm * config.dt_max

    src_lat = lat[seq_len - 1]
    src_lon = lon[seq_len - 1]

    # Velocity for dead reckoning (from last two input positions)
    vlat = lat[seq_len - 1] - lat[seq_len - 2]
    vlon = lon[seq_len - 1] - lon[seq_len - 2]
    last_dt = max(dt_seconds[seq_len - 1], 1.0)
    velocity_lat = vlat / last_dt
    velocity_lon = vlon / last_dt

    # Use actual dt values from the data for cumulative time (this is timing info, not position leakage)
    all_dt = full_features[:, 5].cpu().numpy() * config.dt_max
    future_dt = all_dt[seq_len:seq_len + max_horizon_video]
    cumulative_time = np.cumsum(future_dt)

    # Grid for expected value
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Plot bounds based ONLY on input track + grid range (no future knowledge)
    lat_min = src_lat - grid_range * 2.0
    lat_max = src_lat + grid_range * 2.0
    lon_min = src_lon - grid_range * 2.0
    lon_max = src_lon + grid_range * 2.0

    # Pre-compute all densities
    print(f"    Pre-computing {max_horizon_video} horizon predictions (INPUT ONLY - NO FUTURE DATA)...", flush=True)
    all_densities = []
    batch_size = 50
    with torch.no_grad():
        for start_h in range(1, max_horizon_video + 1, batch_size):
            end_h = min(start_h + batch_size, max_horizon_video + 1)
            horizon_batch = torch.arange(start_h, end_h, device=DEVICE)
            with autocast('cuda', enabled=config.use_amp):
                # Pass the input-only features to the model
                log_densities, _, _, _ = model.forward_train(features_input_only, horizon_batch, causal=False)
            densities = torch.exp(log_densities[0]).cpu().numpy()
            for i in range(densities.shape[0]):
                all_densities.append(densities[i])

    # Generate frames
    frames = []
    density_cmap = create_density_colormap()

    for h in range(1, max_horizon_video + 1):
        fig, ax = plt.subplots(figsize=(8, 8))

        density = all_densities[h - 1]
        cum_t = cumulative_time[h - 1]

        # Set axis limits
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Add map background
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, zoom='auto')
        except Exception:
            pass

        # Plot heatmap
        extent = [src_lon - grid_range, src_lon + grid_range,
                  src_lat - grid_range, src_lat + grid_range]
        im = ax.imshow(density, origin='lower', extent=extent,
                       cmap=density_cmap, aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, label='Probability', shrink=0.8)

        # Plot input trajectory ONLY
        ax.plot(lon[:seq_len], lat[:seq_len], 'b-', linewidth=2, alpha=0.8, label='Input History')

        # NO TARGET POINT - we don't have future data!

        # Dead reckoning prediction
        dr_lat = src_lat + velocity_lat * cum_t
        dr_lon = src_lon + velocity_lon * cum_t
        ax.plot(dr_lon, dr_lat, 'rx', markersize=6, markeredgewidth=2, label='Dead Reckon')

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'LEAK TEST - Horizon {h} | Est. Time: {cum_t:.0f}s ({cum_t/60:.1f} min)\n(NO FUTURE DATA AVAILABLE)',
                     fontsize=12, fontweight='bold', color='red')
        ax.legend(loc='upper left', fontsize=10)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frames.append(img)
        plt.close(fig)

    # Save as GIF
    output_gif = output_path.with_suffix('.gif')
    frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                   duration=20, loop=0)

    print(f"  Saved: {output_gif.name}")


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='LEAK TEST: Generate video with INPUT ONLY (no future data)')
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--max-horizon', type=int, default=400)
    parser.add_argument('--sample-idx', type=int, default=5820, help='Sample index to use')
    args = parser.parse_args()

    print("=" * 70)
    print("LEAK TEST: GENERATING VIDEO WITH INPUT ONLY (NO FUTURE DATA)")
    print("=" * 70)

    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    if not exp_dir.exists():
        print(f"  ERROR: Experiment directory not found: {exp_dir}")
        return

    print(f"\nExperiment: {args.exp_name}")

    # Load config
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        config = Config()
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = Config()

    # Load data
    print("\nLoading validation data...")
    train_data, val_data = load_ais_data(config, max_tracks=2000)
    val_dataset = AISTrackDataset(val_data, config.max_seq_len, config.max_horizon, config)
    print(f"  Validation samples: {len(val_dataset)}")

    # Load model
    print("\nLoading trained model...")
    model = CausalAISModel(config).to(DEVICE)

    checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"  ERROR: No checkpoint found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Generate leak test video
    results_dir = exp_dir / 'results'
    output_path = results_dir / 'leak_test_input_only.gif'

    print(f"\nGenerating LEAK TEST video (sample {args.sample_idx})...")
    generate_leak_test_video(model, val_dataset, config, args.sample_idx, output_path,
                             max_horizon_video=args.max_horizon)

    print("\n" + "=" * 70)
    print("LEAK TEST COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print("\nIf predictions look reasonable, the model is NOT leaking future data.")


if __name__ == '__main__':
    main()
