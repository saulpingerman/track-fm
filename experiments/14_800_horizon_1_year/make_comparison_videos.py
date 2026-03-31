#!/usr/bin/env python3
"""
Generate comparison videos showing 18M vs 100M model predictions for the same track.
Each video shows 800 horizon predictions.
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
import json

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import Config, CausalAISModel, DEVICE

# Try to import contextily for map background
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Note: contextily not available, using plain background")


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


def find_best_checkpoint(exp_dir: Path):
    """Find the best checkpoint file in the experiment directory."""
    checkpoints_dir = exp_dir / 'checkpoints'
    if checkpoints_dir.exists():
        search_dir = checkpoints_dir
    else:
        search_dir = exp_dir

    best_model = search_dir / 'best_model.pt'
    if best_model.exists():
        return best_model, "best_model"

    # Fall back to latest checkpoint
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


def load_model_from_experiment(exp_dir: Path, device):
    """Load a model from an experiment directory."""
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        config = Config()
        for key, value in saved_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        raise ValueError(f"No config.json found in {exp_dir}")

    model = CausalAISModel(config).to(device)
    checkpoint_path, checkpoint_name = find_best_checkpoint(exp_dir)

    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {exp_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 'unknown')
        print(f"  Loaded: {checkpoint_name} (step {step})")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded: {checkpoint_name}")

    model.eval()
    return model, config


def compute_all_densities(model, features, config, max_horizon_video):
    """Pre-compute all density predictions for a model in ONE forward pass."""
    with torch.no_grad():
        # Single forward pass for all horizons
        horizon_batch = torch.arange(1, max_horizon_video + 1, device=DEVICE)
        with autocast('cuda', enabled=config.use_amp):
            log_densities, _, _, _ = model.forward_train(features, horizon_batch, causal=False)
        # Returns (1, num_horizons, grid, grid) -> (num_horizons, grid, grid)
        densities = torch.exp(log_densities[0]).cpu().numpy()
    return densities  # Now a single numpy array instead of list


def generate_comparison_video(model_18m, model_100m, config, val_dataset, sample_idx,
                              output_path, max_horizon_video=800):
    """Generate a side-by-side comparison video for both models."""

    seq_len = config.max_seq_len
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
    max_horizon = config.max_horizon
    future_dt = dt_seconds[seq_len:seq_len + max_horizon]
    cumulative_time = np.cumsum(future_dt)

    # Grid for expected value
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Determine plot bounds based on actual trajectory data
    # Include history and all future positions
    all_lat = np.concatenate([lat[:seq_len], lat[seq_len:seq_len + max_horizon_video]])
    all_lon = np.concatenate([lon[:seq_len], lon[seq_len:seq_len + max_horizon_video]])

    # Calculate bounds with padding (5% of range on each side)
    lat_range = all_lat.max() - all_lat.min()
    lon_range = all_lon.max() - all_lon.min()
    padding = max(lat_range, lon_range) * 0.1  # 10% padding
    padding = max(padding, 0.02)  # Minimum padding of 0.02 degrees

    lat_min = all_lat.min() - padding
    lat_max = all_lat.max() + padding
    lon_min = all_lon.min() - padding
    lon_max = all_lon.max() + padding

    # Make aspect ratio roughly square (important for map display)
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    half_range = max(lat_max - lat_min, lon_max - lon_min) / 2
    lat_min = lat_center - half_range
    lat_max = lat_center + half_range
    lon_min = lon_center - half_range
    lon_max = lon_center + half_range

    # Pre-compute densities for both models
    print(f"    Pre-computing {max_horizon_video} horizon predictions for 18M model...", flush=True)
    densities_18m = compute_all_densities(model_18m, features, config, max_horizon_video)

    print(f"    Pre-computing {max_horizon_video} horizon predictions for 100M model...", flush=True)
    densities_100m = compute_all_densities(model_100m, features, config, max_horizon_video)

    # Generate frames - create fresh plots each frame (reliable approach)
    frames = []
    density_cmap = create_density_colormap()
    extent = [src_lon - grid_range, src_lon + grid_range,
              src_lat - grid_range, src_lat + grid_range]

    # Pre-compute cumulative times
    avg_dt = cumulative_time[-1] / len(cumulative_time)
    cum_times = np.zeros(max_horizon_video)
    for h in range(max_horizon_video):
        if h < len(cumulative_time):
            cum_times[h] = cumulative_time[h]
        else:
            cum_times[h] = cumulative_time[-1] + avg_dt * (h + 1 - len(cumulative_time))

    # Pre-compute all errors
    errors_18m = np.zeros(max_horizon_video)
    errors_100m = np.zeros(max_horizon_video)
    dr_errors = np.zeros(max_horizon_video)

    for h in range(max_horizon_video):
        actual_lat = lat[seq_len + h]
        actual_lon = lon[seq_len + h]
        cum_t = cum_times[h]
        dr_lat = src_lat + velocity_lat * cum_t
        dr_lon = src_lon + velocity_lon * cum_t

        exp_dlat_18m = (densities_18m[h] * xx).sum()
        exp_dlon_18m = (densities_18m[h] * yy).sum()
        exp_dlat_100m = (densities_100m[h] * xx).sum()
        exp_dlon_100m = (densities_100m[h] * yy).sum()

        errors_18m[h] = np.sqrt((exp_dlat_18m - (actual_lat - src_lat))**2 +
                                (exp_dlon_18m - (actual_lon - src_lon))**2) * 111 * 1000
        errors_100m[h] = np.sqrt((exp_dlat_100m - (actual_lat - src_lat))**2 +
                                 (exp_dlon_100m - (actual_lon - src_lon))**2) * 111 * 1000
        dr_errors[h] = np.sqrt((dr_lat - actual_lat)**2 + (dr_lon - actual_lon)**2) * 111 * 1000

    print(f"    Generating {max_horizon_video} frames...", flush=True)

    for h in range(max_horizon_video):
        if (h + 1) % 100 == 0:
            print(f"      Frame {h+1}/{max_horizon_video}", flush=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        cum_t = cum_times[h]
        actual_lat = lat[seq_len + h]
        actual_lon = lon[seq_len + h]
        dr_lat = src_lat + velocity_lat * cum_t
        dr_lon = src_lon + velocity_lon * cum_t

        for ax, density, model_err, title in [
            (ax1, densities_18m[h], errors_18m[h], '18M Model'),
            (ax2, densities_100m[h], errors_100m[h], '100M Model')
        ]:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

            # Add map background if available
            if HAS_CONTEXTILY:
                try:
                    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, zoom='auto')
                except Exception:
                    pass

            # Plot heatmap (auto-scale vmax per frame, fixed 3 decimal format)
            im = ax.imshow(density, origin='lower', extent=extent,
                           cmap=density_cmap, aspect='auto', vmin=0)
            plt.colorbar(im, ax=ax, shrink=0.7, format='%.3f')

            # History trajectory
            ax.plot(lon[:seq_len], lat[:seq_len], 'b-', linewidth=1.5, alpha=0.7)

            # Target position
            ax.plot(actual_lon, actual_lat, 'go', markersize=5,
                    markeredgecolor='white', markeredgewidth=0.5)

            # Dead reckoning
            ax.plot(dr_lon, dr_lat, 'rx', markersize=5, markeredgewidth=1.5)

            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_title(f'{title}\nError: {model_err:.0f}m | DR: {dr_errors[h]:.0f}m',
                         fontsize=11, fontweight='bold')

        fig.suptitle(f'Horizon {h+1} | Forecast: {cum_t:.0f}s ({cum_t/60:.1f} min)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Render to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        frames.append(img)
        plt.close(fig)

    # Save as GIF
    output_gif = output_path.with_suffix('.gif')
    frames[0].save(output_gif, save_all=True, append_images=frames[1:],
                   duration=20, loop=0)
    print(f"  Saved: {output_gif}")
    return output_gif


def find_moving_track(val_dataset, config, num_candidates=100, seed=42):
    """Find a track with significant movement."""
    np.random.seed(seed)
    candidates = np.random.choice(len(val_dataset), min(num_candidates, len(val_dataset)), replace=False)

    best_idx = None
    best_distance = 0

    seq_len = config.max_seq_len

    for idx in candidates:
        feat = val_dataset[idx].numpy()
        lat = feat[:, 0] * config.lat_std + config.lat_mean
        lon = feat[:, 1] * config.lon_std + config.lon_mean

        # Calculate total distance traveled in future portion
        future_lat = lat[seq_len:]
        future_lon = lon[seq_len:]

        if len(future_lat) < 100:
            continue

        total_dist = np.sqrt((future_lat[-1] - future_lat[0])**2 +
                             (future_lon[-1] - future_lon[0])**2) * 111 * 1000  # meters

        if total_dist > best_distance:
            best_distance = total_dist
            best_idx = idx

    print(f"  Selected track {best_idx} with {best_distance/1000:.1f}km of movement")
    return best_idx


class CachedValidationDataset:
    """Simple dataset wrapper for cached validation data."""
    def __init__(self, data_path, seq_len, max_horizon):
        self.data = np.load(data_path, mmap_mode='r')
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        # Total sequence needed: seq_len (history) + max_horizon (future)
        self.total_len = seq_len + max_horizon

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return first total_len timesteps as tensor
        sample = self.data[idx, :self.total_len].copy()
        return torch.from_numpy(sample).float()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate comparison videos for 18M vs 100M models')
    parser.add_argument('--max-horizon', type=int, default=800,
                        help='Maximum horizon to visualize (default: 800)')
    parser.add_argument('--track-idx', type=int, default=None,
                        help='Specific track index to use (default: auto-select moving track)')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING MODEL COMPARISON VIDEOS")
    print("=" * 70)

    base_dir = Path(__file__).parent
    exp_18m_dir = base_dir / 'experiments' / 'run_grid1.2_18M'
    exp_100m_dir = base_dir / 'experiments' / 'run_grid1.2_100M'

    # Load 18M model
    print("\nLoading 18M model...")
    model_18m, config_18m = load_model_from_experiment(exp_18m_dir, DEVICE)
    param_count_18m = sum(p.numel() for p in model_18m.parameters()) / 1e6
    print(f"  Parameters: {param_count_18m:.1f}M")

    # Load 100M model
    print("\nLoading 100M model...")
    model_100m, config_100m = load_model_from_experiment(exp_100m_dir, DEVICE)
    param_count_100m = sum(p.numel() for p in model_100m.parameters()) / 1e6
    print(f"  Parameters: {param_count_100m:.1f}M")

    # Use 18M config for data loading (they should be the same except model params)
    config = config_18m

    # Load cached validation data directly
    print("\nLoading cached validation data...")
    cached_data_path = Path('/home/ec2-user/mds-cache/validation_normalized.npy')
    val_dataset = CachedValidationDataset(cached_data_path, config.max_seq_len, config.max_horizon)
    print(f"  Validation samples: {len(val_dataset)}")

    # Select track
    if args.track_idx is not None:
        sample_idx = args.track_idx
        print(f"\nUsing specified track index: {sample_idx}")
    else:
        print("\nFinding a track with significant movement...")
        sample_idx = find_moving_track(val_dataset, config)

    # Generate comparison video
    output_dir = base_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'comparison_18m_vs_100m_track{sample_idx}_h{args.max_horizon}'

    print(f"\nGenerating comparison video for track {sample_idx} with {args.max_horizon} horizons...")
    generate_comparison_video(model_18m, model_100m, config, val_dataset, sample_idx,
                              output_path, max_horizon_video=args.max_horizon)

    print("\n" + "=" * 70)
    print("COMPARISON VIDEO COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
