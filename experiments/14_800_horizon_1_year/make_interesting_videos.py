#!/usr/bin/env python3
"""
Generate videos/GIFs showing model predictions for INTERESTING tracks.
Focuses on tracks where vessels turn or change direction, not just straight lines.

Each frame shows the prediction for a different horizon (1, 2, ..., N).
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


def compute_track_interestingness(features: torch.Tensor, config: Config, seq_len: int, max_horizon: int) -> dict:
    """
    Compute how "interesting" a track is based on heading changes and trajectory curvature.

    IMPROVED: Favors gradual turns and course changes, penalizes ferry-like back-and-forth behavior.

    Returns a dict with:
    - total_heading_change: Total absolute heading change in degrees
    - max_heading_change: Maximum single-step heading change
    - path_curvature: Ratio of path length to straight-line distance
    - moderate_turns: Count of turns between 20-120 degrees (the interesting ones)
    - reversal_count: Count of U-turns (>150 degrees) - penalized
    - net_heading_change: Net directional change (progressive turns accumulate)
    - is_interesting: Boolean indicating if track is interesting
    """
    feat_np = features.cpu().numpy()

    # Get lat/lon for the full sequence including future
    lat = feat_np[:, 0] * config.lat_std + config.lat_mean
    lon = feat_np[:, 1] * config.lon_std + config.lon_mean

    # Get cog_sin and cog_cos to reconstruct heading
    cog_sin = feat_np[:, 3]
    cog_cos = feat_np[:, 4]
    heading = np.degrees(np.arctan2(cog_sin, cog_cos))  # Heading in degrees

    # Compute heading changes (handle wrap-around at 180/-180)
    heading_diff = np.diff(heading)
    # Normalize to [-180, 180]
    heading_diff = np.mod(heading_diff + 180, 360) - 180
    abs_heading_diff = np.abs(heading_diff)

    # Total absolute heading change
    total_heading_change = abs_heading_diff.sum()
    max_heading_change = abs_heading_diff.max() if len(heading_diff) > 0 else 0

    # Count moderate turns (20-120 degrees) - these are the interesting maneuvers
    moderate_turns = np.sum((abs_heading_diff >= 20) & (abs_heading_diff <= 120))

    # Count reversals/U-turns (>150 degrees) - ferry behavior, penalize these
    reversal_count = np.sum(abs_heading_diff > 150)

    # Net heading change: sum of signed heading changes
    # Progressive turns in same direction will accumulate, back-and-forth will cancel out
    net_heading_change = abs(heading_diff.sum())

    # Path curvature: ratio of total path length to straight-line distance
    # Use the portion of the track we'll visualize (input + prediction horizon)
    vis_len = min(seq_len + max_horizon, len(lat))
    lat_vis = lat[:vis_len]
    lon_vis = lon[:vis_len]

    # Compute path length (sum of segment lengths)
    dlat = np.diff(lat_vis)
    dlon = np.diff(lon_vis)
    segment_lengths = np.sqrt(dlat**2 + dlon**2)
    path_length = segment_lengths.sum()

    # Straight-line distance from start to end
    straight_dist = np.sqrt((lat_vis[-1] - lat_vis[0])**2 + (lon_vis[-1] - lon_vis[0])**2)

    # Curvature ratio (higher = more curved)
    path_curvature = path_length / max(straight_dist, 1e-6)

    # Compute a "smoothness" metric - standard deviation of turn angles
    # Lower values mean more consistent turning (like a gradual arc)
    turn_consistency = np.std(heading_diff) if len(heading_diff) > 1 else 0

    # A track is interesting if it has gradual turns, not just reversals
    # Favor: moderate turns, net heading change, reasonable curvature
    # Penalize: too many reversals, very high curvature (ferry behavior)
    is_interesting = (
        (moderate_turns >= 2) or  # Has at least 2 moderate turns
        (net_heading_change > 60 and reversal_count < 2) or  # Progressive turns
        (path_curvature > 1.3 and path_curvature < 10 and reversal_count < 3)  # Curved but not ferry
    )

    return {
        'total_heading_change': total_heading_change,
        'max_heading_change': max_heading_change,
        'path_curvature': path_curvature,
        'moderate_turns': moderate_turns,
        'reversal_count': reversal_count,
        'net_heading_change': net_heading_change,
        'turn_consistency': turn_consistency,
        'is_interesting': is_interesting
    }


def find_interesting_tracks(val_dataset, config, num_tracks=30, min_interesting=True):
    """
    Scan the validation dataset and find the most interesting tracks.

    Args:
        val_dataset: The validation dataset
        config: Config object
        num_tracks: Number of interesting tracks to return
        min_interesting: If True, only return tracks that meet interestingness criteria

    Returns:
        List of (sample_idx, interestingness_score, metrics_dict) sorted by interestingness
    """
    print(f"\nScanning {len(val_dataset)} samples for interesting tracks...")

    interesting_tracks = []

    # Sample more tracks to find enough interesting ones
    # Use deterministic sampling for reproducibility
    np.random.seed(123)

    # Check all samples (or a large subset if dataset is huge)
    max_to_check = min(len(val_dataset), 10000)
    indices_to_check = np.random.choice(len(val_dataset), max_to_check, replace=False)

    for i, sample_idx in enumerate(indices_to_check):
        if i % 1000 == 0:
            print(f"  Checked {i}/{max_to_check} samples, found {len(interesting_tracks)} interesting so far...")

        features = val_dataset[sample_idx]
        metrics = compute_track_interestingness(features, config, config.max_seq_len, config.max_horizon)

        # IMPROVED scoring: favor gradual turns, penalize ferry-like behavior
        # Reward: moderate turns (20-120°), net heading change (progressive turns)
        # Penalize: reversals (U-turns), extreme curvature (ferry behavior)
        score = (
            metrics['moderate_turns'] * 100.0 +           # Strongly reward moderate turns
            metrics['net_heading_change'] * 2.0 +         # Reward progressive directional change
            min(metrics['path_curvature'], 5.0) * 10.0 -  # Reward curvature, cap at 5x
            metrics['reversal_count'] * 200.0 -           # Strongly penalize U-turns/reversals
            max(0, metrics['path_curvature'] - 10) * 50   # Penalize extreme curvature (ferry)
        )

        if metrics['is_interesting'] or not min_interesting:
            interesting_tracks.append((sample_idx, score, metrics))

    # Sort by interestingness score (descending)
    interesting_tracks.sort(key=lambda x: x[1], reverse=True)

    print(f"  Found {len(interesting_tracks)} interesting tracks total")

    # Return top N
    return interesting_tracks[:num_tracks]


def generate_horizon_video(model, val_dataset, config, sample_idx, output_path, max_horizon_video=40, track_metrics=None):
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

        # Add interestingness info to title if available
        title = f'Horizon {h} | Forecast Time: {cum_t:.0f}s ({cum_t/60:.1f} min)\n{error_text}'
        if track_metrics:
            title += f'\n[Turn: {track_metrics["total_heading_change"]:.0f}° | Curvature: {track_metrics["path_curvature"]:.2f}x]'

        ax.set_title(title, fontsize=12, fontweight='bold')
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

    parser = argparse.ArgumentParser(description='Generate horizon prediction videos/GIFs for INTERESTING tracks')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name (loads from experiments/<exp-name>/)')
    parser.add_argument('--max-horizon', type=int, default=400,
                        help='Maximum horizon to visualize (default: 400)')
    parser.add_argument('--num-tracks', type=int, default=30,
                        help='Number of interesting tracks to generate videos for (default: 30)')
    parser.add_argument('--seed', type=int, default=789,
                        help='Random seed for track selection (default: 789)')
    parser.add_argument('--start-num', type=int, default=1,
                        help='Starting track number for filenames (default: 1)')
    parser.add_argument('--max-val-samples', type=int, default=10000,
                        help='Max validation samples to scan for interesting tracks (default: 10000)')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING HORIZON PREDICTION VIDEOS (INTERESTING TRACKS)")
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

    # Load data - use a reasonable sample size for the larger 3.5B position dataset
    print("\nLoading validation data...")
    train_data, val_data = load_ais_data(config, max_tracks=5000)  # Limit tracks for memory efficiency
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

    # Find interesting tracks
    interesting_tracks = find_interesting_tracks(val_dataset, config, num_tracks=args.num_tracks)

    if len(interesting_tracks) == 0:
        print("ERROR: No interesting tracks found!")
        return

    print(f"\nTop {len(interesting_tracks)} most interesting tracks:")
    for i, (sample_idx, score, metrics) in enumerate(interesting_tracks[:10]):
        print(f"  {i+1}. Sample {sample_idx}: score={score:.1f}, moderate_turns={metrics['moderate_turns']}, "
              f"net_change={metrics['net_heading_change']:.1f}°, reversals={metrics['reversal_count']}, "
              f"curvature={metrics['path_curvature']:.2f}x")

    # Use results subfolder
    results_dir = exp_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = results_dir

    # Generate videos for interesting tracks
    print(f"\nGenerating horizon videos for {len(interesting_tracks)} interesting tracks (max_horizon={args.max_horizon})...")

    for i, (sample_idx, score, metrics) in enumerate(interesting_tracks):
        track_num = args.start_num + i
        print(f"\n  Track {track_num} (sample {sample_idx}, score={score:.1f})...")
        output_path = output_dir / f'interesting_track{track_num}.gif'
        generate_horizon_video(model, val_dataset, config, sample_idx, output_path,
                               max_horizon_video=args.max_horizon, track_metrics=metrics)

    print("\n" + "=" * 70)
    print("GIF GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir}")
    for i in range(len(interesting_tracks)):
        print(f"  - interesting_track{args.start_num + i}.gif")


if __name__ == '__main__':
    main()
