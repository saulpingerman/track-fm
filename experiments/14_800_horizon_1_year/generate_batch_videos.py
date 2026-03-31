#!/usr/bin/env python3
"""Generate batch of comparison videos with varying vessel speeds."""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from make_comparison_videos import (
    load_model_from_experiment, generate_comparison_video,
    CachedValidationDataset
)
from run_experiment import Config, DEVICE


def find_tracks_by_speed(val_dataset, config, num_candidates=5000, seed=42):
    """Find tracks categorized by speed/movement."""
    np.random.seed(seed)
    candidates = np.random.choice(len(val_dataset), min(num_candidates, len(val_dataset)), replace=False)

    seq_len = config.max_seq_len
    sog_max = config.sog_max  # 30.0

    track_info = []

    for idx in candidates:
        feat = val_dataset[idx].numpy()

        # SOG is feature index 2, normalized by sog_max
        sog_norm = feat[:, 2]
        mean_sog = sog_norm[:seq_len].mean() * sog_max  # Denormalize

        # Also compute total distance traveled
        lat = feat[:, 0] * config.lat_std + config.lat_mean
        lon = feat[:, 1] * config.lon_std + config.lon_mean

        future_lat = lat[seq_len:seq_len + 800]
        future_lon = lon[seq_len:seq_len + 800]

        if len(future_lat) < 100:
            continue

        total_dist = np.sqrt((future_lat[-1] - future_lat[0])**2 +
                             (future_lon[-1] - future_lon[0])**2) * 111  # km

        track_info.append({
            'idx': int(idx),
            'mean_sog': float(mean_sog),
            'total_dist_km': float(total_dist)
        })

    return track_info


def select_diverse_tracks(track_info, num_slow=1, num_moving=15):
    """Select diverse tracks: 1 slow + 15 with varying speeds."""

    # Sort by mean SOG
    sorted_by_sog = sorted(track_info, key=lambda x: x['mean_sog'])

    # Find slow movers (SOG < 3)
    slow_movers = [t for t in sorted_by_sog if t['mean_sog'] < 3]

    # Find moving vessels (SOG >= 3) with decent distance
    movers = [t for t in sorted_by_sog if t['mean_sog'] >= 3 and t['total_dist_km'] > 5]

    selected = []

    # Pick 1 slow mover (pick one with some movement to be interesting)
    if slow_movers:
        slow_sorted = sorted(slow_movers, key=lambda x: x['total_dist_km'], reverse=True)
        selected.append(('slow', slow_sorted[0]))
        print(f"  Slow mover: idx={slow_sorted[0]['idx']}, SOG={slow_sorted[0]['mean_sog']:.1f}, dist={slow_sorted[0]['total_dist_km']:.1f}km")

    # Pick 15 movers with varying speeds
    if movers:
        # Sort by SOG and pick from different speed ranges
        movers_sorted = sorted(movers, key=lambda x: x['mean_sog'])
        n = len(movers_sorted)

        # Pick evenly spaced indices to get range of speeds
        if n >= num_moving:
            indices = np.linspace(0, n-1, num_moving, dtype=int)
            for i, idx in enumerate(indices):
                track = movers_sorted[idx]
                speed_category = 'slow-mid' if track['mean_sog'] < 8 else ('mid' if track['mean_sog'] < 15 else 'fast')
                selected.append((speed_category, track))
                print(f"  Mover {i+1}: idx={track['idx']}, SOG={track['mean_sog']:.1f}, dist={track['total_dist_km']:.1f}km")
        else:
            for i, track in enumerate(movers_sorted[:num_moving]):
                selected.append(('mover', track))
                print(f"  Mover {i+1}: idx={track['idx']}, SOG={track['mean_sog']:.1f}, dist={track['total_dist_km']:.1f}km")

    return selected


def main():
    print("=" * 70)
    print("GENERATING BATCH COMPARISON VIDEOS")
    print("=" * 70)

    base_dir = Path(__file__).parent
    exp_18m_dir = base_dir / 'experiments' / 'run_grid1.2_18M'
    exp_100m_dir = base_dir / 'experiments' / 'run_grid1.2_100M'

    # Load models
    print("\nLoading 18M model...")
    model_18m, config_18m = load_model_from_experiment(exp_18m_dir, DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model_18m.parameters()) / 1e6:.1f}M")

    print("\nLoading 100M model...")
    model_100m, config_100m = load_model_from_experiment(exp_100m_dir, DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model_100m.parameters()) / 1e6:.1f}M")

    config = config_18m

    # Load validation data
    print("\nLoading cached validation data...")
    cached_data_path = Path('/home/ec2-user/mds-cache/validation_normalized.npy')
    val_dataset = CachedValidationDataset(cached_data_path, config.max_seq_len, config.max_horizon)
    print(f"  Validation samples: {len(val_dataset)}")

    # Find tracks by speed
    print("\nAnalyzing track speeds...")
    track_info = find_tracks_by_speed(val_dataset, config)
    print(f"  Analyzed {len(track_info)} tracks")

    # Select diverse tracks
    print("\nSelecting diverse tracks (1 slow + 15 movers):")
    selected_tracks = select_diverse_tracks(track_info, num_slow=1, num_moving=15)

    # Create output directory
    output_dir = base_dir / 'results' / 'batch_videos'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate videos
    print(f"\nGenerating {len(selected_tracks)} videos...")
    for i, (category, track) in enumerate(selected_tracks):
        idx = track['idx']
        sog = track['mean_sog']
        dist = track['total_dist_km']

        print(f"\n[{i+1}/{len(selected_tracks)}] Track {idx} ({category}, SOG={sog:.1f}, dist={dist:.1f}km)")

        output_path = output_dir / f'video_{i+1:02d}_{category}_track{idx}_sog{sog:.0f}'
        generate_comparison_video(model_18m, model_100m, config, val_dataset, idx,
                                  output_path, max_horizon_video=800)

    print("\n" + "=" * 70)
    print("BATCH VIDEO GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Generated {len(selected_tracks)} videos")


if __name__ == '__main__':
    main()
