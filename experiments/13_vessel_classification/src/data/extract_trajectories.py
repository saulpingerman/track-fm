"""
Extract labeled trajectories from DMA parquet files for vessel classification.
"""

import polars as pl
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple
import pickle

from .preprocessing import normalize_trajectory


def extract_trajectories_with_labels(
    data_path: str,
    year: int,
    month: int,
    classes: List[str],
    min_length: int = 50,
    max_length: int = 512,
    max_trajectories_per_class: int = 2000,
) -> Tuple[List[np.ndarray], List[int], List[str], Dict]:
    """
    Extract trajectories from DMA parquet files with vessel type labels.

    Args:
        data_path: Path to FSx data directory
        year: Year to extract
        month: Month to extract
        classes: List of class names to include
        min_length: Minimum trajectory length
        max_length: Maximum trajectory length
        max_trajectories_per_class: Max trajectories per class (for balance)

    Returns:
        trajectories: List of trajectory arrays (N, 6)
        labels: List of class indices
        track_ids: List of track IDs
        stats: Dictionary with extraction statistics
    """
    data_path = Path(data_path)
    parquet_pattern = f"{data_path}/year={year}/month={month:02d}/**/*.parquet"

    print(f"Loading data from {parquet_pattern}...")

    # Create class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Load data with lazy evaluation
    df = pl.scan_parquet(parquet_pattern)

    # Filter to target classes
    df = df.filter(pl.col('ship_type').is_in(classes))

    # Select needed columns
    df = df.select([
        'track_id', 'mmsi', 'ship_type', 'timestamp',
        'lat', 'lon', 'sog', 'cog', 'dt_seconds'
    ])

    # Collect (this loads data into memory)
    print("Collecting data...")
    df = df.collect()

    print(f"Total records: {len(df):,}")

    # Get unique track IDs per class
    track_stats = df.group_by(['ship_type', 'track_id']).agg(
        pl.len().alias('length'),
        pl.col('mmsi').first().alias('mmsi')
    )

    # Filter tracks by length
    valid_tracks = track_stats.filter(
        (pl.col('length') >= min_length) &
        (pl.col('length') <= max_length * 2)  # Allow some room for splitting
    )

    print(f"\nValid tracks (length {min_length}-{max_length * 2}):")
    for cls in classes:
        count = valid_tracks.filter(pl.col('ship_type') == cls).height
        print(f"  {cls}: {count:,} tracks")

    # Sample tracks per class
    sampled_track_ids = []
    for cls in classes:
        cls_tracks = valid_tracks.filter(pl.col('ship_type') == cls)['track_id'].to_list()
        np.random.shuffle(cls_tracks)
        n_sample = min(len(cls_tracks), max_trajectories_per_class)
        sampled_track_ids.extend(cls_tracks[:n_sample])
        print(f"  Sampled {n_sample} {cls} tracks")

    # Filter to sampled tracks
    df = df.filter(pl.col('track_id').is_in(sampled_track_ids))

    # Sort by track_id and timestamp
    df = df.sort(['track_id', 'timestamp'])

    # Extract trajectories
    print("\nExtracting trajectories...")
    trajectories = []
    labels = []
    track_ids = []

    for track_id in sampled_track_ids:
        track_data = df.filter(pl.col('track_id') == track_id)

        if track_data.height < min_length:
            continue

        ship_type = track_data['ship_type'][0]
        if ship_type not in class_to_idx:
            continue

        # Extract arrays
        lat = track_data['lat'].to_numpy()
        lon = track_data['lon'].to_numpy()
        sog = track_data['sog'].to_numpy()
        cog = track_data['cog'].to_numpy()
        dt = track_data['dt_seconds'].to_numpy()

        # Handle NaN/None values
        sog = np.nan_to_num(sog, nan=0.0)
        cog = np.nan_to_num(cog, nan=0.0)
        dt = np.nan_to_num(dt, nan=0.0)

        # Normalize features
        features = normalize_trajectory(lat, lon, sog, cog, dt)

        # Split if too long
        if len(features) > max_length:
            n_windows = len(features) // max_length
            for i in range(n_windows):
                start = i * max_length
                end = start + max_length
                trajectories.append(features[start:end])
                labels.append(class_to_idx[ship_type])
                track_ids.append(f"{track_id}_{i}")
        else:
            trajectories.append(features)
            labels.append(class_to_idx[ship_type])
            track_ids.append(track_id)

    # Compute stats
    stats = {
        'total_trajectories': len(trajectories),
        'class_distribution': {},
        'classes': classes,
    }
    for cls_idx, cls in enumerate(classes):
        count = sum(1 for l in labels if l == cls_idx)
        stats['class_distribution'][cls] = count
        print(f"  {cls}: {count} trajectories")

    return trajectories, labels, track_ids, stats


def create_splits(
    trajectories: List[np.ndarray],
    labels: List[int],
    track_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits, stratified by class.
    Split by base track_id (MMSI) to prevent data leakage.
    """
    np.random.seed(seed)

    # Get base track IDs (remove window suffix)
    base_track_ids = [tid.split('_')[0] + '_' + tid.split('_')[1] if '_' in tid else tid
                      for tid in track_ids]

    # Group indices by class and base track
    class_tracks = defaultdict(lambda: defaultdict(list))
    for idx, (label, base_tid) in enumerate(zip(labels, base_track_ids)):
        class_tracks[label][base_tid].append(idx)

    splits = {'train': [], 'val': [], 'test': []}

    for class_idx, tracks in class_tracks.items():
        track_list = list(tracks.keys())
        np.random.shuffle(track_list)

        n = len(track_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_tracks = set(track_list[:n_train])
        val_tracks = set(track_list[n_train:n_train + n_val])
        test_tracks = set(track_list[n_train + n_val:])

        for track_id, indices in tracks.items():
            if track_id in train_tracks:
                splits['train'].extend(indices)
            elif track_id in val_tracks:
                splits['val'].extend(indices)
            else:
                splits['test'].extend(indices)

    # Shuffle within splits
    for split in splits.values():
        np.random.shuffle(split)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")

    return splits


def save_processed_data(
    trajectories: List[np.ndarray],
    labels: List[int],
    track_ids: List[str],
    splits: Dict[str, List[int]],
    stats: Dict,
    output_dir: str,
):
    """Save processed data to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save trajectories and labels
    np.save(output_path / 'trajectories.npy', np.array(trajectories, dtype=object), allow_pickle=True)
    np.save(output_path / 'labels.npy', np.array(labels))

    # Save track IDs
    with open(output_path / 'track_ids.json', 'w') as f:
        json.dump(track_ids, f)

    # Save splits
    with open(output_path / 'splits.json', 'w') as f:
        json.dump(splits, f)

    # Save stats
    with open(output_path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved processed data to {output_path}")


def load_processed_data(data_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[str, List[int]], Dict]:
    """Load processed data from disk."""
    data_path = Path(data_dir)

    trajectories = np.load(data_path / 'trajectories.npy', allow_pickle=True).tolist()
    labels = np.load(data_path / 'labels.npy').tolist()

    with open(data_path / 'splits.json', 'r') as f:
        splits = json.load(f)

    with open(data_path / 'stats.json', 'r') as f:
        stats = json.load(f)

    return trajectories, labels, splits, stats
