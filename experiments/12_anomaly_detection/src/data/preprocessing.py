"""
Preprocessing utilities for trajectory data.

Converts raw trajectories to normalized format matching TrackFM experiment 11.
"""

import numpy as np
from typing import List
from pathlib import Path
import pickle


def preprocess_trajectories(
    trajectories: List[np.ndarray],
    config: dict
) -> List[np.ndarray]:
    """
    Preprocess trajectories to match TrackFM format.

    Applies normalization consistent with experiment 11:
    - lat: (lat - lat_mean) / lat_std
    - lon: (lon - lon_mean) / lon_std
    - sog: sog / sog_max (clipped to [0, sog_max])
    - cog_sin: already in [-1, 1]
    - cog_cos: already in [-1, 1]
    - dt: dt / dt_max (clipped to [0, dt_max])

    Args:
        trajectories: List of raw trajectory arrays (seq_len, 6)
                     Features: [lat, lon, sog, cog_sin, cog_cos, dt]
        config: Configuration dict with normalization parameters

    Returns:
        List of normalized trajectory arrays
    """
    data_config = config['data']

    lat_mean = data_config['lat_mean']
    lat_std = data_config['lat_std']
    lon_mean = data_config['lon_mean']
    lon_std = data_config['lon_std']
    sog_max = data_config['sog_max']
    dt_max = data_config['dt_max']

    normalized = []

    for traj in trajectories:
        traj_norm = traj.copy()

        # Normalize lat
        traj_norm[:, 0] = (traj[:, 0] - lat_mean) / lat_std

        # Normalize lon
        traj_norm[:, 1] = (traj[:, 1] - lon_mean) / lon_std

        # Normalize sog (clip and scale)
        traj_norm[:, 2] = np.clip(traj[:, 2], 0, sog_max) / sog_max

        # cog_sin and cog_cos already in [-1, 1], no change needed
        # traj_norm[:, 3] = traj[:, 3]  # cog_sin
        # traj_norm[:, 4] = traj[:, 4]  # cog_cos

        # Normalize dt (clip and scale)
        traj_norm[:, 5] = np.clip(traj[:, 5], 0, dt_max) / dt_max

        normalized.append(traj_norm.astype(np.float32))

    return normalized


def save_processed_data(
    trajectories: List[np.ndarray],
    labels: np.ndarray,
    output_path: str
):
    """Save preprocessed data to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    np.save(output_path / 'labels.npy', labels)

    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def load_processed_data(processed_path: str):
    """Load preprocessed data from disk."""
    processed_path = Path(processed_path)

    with open(processed_path / 'trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    labels = np.load(processed_path / 'labels.npy')

    return trajectories, labels


def filter_trajectories(
    trajectories: List[np.ndarray],
    labels: np.ndarray,
    min_length: int = 10,
    max_length: int = None
) -> tuple:
    """
    Filter trajectories by length.

    Args:
        trajectories: List of trajectory arrays
        labels: Corresponding labels
        min_length: Minimum trajectory length
        max_length: Maximum trajectory length (None for no limit)

    Returns:
        Filtered trajectories and labels
    """
    filtered_trajs = []
    filtered_labels = []

    for traj, label in zip(trajectories, labels):
        length = len(traj)
        if length >= min_length:
            if max_length is None or length <= max_length:
                filtered_trajs.append(traj)
                filtered_labels.append(label)

    return filtered_trajs, np.array(filtered_labels)


def compute_trajectory_stats(trajectories: List[np.ndarray]) -> dict:
    """
    Compute statistics over all trajectories.

    Useful for verifying data distribution matches training data.
    """
    all_data = np.vstack(trajectories)

    stats = {
        'lat': {
            'mean': float(all_data[:, 0].mean()),
            'std': float(all_data[:, 0].std()),
            'min': float(all_data[:, 0].min()),
            'max': float(all_data[:, 0].max())
        },
        'lon': {
            'mean': float(all_data[:, 1].mean()),
            'std': float(all_data[:, 1].std()),
            'min': float(all_data[:, 1].min()),
            'max': float(all_data[:, 1].max())
        },
        'sog': {
            'mean': float(all_data[:, 2].mean()),
            'std': float(all_data[:, 2].std()),
            'min': float(all_data[:, 2].min()),
            'max': float(all_data[:, 2].max())
        },
        'dt': {
            'mean': float(all_data[:, 5].mean()),
            'std': float(all_data[:, 5].std()),
            'min': float(all_data[:, 5].min()),
            'max': float(all_data[:, 5].max())
        },
        'trajectory_lengths': {
            'mean': float(np.mean([len(t) for t in trajectories])),
            'std': float(np.std([len(t) for t in trajectories])),
            'min': int(min(len(t) for t in trajectories)),
            'max': int(max(len(t) for t in trajectories))
        }
    }

    return stats
