"""Sliding-window extraction from cleaned tracks.

`extract_windows_from_track` is ported verbatim from the legacy
materialize_samples.py (ais-analysis history, commit a9d516f). Feature
order matches the materialized format documented in MATERIALIZED_DATA.md.
"""
from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = ["lat", "lon", "sog", "cog", "dt_seconds"]
NUM_FEATURES = len(FEATURE_COLUMNS)


def extract_windows_from_track(
    features: np.ndarray,
    window_size: int = 928,
    stride: int = 32,
) -> np.ndarray:
    """Extract sliding windows from a track's features.

    Args:
        features: (num_positions, NUM_FEATURES) array

    Returns:
        (num_windows, window_size, NUM_FEATURES) float32 array
    """
    num_positions = len(features)
    if num_positions < window_size:
        return np.array([], dtype=np.float32).reshape(0, window_size, features.shape[1])

    num_windows = (num_positions - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size, features.shape[1]), dtype=np.float32)

    for i in range(num_windows):
        start = i * stride
        windows[i] = features[start:start + window_size]

    return windows


def normalize_features(windows: np.ndarray, lat_center: float = 56.25, lat_scale: float = 1.0,
                       lon_center: float = 11.5, lon_scale: float = 2.0,
                       sog_scale: float = 30.0, dt_scale: float = 300.0) -> np.ndarray:
    """Raw 5-feature windows -> normalized 6-feature model input.

    (…, 5)[lat, lon, sog, cog_deg, dt_s] -> (…, 6)[lat, lon, sog, cog_sin, cog_cos, dt],
    matching experiment 11's feature normalization (HYPERPARAMETERS.md).
    """
    out = np.empty((*windows.shape[:-1], 6), dtype=np.float32)
    out[..., 0] = (windows[..., 0] - lat_center) / lat_scale
    out[..., 1] = (windows[..., 1] - lon_center) / lon_scale
    out[..., 2] = windows[..., 2] / sog_scale
    cog_rad = np.deg2rad(windows[..., 3])
    out[..., 3] = np.sin(cog_rad)
    out[..., 4] = np.cos(cog_rad)
    out[..., 5] = windows[..., 4] / dt_scale
    return out
