"""Sliding-window extraction from cleaned tracks.

`extract_windows_from_track` is ported verbatim from the legacy
materialize_samples.py (ais-analysis history, commit a9d516f). Feature
order matches the materialized format documented in MATERIALIZED_DATA.md.
"""
from __future__ import annotations

import numpy as np

FEATURE_COLUMNS = ["lat", "lon", "sog", "cog", "dt_seconds"]
NUM_FEATURES = len(FEATURE_COLUMNS)

# ---- v3 format: adds per-position heading + per-window metadata ----------
# Row layout: [META_FLOATS meta slots][window_size * NUM_FEATURES_V3 floats].
# Absolute time of position j = t0 + cumsum(dt_seconds)[j] (dt[0] belongs to
# the gap BEFORE the first stored position and is excluded), so one t0 per
# window recovers every position's timestamp — the weather-join key.
FEATURE_COLUMNS_V3 = ["lat", "lon", "sog", "cog", "dt_seconds", "heading"]
NUM_FEATURES_V3 = len(FEATURE_COLUMNS_V3)
META_FLOATS = 4                      # [day_num, sec_in_day, mmsi_hi, mmsi_lo]
HEADING_MISSING = -1.0               # AIS 511 / null sentinel (0 is a valid heading)


def pack_window_meta(t0_epoch_s: np.ndarray, mmsi: int) -> np.ndarray:
    """(N,) int64 epoch seconds + mmsi -> (N, META_FLOATS) float32, EXACT.

    float32 cannot hold an epoch (needs 31 bits); day number (<2^24) and
    second-in-day (<2^17) fit exactly, as do the two mmsi halves.
    """
    t0 = np.asarray(t0_epoch_s, dtype=np.int64)
    out = np.empty((len(t0), META_FLOATS), dtype=np.float32)
    out[:, 0] = t0 // 86400
    out[:, 1] = t0 % 86400
    out[:, 2] = mmsi // 100_000
    out[:, 3] = mmsi % 100_000
    return out


def unpack_window_meta(meta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, META_FLOATS) float32 -> (t0 epoch-seconds int64, mmsi int64)."""
    m = np.asarray(meta, dtype=np.float64)
    t0 = (m[:, 0] * 86400 + m[:, 1]).astype(np.int64)
    mmsi = (m[:, 2] * 100_000 + m[:, 3]).astype(np.int64)
    return t0, mmsi


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
