"""
Preprocessing utilities for vessel classification.

IMPORTANT: Normalization must match Experiment 11 (pre-training) exactly!
"""

import numpy as np


# Normalization constants from Experiment 11 (must match exactly!)
LAT_MEAN = 56.25  # Center of Danish EEZ
LAT_STD = 1.0
LON_MEAN = 11.5
LON_STD = 2.0
SOG_MAX = 30.0    # Max SOG for normalization
DT_MAX = 300.0    # Max dt (5 minutes)


def normalize_trajectory(
    lat: np.ndarray,
    lon: np.ndarray,
    sog: np.ndarray,
    cog: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """
    Normalize trajectory features to match pre-training format.

    MUST use same normalization as Experiment 11!

    Args:
        lat: Latitude values
        lon: Longitude values
        sog: Speed over ground (knots)
        cog: Course over ground (degrees, 0-360)
        dt: Time delta (seconds)

    Returns:
        features: Array of shape (seq_len, 6) with normalized features
    """
    # Normalize lat/lon using GLOBAL mean/std (same as Exp 11!)
    lat_norm = (lat - LAT_MEAN) / LAT_STD
    lon_norm = (lon - LON_MEAN) / LON_STD

    # Normalize SOG (same max as Exp 11: 30 knots)
    sog_norm = np.clip(sog, 0, SOG_MAX) / SOG_MAX

    # Convert COG to sin/cos
    cog_rad = np.deg2rad(cog)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)

    # Normalize dt (same max as Exp 11: 300 seconds)
    dt_clipped = np.clip(dt, 0, DT_MAX)
    dt_norm = dt_clipped / DT_MAX

    # Stack features: [lat, lon, sog, cog_sin, cog_cos, dt]
    features = np.stack([
        lat_norm, lon_norm, sog_norm, cog_sin, cog_cos, dt_norm
    ], axis=1).astype(np.float32)

    return features
