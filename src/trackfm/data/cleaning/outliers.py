"""Single outlier detection and removal using velocity-skip method.

Problem Type 1: Single Outlier Positions
Signature: One position wildly far from neighbors, then track continues normally.
Cause: GPS glitch, transcription error, atmospheric interference.

Detection Algorithm: Velocity-Based Outlier

For each position P[i] in a track:
  1. Calculate velocity from P[i-1] to P[i]: v_before
  2. Calculate velocity from P[i] to P[i+1]: v_after
  3. Calculate velocity from P[i-1] to P[i+1]: v_skip (as if P[i] didn't exist)

  If (v_before > MAX_VELOCITY AND v_after > MAX_VELOCITY AND v_skip < MAX_VELOCITY):
      → P[i] is a single outlier, remove it
"""
import polars as pl
from typing import Dict, List, Optional
import logging

from ..utils.geo import haversine_km

logger = logging.getLogger(__name__)


def velocity_knots_from_positions(
    lat1: float, lon1: float, t1,
    lat2: float, lon2: float, t2
) -> float:
    """Calculate velocity in knots between two positions."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return float('inf')

    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    dt_seconds = (t2 - t1).total_seconds()

    if dt_seconds <= 0:
        return float('inf')

    dt_hours = dt_seconds / 3600
    speed_kmh = dist_km / dt_hours
    return speed_kmh / 1.852  # Convert to knots


def remove_single_outliers(
    df: pl.DataFrame,
    max_velocity_knots: float = 50.0,
    velocity_by_ship_type: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Remove single outlier positions using velocity-skip method.

    For each position, checks if it's a single outlier by:
    1. Computing velocity from previous position to current
    2. Computing velocity from current to next position
    3. Computing velocity from previous to next (skipping current)

    If both incoming and outgoing velocities exceed threshold but skip velocity
    is reasonable, the current position is a single outlier.

    Args:
        df: DataFrame sorted by timestamp for a single MMSI
        max_velocity_knots: Maximum allowed velocity in knots
        velocity_by_ship_type: Optional dict mapping ship types to max velocities

    Returns:
        DataFrame with single outliers removed
    """
    if df.height < 3:
        # Need at least 3 positions to detect outliers
        return df

    # Get data as lists for efficient iteration
    timestamps = df["timestamp"].to_list()
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()

    # Get ship-type specific threshold if available
    max_vel = max_velocity_knots
    if velocity_by_ship_type and "ship_type" in df.columns:
        ship_type = df.select("ship_type").item(0, 0)
        if ship_type in velocity_by_ship_type:
            max_vel = velocity_by_ship_type[ship_type]

    # Track which indices to keep (start with all True)
    keep_indices = [True] * len(timestamps)

    # Always keep first and last positions
    # Check middle positions for outliers
    for i in range(1, len(timestamps) - 1):
        lat_prev, lon_prev, t_prev = lats[i-1], lons[i-1], timestamps[i-1]
        lat_curr, lon_curr, t_curr = lats[i], lons[i], timestamps[i]
        lat_next, lon_next, t_next = lats[i+1], lons[i+1], timestamps[i+1]

        # Skip if any positions are null
        if None in [lat_prev, lon_prev, lat_curr, lon_curr, lat_next, lon_next]:
            continue

        # Calculate velocities
        v_before = velocity_knots_from_positions(
            lat_prev, lon_prev, t_prev,
            lat_curr, lon_curr, t_curr
        )
        v_after = velocity_knots_from_positions(
            lat_curr, lon_curr, t_curr,
            lat_next, lon_next, t_next
        )
        v_skip = velocity_knots_from_positions(
            lat_prev, lon_prev, t_prev,
            lat_next, lon_next, t_next
        )

        # Check if this is a single outlier
        if v_before > max_vel and v_after > max_vel and v_skip <= max_vel:
            keep_indices[i] = False
            logger.debug(
                f"Removing single outlier at index {i}: "
                f"v_before={v_before:.1f}, v_after={v_after:.1f}, v_skip={v_skip:.1f}"
            )

    # Filter the dataframe
    indices_to_keep = [i for i, keep in enumerate(keep_indices) if keep]

    if len(indices_to_keep) == len(timestamps):
        return df

    return df[indices_to_keep]


def remove_outliers_iterative(
    df: pl.DataFrame,
    max_velocity_knots: float = 50.0,
    max_iterations: int = 3,
) -> pl.DataFrame:
    """Iteratively remove outliers until no more are found.

    Sometimes removing one outlier reveals another. This function
    iterates until convergence or max_iterations is reached.

    Args:
        df: DataFrame sorted by timestamp for a single MMSI
        max_velocity_knots: Maximum allowed velocity in knots
        max_iterations: Maximum number of iterations

    Returns:
        DataFrame with all single outliers removed
    """
    prev_count = df.height

    for iteration in range(max_iterations):
        df = remove_single_outliers(df, max_velocity_knots)

        if df.height == prev_count:
            # No more outliers found
            break

        prev_count = df.height

    return df
