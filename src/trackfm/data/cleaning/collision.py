"""MMSI collision detection - detecting when two vessels share the same identifier.

Problem Type 2: MMSI Collision (Two Vessels, One ID)
Signature: Track "bounces" between two distant locations repeatedly.
Cause: MMSI spoofing, misconfigured transponders, reused MMSIs (common in fishing fleets)

Detection Algorithm: Spatial Clustering with Bounce Detection

Phase 1: Detect Potential Collision
  For each MMSI with positions sorted by time:
    1. Compute pairwise distances between consecutive positions
    2. Flag if distance > COLLISION_THRESHOLD_KM (e.g., 50 km)
       AND elapsed time < reasonable transit time at max speed
    3. If flagged, examine pattern over next N positions

Phase 2: Classify Bounce Pattern
  For flagged sequences:
    1. Extract positions in sliding window of size W (e.g., 10 positions)
    2. Apply DBSCAN clustering with eps=5km, min_samples=2
    3. If exactly 2 clusters found AND positions alternate between them:
       → MMSI collision confirmed
    4. If 1 cluster with outliers:
       → Single outliers, use velocity filter

Phase 3: Split Tracks
  For confirmed collisions:
    1. Assign each position to nearest cluster centroid
    2. Create two synthetic track IDs: {MMSI}_A and {MMSI}_B
    3. Maintain cluster assignment for cross-file continuity
"""
import polars as pl
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..utils.geo import haversine_km

logger = logging.getLogger(__name__)


@dataclass
class CollisionInfo:
    """Information about a detected MMSI collision."""
    mmsi: int
    cluster_a_centroid: Tuple[float, float]  # (lat, lon)
    cluster_b_centroid: Tuple[float, float]  # (lat, lon)
    bounce_count: int
    detection_timestamp: str


def detect_potential_collision(
    df: pl.DataFrame,
    distance_threshold_km: float = 50.0,
    max_speed_knots: float = 50.0,
) -> bool:
    """Check if a track has potential MMSI collision signatures.

    A potential collision is flagged when:
    - Distance between consecutive positions > threshold
    - Time between positions < what would be needed at max speed

    Args:
        df: DataFrame sorted by timestamp for single MMSI
        distance_threshold_km: Minimum distance to consider as potential collision
        max_speed_knots: Maximum realistic vessel speed

    Returns:
        True if potential collision detected
    """
    if df.height < 4:
        return False

    timestamps = df["timestamp"].to_list()
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()

    bounce_count = 0

    for i in range(1, len(timestamps)):
        if lats[i] is None or lons[i] is None or lats[i-1] is None or lons[i-1] is None:
            continue

        dist_km = haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
        dt_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600

        if dt_hours <= 0:
            continue

        # Calculate minimum time needed to cover this distance at max speed
        max_speed_kmh = max_speed_knots * 1.852
        min_time_needed = dist_km / max_speed_kmh if max_speed_kmh > 0 else float('inf')

        # If actual time is less than half of minimum needed, it's impossible
        if dist_km > distance_threshold_km and dt_hours < min_time_needed * 0.5:
            bounce_count += 1

            if bounce_count >= 2:
                return True

    return False


def cluster_positions(
    lats: List[float],
    lons: List[float],
    eps_km: float = 5.0,
    min_samples: int = 2,
) -> np.ndarray:
    """Cluster positions using DBSCAN.

    Args:
        lats: List of latitudes
        lons: List of longitudes
        eps_km: Maximum distance in km for DBSCAN
        min_samples: Minimum samples per cluster

    Returns:
        Array of cluster labels (-1 for noise)
    """
    # Filter out None values
    valid_indices = [i for i in range(len(lats)) if lats[i] is not None and lons[i] is not None]
    if len(valid_indices) < min_samples:
        return np.array([-1] * len(lats))

    valid_lats = [lats[i] for i in valid_indices]
    valid_lons = [lons[i] for i in valid_indices]

    # Convert to radians for haversine-based distance
    coords = np.radians(np.column_stack([valid_lats, valid_lons]))

    # DBSCAN with haversine metric
    # eps needs to be in radians: km / earth_radius
    eps_rad = eps_km / 6371.0

    clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    labels = clustering.fit_predict(coords)

    # Map labels back to original indices
    full_labels = np.array([-1] * len(lats))
    for idx, label in zip(valid_indices, labels):
        full_labels[idx] = label

    return full_labels


def detect_bounce_pattern(
    labels: np.ndarray,
    min_bounce_count: int = 3,
) -> bool:
    """Check if cluster labels show a bounce pattern (alternating between 2 clusters).

    Args:
        labels: Array of cluster labels
        min_bounce_count: Minimum number of bounces required

    Returns:
        True if bounce pattern detected
    """
    # Filter out noise labels (-1)
    valid_labels = labels[labels >= 0]

    if len(valid_labels) < min_bounce_count * 2:
        return False

    # Check for exactly 2 distinct clusters
    unique_labels = set(valid_labels)
    if len(unique_labels) != 2:
        return False

    # Count alternations
    alternations = 0
    for i in range(1, len(valid_labels)):
        if valid_labels[i] != valid_labels[i-1]:
            alternations += 1

    return alternations >= min_bounce_count


def compute_cluster_centroids(
    lats: List[float],
    lons: List[float],
    labels: np.ndarray,
) -> Dict[int, Tuple[float, float]]:
    """Compute centroid for each cluster.

    Args:
        lats: List of latitudes
        lons: List of longitudes
        labels: Cluster labels

    Returns:
        Dictionary mapping cluster label to (lat, lon) centroid
    """
    centroids = {}
    unique_labels = set(labels[labels >= 0])

    for label in unique_labels:
        mask = labels == label
        cluster_lats = [lats[i] for i in range(len(lats)) if mask[i] and lats[i] is not None]
        cluster_lons = [lons[i] for i in range(len(lons)) if mask[i] and lons[i] is not None]

        if cluster_lats and cluster_lons:
            centroids[label] = (
                sum(cluster_lats) / len(cluster_lats),
                sum(cluster_lons) / len(cluster_lons)
            )

    return centroids


def detect_mmsi_collision(
    df: pl.DataFrame,
    distance_threshold_km: float = 50.0,
    dbscan_eps_km: float = 5.0,
    min_bounce_count: int = 3,
    lookback_window: int = 10,
) -> Optional[CollisionInfo]:
    """Detect if an MMSI has collision (two vessels sharing same ID).

    Args:
        df: DataFrame sorted by timestamp for single MMSI
        distance_threshold_km: Minimum distance for potential collision
        dbscan_eps_km: DBSCAN epsilon in km
        min_bounce_count: Minimum bounces to confirm collision
        lookback_window: Number of positions to analyze

    Returns:
        CollisionInfo if collision detected, None otherwise
    """
    if df.height < lookback_window:
        return None

    # Quick check for potential collision
    if not detect_potential_collision(df, distance_threshold_km):
        return None

    # Get positions for clustering
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()

    # Cluster positions
    labels = cluster_positions(lats, lons, dbscan_eps_km)

    # Check for bounce pattern
    if not detect_bounce_pattern(labels, min_bounce_count):
        return None

    # Compute centroids
    centroids = compute_cluster_centroids(lats, lons, labels)

    if len(centroids) != 2:
        return None

    # Get MMSI and create collision info
    mmsi = df.select("mmsi").item(0, 0)
    centroid_list = list(centroids.values())

    # Count actual bounces
    valid_labels = labels[labels >= 0]
    alternations = sum(1 for i in range(1, len(valid_labels)) if valid_labels[i] != valid_labels[i-1])

    return CollisionInfo(
        mmsi=mmsi,
        cluster_a_centroid=centroid_list[0],
        cluster_b_centroid=centroid_list[1],
        bounce_count=alternations,
        detection_timestamp=str(df.select("timestamp").min().item()),
    )


def assign_to_cluster(
    lat: float,
    lon: float,
    centroid_a: Tuple[float, float],
    centroid_b: Tuple[float, float],
) -> str:
    """Assign a position to the nearest cluster.

    Args:
        lat: Latitude of position
        lon: Longitude of position
        centroid_a: (lat, lon) of cluster A
        centroid_b: (lat, lon) of cluster B

    Returns:
        "A" or "B" indicating cluster assignment
    """
    if lat is None or lon is None:
        return "A"  # Default assignment

    dist_a = haversine_km(lat, lon, centroid_a[0], centroid_a[1])
    dist_b = haversine_km(lat, lon, centroid_b[0], centroid_b[1])

    return "A" if dist_a <= dist_b else "B"


def split_collision_tracks(
    df: pl.DataFrame,
    collision_info: CollisionInfo,
) -> pl.DataFrame:
    """Split a collision track into two separate track IDs.

    Args:
        df: DataFrame with collision MMSI
        collision_info: Information about the collision

    Returns:
        DataFrame with cluster_assignment column ("A" or "B")
    """
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()

    assignments = [
        assign_to_cluster(
            lat, lon,
            collision_info.cluster_a_centroid,
            collision_info.cluster_b_centroid
        )
        for lat, lon in zip(lats, lons)
    ]

    return df.with_columns(
        pl.Series(name="cluster_assignment", values=assignments)
    )
