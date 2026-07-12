"""Track segmentation - breaking tracks by time gaps.

Step 4 of the cleaning pipeline:
- Sort by timestamp
- Break track where gap > threshold (e.g., 4 hours)
- Assign unique track_id: {MMSI}_{segment_number} or {MMSI}_{cluster}_{segment}
"""
import polars as pl
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def segment_tracks(
    df: pl.DataFrame,
    gap_hours: float = 4.0,
    min_track_points: int = 2,
    starting_segment: int = 0,
    cluster_assignment: Optional[str] = None,
) -> pl.DataFrame:
    """Segment a single MMSI's positions into tracks based on time gaps.

    Args:
        df: DataFrame sorted by timestamp for single MMSI
        gap_hours: Time gap threshold in hours to start new segment
        min_track_points: Minimum positions required for a valid track
        starting_segment: Starting segment number (for cross-file continuity)
        cluster_assignment: If set ("A" or "B"), include in track_id for collision tracks

    Returns:
        DataFrame with track_id column added
    """
    if df.is_empty():
        return df.with_columns(pl.lit("").alias("track_id"))

    if df.height == 1:
        mmsi = df.select("mmsi").item(0, 0)
        if cluster_assignment:
            track_id = f"{mmsi}_{cluster_assignment}_{starting_segment}"
        else:
            track_id = f"{mmsi}_{starting_segment}"
        return df.with_columns(pl.lit(track_id).alias("track_id"))

    gap_seconds = gap_hours * 3600

    # Calculate time differences and identify new segments
    df_with_gaps = df.with_columns([
        (pl.col("timestamp").diff().dt.total_seconds() > gap_seconds)
        .fill_null(False)
        .alias("new_segment")
    ])

    # Create segment numbers
    df_with_segments = df_with_gaps.with_columns([
        pl.col("new_segment").cum_sum().alias("segment_offset")
    ]).with_columns([
        (pl.lit(starting_segment) + pl.col("segment_offset")).alias("segment_num")
    ])

    # Create track_id based on whether this is a collision track
    mmsi = df.select("mmsi").item(0, 0)

    if cluster_assignment:
        df_with_tracks = df_with_segments.with_columns([
            (pl.lit(f"{mmsi}_{cluster_assignment}_") + pl.col("segment_num").cast(pl.Utf8))
            .alias("track_id")
        ])
    else:
        df_with_tracks = df_with_segments.with_columns([
            (pl.lit(f"{mmsi}_") + pl.col("segment_num").cast(pl.Utf8))
            .alias("track_id")
        ])

    # Drop temporary columns
    df_with_tracks = df_with_tracks.drop(["new_segment", "segment_offset", "segment_num"])

    return df_with_tracks


def filter_short_tracks(
    df: pl.DataFrame,
    min_track_points: int = 2,
) -> pl.DataFrame:
    """Remove tracks with fewer than min_track_points positions.

    Args:
        df: DataFrame with track_id column
        min_track_points: Minimum positions required

    Returns:
        DataFrame with short tracks removed
    """
    if "track_id" not in df.columns or df.is_empty():
        return df

    # Get track counts
    track_counts = (
        df.group_by("track_id")
        .agg(pl.count().alias("count"))
        .filter(pl.col("count") >= min_track_points)
        .select("track_id")
    )

    # Filter to keep only valid tracks
    return df.join(track_counts, on="track_id", how="inner")


def get_final_segment_number(df: pl.DataFrame) -> int:
    """Get the final segment number from a DataFrame with track_id.

    Args:
        df: DataFrame with track_id column

    Returns:
        Final segment number, or 0 if not found
    """
    if "track_id" not in df.columns or df.is_empty():
        return 0

    # Get the last track_id
    last_track_id = df.tail(1).select("track_id").item(0, 0)

    if not last_track_id:
        return 0

    # Parse segment number from track_id
    # Format: {mmsi}_{segment} or {mmsi}_{cluster}_{segment}
    parts = last_track_id.split("_")

    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def add_dt_seconds(df: pl.DataFrame) -> pl.DataFrame:
    """Add dt_seconds column - time delta from previous position in same track.

    Args:
        df: DataFrame sorted by track_id, timestamp

    Returns:
        DataFrame with dt_seconds column
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("dt_seconds"))

    # Calculate time difference within each track
    df = df.sort(["track_id", "timestamp"])

    df = df.with_columns([
        pl.col("timestamp")
        .diff()
        .dt.total_seconds()
        .over("track_id")
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("dt_seconds")
    ])

    return df
