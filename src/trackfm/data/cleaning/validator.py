"""Basic validation functions for AIS position data."""
import polars as pl
from typing import Dict, Any


def validate_positions(
    df: pl.DataFrame,
    bounds: Dict[str, float] = None,
    max_sog: float = 102.3,  # AIS max speed
) -> pl.DataFrame:
    """Apply basic validation filters to AIS position data.

    Step 1 of the cleaning pipeline:
    - Remove null lat/lon
    - Remove positions outside bounding box (if specified)
    - Remove SOG > max_sog (AIS specification max is 102.3 knots)
    - Remove duplicate timestamps per MMSI

    Args:
        df: Input DataFrame with columns: lat, lon, mmsi, timestamp, sog (optional)
        bounds: Dictionary with lat_min, lat_max, lon_min, lon_max (optional)
        max_sog: Maximum allowed SOG value in knots

    Returns:
        Filtered DataFrame
    """
    if df.is_empty():
        return df

    # Remove null coordinates
    df = df.filter(
        pl.col("lat").is_not_null() &
        pl.col("lon").is_not_null()
    )

    # Remove invalid coordinate ranges
    df = df.filter(
        pl.col("lat").is_between(-90, 90) &
        pl.col("lon").is_between(-180, 180)
    )

    # Apply bounding box filter if specified (Danish EEZ)
    if bounds:
        df = df.filter(
            pl.col("lat").is_between(bounds["lat_min"], bounds["lat_max"]) &
            pl.col("lon").is_between(bounds["lon_min"], bounds["lon_max"])
        )

    # Remove invalid SOG values
    if "sog" in df.columns:
        df = df.filter(
            pl.col("sog").is_null() |
            (pl.col("sog").is_between(0, max_sog))
        )

    # Remove duplicate timestamps per MMSI
    df = df.unique(subset=["mmsi", "timestamp"], keep="first")

    return df


def get_validation_stats(original_df: pl.DataFrame, validated_df: pl.DataFrame) -> Dict[str, Any]:
    """Get statistics about validation filtering.

    Args:
        original_df: DataFrame before validation
        validated_df: DataFrame after validation

    Returns:
        Dictionary with validation statistics
    """
    original_count = original_df.height
    validated_count = validated_df.height
    removed_count = original_count - validated_count

    return {
        "original_records": original_count,
        "validated_records": validated_count,
        "removed_records": removed_count,
        "removal_percentage": (removed_count / original_count * 100) if original_count > 0 else 0,
    }
