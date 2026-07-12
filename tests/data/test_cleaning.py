"""Cleaning-pipeline invariants on synthetic tracks."""
from datetime import datetime, timedelta

import polars as pl
import pytest

from trackfm.data.cleaning.outliers import remove_single_outliers
from trackfm.data.cleaning.segmentation import add_dt_seconds, filter_short_tracks, segment_tracks
from trackfm.data.cleaning.validator import validate_positions

T0 = datetime(2024, 3, 15, 12, 0, 0)
BOUNDS = {"lat_min": 54.0, "lat_max": 58.5, "lon_min": 7.0, "lon_max": 16.0}


def track(rows):
    """rows: list of (minutes_offset, lat, lon, sog)."""
    return pl.DataFrame({
        "timestamp": [T0 + timedelta(minutes=m) for m, *_ in rows],
        "mmsi": [219000001] * len(rows),
        "lat": [r[1] for r in rows],
        "lon": [r[2] for r in rows],
        "sog": [r[3] for r in rows],
        "cog": [90.0] * len(rows),
    })


# ------------------------------------------------------------------ validator
def test_validator_bbox_and_nulls():
    df = track([(0, 56.0, 11.0, 10.0), (1, 60.0, 11.0, 10.0),   # lat out of bbox
                (2, 56.0, 20.0, 10.0),                           # lon out of bbox
                (3, 56.1, 11.1, 10.0)])
    df = pl.concat([df, df.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("lat"))],
                   how="diagonal")
    out = validate_positions(df, bounds=BOUNDS)
    assert out.height == 2
    assert out["lat"].is_between(54.0, 58.5).all()
    assert out["lon"].is_between(7.0, 16.0).all()


def test_validator_max_sog_and_duplicate_timestamps():
    df = track([(0, 56.0, 11.0, 10.0), (0, 56.0, 11.0, 10.0),   # duplicate ts
                (1, 56.0, 11.0, 150.0)])                         # impossible SOG
    out = validate_positions(df, bounds=BOUNDS)
    assert out.height == 1


# ------------------------------------------------------------------- outliers
def test_velocity_skip_removes_single_gps_glitch():
    # Steady 56.0 -> 56.04 track with one glitch jumping ~1 degree away
    rows = [(i * 5, 56.0 + i * 0.005, 11.0, 10.0) for i in range(9)]
    rows[4] = (20, 57.0, 12.0, 10.0)  # glitch: ~130km off-course for one ping
    df = track(rows)
    out = remove_single_outliers(df, max_velocity_knots=50.0)
    assert out.height == 8
    assert not ((out["lat"] - 57.0).abs() < 1e-9).any()


def test_outliers_keeps_clean_track_intact():
    rows = [(i * 5, 56.0 + i * 0.005, 11.0, 10.0) for i in range(9)]
    out = remove_single_outliers(track(rows), max_velocity_knots=50.0)
    assert out.height == 9


# --------------------------------------------------------------- segmentation
def test_gap_segmentation_splits_and_names_tracks():
    rows = [(0, 56.0, 11.0, 10.0), (5, 56.01, 11.0, 10.0),
            (5 * 60 + 10, 56.02, 11.0, 10.0), (5 * 60 + 15, 56.03, 11.0, 10.0)]  # 5h gap
    out = segment_tracks(track(rows), gap_hours=4.0, starting_segment=0)
    ids = out["track_id"].unique().sort().to_list()
    assert ids == ["219000001_0", "219000001_1"]


def test_segmentation_continuity_starting_segment():
    rows = [(0, 56.0, 11.0, 10.0), (5, 56.01, 11.0, 10.0)]
    out = segment_tracks(track(rows), gap_hours=4.0, starting_segment=7)
    assert out["track_id"].unique().to_list() == ["219000001_7"]


def test_filter_short_tracks():
    rows = [(0, 56.0, 11.0, 10.0), (5, 56.01, 11.0, 10.0),
            (5 * 60 + 10, 56.02, 11.0, 10.0)]  # second segment has 1 point
    seg = segment_tracks(track(rows), gap_hours=4.0)
    out = filter_short_tracks(seg, min_track_points=2)
    assert out["track_id"].unique().to_list() == ["219000001_0"]


def test_add_dt_seconds():
    rows = [(0, 56.0, 11.0, 10.0), (5, 56.01, 11.0, 10.0), (12, 56.02, 11.0, 10.0)]
    seg = segment_tracks(track(rows), gap_hours=4.0)
    out = add_dt_seconds(seg)
    assert out["dt_seconds"].to_list() == [0.0, 300.0, 420.0]
