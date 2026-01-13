from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl
from pyproj.geod import Geod


def generate_linear_track_in_box(
    top: float,
    bottom: float,
    left: float,
    right: float,
    min_posits: int = 128,
    max_posits: int = 2048,
    min_speed_kt: float = 1.0,
    max_speed_kt: float = 30.0,
    min_start_date: datetime = datetime(2022, 1, 1),
    max_start_date: datetime = datetime(2026, 1, 1),
    seed: int | None = None,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    geod = Geod(ellps="WGS84")

    # Generate start date of track
    max_start_delta = int((max_start_date - min_start_date).total_seconds())
    start_delta = int(rng.integers(0, max_start_delta))
    start_date = min_start_date + timedelta(seconds=start_delta)

    # Generate start and end position of track
    start_lat, end_lat = rng.uniform(bottom, top, size=2)
    start_lon, end_lon = rng.uniform(left, right, size=2)

    # Generate track speed and length
    speed_kt = rng.uniform(bottom, top)
    KNOTS_TO_METERS_PER_SECOND = 0.514444
    speed_m_s = speed_kt * KNOTS_TO_METERS_PER_SECOND
    num_posits = int(rng.integers(min_posits, max_posits))

    # Calculate posit timestamps
    az, _, track_distance_m = geod.inv(start_lon, start_lat, end_lon, end_lat, return_back_azimuth=False)
    track_duration_s = track_distance_m / speed_m_s
    end_date = start_date + timedelta(seconds=track_duration_s)
    num_intermediate_points = num_posits - 2
    time_deltas = sorted(rng.uniform(0, track_duration_s, num_intermediate_points))
    timestamps = [start_date] + [start_date + timedelta(seconds=x) for x in time_deltas] + [end_date]

    # Generate Track ID
    track_id = str(UUID(int=seed)) if seed is not None else None

    # Calculate coordinates
    dists = [speed_m_s * t for t in time_deltas]
    intermediate_lons, intermediate_lats, _ = geod.fwd(
        lons=[start_lon]*num_intermediate_points,
        lats=[start_lat]*num_intermediate_points,
        az=[az]*num_intermediate_points,
        dist=dists,
    )

    return pl.DataFrame({
        "time": timestamps,
        "lat": [start_lat] + intermediate_lats + [end_lat],
        "lon": [start_lon] + intermediate_lons + [end_lon],
        "track_id": track_id
    })

starting_lat = 30.0
starting_lon = -70
box_size_mi = 200
half_size_m = (box_size_mi / 2) * 1609.34

geod = Geod(ellps="WGS84")
lons, lats, _ = geod.fwd(
    lons=[starting_lon] * 4,
    lats=[starting_lat] * 4,
    az=[0, 90, 180, 270],
    dist=[half_size_m] * 4,
    return_back_azimuth=False,
)
top = max(lats)
bottom = min(lats)
left = min(lons)
right = max(lons)

n_train = 10_000
train = []
for i in range(n_train):
    train.append(
        generate_linear_track_in_box(
            top,
            bottom,
            left,
            right,
            seed=i
        )
    )
train = pl.concat(train)

n_val = 1000
val = []
for i in range(n_train, n_train+n_val):
    val.append(
        generate_linear_track_in_box(
            top,
            bottom,
            left,
            right,
            seed=i
        )
    )
val = pl.concat(val)

n_test = 1000
test = []
for i in range(n_train+n_val, n_train+n_val+n_test):
    test.append(
        generate_linear_track_in_box(
            top,
            bottom,
            left,
            right,
            seed=i
        )
    )
test = pl.concat(test)

dataset_path = Path("./data")
dataset_path.mkdir(exist_ok=True)
train_path = dataset_path / "train.parquet"
val_path = dataset_path / "val.parquet"
test_path = dataset_path / "test.parquet"
train.write_parquet(train_path)
val.write_parquet(val_path)
test.write_parquet(test_path)