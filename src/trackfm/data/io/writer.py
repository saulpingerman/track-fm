"""Writing Parquet output to local filesystem."""
import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def write_parquet(
    df: pl.DataFrame,
    path: Path,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100_000,
) -> bool:
    """Write a DataFrame to a single local parquet file.

    The parent directory is created if missing.
    """
    path = Path(path)
    try:
        if "track_id" in df.columns and "timestamp" in df.columns:
            df = df.sort(["track_id", "timestamp"])

        path.parent.mkdir(parents=True, exist_ok=True)
        table = df.to_arrow()
        pq.write_table(
            table,
            path,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
        )

        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Wrote {df.height} rows ({file_size_mb:.1f} MB) to {path}")
        return True

    except Exception as e:
        logger.error(f"Error writing parquet to {path}: {e}")
        return False


def write_partitioned_parquet(
    df: pl.DataFrame,
    base_dir: Path,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100_000,
) -> bool:
    """Write DataFrame to base_dir partitioned by year/month/day.

    Output layout:
        <base_dir>/year=YYYY/month=MM/day=DD/tracks.parquet
    """
    base_dir = Path(base_dir)

    if "timestamp" not in df.columns:
        logger.error("DataFrame must have timestamp column for partitioning")
        return False

    try:
        df = df.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
        ])

        partition_groups = df.group_by(["year", "month", "day"])

        success = True
        for partition_keys, partition_df in partition_groups:
            year, month, day = partition_keys

            partition_path = (
                base_dir
                / f"year={year}"
                / f"month={month:02d}"
                / f"day={day:02d}"
                / "tracks.parquet"
            )

            partition_df = partition_df.drop(["year", "month", "day"])

            if not write_parquet(
                partition_df,
                partition_path,
                compression=compression,
                compression_level=compression_level,
                row_group_size=row_group_size,
            ):
                success = False

        return success

    except Exception as e:
        logger.error(f"Error writing partitioned parquet: {e}")
        return False


def generate_track_catalog(df: pl.DataFrame) -> pl.DataFrame:
    """Generate a per-track summary catalog for efficient lookup during training."""
    if "track_id" not in df.columns or "timestamp" not in df.columns:
        logger.error("DataFrame must have track_id and timestamp columns")
        return pl.DataFrame()

    catalog = (
        df.group_by("track_id")
        .agg([
            pl.col("mmsi").first().alias("mmsi"),
            pl.col("timestamp").min().alias("start_time"),
            pl.col("timestamp").max().alias("end_time"),
            pl.count().alias("num_positions"),
        ])
        .with_columns([
            ((pl.col("end_time") - pl.col("start_time")).dt.total_seconds() / 3600)
            .alias("duration_hours")
        ])
    )

    return catalog


def write_track_catalog(catalog_df: pl.DataFrame, base_dir: Path) -> bool:
    """Write track catalog to <base_dir>/track_catalog.parquet."""
    base_dir = Path(base_dir)
    catalog_path = base_dir / "track_catalog.parquet"
    return write_parquet(catalog_df, catalog_path)
