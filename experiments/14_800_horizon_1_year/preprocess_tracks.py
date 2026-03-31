#!/usr/bin/env python3
"""
Preprocess AIS data: reorganize from day-partitioned to track-partitioned format.

This script reads the day-organized parquet files and creates new parquet files
organized by track_id, making subsequent loading MUCH faster since we don't
need to piece together tracks from 363+ day files.

Input:  /mnt/fsx/data/year=*/month=*/day=*/tracks.parquet  (day-partitioned)
Output: /mnt/fsx/data/tracks_by_id/*.parquet               (track-partitioned)

The output format groups tracks into ~100 files for efficient parallel loading.
"""

import polars as pl
import numpy as np
import glob
import os
import time
from pathlib import Path
from collections import defaultdict
import argparse


def preprocess_tracks(
    data_path: str = "/mnt/fsx/data",
    catalog_path: str = "/mnt/fsx/data/track_catalog.parquet",
    output_dir: str = "/mnt/fsx/data/tracks_by_id",
    min_track_length: int = 1000,
    min_sog: float = 3.0,
    num_output_files: int = 100,
):
    """
    Reorganize day-partitioned data into track-partitioned format.

    Args:
        data_path: Base path for input data
        catalog_path: Path to track catalog parquet
        output_dir: Directory for output track-partitioned files
        min_track_length: Minimum track length to include
        min_sog: Minimum speed over ground filter
        num_output_files: Number of output files to create (for parallelism)
    """
    start_time = time.time()

    print("=" * 70)
    print("PREPROCESSING: DAY-PARTITIONED -> TRACK-PARTITIONED")
    print("=" * 70)
    print(f"\nInput: {data_path}/year=*/month=*/day=*/*.parquet")
    print(f"Output: {output_dir}/")
    print(f"Min track length: {min_track_length}")
    print(f"Min SOG: {min_sog}")
    print(f"Output files: {num_output_files}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load track catalog to get valid track IDs
    print("\nStep 1: Loading track catalog...")
    catalog = pl.read_parquet(catalog_path)
    valid_tracks = catalog.filter(pl.col("num_positions") >= min_track_length)
    valid_track_ids = set(valid_tracks["track_id"].to_list())
    print(f"  Found {len(valid_track_ids):,} valid tracks (length >= {min_track_length})")

    # Find all parquet files
    print("\nStep 2: Finding input files...")
    parquet_files = sorted(glob.glob(f"{data_path}/year=*/**/*.parquet", recursive=True))
    print(f"  Found {len(parquet_files)} day-partitioned files")

    # Assign tracks to output files (round-robin)
    track_to_file = {}
    sorted_track_ids = sorted(valid_track_ids)
    for idx, tid in enumerate(sorted_track_ids):
        track_to_file[tid] = idx % num_output_files

    # Process all input files and accumulate data by track
    print("\nStep 3: Reading and reorganizing data...")
    track_data = defaultdict(list)  # track_id -> list of DataFrames

    for file_idx, pf in enumerate(parquet_files):
        try:
            df = pl.read_parquet(pf)

            # Filter for valid tracks and moving vessels
            df = df.filter(
                (pl.col("track_id").is_in(list(valid_track_ids))) &
                (pl.col("sog") >= min_sog)
            )

            if df.height == 0:
                continue

            # Keep only necessary columns
            df = df.select([
                "track_id", "timestamp", "lat", "lon", "sog", "cog", "dt_seconds"
            ])

            # Group by track and store
            for track_df in df.partition_by("track_id", maintain_order=True):
                tid = track_df["track_id"][0]
                track_data[tid].append(track_df)

            del df

        except Exception as e:
            print(f"    Warning: Error processing {pf}: {e}")
            continue

        if (file_idx + 1) % 20 == 0 or file_idx == len(parquet_files) - 1:
            print(f"    Processed {file_idx + 1}/{len(parquet_files)} files, "
                  f"tracks accumulated: {len(track_data):,}")

    # Merge track data and write output files
    print(f"\nStep 4: Merging and writing {num_output_files} output files...")

    # Prepare output file writers
    output_data = defaultdict(list)  # file_idx -> list of DataFrames

    tracks_written = 0
    for tid, dfs in track_data.items():
        # Concatenate all data for this track and sort by timestamp
        if len(dfs) == 1:
            merged = dfs[0]
        else:
            merged = pl.concat(dfs)

        merged = merged.sort("timestamp")

        # Assign to output file
        file_idx = track_to_file[tid]
        output_data[file_idx].append(merged)
        tracks_written += 1

        if tracks_written % 5000 == 0:
            print(f"    Merged {tracks_written:,}/{len(track_data):,} tracks")

    print(f"    Merged all {tracks_written:,} tracks")

    # Write output files
    print("\nStep 5: Writing output files...")
    total_rows = 0
    for file_idx in range(num_output_files):
        if file_idx not in output_data:
            continue

        output_file = Path(output_dir) / f"tracks_{file_idx:03d}.parquet"
        combined = pl.concat(output_data[file_idx])
        combined.write_parquet(output_file)

        total_rows += combined.height

        if (file_idx + 1) % 10 == 0 or file_idx == num_output_files - 1:
            print(f"    Written {file_idx + 1}/{num_output_files} files, "
                  f"total rows: {total_rows:,}")

        del output_data[file_idx]

    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"PREPROCESSING COMPLETE")
    print(f"=" * 70)
    print(f"  Tracks processed: {tracks_written:,}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Output files: {num_output_files}")
    print(f"  Output location: {output_dir}/")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Create index file for fast lookups
    print("\nCreating track index file...")
    index_data = []
    for file_idx in range(num_output_files):
        output_file = Path(output_dir) / f"tracks_{file_idx:03d}.parquet"
        if output_file.exists():
            df = pl.read_parquet(output_file, columns=["track_id"])
            track_ids_in_file = df["track_id"].unique().to_list()
            for tid in track_ids_in_file:
                index_data.append({"track_id": tid, "file_idx": file_idx})
            del df

    index_df = pl.DataFrame(index_data)
    index_file = Path(output_dir) / "track_index.parquet"
    index_df.write_parquet(index_file)
    print(f"  Index file: {index_file} ({len(index_df)} entries)")

    print("\nDone! You can now use --track-data-dir for fast loading.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AIS data by track")
    parser.add_argument("--data-path", type=str, default="/mnt/fsx/data",
                       help="Base path for input data")
    parser.add_argument("--catalog-path", type=str,
                       default="/mnt/fsx/data/track_catalog.parquet",
                       help="Path to track catalog")
    parser.add_argument("--output-dir", type=str,
                       default="/mnt/fsx/data/tracks_by_id",
                       help="Output directory for track-partitioned files")
    parser.add_argument("--min-track-length", type=int, default=1000,
                       help="Minimum track length")
    parser.add_argument("--min-sog", type=float, default=3.0,
                       help="Minimum speed over ground")
    parser.add_argument("--num-files", type=int, default=100,
                       help="Number of output files")

    args = parser.parse_args()

    preprocess_tracks(
        data_path=args.data_path,
        catalog_path=args.catalog_path,
        output_dir=args.output_dir,
        min_track_length=args.min_track_length,
        min_sog=args.min_sog,
        num_output_files=args.num_files,
    )
