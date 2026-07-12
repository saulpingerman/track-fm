"""Main AIS data processing pipeline (local filesystem).

Orchestrates the complete cleaning workflow:
1. Basic Validation
2. MMSI Collision Detection
3. Single Outlier Removal
4. Track Segmentation
5. Track Validation

Handles cross-file track continuity and checkpointing.
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from tqdm import tqdm

from .cleaning.collision import detect_mmsi_collision, split_collision_tracks
from .cleaning.outliers import remove_single_outliers
from .cleaning.segmentation import (
    add_dt_seconds,
    filter_short_tracks,
    get_final_segment_number,
    segment_tracks,
)
from .cleaning.validator import validate_positions
from .config import PipelineConfig
from .io.reader import iter_zip_csvs, list_raw_files
from .io.writer import generate_track_catalog, write_partitioned_parquet, write_track_catalog
from .state.checkpoint import ProcessingCheckpoint, load_checkpoint, save_checkpoint
from .state.continuity import TrackContinuityState, load_state, save_state

logger = logging.getLogger(__name__)


class AISPipeline:
    """Main AIS data processing pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state: Optional[TrackContinuityState] = None
        self.checkpoint: Optional[ProcessingCheckpoint] = None
        self.stats: Dict[str, Any] = {
            "files_processed": 0,
            "total_records_input": 0,
            "total_records_output": 0,
            "records_removed_validation": 0,
            "records_removed_outliers": 0,
            "collisions_detected": 0,
            "unique_mmsis": set(),
            "unique_tracks": set(),
        }

    def initialize(self, resume: bool = False) -> bool:
        try:
            state_dir = self.config.storage.state_path

            if resume:
                self.state = load_state(state_dir)
                self.checkpoint = load_checkpoint(state_dir)
                logger.info(
                    f"Resumed from checkpoint: {len(self.checkpoint.processed_files)} files already processed"
                )
            else:
                self.state = TrackContinuityState()
                self.checkpoint = ProcessingCheckpoint()
                self.checkpoint.processing_started = datetime.utcnow().isoformat()
                logger.info("Starting fresh processing run")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def process_mmsi_group(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Process a single MMSI group through the cleaning pipeline."""
        if df.is_empty():
            return None

        mmsi = df.select("mmsi").item(0, 0)

        cluster_assignment = None
        if self.state.is_known_collision(mmsi):
            centroids = self.state.get_collision_centroids(mmsi)
            if centroids:
                from .cleaning.collision import CollisionInfo
                known_collision_info = CollisionInfo(
                    mmsi=mmsi,
                    cluster_a_centroid=centroids[0],
                    cluster_b_centroid=centroids[1],
                    bounce_count=0,
                    detection_timestamp="",
                )
                df = split_collision_tracks(df, known_collision_info)
                cluster_assignment = df.select("cluster_assignment").item(0, 0)
        else:
            collision_info = detect_mmsi_collision(
                df,
                distance_threshold_km=self.config.cleaning.collision_distance_threshold_km,
                dbscan_eps_km=self.config.cleaning.collision_dbscan_eps_km,
                min_bounce_count=self.config.cleaning.collision_min_bounce_count,
                lookback_window=self.config.cleaning.collision_lookback_window,
            )

            if collision_info:
                logger.info(
                    f"Detected MMSI collision for {mmsi}: {collision_info.bounce_count} bounces"
                )
                self.stats["collisions_detected"] += 1

                self.state.register_collision(
                    mmsi=mmsi,
                    centroid_a=collision_info.cluster_a_centroid,
                    centroid_b=collision_info.cluster_b_centroid,
                    detected_date=datetime.utcnow().strftime("%Y-%m-%d"),
                )

                df = split_collision_tracks(df, collision_info)
                cluster_assignment = df.select("cluster_assignment").item(0, 0)

        pre_outlier_count = df.height
        df = remove_single_outliers(
            df,
            max_velocity_knots=self.config.cleaning.max_velocity_knots,
            velocity_by_ship_type=self.config.cleaning.velocity_by_ship_type,
        )
        self.stats["records_removed_outliers"] += pre_outlier_count - df.height

        if df.is_empty():
            return None

        first_timestamp = df.select("timestamp").item(0, 0)
        starting_segment = self.state.get_starting_segment(
            mmsi,
            first_timestamp,
            self.config.cleaning.track_gap_hours,
        )

        df = segment_tracks(
            df,
            gap_hours=self.config.cleaning.track_gap_hours,
            min_track_points=self.config.cleaning.min_track_points,
            starting_segment=starting_segment,
            cluster_assignment=cluster_assignment,
        )

        df = filter_short_tracks(df, self.config.cleaning.min_track_points)

        if df.is_empty():
            return None

        df = add_dt_seconds(df)

        last_row = df.tail(1)
        self.state.update_mmsi_state(
            mmsi=mmsi,
            last_position=(
                last_row.select("lat").item(0, 0),
                last_row.select("lon").item(0, 0),
            ),
            last_timestamp=last_row.select("timestamp").item(0, 0),
            current_segment=get_final_segment_number(df),
            cluster_assignment=cluster_assignment,
        )

        return df

    def process_file(self, zip_path: Path) -> Optional[pl.DataFrame]:
        """Process a single ZIP file, one CSV member (day) at a time.

        Monthly zips bundle ~31 daily CSVs; streaming members keeps peak
        memory bounded to one day while preserving cross-day track
        continuity (members are processed in date order, exactly like a
        sequence of daily zips).
        """
        member_results: List[pl.DataFrame] = []
        members_read = 0

        for member_name, member_df in iter_zip_csvs(zip_path):
            members_read += 1
            logger.info(f"  {zip_path.name}::{member_name}: {member_df.height} records")
            result = self._process_dataframe(member_df, source=f"{zip_path.name}::{member_name}")
            if result is not None:
                member_results.append(result)

        if members_read == 0:
            logger.warning(f"No data read from {zip_path}")
            return None
        if not member_results:
            return None

        result_df = pl.concat(member_results, how="diagonal")
        self.stats["total_records_output"] += result_df.height
        return result_df

    def iter_processed_members(self, zip_path: Path):
        """Yield one cleaned DataFrame per CSV member (day) of a zip.

        Streaming counterpart of process_file: callers write each member's
        result immediately, so peak memory stays bounded to ~one day even
        for monthly zips (process_file's whole-month accumulation peaked at
        ~115GB RSS on aisdk-2023-01).
        """
        members_read = 0
        for member_name, member_df in iter_zip_csvs(zip_path):
            members_read += 1
            logger.info(f"  {zip_path.name}::{member_name}: {member_df.height} records")
            result = self._process_dataframe(member_df, source=f"{zip_path.name}::{member_name}")
            if result is not None and not result.is_empty():
                self.stats["total_records_output"] += result.height
                yield result

        if members_read == 0:
            logger.warning(f"No data read from {zip_path}")

    def _process_dataframe(self, raw_df: pl.DataFrame, source: str = "") -> Optional[pl.DataFrame]:
        """Run one day's DataFrame through validation and per-MMSI cleaning."""
        self.stats["total_records_input"] += raw_df.height

        validated_df = validate_positions(
            raw_df,
            bounds=self.config.cleaning.bounds,
        )

        self.stats["records_removed_validation"] += raw_df.height - validated_df.height

        if validated_df.is_empty():
            logger.warning(f"All records filtered out from {source}")
            return None

        validated_df = validated_df.sort(["mmsi", "timestamp"])

        all_processed: List[pl.DataFrame] = []
        mmsi_groups = validated_df.partition_by("mmsi")

        for mmsi_df in mmsi_groups:
            mmsi = mmsi_df.select("mmsi").item(0, 0)
            self.stats["unique_mmsis"].add(mmsi)

            processed_df = self.process_mmsi_group(mmsi_df)
            if processed_df is not None:
                all_processed.append(processed_df)

                for track_id in processed_df.select("track_id").unique().to_series().to_list():
                    self.stats["unique_tracks"].add(track_id)

        if not all_processed:
            return None

        return pl.concat(all_processed, how="diagonal")

    def run(self, resume: bool = False, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete processing pipeline."""
        start_time = time.time()

        if not self.initialize(resume):
            return {"error": "Failed to initialize"}

        raw_dir = self.config.storage.raw_path
        clean_dir = self.config.storage.clean_path
        state_dir = self.config.storage.state_path

        all_files = list_raw_files(raw_dir)
        if not all_files:
            return {"error": f"No files found in {raw_dir}"}

        all_keys = [str(p) for p in all_files]
        pending_keys = self.checkpoint.get_pending_files(all_keys)

        if max_files:
            pending_keys = pending_keys[:max_files]

        logger.info(f"Processing {len(pending_keys)} files (of {len(all_files)} total)")

        all_catalogs: List[pl.DataFrame] = []

        for zip_key in tqdm(pending_keys, desc="Processing files"):
            zip_path = Path(zip_key)
            try:
                for member_df in self.iter_processed_members(zip_path):
                    write_partitioned_parquet(
                        member_df,
                        clean_dir,
                        compression=self.config.output.compression,
                        compression_level=self.config.output.compression_level,
                        row_group_size=self.config.output.row_group_size,
                    )

                    file_catalog = generate_track_catalog(member_df)
                    if not file_catalog.is_empty():
                        all_catalogs.append(file_catalog)

                self.checkpoint.mark_processed(zip_key)
                self.stats["files_processed"] += 1

                if self.stats["files_processed"] % self.config.processing.checkpoint_interval == 0:
                    save_state(self.state, state_dir)
                    save_checkpoint(self.checkpoint, state_dir)

            except Exception as e:
                logger.error(f"Error processing {zip_key}: {e}")
                self.checkpoint.mark_failed(zip_key)

        if all_catalogs:
            combined_catalog = pl.concat(all_catalogs, how="diagonal")
            write_track_catalog(combined_catalog, clean_dir)

        self.state.last_file_processed = pending_keys[-1] if pending_keys else ""
        save_state(self.state, state_dir)
        save_checkpoint(self.checkpoint, state_dir)

        elapsed_time = time.time() - start_time

        final_stats = {
            "files_processed": self.stats["files_processed"],
            "total_records_input": self.stats["total_records_input"],
            "total_records_output": self.stats["total_records_output"],
            "records_removed_validation": self.stats["records_removed_validation"],
            "records_removed_outliers": self.stats["records_removed_outliers"],
            "collisions_detected": self.stats["collisions_detected"],
            "unique_mmsis": len(self.stats["unique_mmsis"]),
            "unique_tracks": len(self.stats["unique_tracks"]),
            "elapsed_seconds": elapsed_time,
            "records_per_second": (
                self.stats["total_records_input"] / elapsed_time if elapsed_time > 0 else 0
            ),
        }

        logger.info(f"Processing complete in {elapsed_time:.1f} seconds")
        return final_stats
