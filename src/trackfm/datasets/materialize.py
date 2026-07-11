"""Materialize pre-shuffled training shards from cleaned day-partitioned parquet.

Local-filesystem port of the legacy two-pass "Jane Street" shuffle
(ais-analysis history, materialize_samples.py @ a9d516f):

  Pass 0  Assign every track to a temporal split (train/val/test) by its
          START date, so no window leaks across the split boundary.
  Pass 1  Stream day partitions in date order, buffer rows per track until a
          track is complete (a day boundary with a newer day present, or end
          of data), window it, and append each window to a random pile file.
  Pass 2  Shuffle each pile in memory (memmap) and write one zstd parquet
          shard per pile: column `features`, FixedSizeList<float32>[4640].

Reading any output shard sequentially yields random samples from the whole
dataset — no shuffling needed at training time.

Resumable: checkpoint JSON in the output dir tracks completed days/piles.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from trackfm.config import MaterializeConfig
from trackfm.datasets.windowing import FEATURE_COLUMNS, extract_windows_from_track

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers
def list_day_partitions(clean_dir: Path) -> list[Path]:
    """All day partition files, sorted temporally (year=/month=/day= layout)."""
    return sorted(clean_dir.glob("year=*/month=*/day=*/tracks.parquet"))


def partition_date(p: Path) -> str:
    """'.../year=2024/month=03/day=15/tracks.parquet' -> '2024-03-15'."""
    parts = {kv.split("=")[0]: kv.split("=")[1] for kv in p.parent.parts[-3:] if "=" in kv}
    return f"{parts['year']}-{parts['month']}-{parts['day']}"


def temporal_split(days: list[Path], train_frac: float, val_frac: float):
    """Split day files temporally: first train_frac -> train, then val, then test."""
    n = len(days)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": days[:n_train],
        "val": days[n_train:n_train + n_val],
        "test": days[n_train + n_val:],
    }


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_checkpoint(path: Path, ckpt: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt))
    tmp.replace(path)


# ------------------------------------------------------------------- pass 1
def pass1_extract(cfg: MaterializeConfig, split_days: list[Path], out_dir: Path,
                  rng: np.random.Generator, num_piles: int, ckpt_path: Path,
                  split_name: str) -> None:
    """Stream day partitions, window completed tracks, distribute to piles.

    Tracks span consecutive days, so rows are buffered per track_id and a
    track is finalized once a day arrives that no longer contains it (tracks
    are gap-segmented at 4h, so one missing day ends the track) or when the
    split's day list is exhausted.
    """
    ckpt = _load_checkpoint(ckpt_path)
    done_key = f"pass1_{split_name}_done_days"
    done_days = set(ckpt.get(done_key, []))
    if done_days:
        logger.info(f"[{split_name}] resuming pass 1: {len(done_days)} days already done")

    temp_dir = out_dir / f"_piles_{split_name}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pile_files = [open(temp_dir / f"pile_{i:03d}.bin", "ab") for i in range(num_piles)]

    buffers: dict[str, list[np.ndarray]] = defaultdict(list)
    last_seen: dict[str, str] = {}

    def finalize_track(track_id: str) -> int:
        chunks = buffers.pop(track_id)
        last_seen.pop(track_id, None)
        feats = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        if len(feats) < cfg.min_track_length:
            return 0
        windows = extract_windows_from_track(feats, cfg.window_size, cfg.stride)
        if len(windows) == 0:
            return 0
        assignment = rng.integers(0, num_piles, size=len(windows))
        for pile_id in np.unique(assignment):
            sel = windows[assignment == pile_id]
            pile_files[pile_id].write(np.ascontiguousarray(sel).tobytes())
        return len(windows)

    total_windows = 0
    try:
        for day_path in tqdm(split_days, desc=f"pass1[{split_name}]"):
            day = partition_date(day_path)
            if day in done_days:
                continue

            df = pl.read_parquet(
                day_path, columns=["track_id", "timestamp", *FEATURE_COLUMNS]
            )
            if cfg.min_sog_knots > 0:
                # NOTE: drops individual slow positions; dt_seconds is NOT
                # recomputed (see docs/MIGRATION.md). Legacy materialization
                # applied no SOG filter — keep 0.0 to reproduce it.
                df = df.filter(pl.col("sog") >= cfg.min_sog_knots)
            df = df.sort(["track_id", "timestamp"])
            day_tracks = set()
            for track_df in df.partition_by("track_id"):
                tid = track_df.item(0, "track_id")
                day_tracks.add(tid)
                feats = (
                    track_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
                )
                buffers[tid].append(np.nan_to_num(feats, nan=0.0))
                last_seen[tid] = day

            # Finalize tracks not present in this day (they've ended)
            for tid in [t for t, d in last_seen.items() if d != day]:
                total_windows += finalize_track(tid)

            done_days.add(day)
            ckpt[done_key] = sorted(done_days)
            _save_checkpoint(ckpt_path, ckpt)

        # End of split: flush all remaining buffers
        for tid in list(buffers):
            total_windows += finalize_track(tid)
    finally:
        for f in pile_files:
            f.close()

    logger.info(f"[{split_name}] pass 1 complete: {total_windows:,} windows")


# ------------------------------------------------------------------- pass 2
def pass2_shuffle(cfg: MaterializeConfig, out_dir: Path, rng: np.random.Generator,
                  num_piles: int, ckpt_path: Path, split_name: str) -> None:
    """Shuffle each pile and write final parquet shards."""
    ckpt = _load_checkpoint(ckpt_path)
    done_key = f"pass2_{split_name}_done_piles"
    done = set(ckpt.get(done_key, []))

    temp_dir = out_dir / f"_piles_{split_name}"
    dest_dir = out_dir / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    sample_floats = cfg.window_size * len(FEATURE_COLUMNS)
    bytes_per_sample = sample_floats * 4

    for pile_id in tqdm(range(num_piles), desc=f"pass2[{split_name}]"):
        if pile_id in done:
            continue
        pile_path = temp_dir / f"pile_{pile_id:03d}.bin"
        if not pile_path.exists() or pile_path.stat().st_size == 0:
            pile_path.unlink(missing_ok=True)
            done.add(pile_id)
            continue

        num_samples = pile_path.stat().st_size // bytes_per_sample
        data = np.memmap(pile_path, dtype=np.float32, mode="r",
                         shape=(num_samples, sample_floats))
        order = rng.permutation(num_samples)
        shuffled = np.ascontiguousarray(data[order])

        flat = pa.array(shuffled.ravel(), type=pa.float32())
        table = pa.table({
            "features": pa.FixedSizeListArray.from_arrays(flat, sample_floats)
        })
        pq.write_table(table, dest_dir / f"samples_{pile_id:03d}.parquet",
                       compression="zstd", compression_level=3)

        del data, shuffled
        pile_path.unlink()

        done.add(pile_id)
        ckpt[done_key] = sorted(done)
        _save_checkpoint(ckpt_path, ckpt)

    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    logger.info(f"[{split_name}] pass 2 complete")


# -------------------------------------------------------------------- entry
def materialize_dataset(cfg: MaterializeConfig, subset_days: int | None = None) -> None:
    """Full materialization: temporal split -> pass 1 -> pass 2, per split."""
    clean_dir = cfg.clean_dir
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "materialize_checkpoint.json"

    days = list_day_partitions(clean_dir)
    if not days:
        raise FileNotFoundError(f"No day partitions under {clean_dir}")
    if subset_days:
        days = days[:subset_days]

    splits = temporal_split(days, cfg.train_frac, cfg.val_frac)
    logger.info(
        f"{len(days)} days -> train {len(splits['train'])}, "
        f"val {len(splits['val'])}, test {len(splits['test'])}"
    )

    # val/test are small: a handful of piles is plenty
    piles_per_split = {"train": cfg.num_output_shards, "val": 8, "test": 8}

    rng = np.random.default_rng(cfg.seed)
    for split_name, split_days in splits.items():
        if not split_days:
            continue
        pass1_extract(cfg, split_days, out_dir, rng, piles_per_split[split_name],
                      ckpt_path, split_name)
        pass2_shuffle(cfg, out_dir, rng, piles_per_split[split_name],
                      ckpt_path, split_name)

    manifest = {
        "clean_dir": str(clean_dir),
        "num_days": len(days),
        "first_day": partition_date(days[0]),
        "last_day": partition_date(days[-1]),
        "splits": {k: [partition_date(v[0]), partition_date(v[-1])] if v else None
                   for k, v in splits.items()},
        "config": cfg.model_dump(mode="json"),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"Materialization complete -> {out_dir}")
