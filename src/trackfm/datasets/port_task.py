"""Materialize the port-prediction task dataset.

For every labeled track (origin + destination known), extracts 128-position
input windows and attaches three targets:

  origin       -- where the vessel came from (port name or ENTRY_* edge)
  destination  -- where it is going (port name or EXIT_* edge)
  remaining_s  -- seconds from the window's last position until track end
                  (time-to-arrival when the destination is a port)

Windows near the very end of a track are kept — predicting "almost there"
is part of the ETA task. Output: per-split parquet shards + labels.json
vocabulary (built from the train split; rare classes -> OTHER).
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from trackfm.datasets.materialize import list_day_partitions, partition_date, temporal_split
from trackfm.datasets.windowing import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

OTHER = "OTHER"


def build_port_task_dataset(
    clean_dir: Path,
    labeled_tracks: pl.DataFrame,
    out_dir: Path,
    input_len: int = 128,
    stride: int = 64,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    min_class_count: int = 50,
    subset_days: int | None = None,
) -> None:
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled = labeled_tracks.filter(
        pl.col("origin").is_not_null() & pl.col("destination").is_not_null())
    meta = {
        r["track_id"]: (r["origin"], r["destination"], r["end_ts"])
        for r in labeled.iter_rows(named=True)
    }
    logger.info(f"{len(meta):,} fully-labeled tracks")

    days = list_day_partitions(Path(clean_dir).expanduser())
    if subset_days:
        days = days[:subset_days]
    splits = temporal_split(days, train_frac, val_frac)

    for split_name, split_days in splits.items():
        if not split_days:
            continue
        _write_split(split_days, meta, out_dir, split_name, input_len, stride)

    _write_vocab(out_dir, min_class_count)
    manifest = {
        "input_len": input_len, "stride": stride,
        "features": FEATURE_COLUMNS,
        "splits": {k: [partition_date(v[0]), partition_date(v[-1])] if v else None
                   for k, v in splits.items()},
        "n_labeled_tracks": len(meta),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"port task dataset -> {out_dir}")


def _write_split(split_days, meta, out_dir: Path, split_name: str,
                 input_len: int, stride: int) -> None:
    """Stream days, buffer labeled tracks, window on completion."""
    buffers: dict[str, list] = defaultdict(list)
    last_seen: dict[str, str] = {}
    rows: dict[str, list] = {"features": [], "origin": [], "destination": [],
                             "remaining_s": [], "track_id": []}

    def finalize(tid: str) -> None:
        chunks = buffers.pop(tid)
        last_seen.pop(tid, None)
        origin, dest, end_ts = meta[tid]
        feats = np.vstack([c[0] for c in chunks])
        ts = np.concatenate([c[1] for c in chunks])
        if len(feats) < input_len:
            return
        end_epoch = end_ts.timestamp()
        for start in range(0, len(feats) - input_len + 1, stride):
            w = feats[start:start + input_len]
            rows["features"].append(w.astype(np.float32).ravel())
            rows["origin"].append(origin)
            rows["destination"].append(dest)
            rows["remaining_s"].append(max(0.0, end_epoch - ts[start + input_len - 1]))
            rows["track_id"].append(tid)

    for day_path in tqdm(split_days, desc=f"port-task[{split_name}]"):
        day = partition_date(day_path)
        df = pl.read_parquet(day_path,
                             columns=["track_id", "timestamp", *FEATURE_COLUMNS])
        df = df.filter(pl.col("track_id").is_in(list(meta))).sort(
            ["track_id", "timestamp"])
        for track_df in df.partition_by("track_id"):
            tid = track_df.item(0, "track_id")
            feats = np.nan_to_num(
                track_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32), nan=0.0)
            epochs = track_df["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.float64)
            buffers[tid].append((feats, epochs))
            last_seen[tid] = day
        for tid in [t for t, d in last_seen.items() if d != day]:
            finalize(tid)
    for tid in list(buffers):
        finalize(tid)

    if not rows["features"]:
        logger.warning(f"[{split_name}] no windows")
        return
    n = len(rows["features"])
    flat = pa.array(np.concatenate(rows["features"]), type=pa.float32())
    table = pa.table({
        "features": pa.FixedSizeListArray.from_arrays(flat, input_len * len(FEATURE_COLUMNS)),
        "origin": pa.array(rows["origin"]),
        "destination": pa.array(rows["destination"]),
        "remaining_s": pa.array(rows["remaining_s"], type=pa.float32()),
        "track_id": pa.array(rows["track_id"]),
    })
    dest_dir = out_dir / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest_dir / "windows.parquet",
                   compression="zstd", compression_level=3)
    logger.info(f"[{split_name}] {n:,} windows written")


def _write_vocab(out_dir: Path, min_class_count: int) -> None:
    """Class vocabularies from the train split; rare classes -> OTHER."""
    train = out_dir / "train" / "windows.parquet"
    if not train.exists():
        return
    df = pl.read_parquet(train, columns=["origin", "destination"])
    vocab = {}
    for col in ("origin", "destination"):
        counts = Counter(df[col].to_list())
        kept = sorted(c for c, n in counts.items() if n >= min_class_count)
        vocab[col] = {name: i for i, name in enumerate(kept + [OTHER])}
    (out_dir / "labels.json").write_text(json.dumps(vocab, indent=2))
    logger.info("vocab: %d origin classes, %d destination classes",
                len(vocab["origin"]), len(vocab["destination"]))
