"""Materialize the port-prediction task dataset (v2: voyage-based).

Pass A discovers ports/anchorages from dwell clusters (see ports.py).
Pass B streams tracks again, splits each into voyages at dwell events, and
extracts 128-position windows per voyage with targets:

  origin       port/anchorage the voyage departed (or ENTRY_* edge)
  destination  port/anchorage it arrives at (or EXIT_* edge)
  remaining_s  seconds from the window's last position to voyage arrival

Only voyages with BOTH labels known produce windows. Output: per-split
parquet + labels.json vocabulary (train split; rare classes -> OTHER).
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from trackfm.datasets.materialize import list_day_partitions, partition_date, temporal_split
from trackfm.datasets.ports import (
    PortIndex, PortLabelConfig, collect_dwell_centroids, discover_ports,
    split_voyages, stream_tracks,
)
from trackfm.datasets.windowing import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

OTHER = "OTHER"


def build_port_dataset(
    clean_dir: Path,
    out_dir: Path,
    cfg: PortLabelConfig = PortLabelConfig(),
    input_len: int = 128,
    stride: int = 64,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    min_class_count: int = 50,
    subset_days: int | None = None,
) -> None:
    clean_dir = Path(clean_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    days = list_day_partitions(clean_dir)
    if subset_days:
        days = days[:subset_days]
    splits = temporal_split(days, train_frac, val_frac)

    # ---- Pass A: dwells -> port/anchorage table (train days only, no leakage)
    dwells = collect_dwell_centroids(splits["train"], cfg)
    dwells.write_parquet(out_dir / "dwells.parquet")
    ports = discover_ports(dwells, cfg)
    ports.write_parquet(out_dir / "ports.parquet")
    index = PortIndex(ports, cfg)

    # ---- Pass B: voyages -> labeled windows, per split
    stats = {}
    for split_name, split_days in splits.items():
        if not split_days:
            continue
        stats[split_name] = _write_split(
            split_days, index, cfg, out_dir, split_name, input_len, stride)

    _write_vocab(out_dir, min_class_count)
    manifest = {
        "input_len": input_len, "stride": stride, "features": FEATURE_COLUMNS,
        "label_config": cfg.__dict__,
        "splits": {k: [partition_date(v[0]), partition_date(v[-1])] if v else None
                   for k, v in splits.items()},
        "stats": stats,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"port task dataset -> {out_dir}")


def _write_split(split_days, index: PortIndex, cfg: PortLabelConfig, out_dir: Path,
                 split_name: str, input_len: int, stride: int) -> dict:
    rows = {"features": [], "origin": [], "destination": [],
            "remaining_s": [], "track_id": [], "voyage": []}
    n_voyages = n_labeled = 0

    for tid, feats, epochs in stream_tracks(split_days, cfg.min_track_positions):
        for k, v in enumerate(split_voyages(tid, feats, epochs, cfg)):
            n_voyages += 1
            origin, dest = index.label_voyage(v, feats)
            if origin is None or dest is None:
                continue
            n_labeled += 1
            vf = feats[v.start:v.end]
            ve = epochs[v.start:v.end]
            for s in range(0, len(vf) - input_len + 1, stride):
                w = vf[s:s + input_len]
                rows["features"].append(w.astype(np.float32).ravel())
                rows["origin"].append(origin)
                rows["destination"].append(dest)
                rows["remaining_s"].append(
                    max(0.0, v.arrival_epoch - ve[s + input_len - 1]))
                rows["track_id"].append(tid)
                rows["voyage"].append(k)

    stats = {"voyages": n_voyages, "labeled_voyages": n_labeled,
             "windows": len(rows["features"])}
    logger.info(f"[{split_name}] {stats}")
    if not rows["features"]:
        return stats

    flat = pa.array(np.concatenate(rows["features"]), type=pa.float32())
    table = pa.table({
        "features": pa.FixedSizeListArray.from_arrays(flat, input_len * len(FEATURE_COLUMNS)),
        "origin": pa.array(rows["origin"]),
        "destination": pa.array(rows["destination"]),
        "remaining_s": pa.array(rows["remaining_s"], type=pa.float32()),
        "track_id": pa.array(rows["track_id"]),
        "voyage": pa.array(rows["voyage"], type=pa.int32()),
    })
    dest_dir = out_dir / split_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest_dir / "windows.parquet",
                   compression="zstd", compression_level=3)
    return stats


def _write_vocab(out_dir: Path, min_class_count: int) -> None:
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
