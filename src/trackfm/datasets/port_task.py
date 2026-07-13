"""Materialize the port-prediction task dataset (v2: voyage-based).

Pass A discovers ports/anchorages from dwell clusters (see ports.py).
Pass B streams tracks again, splits each into voyages at dwell events, and
extracts 128-position windows per voyage with targets:

  origin       port/anchorage the voyage departed (or ENTRY_* edge)
  destination  port/anchorage it arrives at (or EXIT_* edge)
  remaining_s  seconds from the window's last position to voyage arrival

Only voyages with BOTH labels known produce windows. Windows stream into
~512MB parquet shards (split recorded per row) instead of accumulating in
memory; rerunning an interrupted build skips shards already on disk.

Output layout:
  ports.parquet                     port/anchorage table (pass A)
  port_windows/shard_XXXX.parquet   labeled windows, all splits
  labels.json                       train-split label vocabulary
  MANIFEST.json
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
SHARD_BYTES = 512 * 2**20


class _ShardWriter:
    """Accumulate window rows; flush a parquet shard every ~shard_bytes.

    Deterministic over a fixed input stream: shard boundaries depend only on
    row sizes, so a rerun regenerates identical shards. A shard whose file
    already exists is skipped (buffer dropped, not rewritten) — an
    interrupted build resumes by simply rerunning it. Writes are atomic
    (tmp file + rename), so a kill mid-write never leaves a corrupt shard.
    """

    def __init__(self, dest_dir: Path, sample_floats: int,
                 shard_bytes: int = SHARD_BYTES):
        self.dest_dir = dest_dir
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.sample_floats = sample_floats
        self.shard_bytes = shard_bytes
        self.shards: list[str] = []
        self.rows_per_shard: list[int] = []
        self.skipped = 0
        self._idx = 0
        self._reset()

    def _reset(self) -> None:
        self._rows: dict[str, list] = {
            "features": [], "origin": [], "destination": [],
            "remaining_s": [], "track_id": [], "voyage": [], "split": []}
        self._bytes = 0

    def add(self, features: np.ndarray, origin: str, destination: str,
            remaining_s: float, track_id: str, voyage: int, split: str) -> None:
        r = self._rows
        r["features"].append(features)
        r["origin"].append(origin)
        r["destination"].append(destination)
        r["remaining_s"].append(remaining_s)
        r["track_id"].append(track_id)
        r["voyage"].append(voyage)
        r["split"].append(split)
        self._bytes += features.nbytes + 64
        if self._bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        r = self._rows
        if not r["features"]:
            return
        path = self.dest_dir / f"shard_{self._idx:04d}.parquet"
        self._idx += 1
        self.shards.append(path.name)
        self.rows_per_shard.append(len(r["features"]))
        if path.exists():  # resumed run: shard already materialized
            self.skipped += 1
            self._reset()
            return
        flat = pa.array(np.concatenate(r["features"]), type=pa.float32())
        table = pa.table({
            "features": pa.FixedSizeListArray.from_arrays(flat, self.sample_floats),
            "origin": pa.array(r["origin"]),
            "destination": pa.array(r["destination"]),
            "remaining_s": pa.array(r["remaining_s"], type=pa.float32()),
            "track_id": pa.array(r["track_id"]),
            "voyage": pa.array(r["voyage"], type=pa.int32()),
            "split": pa.array(r["split"]),
        })
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd", compression_level=3)
        tmp.replace(path)
        self._reset()


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
    shard_bytes: int = SHARD_BYTES,
    ports_path: Path | None = None,
) -> None:
    clean_dir = Path(clean_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    days = list_day_partitions(clean_dir)
    if subset_days:
        days = days[:subset_days]
    splits = temporal_split(days, train_frac, val_frac)

    # ---- Pass A: dwells -> port/anchorage table (train days only, no leakage)
    if ports_path is not None:
        ports = pl.read_parquet(Path(ports_path).expanduser())
        if Path(ports_path).expanduser().resolve() != (out_dir / "ports.parquet").resolve():
            ports.write_parquet(out_dir / "ports.parquet")
    else:
        dwells = collect_dwell_centroids(splits["train"], cfg)
        dwells.write_parquet(out_dir / "dwells.parquet")
        ports = discover_ports(dwells, cfg)
        ports.write_parquet(out_dir / "ports.parquet")
    index = PortIndex(ports, cfg)

    # ---- Pass B: voyages -> labeled windows, streamed into shards
    writer = _ShardWriter(out_dir / "port_windows",
                          input_len * len(FEATURE_COLUMNS), shard_bytes)
    stats = {}
    for split_name, split_days in splits.items():
        if not split_days:
            continue
        stats[split_name] = _stream_split(
            split_days, index, cfg, writer, split_name, input_len, stride)
    writer.flush()
    if writer.skipped:
        logger.info(f"resume: {writer.skipped}/{len(writer.shards)} shards already on disk")

    _write_vocab(out_dir, min_class_count)
    manifest = {
        "input_len": input_len, "stride": stride, "features": FEATURE_COLUMNS,
        "label_config": cfg.__dict__,
        "shard_bytes": shard_bytes,
        "shards": writer.shards,
        "rows_per_shard": writer.rows_per_shard,
        "splits": {k: [partition_date(v[0]), partition_date(v[-1])] if v else None
                   for k, v in splits.items()},
        "stats": stats,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"port task dataset -> {out_dir}")


def _stream_split(split_days, index: PortIndex, cfg: PortLabelConfig,
                  writer: _ShardWriter, split_name: str, input_len: int,
                  stride: int) -> dict:
    n_voyages = n_labeled = n_windows = 0
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
                writer.add(
                    vf[s:s + input_len].astype(np.float32).ravel(),
                    origin, dest,
                    max(0.0, v.arrival_epoch - ve[s + input_len - 1]),
                    tid, k, split_name)
                n_windows += 1

    stats = {"voyages": n_voyages, "labeled_voyages": n_labeled,
             "windows": n_windows}
    logger.info(f"[{split_name}] {stats}")
    return stats


def _write_vocab(out_dir: Path, min_class_count: int) -> None:
    shards = sorted((out_dir / "port_windows").glob("shard_*.parquet"))
    if not shards:
        return
    df = (pl.scan_parquet(shards)
          .filter(pl.col("split") == "train")
          .select("origin", "destination").collect())
    vocab = {}
    for col in ("origin", "destination"):
        counts = Counter(df[col].to_list())
        kept = sorted(c for c, n in counts.items() if n >= min_class_count)
        vocab[col] = {name: i for i, name in enumerate(kept + [OTHER])}
    (out_dir / "labels.json").write_text(json.dumps(vocab, indent=2))
    logger.info("vocab: %d origin classes, %d destination classes",
                len(vocab["origin"]), len(vocab["destination"]))
