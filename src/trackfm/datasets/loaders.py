"""Torch datasets for pre-shuffled parquet shards.

Shards are globally pre-shuffled (two-pass shuffle at materialization), so
sequential reads ARE random samples; workers just take disjoint shard
subsets and shuffle shard order per epoch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from trackfm.config import NormalizationConfig
from trackfm.datasets.windowing import (META_FLOATS, NUM_FEATURES,
                                        NUM_FEATURES_V3, normalize_features,
                                        unpack_window_meta)


class ShardedWindowDataset(IterableDataset):
    """Streams normalized (B, window, 6) batches from pre-shuffled shards.

    Yields ready-made batches (DataLoader should use batch_size=None).

    Shard format is autodetected from row width: v1 rows are flat 5-feature
    windows; v3 rows prepend META_FLOATS [t0, mmsi] slots and add a raw
    heading feature. The DEFAULT yield is identical for both formats (the
    model's 6 normalized kinematic features) so all existing training code
    is format-agnostic. v3 extras are opt-in: return_meta=True switches the
    yield to dicts {features, heading_deg, t0, mmsi} for conditioning work
    (heading_deg keeps the raw degrees with -1 = missing).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        batch_size: int = 64,
        window_size: int = 928,
        norm: Optional[NormalizationConfig] = None,
        shuffle_shards: bool = True,
        seed: int = 0,
        return_meta: bool = False,
    ):
        super().__init__()
        self.shard_paths = sorted(Path(shard_dir).expanduser().glob("samples_*.parquet"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No samples_*.parquet under {shard_dir}")
        self.batch_size = batch_size
        self.window_size = window_size
        self.norm = norm or NormalizationConfig()
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0
        self.return_meta = return_meta
        row_floats = pq.ParquetFile(self.shard_paths[0]).schema_arrow[0].type.list_size
        if row_floats == window_size * NUM_FEATURES:
            self.format_version = 1
        elif row_floats == META_FLOATS + window_size * NUM_FEATURES_V3:
            self.format_version = 3
        else:
            raise ValueError(f"Unrecognized shard row width {row_floats} "
                             f"for window_size={window_size}")
        if return_meta and self.format_version < 3:
            raise ValueError("return_meta requires v3 shards (this dir is v1)")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _my_shards(self) -> list[Path]:
        info = get_worker_info()
        shards = list(self.shard_paths)
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(shards)
        if info is None:
            return shards
        return shards[info.id::info.num_workers]

    def __iter__(self) -> Iterator:
        n = self.norm
        v3 = self.format_version >= 3
        for shard in self._my_shards():
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=self.batch_size, columns=["features"]):
                flat = np.asarray(batch.column("features").flatten(), dtype=np.float32)
                if v3:
                    rows = flat.reshape(len(batch), -1)
                    meta, body = rows[:, :META_FLOATS], rows[:, META_FLOATS:]
                    raw6 = body.reshape(-1, self.window_size, NUM_FEATURES_V3)
                    raw = raw6[..., :NUM_FEATURES]
                else:
                    raw = flat.reshape(-1, self.window_size, NUM_FEATURES)
                feats = normalize_features(
                    np.ascontiguousarray(raw),
                    lat_center=n.lat_center, lat_scale=n.lat_scale,
                    lon_center=n.lon_center, lon_scale=n.lon_scale,
                    sog_scale=n.sog_scale, dt_scale=n.dt_scale,
                )
                if not self.return_meta:
                    yield torch.from_numpy(feats)
                    continue
                t0, mmsi = unpack_window_meta(meta)
                yield {
                    "features": torch.from_numpy(feats),
                    "heading_deg": torch.from_numpy(
                        np.ascontiguousarray(raw6[..., NUM_FEATURES])),
                    "t0": torch.from_numpy(t0),
                    "mmsi": torch.from_numpy(mmsi),
                }


def num_samples(shard_dir: str | Path, pattern: str = "samples_*.parquet") -> int:
    """Total rows across shards (parquet metadata only — fast)."""
    return sum(
        pq.ParquetFile(p).metadata.num_rows
        for p in sorted(Path(shard_dir).expanduser().glob(pattern))
    )
