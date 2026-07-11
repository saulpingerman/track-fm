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
from trackfm.datasets.windowing import NUM_FEATURES, normalize_features


class ShardedWindowDataset(IterableDataset):
    """Streams normalized (B, window, 6) batches from pre-shuffled shards.

    Yields ready-made batches (DataLoader should use batch_size=None).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        batch_size: int = 64,
        window_size: int = 928,
        norm: Optional[NormalizationConfig] = None,
        shuffle_shards: bool = True,
        seed: int = 0,
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

    def __iter__(self) -> Iterator[torch.Tensor]:
        n = self.norm
        for shard in self._my_shards():
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=self.batch_size, columns=["features"]):
                flat = np.asarray(batch.column("features").flatten(), dtype=np.float32)
                raw = flat.reshape(-1, self.window_size, NUM_FEATURES)
                feats = normalize_features(
                    raw,
                    lat_center=n.lat_center, lat_scale=n.lat_scale,
                    lon_center=n.lon_center, lon_scale=n.lon_scale,
                    sog_scale=n.sog_scale, dt_scale=n.dt_scale,
                )
                yield torch.from_numpy(feats)


def num_samples(shard_dir: str | Path, pattern: str = "samples_*.parquet") -> int:
    """Total rows across shards (parquet metadata only — fast)."""
    return sum(
        pq.ParquetFile(p).metadata.num_rows
        for p in sorted(Path(shard_dir).expanduser().glob(pattern))
    )
