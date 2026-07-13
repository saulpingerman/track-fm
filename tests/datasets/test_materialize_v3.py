"""v3 materialization: per-window [t0, mmsi] meta + heading feature.

Covers the exact float32 meta encoding, the v3 pass1/pass2 roundtrip
against synthetic clean partitions, heading sentinel handling, and the
guarantee that v1 and v3 shards feed the model IDENTICAL normalized
features (training code is format-agnostic).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from trackfm.config import MaterializeConfig
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.datasets.materialize import floats_per_sample, materialize_dataset
from trackfm.datasets.windowing import (HEADING_MISSING, META_FLOATS,
                                        pack_window_meta, unpack_window_meta)

WINDOW, STRIDE = 8, 2


def test_meta_pack_roundtrip_is_exact():
    t0 = np.array([1672531200, 1798761599, 1735689600], dtype=np.int64)  # 2023..2026
    for mmsi in (999_999_999, 219_000_001, 1):
        packed = pack_window_meta(t0, mmsi)
        assert packed.dtype == np.float32 and packed.shape == (3, META_FLOATS)
        t0_out, mmsi_out = unpack_window_meta(packed)
        np.testing.assert_array_equal(t0_out, t0)
        assert (mmsi_out == mmsi).all()


def _write_clean_days(root, n_days=5, positions_per_day=40, mmsi=219_012_345):
    """Synthetic day partitions in the clean schema (one track per day)."""
    base = datetime(2024, 3, 1)
    for d in range(n_days):
        day = base + timedelta(days=d)
        ts = [day + timedelta(seconds=30 * i) for i in range(positions_per_day)]
        heading = [(90 + i) % 360 for i in range(positions_per_day)]
        heading[3] = 511                       # AIS "not available"
        df = pl.DataFrame({
            "track_id": [f"trk_{d}"] * positions_per_day,
            "timestamp": ts,
            "mmsi": [mmsi + d] * positions_per_day,
            "lat": np.linspace(55.0, 55.1, positions_per_day),
            "lon": np.linspace(11.0, 11.1, positions_per_day),
            "sog": np.full(positions_per_day, 10.0),
            "cog": np.linspace(0, 90, positions_per_day),
            "heading": heading,
            "dt_seconds": np.full(positions_per_day, 30.0),
        }).with_columns(pl.col("heading").cast(pl.Int64),
                        pl.col("timestamp").cast(pl.Datetime("us")))
        out = root / f"year=2024/month=03/day={d + 1:02d}"
        out.mkdir(parents=True)
        df.write_parquet(out / "tracks.parquet")


def _cfg(clean, out, version):
    return MaterializeConfig(
        clean_dir=clean, out_dir=out, window_size=WINDOW, stride=STRIDE,
        min_track_length=WINDOW, num_output_shards=8, format_version=version,
        train_frac=0.8, val_frac=0.0, seed=7, num_workers=1)


def test_v3_roundtrip_meta_and_heading(tmp_path):
    _write_clean_days(tmp_path / "clean")
    cfg = _cfg(tmp_path / "clean", tmp_path / "v3", 3)
    materialize_dataset(cfg)

    ds = ShardedWindowDataset(tmp_path / "v3" / "train", batch_size=64,
                              window_size=WINDOW, shuffle_shards=False,
                              return_meta=True)
    assert ds.format_version == 3
    batch = next(iter(ds))
    t0, mmsi = batch["t0"].numpy(), batch["mmsi"].numpy()
    # every t0 must be an exact synthetic timestamp: day start + 30s * (2k)
    base = np.int64(datetime(2024, 3, 1).timestamp())
    rel = (t0 - base) % 86400
    assert ((rel % 30) == 0).all() and ((rel // 30) % STRIDE == 0).all()
    assert set(np.unique(mmsi)) <= {219_012_345 + d for d in range(5)}
    # windows overlapping position 3 carry the 511 -> -1 sentinel
    h = batch["heading_deg"].numpy()
    assert (h == HEADING_MISSING).any() and h.max() < 360 and h.min() >= -1
    # t0 must agree with the mmsi's day (same track => same day partition)
    assert ((t0 - base) // 86400 == mmsi - 219_012_345).all()


def test_v1_and_v3_feed_identical_model_features(tmp_path):
    _write_clean_days(tmp_path / "clean")
    materialize_dataset(_cfg(tmp_path / "clean", tmp_path / "v1", 1))
    materialize_dataset(_cfg(tmp_path / "clean", tmp_path / "v3b", 3))

    def all_feats(d):
        ds = ShardedWindowDataset(d, batch_size=1024, window_size=WINDOW,
                                  shuffle_shards=False)
        return torch.cat(list(ds))

    f1, f3 = all_feats(tmp_path / "v1" / "train"), all_feats(tmp_path / "v3b" / "train")
    assert f1.shape == f3.shape and f1.shape[-1] == 6
    # same seed => same pile assignment & shuffle => elementwise identical
    torch.testing.assert_close(f1, f3, rtol=0, atol=0)


def test_v1_loader_rejects_return_meta(tmp_path):
    _write_clean_days(tmp_path / "clean")
    cfg = _cfg(tmp_path / "clean", tmp_path / "v1b", 1)
    materialize_dataset(cfg)
    assert floats_per_sample(cfg) == WINDOW * 5
    with pytest.raises(ValueError, match="return_meta"):
        ShardedWindowDataset(tmp_path / "v1b" / "train", window_size=WINDOW,
                             return_meta=True)
