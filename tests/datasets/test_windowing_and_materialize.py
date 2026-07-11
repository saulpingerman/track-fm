"""Windowing, normalization, materialization round-trip, and loader tests."""
import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from trackfm.config import MaterializeConfig, NormalizationConfig
from trackfm.datasets.loaders import ShardedWindowDataset, num_samples
from trackfm.datasets.materialize import materialize_dataset
from trackfm.datasets.windowing import (
    FEATURE_COLUMNS,
    extract_windows_from_track,
    normalize_features,
)


# ---------------------------------------------------------------- windowing
def test_window_shape_and_stride():
    feats = np.arange(1000 * 5, dtype=np.float32).reshape(1000, 5)
    w = extract_windows_from_track(feats, window_size=928, stride=32)
    assert w.shape == ((1000 - 928) // 32 + 1, 928, 5)
    np.testing.assert_array_equal(w[0], feats[:928])
    np.testing.assert_array_equal(w[1], feats[32:960])


def test_window_too_short_track():
    w = extract_windows_from_track(np.zeros((927, 5), dtype=np.float32))
    assert w.shape == (0, 928, 5)


def test_normalize_matches_hyperparameters_md():
    raw = np.array([[[56.25, 11.5, 15.0, 90.0, 300.0]]], dtype=np.float32)
    out = normalize_features(raw)
    np.testing.assert_allclose(
        out[0, 0], [0.0, 0.0, 0.5, 1.0, 0.0, 1.0], atol=1e-6
    )  # lat/lon centered, sog/30, sin90=1, cos90=0, dt/300


# ------------------------------------------------- materialize + loader e2e
@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Synthetic clean/ tree: 10 days, a few long tracks per day."""
    root = tmp_path_factory.mktemp("clean")
    rng = np.random.default_rng(0)
    t0 = datetime(2024, 1, 1)
    for d in range(10):
        day = t0 + timedelta(days=d)
        rows = []
        for v in range(3):
            n = 1100  # enough for multiple windows
            ts = [day + timedelta(seconds=30 * i) for i in range(n)]
            rows.append(pl.DataFrame({
                "timestamp": ts,
                "track_id": [f"m{v}_{d}"] * n,
                "lat": (56.0 + rng.normal(0, 0.01, n)).astype(np.float64),
                "lon": (11.0 + rng.normal(0, 0.01, n)).astype(np.float64),
                "sog": np.full(n, 10.0),
                "cog": np.full(n, 45.0),
                "dt_seconds": np.full(n, 30.0, dtype=np.float32),
            }))
        df = pl.concat(rows)
        out = root / f"year=2024/month=01/day={d+1:02d}"
        out.mkdir(parents=True)
        df.write_parquet(out / "tracks.parquet")
    return root


def test_materialize_end_to_end(tiny_dataset, tmp_path):
    out_dir = tmp_path / "mat"
    cfg = MaterializeConfig(
        clean_dir=tiny_dataset, out_dir=out_dir,
        num_output_shards=4, seed=7,
    )
    materialize_dataset(cfg)

    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    assert manifest["num_days"] == 10
    assert manifest["splits"]["train"] == ["2024-01-01", "2024-01-08"]

    # 8 train days * 3 tracks * 6 windows each ((1100-928)//32+1)
    expected_train = 8 * 3 * 6
    assert num_samples(out_dir / "train") == expected_train
    assert num_samples(out_dir / "val") == 1 * 3 * 6
    assert num_samples(out_dir / "test") == 1 * 3 * 6

    # temp piles cleaned up
    assert not list(out_dir.glob("_piles_*/*"))


def test_loader_batches(tiny_dataset, tmp_path):
    out_dir = tmp_path / "mat2"
    cfg = MaterializeConfig(clean_dir=tiny_dataset, out_dir=out_dir,
                            num_output_shards=2, seed=7)
    materialize_dataset(cfg)

    ds = ShardedWindowDataset(out_dir / "train", batch_size=16)
    batch = next(iter(ds))
    assert batch.shape == (16, 928, 6)
    assert batch.dtype == torch.float32
    # normalized lat centered near (56.0-56.25)/1.0 = -0.25
    assert abs(batch[..., 0].mean().item() + 0.25) < 0.05
    # cog 45deg -> sin == cos
    torch.testing.assert_close(batch[..., 3], batch[..., 4], atol=1e-5, rtol=1e-5)

    total = sum(b.shape[0] for b in ds)
    assert total == num_samples(out_dir / "train")
