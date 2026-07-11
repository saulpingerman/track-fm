"""Voyage splitting / port discovery / task-dataset tests on synthetic tracks."""
import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from trackfm.datasets.port_task import build_port_dataset
from trackfm.datasets.ports import (
    BBOX, PortIndex, PortLabelConfig, discover_ports, find_dwells, split_voyages,
)

# Fake ports: A matches a fake OSM registry entry, B does not (anchorage)
PORT_A = (56.00, 10.00)
PORT_B = (56.80, 11.50)
REGISTRY = pl.DataFrame({"name": ["testhavn"], "lat": [PORT_A[0]], "lon": [PORT_A[1]]})
T0 = datetime(2024, 1, 1)

CFG = PortLabelConfig(min_samples=5, min_track_positions=50, min_voyage_positions=50)


def leg(src, dst, n, sog=12.0):
    f = np.linspace(0, 1, n)
    lat = src[0] + (dst[0] - src[0]) * f
    lon = src[1] + (dst[1] - src[1]) * f
    return np.c_[lat, lon, np.full(n, sog), np.full(n, 45.0), np.full(n, 60.0)]


def dwell(at, n, jitter=1e-4):
    rng = np.random.default_rng(0)
    lat = at[0] + rng.normal(0, jitter, n)
    lon = at[1] + rng.normal(0, jitter, n)
    return np.c_[lat, lon, np.zeros(n), np.full(n, 45.0), np.full(n, 60.0)]


def multi_leg_track(day):
    """A -> (dwell 2h at B) -> back to A: one track, TWO voyages."""
    parts = [leg(PORT_A, PORT_B, 200), dwell(PORT_B, 120), leg(PORT_B, PORT_A, 200)]
    feats = np.vstack(parts).astype(np.float32)
    ts0 = T0 + timedelta(days=day)
    epochs = np.arange(len(feats)) * 60.0
    return feats, epochs, ts0


def track_frame(tid, feats, ts0):
    ts = [ts0 + timedelta(seconds=60 * i) for i in range(len(feats))]
    return pl.DataFrame({
        "timestamp": ts, "track_id": [tid] * len(feats),
        "lat": feats[:, 0].astype(np.float64), "lon": feats[:, 1].astype(np.float64),
        "sog": feats[:, 2].astype(np.float64), "cog": feats[:, 3].astype(np.float64),
        "dt_seconds": feats[:, 4].astype(np.float32),
    })


# --------------------------------------------------------------- unit level
def test_find_dwells():
    feats, epochs, _ = multi_leg_track(0)
    d = find_dwells(feats, epochs, stop_sog=0.5, min_dwell_s=3600)
    assert len(d) == 1
    a, b = d[0]
    assert (a, b) == (200, 320)  # exactly the parked segment


def test_split_voyages_multi_leg():
    feats, epochs, _ = multi_leg_track(0)
    v = split_voyages("t", feats, epochs, CFG)
    assert len(v) == 2
    # first voyage: no preceding dwell, arrives at B's dwell
    assert v[0].origin_pos is None and v[0].dest_pos is not None
    assert abs(v[0].dest_pos[0] - PORT_B[0]) < 0.01
    assert v[0].arrival_epoch == epochs[200]
    # second voyage departs B's dwell, ends the track
    assert v[1].origin_pos is not None and v[1].ends_track


# ------------------------------------------------------------ e2e materialize
def full_roundtrip_track(day):
    """dwell@A -> voyage to B -> dwell@B -> voyage back -> dwell@A.

    Both voyages have labeled origin AND destination dwells.
    """
    parts = [dwell(PORT_A, 90), leg(PORT_A, PORT_B, 200), dwell(PORT_B, 120),
             leg(PORT_B, PORT_A, 200), dwell(PORT_A, 90)]
    feats = np.vstack(parts).astype(np.float32)
    return feats, T0 + timedelta(days=day)


@pytest.fixture(scope="module")
def clean_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean")
    for day in range(10):
        frames = []
        for k in range(6):
            feats, ts0 = full_roundtrip_track(day)
            frames.append(track_frame(f"m{k}_{day}", feats, ts0))
        df = pl.concat(frames)
        out = root / f"year=2024/month=01/day={day+1:02d}"
        out.mkdir(parents=True)
        df.write_parquet(out / "tracks.parquet")
    return root


@pytest.fixture(scope="module")
def built(clean_tree, tmp_path_factory, monkeypatch=None):
    out = tmp_path_factory.mktemp("ports_ds")
    import trackfm.datasets.ports as P
    orig = P.load_registry
    P.load_registry = lambda path=None: REGISTRY
    try:
        build_port_dataset(clean_tree, out, cfg=CFG, input_len=128, stride=64,
                           min_class_count=1)
    finally:
        P.load_registry = orig
    return out


def test_registry_vs_anchorage(built):
    ports = pl.read_parquet(built / "ports.parquet")
    assert ports.height == 2
    kinds = dict(zip(ports["name"].to_list(), ports["kind"].to_list()))
    assert kinds.get("testhavn") == "port"          # matched fake OSM registry
    anch = [n for n, k in kinds.items() if k == "anchorage"]
    assert len(anch) == 1                            # PORT_B unmatched -> anchorage


def test_voyage_labels_and_eta(built):
    df = pl.read_parquet(built / "train" / "windows.parquet")
    assert df.height > 0
    # outbound voyage: port A (testhavn) -> anchorage@B; return: reverse
    out_v = df.filter(pl.col("origin") == "testhavn")
    back_v = df.filter(pl.col("destination") == "testhavn")
    assert out_v.height > 0 and back_v.height > 0
    assert out_v["destination"].unique().to_list()[0].startswith("anchorage")
    # remaining_s decreases across consecutive windows of one voyage
    one = df.filter((pl.col("track_id") == "m0_0") & (pl.col("voyage") == 0))
    assert one.height >= 2
    r = one["remaining_s"].to_list()
    assert all(r[i] > r[i + 1] for i in range(len(r) - 1))


def test_vocab_written(built):
    vocab = json.loads((built / "labels.json").read_text())
    assert set(vocab) == {"origin", "destination"}
    assert all("OTHER" in v for v in vocab.values())


def test_heads_forward():
    import torch

    from trackfm.config import ModelConfig
    from trackfm.models.factory import build_model
    from trackfm.models.heads import EtaRegressor, PortClassifier

    enc = build_model(ModelConfig(d_model=32, nhead=2, num_layers=1,
                                  dim_feedforward=64, grid_size=16, num_freqs=3))
    x = torch.randn(4, 128, 6) * 0.1
    clf = PortClassifier(enc, num_classes=7, pooling="mean")
    assert clf(x).shape == (4, 7)
    reg = EtaRegressor(enc, pooling="last")
    out = reg(x)
    assert out.shape == (4,)
    assert (EtaRegressor.to_seconds(out) >= 0).all()
