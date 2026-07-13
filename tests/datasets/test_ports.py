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


def _read_split(built, split):
    shards = sorted((built / "port_windows").glob("shard_*.parquet"))
    return pl.scan_parquet(shards).filter(pl.col("split") == split).collect()


def test_voyage_labels_and_eta(built):
    df = _read_split(built, "train")
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


# ----------------------------------------------------- binned DBSCAN equiv
def test_binned_dbscan_matches_plain_dbscan():
    """Grid-binned weighted DBSCAN partitions points like plain DBSCAN."""
    from sklearn.cluster import DBSCAN

    from trackfm.datasets.ports import EARTH_RADIUS_KM, _binned_dbscan

    rng = np.random.default_rng(3)
    pts = np.vstack([
        np.c_[rng.normal(PORT_A[0], 0.005, 400), rng.normal(PORT_A[1], 0.005, 400)],
        np.c_[rng.normal(PORT_B[0], 0.005, 300), rng.normal(PORT_B[1], 0.005, 300)],
        [[55.0, 9.0], [57.5, 13.0], [54.8, 14.0]],  # isolated noise
    ])
    lat, lon = pts[:, 0], pts[:, 1]

    plain = DBSCAN(eps=CFG.eps_km / EARTH_RADIUS_KM, min_samples=CFG.min_samples,
                   metric="haversine").fit(np.radians(pts)).labels_
    binned = _binned_dbscan(lat, lon, CFG, BBOX)

    # identical partition up to label permutation, identical noise set
    np.testing.assert_array_equal(binned == -1, plain == -1)
    plain_sets = {frozenset(np.flatnonzero(plain == c)) for c in set(plain) - {-1}}
    binned_sets = {frozenset(np.flatnonzero(binned == c)) for c in set(binned) - {-1}}
    assert plain_sets == binned_sets

    # discover_ports on the same points: exact sizes and centroids, right kinds
    dwells = pl.DataFrame({"track_id": [f"t{i}" for i in range(len(pts))],
                           "lat": lat, "lon": lon,
                           "duration_s": [3600.0] * len(pts)})
    ports = discover_ports(dwells, CFG, registry=REGISTRY)
    assert sorted(ports["n_dwells"].to_list()) == [300, 400]
    kinds = dict(zip(ports["name"].to_list(), ports["kind"].to_list()))
    assert kinds.get("testhavn") == "port"
    a = ports.filter(pl.col("name") == "testhavn")
    assert abs(a.item(0, "lat") - lat[:400].mean()) < 1e-9
    assert abs(a.item(0, "lon") - lon[:400].mean()) < 1e-9


# ------------------------------------------------- shard writer roll/resume
def _writer_rows(n):
    return [(np.full(128 * 5, float(i), dtype=np.float32),
             "A", "B", 60.0 * i, f"t{i}", 0, "train") for i in range(n)]


def test_shard_writer_rollover_and_resume(tmp_path):
    from trackfm.datasets.port_task import _ShardWriter

    dest = tmp_path / "port_windows"
    row_bytes = 128 * 5 * 4 + 64
    n = 25

    w = _ShardWriter(dest, 128 * 5, shard_bytes=10 * row_bytes)  # 10 rows/shard
    for row in _writer_rows(n):
        w.add(*row)
    w.flush()

    files = sorted(dest.glob("shard_*.parquet"))
    assert [f.name for f in files] == [f"shard_{i:04d}.parquet" for i in range(3)]
    assert w.rows_per_shard == [10, 10, 5] and w.skipped == 0
    df = pl.read_parquet(files)
    assert df.height == n
    assert df["track_id"].to_list() == [f"t{i}" for i in range(n)]  # order kept
    assert df["features"].dtype == pl.Array(pl.Float32, 128 * 5)
    assert df.columns == ["features", "origin", "destination", "remaining_s",
                          "track_id", "voyage", "split"]

    # resume: drop the last shard, rerun the same stream -> only it is rebuilt
    mtimes = {f.name: f.stat().st_mtime_ns for f in files}
    files[-1].unlink()
    w2 = _ShardWriter(dest, 128 * 5, shard_bytes=10 * row_bytes)
    for row in _writer_rows(n):
        w2.add(*row)
    w2.flush()
    assert w2.skipped == 2 and w2.rows_per_shard == [10, 10, 5]
    for f in sorted(dest.glob("shard_*.parquet"))[:2]:
        assert f.stat().st_mtime_ns == mtimes[f.name]
    tail = pl.read_parquet(dest / "shard_0002.parquet")
    assert tail["track_id"].to_list() == [f"t{i}" for i in range(20, n)]


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
