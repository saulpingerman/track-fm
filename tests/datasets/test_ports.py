"""Port discovery / labeling / task-dataset tests on synthetic voyages."""
import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from trackfm.datasets.port_task import build_port_task_dataset
from trackfm.datasets.ports import (
    BBOX, PortLabelConfig, discover_ports, extract_track_endpoints, label_tracks,
)

# Two fake "ports" well inside the bbox
PORT_A = (56.00, 10.00)
PORT_B = (56.80, 11.50)
T0 = datetime(2024, 1, 1)

CFG = PortLabelConfig(min_samples=5, min_track_positions=50)


def make_voyage(tid: str, day: int, src, dst, n=300, end_sog=0.0, start_sog=0.0):
    """A straight-line voyage src->dst within one day."""
    f = np.linspace(0, 1, n)
    lat = src[0] + (dst[0] - src[0]) * f
    lon = src[1] + (dst[1] - src[1]) * f
    sog = np.full(n, 12.0)
    sog[0] = start_sog
    sog[-1] = end_sog
    ts = [T0 + timedelta(days=day, seconds=60 * i) for i in range(n)]
    return pl.DataFrame({
        "timestamp": ts, "track_id": [tid] * n,
        "lat": lat, "lon": lon, "sog": sog,
        "cog": np.full(n, 45.0), "dt_seconds": np.full(n, 60.0, dtype=np.float32),
    })


@pytest.fixture(scope="module")
def clean_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean")
    exit_pt = (56.2, 15.95)  # near the east edge
    for day in range(10):
        frames = []
        for v in range(8):  # A->B voyages (arrivals)
            frames.append(make_voyage(f"ab{v}_{day}", day, PORT_A, PORT_B))
        for v in range(8):  # B->A voyages
            frames.append(make_voyage(f"ba{v}_{day}", day, PORT_B, PORT_A))
        # departures from A that leave the region east (still moving at end)
        frames.append(make_voyage(f"ax{day}", day, PORT_A, exit_pt, end_sog=14.0))
        df = pl.concat(frames)
        out = root / f"year=2024/month=01/day={day+1:02d}"
        out.mkdir(parents=True)
        df.write_parquet(out / "tracks.parquet")
    return root


@pytest.fixture(scope="module")
def labeled(clean_tree):
    tracks = extract_track_endpoints(clean_tree, min_positions=CFG.min_track_positions)
    ports = discover_ports(tracks, CFG)
    return tracks, ports, label_tracks(tracks, ports, CFG)


def test_discovers_two_ports(labeled):
    _, ports, _ = labeled
    assert ports.height == 2
    for want in (PORT_A, PORT_B):
        d = np.hypot(ports["lat"].to_numpy() - want[0], ports["lon"].to_numpy() - want[1])
        assert d.min() < 0.05, f"no cluster near {want}"


def test_labels_arrivals_and_exits(labeled):
    _, ports, lab = labeled
    ab = lab.filter(pl.col("track_id").str.starts_with("ab"))
    assert ab.height == 80
    # all A->B voyages: origin = port at A, destination = port at B
    assert ab["origin"].n_unique() == 1 and ab["destination"].n_unique() == 1
    assert ab["origin"][0] != ab["destination"][0]
    # eastbound leavers get EXIT_E
    ax = lab.filter(pl.col("track_id").str.starts_with("ax"))
    assert set(ax["destination"].to_list()) == {"EXIT_E"}
    assert ax["origin"].n_unique() == 1  # departed from port A


def test_task_dataset_targets(labeled, clean_tree, tmp_path):
    _, _, lab = labeled
    out = tmp_path / "ports"
    build_port_task_dataset(clean_tree, lab, out, input_len=128, stride=64,
                            min_class_count=1)
    vocab = json.loads((out / "labels.json").read_text())
    assert set(vocab) == {"origin", "destination"}
    assert "EXIT_E" in vocab["destination"]

    df = pl.read_parquet(out / "train" / "windows.parquet")
    assert df["features"].dtype == pl.Array(pl.Float32, 128 * 5)
    # voyage = 300 min, windows every 64 pos: remaining time decreases per window
    one = df.filter(pl.col("track_id") == "ab0_0").sort("remaining_s", descending=True)
    r = one["remaining_s"].to_list()
    assert len(r) == (300 - 128) // 64 + 1
    # first window ends at position 127 => 300-1-127 = 172 min remain
    assert abs(r[0] - 172 * 60) < 1
    assert abs(r[-1] - (r[0] - 64 * 60 * (len(r) - 1))) < 1


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
    secs = EtaRegressor.to_seconds(out)
    assert (secs >= 0).all()
