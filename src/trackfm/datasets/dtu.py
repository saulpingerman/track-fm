"""DTU Danish-waters anomaly dataset (exp-12 port).

Loads the labelled evaluation set (521 trajectories, 25 anomalies; article
21511815, variant *_600_43200_120) from the figshare pickle format and
converts to model-ready arrays. Conversion + normalization are ported
verbatim from legacy exp 12 (`src/data/{dtu_dataset,preprocessing}.py`):
note SOG and dt are CLIPPED before scaling (unlike pretraining's unclipped
transform) — kept for paper comparability.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("~/data/trackfm/dtu_anomaly/raw")


def load_dtu(raw_dir: Path = DEFAULT_RAW) -> tuple[list[np.ndarray], np.ndarray]:
    """Returns (trajectories [(len,6) raw-feature arrays], labels (N,))."""
    raw_dir = Path(raw_dir).expanduser()
    data_files = sorted(raw_dir.glob("data_*.pkl"))
    info_files = sorted(raw_dir.glob("datasetInfo_*.pkl"))
    if len(data_files) != 1 or len(info_files) != 1:
        raise FileNotFoundError(
            f"expected exactly one data_/datasetInfo_ pair in {raw_dir} "
            f"(found {len(data_files)}/{len(info_files)}); the legacy loader's "
            "glob[0] pairing bug is avoided by keeping one variant per dir")

    raw = []
    with open(data_files[0], "rb") as f:
        try:
            while True:
                raw.append(pickle.load(f))
        except EOFError:
            pass
    with open(info_files[0], "rb") as f:
        labels = np.array(pickle.load(f)["outlierLabels"])

    trajectories = [t for t in (_convert(d) for d in raw) if t is not None]
    if len(trajectories) != len(labels):
        raise ValueError(f"{len(trajectories)} trajectories vs {len(labels)} labels")
    logger.info(f"DTU: {len(trajectories)} trajectories, {int(labels.sum())} anomalies")
    return trajectories, labels


def _convert(traj_dict: dict) -> np.ndarray | None:
    """dict(lat,lon,speed,course,timestamp) -> (len, 6) [lat,lon,sog,cog_sin,cog_cos,dt]."""
    try:
        lat = np.array(traj_dict["lat"], dtype=np.float64)
        lon = np.array(traj_dict["lon"], dtype=np.float64)
        sog = np.array(traj_dict["speed"], dtype=np.float64)
        cog = np.radians(np.array(traj_dict["course"], dtype=np.float64))
        ts = np.array(traj_dict["timestamp"], dtype=np.float64)

        dt = np.diff(ts, prepend=ts[0])
        dt[0] = 0
        dt = np.clip(dt, 0, 3600)

        out = np.stack([lat, lon, sog, np.sin(cog), np.cos(cog), dt], axis=1)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)
    except Exception as e:
        logger.warning(f"skipping malformed trajectory: {e}")
        return None


def normalize_exp12(traj: np.ndarray, lat_mean: float = 56.25, lat_std: float = 1.0,
                    lon_mean: float = 11.5, lon_std: float = 2.0,
                    sog_max: float = 30.0, dt_max: float = 300.0) -> np.ndarray:
    """Exp-12 normalization: like pretraining but SOG/dt are clipped."""
    out = traj.copy()
    out[:, 0] = (traj[:, 0] - lat_mean) / lat_std
    out[:, 1] = (traj[:, 1] - lon_mean) / lon_std
    out[:, 2] = np.clip(traj[:, 2], 0, sog_max) / sog_max
    out[:, 5] = np.clip(traj[:, 5], 0, dt_max) / dt_max
    return out.astype(np.float32)


def to_padded_arrays(trajectories: list[np.ndarray], labels: np.ndarray,
                     max_len: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncate (keep most recent)/zero-pad to (N, max_len, 6) + lengths.

    Returns (x, lengths, y); lengths enable masked pooling in the heads.
    """
    n = len(trajectories)
    x = np.zeros((n, max_len, 6), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    for i, t in enumerate(trajectories):
        t = normalize_exp12(t)
        t = t[-max_len:]
        x[i, :len(t)] = t
        lengths[i] = len(t)
    return x, lengths, labels.astype(np.int64)
