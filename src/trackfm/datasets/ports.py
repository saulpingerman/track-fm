"""Self-supervised port discovery and voyage labeling (v2).

Tracks end only at AIS *silence* (4h gaps), so a single track can span
months and many port calls. Voyage extraction therefore splits tracks at
DWELL events — sustained stops (SOG below threshold for >= min_dwell) —
and labels each voyage:

  origin       port/anchorage at the preceding dwell, or ENTRY_<edge>
  destination  port/anchorage at the terminating dwell, or EXIT_<edge>
  arrival      timestamp the terminating dwell begins

Port identity is the intersection of two sources:
  * data-driven: DBSCAN clusters of dwell centroids
  * authoritative: OSM harbour registry (configs/data/port_registry_osm.json)
Clusters matched to the registry are kind="port" and take the OSM name;
unmatched dwell clusters are kind="anchorage" (real, predictable
destinations — but never conflated with ports).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from trackfm.datasets.windowing import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
SOG_IDX = FEATURE_COLUMNS.index("sog")

# Danish EEZ processing bbox (must match the cleaning config)
BBOX = {"lat_min": 54.0, "lat_max": 58.5, "lon_min": 7.0, "lon_max": 16.0}

REGISTRY_PATH = Path(__file__).parents[3] / "configs/data/port_registry_osm.json"


@dataclass
class PortLabelConfig:
    stop_sog: float = 0.5              # below this = stopped
    min_dwell_s: float = 3600.0        # sustained stop >= 1h = dwell event
    eps_km: float = 0.75               # DBSCAN radius; validated 2026-07-13:
                                       # recovers 22/22 known major ports at
                                       # full dwell density (3.0 chains the
                                       # whole Oresund into one mega-cluster)
    min_samples: int = 30              # min dwells to form a port/anchorage
    bin_km: float = 0.15               # pre-clustering grid pitch (memory bound)
    edge_margin_km: float = 10.0       # within this of bbox edge = entry/exit
    min_track_positions: int = 200
    min_voyage_positions: int = 128    # voyages shorter than a window are useless
    registry_match_km: float = 3.0     # cluster<->OSM-harbour match radius


# ------------------------------------------------------------------ streaming
def stream_tracks(day_files: list[Path], min_positions: int = 200,
                  ) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (track_id, features(N,5), epoch_seconds(N)) per completed track.

    Buffers tracks across day partitions; a track completes when it stops
    appearing (tracks are gap-segmented, so one missing day ends it).
    """
    from collections import defaultdict

    buffers: dict[str, list] = defaultdict(list)
    last_seen: dict[str, str] = {}

    def finalize(tid: str):
        chunks = buffers.pop(tid)
        last_seen.pop(tid, None)
        feats = np.vstack([c[0] for c in chunks])
        epochs = np.concatenate([c[1] for c in chunks])
        if len(feats) >= min_positions:
            return tid, feats, epochs
        return None

    for day_path in day_files:
        day = day_path.parent.as_posix()
        df = pl.read_parquet(day_path, columns=["track_id", "timestamp", *FEATURE_COLUMNS])
        df = df.sort(["track_id", "timestamp"])
        for track_df in df.partition_by("track_id"):
            tid = track_df.item(0, "track_id")
            feats = np.nan_to_num(
                track_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32), nan=0.0)
            epochs = track_df["timestamp"].dt.epoch(time_unit="s").to_numpy().astype(np.float64)
            buffers[tid].append((feats, epochs))
            last_seen[tid] = day
        for tid in [t for t, d in last_seen.items() if d != day]:
            done = finalize(tid)
            if done:
                yield done
    for tid in list(buffers):
        done = finalize(tid)
        if done:
            yield done


# -------------------------------------------------------------------- dwells
def find_dwells(feats: np.ndarray, epochs: np.ndarray, stop_sog: float,
                min_dwell_s: float) -> list[tuple[int, int]]:
    """Index ranges [i0, i1) of sustained stops (dwell events)."""
    stopped = feats[:, SOG_IDX] < stop_sog
    dwells = []
    i = 0
    n = len(feats)
    while i < n:
        if stopped[i]:
            j = i
            while j + 1 < n and stopped[j + 1]:
                j += 1
            if epochs[j] - epochs[i] >= min_dwell_s:
                dwells.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return dwells


@dataclass
class Voyage:
    track_id: str
    start: int                  # index into track arrays
    end: int                    # exclusive; voyage positions = feats[start:end]
    origin_pos: tuple | None    # (lat, lon) of preceding dwell centroid
    dest_pos: tuple | None      # (lat, lon) of terminating dwell centroid
    arrival_epoch: float        # when the terminating dwell begins (= epochs[end-1] if none)
    starts_track: bool          # voyage begins at track start (possible region entry)
    ends_track: bool            # voyage ends at track end (possible region exit)


def split_voyages(track_id: str, feats: np.ndarray, epochs: np.ndarray,
                  cfg: PortLabelConfig) -> list[Voyage]:
    """Split one track into voyages at dwell events."""
    dwells = find_dwells(feats, epochs, cfg.stop_sog, cfg.min_dwell_s)
    centroids = [(float(feats[a:b, 0].mean()), float(feats[a:b, 1].mean()))
                 for a, b in dwells]

    voyages = []
    bounds = [(None, 0, None)]  # (origin_centroid, start_idx, _)
    prev_end = 0
    prev_centroid = None
    for (a, b), c in zip(dwells, centroids):
        if a > prev_end:
            voyages.append(Voyage(
                track_id=track_id, start=prev_end, end=a,
                origin_pos=prev_centroid, dest_pos=c,
                arrival_epoch=float(epochs[a]),
                starts_track=prev_end == 0, ends_track=False,
            ))
        prev_end = b
        prev_centroid = c
    if prev_end < len(feats):
        voyages.append(Voyage(
            track_id=track_id, start=prev_end, end=len(feats),
            origin_pos=prev_centroid, dest_pos=None,
            arrival_epoch=float(epochs[-1]),
            starts_track=prev_end == 0, ends_track=True,
        ))
    return [v for v in voyages if v.end - v.start >= cfg.min_voyage_positions]


# ----------------------------------------------------------------- geometry
def _km_to_edge(lat, lon, bbox: dict):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_km = 111.32
    lon_km = 111.32 * np.cos(np.radians(lat))
    return np.minimum.reduce([
        (lat - bbox["lat_min"]) * lat_km,
        (bbox["lat_max"] - lat) * lat_km,
        (lon - bbox["lon_min"]) * lon_km,
        (bbox["lon_max"] - lon) * lon_km,
    ])


def _edge_name(lat: float, lon: float, bbox: dict) -> str:
    lat_km = 111.32
    lon_km = 111.32 * math.cos(math.radians(lat))
    dists = {
        "S": (lat - bbox["lat_min"]) * lat_km,
        "N": (bbox["lat_max"] - lat) * lat_km,
        "W": (lon - bbox["lon_min"]) * lon_km,
        "E": (bbox["lon_max"] - lon) * lon_km,
    }
    return min(dists, key=dists.get)


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2)) - np.radians(np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ registry
def load_registry(path: Path | None = None) -> pl.DataFrame:
    """Harbour registry: name, lat, lon.

    Base OSM extraction (marina-biased tags) + the major-commercial-port
    supplement (the OSM pass missed Copenhagen, Aalborg, Fredericia, ...;
    supplement coordinates are snapped to our own dwell-mass medians).
    An explicit path loads that file alone (tests, alternate registries).
    """
    if path is not None:
        return pl.DataFrame(json.loads(Path(path).read_text()))
    entries = json.loads(REGISTRY_PATH.read_text())
    sup_path = REGISTRY_PATH.parent / "port_registry_supplement.json"
    if sup_path.exists():
        entries += [{"name": e["name"], "lat": e["lat"], "lon": e["lon"]}
                    for e in json.loads(sup_path.read_text())]
    return pl.DataFrame(entries)


# ------------------------------------------------------- pass A: discovery
def collect_dwell_centroids(day_files: list[Path], cfg: PortLabelConfig,
                            ) -> pl.DataFrame:
    """All dwell centroids across tracks (pass A input to clustering)."""
    rows = []
    for tid, feats, epochs in stream_tracks(day_files, cfg.min_track_positions):
        for a, b in find_dwells(feats, epochs, cfg.stop_sog, cfg.min_dwell_s):
            rows.append({
                "track_id": tid,
                "lat": float(feats[a:b, 0].mean()),
                "lon": float(feats[a:b, 1].mean()),
                "duration_s": float(epochs[b - 1] - epochs[a]),
            })
    df = pl.DataFrame(rows)
    logger.info(f"{df.height:,} dwell events collected")
    return df


def _bin_centroids(lat: np.ndarray, lon: np.ndarray, bin_km: float, bbox: dict,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snap points to a fixed ~bin_km grid; aggregate count + mean per bin.

    Returns (bin_lat, bin_lon, counts, inverse). Bin positions are the exact
    means of member points, so count-weighted cluster centroids match the
    unbinned ones. Deterministic: the grid is anchored at (0, 0) with a pitch
    derived from the bbox mid-latitude.
    """
    lat_step = bin_km / 111.32
    lon_step = bin_km / (111.32 * math.cos(
        math.radians((bbox["lat_min"] + bbox["lat_max"]) / 2)))
    code = (np.floor(lat / lat_step).astype(np.int64) << 32) \
        + np.floor(lon / lon_step).astype(np.int64)
    _, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
    bin_lat = np.bincount(inverse, weights=lat) / counts
    bin_lon = np.bincount(inverse, weights=lon) / counts
    return bin_lat, bin_lon, counts.astype(np.float64), inverse


def _binned_dbscan(lat: np.ndarray, lon: np.ndarray, cfg: PortLabelConfig,
                   bbox: dict) -> np.ndarray:
    """DBSCAN over grid-binned points; returns a cluster label per input point.

    Unique bins are clustered with per-bin dwell counts as sample_weight, so
    min_samples still counts dwells (not bins) while memory stays O(bins)
    instead of O(dwells^2) neighborhoods. Each point inherits its bin's label.
    """
    from sklearn.cluster import DBSCAN

    bin_lat, bin_lon, counts, inverse = _bin_centroids(lat, lon, cfg.bin_km, bbox)
    db = DBSCAN(eps=cfg.eps_km / EARTH_RADIUS_KM, min_samples=cfg.min_samples,
                metric="haversine").fit(np.radians(np.c_[bin_lat, bin_lon]),
                                        sample_weight=counts)
    return db.labels_[inverse]


def discover_ports(dwells: pl.DataFrame, cfg: PortLabelConfig = PortLabelConfig(),
                   bbox: dict = BBOX, registry: pl.DataFrame | None = None,
                   ) -> pl.DataFrame:
    """Cluster dwell centroids; classify clusters via the OSM registry.

    Returns: port_id, lat, lon, n_dwells, kind ('port'|'anchorage'), name.
    """
    if registry is None:
        registry = load_registry()

    lat = dwells["lat"].to_numpy()
    lon = dwells["lon"].to_numpy()
    interior = _km_to_edge(lat, lon, bbox) > cfg.edge_margin_km
    lat, lon = lat[interior], lon[interior]

    labels = _binned_dbscan(lat, lon, cfg, bbox)

    r_lat = registry["lat"].to_numpy()
    r_lon = registry["lon"].to_numpy()
    r_name = registry["name"].to_list()

    rows = []
    for c in range(labels.max() + 1):
        m = labels == c
        c_lat, c_lon = float(lat[m].mean()), float(lon[m].mean())
        d = _haversine_km(c_lat, c_lon, r_lat, r_lon)
        j = int(d.argmin())
        if d[j] <= cfg.registry_match_km:
            kind, name = "port", r_name[j]
        else:
            kind, name = "anchorage", f"anchorage_{c:03d}"
        rows.append({"port_id": c, "lat": c_lat, "lon": c_lon,
                     "n_dwells": int(m.sum()), "kind": kind, "name": name})

    # Deduplicate names (several clusters can match one big harbour complex)
    ports = pl.DataFrame(rows).sort("n_dwells", descending=True)
    seen: dict[str, int] = {}
    names = []
    for n in ports["name"].to_list():
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    ports = ports.with_columns(pl.Series("name", names))

    n_port = ports.filter(pl.col("kind") == "port").height
    logger.info(f"discovered {ports.height} stop-clusters: {n_port} ports, "
                f"{ports.height - n_port} anchorages "
                f"({(labels == -1).mean():.0%} dwells unclustered)")
    return ports


# ------------------------------------------------------------------ labeling
class PortIndex:
    """Nearest-cluster lookup for voyage labeling."""

    def __init__(self, ports: pl.DataFrame, cfg: PortLabelConfig, bbox: dict = BBOX):
        self.lat = ports["lat"].to_numpy()
        self.lon = ports["lon"].to_numpy()
        self.name = ports["name"].to_list()
        self.kind = ports["kind"].to_list()
        self.cfg = cfg
        self.bbox = bbox

    def label_dwell(self, pos: tuple | None) -> tuple[str | None, str | None]:
        """(label, kind) for a dwell centroid; None if it matches no cluster."""
        if pos is None or len(self.lat) == 0:
            return None, None
        d = _haversine_km(pos[0], pos[1], self.lat, self.lon)
        j = int(d.argmin())
        if d[j] <= self.cfg.eps_km:
            return self.name[j], self.kind[j]
        return None, None

    def label_voyage(self, v: Voyage, feats: np.ndarray) -> tuple[str | None, str | None]:
        """(origin, destination) labels for a voyage; None where unknown."""
        origin, _ = self.label_dwell(v.origin_pos)
        if origin is None and v.starts_track:
            s_lat, s_lon = float(feats[v.start, 0]), float(feats[v.start, 1])
            if _km_to_edge(s_lat, s_lon, self.bbox) <= self.cfg.edge_margin_km:
                origin = f"ENTRY_{_edge_name(s_lat, s_lon, self.bbox)}"

        dest, _ = self.label_dwell(v.dest_pos)
        if dest is None and v.ends_track:
            e_lat, e_lon = float(feats[v.end - 1, 0]), float(feats[v.end - 1, 1])
            if _km_to_edge(e_lat, e_lon, self.bbox) <= self.cfg.edge_margin_km:
                dest = f"EXIT_{_edge_name(e_lat, e_lon, self.bbox)}"
        return origin, dest
