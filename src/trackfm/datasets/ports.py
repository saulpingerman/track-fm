"""Self-supervised port discovery and track labeling.

The cleaning pipeline segments tracks at 4h gaps, so a vessel parked in
port for >4h terminates its track there: track starts approximate port
departures, track ends approximate arrivals. This module

  1. extracts per-track endpoints from the cleaned day-partitioned parquet,
  2. discovers "ports" by DBSCAN-clustering endpoints where vessels actually
     stopped (low SOG, away from the region boundary),
  3. labels each track with an origin (port or region-entry edge) and a
     destination (port or region-exit edge) plus the arrival timestamp.

Endpoints that are neither stopped nor near the boundary (mid-sea AIS
dropouts) get no label and are excluded from the port tasks.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# Danish EEZ processing bbox (must match the cleaning config)
BBOX = {"lat_min": 54.0, "lat_max": 58.5, "lon_min": 7.0, "lon_max": 16.0}

# Best-effort names for discovered clusters (cosmetic; labels are cluster ids).
KNOWN_PORTS = {
    "copenhagen": (55.69, 12.60), "gothenburg": (57.69, 11.88),
    "aarhus": (56.15, 10.22), "aalborg": (57.05, 9.92),
    "esbjerg": (55.47, 8.43), "fredericia": (55.56, 9.74),
    "frederikshavn": (57.44, 10.55), "hirtshals": (57.59, 9.96),
    "skagen": (57.72, 10.59), "grenaa": (56.41, 10.93),
    "kalundborg": (55.68, 11.08), "koege": (55.45, 12.20),
    "roenne": (55.10, 14.70), "helsingborg": (56.04, 12.69),
    "helsingoer": (56.03, 12.61), "malmoe": (55.61, 12.99),
    "trelleborg": (55.37, 13.15), "ystad": (55.42, 13.82),
    "kiel": (54.36, 10.15), "rostock": (54.15, 12.10),
    "travemuende": (53.96, 10.87), "puttgarden": (54.50, 11.22),
    "roedby": (54.66, 11.35), "gedser": (54.57, 11.93),
    "sassnitz": (54.51, 13.63), "swinoujscie": (53.91, 14.25),
    "varberg": (57.11, 12.24), "halmstad": (56.66, 12.85),
    "falkenberg": (56.90, 12.49), "frederiksvaerk": (55.97, 12.02),
}


@dataclass
class PortLabelConfig:
    eps_km: float = 3.0                # DBSCAN radius for port clustering
    min_samples: int = 30              # min stop-endpoints to form a port
    max_stop_sog: float = 1.0          # endpoint counts as "stopped" below this
    edge_margin_km: float = 10.0       # within this of bbox edge = entry/exit
    min_track_positions: int = 200     # ignore shorter tracks
    name_match_km: float = 6.0         # cluster<->known-port naming radius


# ------------------------------------------------------------------ extract
def extract_track_endpoints(clean_dir: Path, day_files: list[Path] | None = None,
                            min_positions: int = 200) -> pl.DataFrame:
    """One row per track: start/end position, timestamp, SOG, total length.

    Streams day partitions; tracks spanning days are resolved by taking the
    first day's first row and the last day's last row per track_id.
    """
    if day_files is None:
        day_files = sorted(Path(clean_dir).glob("year=*/month=*/day=*/tracks.parquet"))

    per_day = []
    for f in day_files:
        df = pl.read_parquet(f, columns=["track_id", "timestamp", "lat", "lon", "sog"])
        g = df.sort("timestamp").group_by("track_id").agg(
            pl.col("timestamp").first().alias("start_ts"),
            pl.col("lat").first().alias("start_lat"),
            pl.col("lon").first().alias("start_lon"),
            pl.col("sog").first().alias("start_sog"),
            pl.col("timestamp").last().alias("end_ts"),
            pl.col("lat").last().alias("end_lat"),
            pl.col("lon").last().alias("end_lon"),
            pl.col("sog").last().alias("end_sog"),
            pl.len().alias("n"),
        )
        per_day.append(g)

    all_days = pl.concat(per_day)
    tracks = (
        all_days.sort("start_ts")
        .group_by("track_id")
        .agg(
            pl.col("start_ts").first(), pl.col("start_lat").first(),
            pl.col("start_lon").first(), pl.col("start_sog").first(),
            pl.col("end_ts").last(), pl.col("end_lat").last(),
            pl.col("end_lon").last(), pl.col("end_sog").last(),
            pl.col("n").sum().alias("n_positions"),
        )
        .filter(pl.col("n_positions") >= min_positions)
    )
    logger.info(f"{tracks.height:,} tracks with >= {min_positions} positions")
    return tracks


# ----------------------------------------------------------------- geometry
def _km_to_edge(lat: np.ndarray, lon: np.ndarray, bbox: dict) -> np.ndarray:
    """Approximate distance (km) to the nearest bbox edge."""
    lat_km = 111.32
    lon_km = 111.32 * np.cos(np.radians(lat))
    d = np.minimum.reduce([
        (lat - bbox["lat_min"]) * lat_km,
        (bbox["lat_max"] - lat) * lat_km,
        (lon - bbox["lon_min"]) * lon_km,
        (bbox["lon_max"] - lon) * lon_km,
    ])
    return d


def _edge_name(lat: float, lon: float, bbox: dict) -> str:
    """Which edge of the bbox is nearest (for entry/exit pseudo-labels)."""
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
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------- discover
def discover_ports(tracks: pl.DataFrame, cfg: PortLabelConfig = PortLabelConfig(),
                   bbox: dict = BBOX) -> pl.DataFrame:
    """Cluster stopped endpoints into a port table.

    Pools track starts AND ends (departures and arrivals both mark ports),
    keeping only endpoints that are stopped and not near the region edge.
    Returns: port_id, lat, lon, n_endpoints, name.
    """
    from sklearn.cluster import DBSCAN

    pts = []
    for side in ("start", "end"):
        sub = tracks.filter(pl.col(f"{side}_sog") < cfg.max_stop_sog).select(
            pl.col(f"{side}_lat").alias("lat"), pl.col(f"{side}_lon").alias("lon"))
        pts.append(sub)
    stops = pl.concat(pts)
    lat = stops["lat"].to_numpy()
    lon = stops["lon"].to_numpy()
    interior = _km_to_edge(lat, lon, bbox) > cfg.edge_margin_km
    lat, lon = lat[interior], lon[interior]
    logger.info(f"{len(lat):,} stopped interior endpoints for clustering")

    db = DBSCAN(eps=cfg.eps_km / EARTH_RADIUS_KM, min_samples=cfg.min_samples,
                metric="haversine").fit(np.radians(np.c_[lat, lon]))
    labels = db.labels_

    rows = []
    for c in range(labels.max() + 1):
        m = labels == c
        c_lat, c_lon = float(lat[m].mean()), float(lon[m].mean())
        name = f"port_{c:03d}"
        best = min(KNOWN_PORTS.items(),
                   key=lambda kv: _haversine_km(c_lat, c_lon, kv[1][0], kv[1][1]))
        if _haversine_km(c_lat, c_lon, best[1][0], best[1][1]) < cfg.name_match_km:
            name = best[0]
        rows.append({"port_id": c, "lat": c_lat, "lon": c_lon,
                     "n_endpoints": int(m.sum()), "name": name})

    ports = pl.DataFrame(rows).sort("n_endpoints", descending=True)
    logger.info(f"discovered {ports.height} ports "
                f"({(labels == -1).mean():.0%} endpoints unclustered)")
    return ports


# -------------------------------------------------------------------- label
def _assign(lat: np.ndarray, lon: np.ndarray, sog: np.ndarray, ports: pl.DataFrame,
            cfg: PortLabelConfig, bbox: dict, edge_prefix: str) -> list[str | None]:
    """Label each endpoint: a port name, an edge pseudo-label, or None."""
    p_lat = ports["lat"].to_numpy()
    p_lon = ports["lon"].to_numpy()
    p_name = ports["name"].to_list()

    out: list[str | None] = []
    edge_km = _km_to_edge(lat, lon, bbox)
    for i in range(len(lat)):
        d = _haversine_km(lat[i], lon[i], p_lat, p_lon)
        j = int(d.argmin()) if len(d) else -1
        if j >= 0 and d[j] <= cfg.eps_km and sog[i] < cfg.max_stop_sog:
            out.append(p_name[j])
        elif edge_km[i] <= cfg.edge_margin_km:
            out.append(f"{edge_prefix}_{_edge_name(lat[i], lon[i], bbox)}")
        else:
            out.append(None)  # mid-sea AIS dropout — unlabeled
    return out


def label_tracks(tracks: pl.DataFrame, ports: pl.DataFrame,
                 cfg: PortLabelConfig = PortLabelConfig(),
                 bbox: dict = BBOX) -> pl.DataFrame:
    """Origin/destination labels + arrival timestamp per track.

    origin: port name or ENTRY_{N,S,E,W}; destination: port name or
    EXIT_{N,S,E,W}; None where the endpoint is a mid-sea dropout.
    """
    origin = _assign(tracks["start_lat"].to_numpy(), tracks["start_lon"].to_numpy(),
                     tracks["start_sog"].to_numpy(), ports, cfg, bbox, "ENTRY")
    dest = _assign(tracks["end_lat"].to_numpy(), tracks["end_lon"].to_numpy(),
                   tracks["end_sog"].to_numpy(), ports, cfg, bbox, "EXIT")

    labeled = tracks.with_columns(
        pl.Series("origin", origin), pl.Series("destination", dest))
    n_both = labeled.filter(pl.col("origin").is_not_null()
                            & pl.col("destination").is_not_null()).height
    logger.info(f"labels: {n_both:,}/{labeled.height:,} tracks have both "
                f"origin and destination")
    return labeled
