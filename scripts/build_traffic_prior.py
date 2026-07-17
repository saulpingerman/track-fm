"""Build the static traffic-prior rasters from the TRAIN period only.

Leakage guard: the temporal split puts val/test after ~2024-09-21 (v1
val window starts 2024-09-22); this scan hard-stops at TRAIN_END. The
output is a STATIC conditioning channel — historical shipping-lane
structure — usable for any window regardless of its own date.

Rasters (bbox 54-58.5N, 7-16E, 0.005 deg lat x 0.01 deg lon ~ 550m):
  count_all    — posit count per cell (log1p at load time)
  count_moving — posits with sog >= 2 kn
  flow_u/flow_v — mean velocity components (kn) over moving posits:
                  the lane DIRECTION field
Written to ~/data/trackfm/context_static/traffic_prior.npz with grid
metadata for the crop sampler.
"""
from __future__ import annotations

import glob
import time

import numpy as np
import polars as pl

import sys
TRAIN_END = sys.argv[1] if len(sys.argv) > 1 else "2024-09-21"   # v1 split boundary; MUST match the split of the dataset the prior will condition (audit F4)
LAT0, LAT1, LON0, LON1 = 54.0, 58.5, 7.0, 16.0
DLAT, DLON = 0.005, 0.01
NLAT = int(round((LAT1 - LAT0) / DLAT))         # 900
NLON = int(round((LON1 - LON0) / DLON))         # 900

count_all = np.zeros((NLAT, NLON), dtype=np.int64)
count_mov = np.zeros((NLAT, NLON), dtype=np.int64)
sum_u = np.zeros((NLAT, NLON), dtype=np.float64)
sum_v = np.zeros((NLAT, NLON), dtype=np.float64)

days = sorted(glob.glob("/home/paul/data/ais/clean/year=*/month=*/day=*/tracks.parquet"))


def day_key(p: str) -> str:
    parts = dict(kv.split("=") for kv in p.split("/") if "=" in kv)
    return f"{parts['year']}-{parts['month']}-{parts['day']}"


days = [d for d in days if day_key(d) <= TRAIN_END]
print(f"{len(days)} train-period days through {TRAIN_END}")

t0 = time.time()
for i, p in enumerate(days):
    df = pl.read_parquet(p, columns=["lat", "lon", "sog", "cog"])
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    sog = df["sog"].to_numpy(); cog = np.radians(df["cog"].to_numpy())
    iy = ((lat - LAT0) / DLAT).astype(np.int64)
    ix = ((lon - LON0) / DLON).astype(np.int64)
    # AIS cog/sog can be NaN (course unavailable) — one NaN posit poisons
    # its cell's flow sums permanently. Keep such posits for counts but
    # exclude them from the flow accumulation below.
    ok = (iy >= 0) & (iy < NLAT) & (ix >= 0) & (ix < NLON)
    iy, ix, sog, cog = iy[ok], ix[ok], sog[ok], cog[ok]
    flat = iy * NLON + ix
    np.add.at(count_all.ravel(), flat, 1)
    mov = (sog >= 2.0) & np.isfinite(sog) & np.isfinite(cog)
    fm = flat[mov]
    np.add.at(count_mov.ravel(), fm, 1)
    np.add.at(sum_u.ravel(), fm, (sog[mov] * np.sin(cog[mov])))
    np.add.at(sum_v.ravel(), fm, (sog[mov] * np.cos(cog[mov])))
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(days)} days, {el:.0f}s elapsed, "
              f"eta {el/(i+1)*(len(days)-i-1):.0f}s", flush=True)

denom = np.maximum(count_mov, 1)
flow_u = (sum_u / denom).astype(np.float32)
flow_v = (sum_v / denom).astype(np.float32)

out = "/home/paul/data/trackfm/context_static/traffic_prior.npz"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out,
                    count_all=count_all, count_moving=count_mov,
                    flow_u=flow_u, flow_v=flow_v,
                    lat0=LAT0, lat1=LAT1, lon0=LON0, lon1=LON1,
                    dlat=DLAT, dlon=DLON, train_end=TRAIN_END)
print(f"written {out}")
print(f"cells with traffic: {(count_all > 0).mean()*100:.1f}% | "
      f"max cell count {count_all.max():,}")
