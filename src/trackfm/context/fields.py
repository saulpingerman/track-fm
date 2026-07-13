"""Time-indexed gridded context fields with forecast-leakage protection.

A FieldStore serves values of gridded environmental fields (wind, currents,
bathymetry, ...) at arbitrary (lat, lon, t) — the join key the v3 windows
carry as t0 + cumsum(dt). Two access disciplines, chosen per experiment:

  * analysis mode   — fields indexed by VALID time only (ERA5/reanalysis).
    Legitimate for training and for reanalysis-conditioned evaluation.
  * operational mode — every field slice carries the ISSUANCE time of the
    forecast run that produced it; queries must state their decision time
    (the window's forecast origin t0), and slices issued AFTER t0 are
    invisible. This is what makes forecast-interval conditioning honest at
    evaluation: the model may see the future VALID time, never a run that
    had not been published yet.

Storage is deliberately dumb: one float32 array (T, H, W) per field plus
axis vectors, memory-mapped; our bbox subsets are hundreds of MB at most.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Field3D:
    """One gridded variable: values (T, H, W) over (valid_time, lat, lon)."""
    name: str
    values: np.ndarray               # float32 (T, H, W); NaN = missing
    valid_time: np.ndarray           # int64 epoch s, ascending, len T
    lat: np.ndarray                  # float64, len H (ascending or descending)
    lon: np.ndarray                  # float64, len W
    issued_time: np.ndarray | None = None   # int64 epoch s per slice; None = analysis

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.valid_time), len(self.lat), len(self.lon)):
            raise ValueError(f"{self.name}: shape {self.values.shape} does not "
                             f"match axes ({len(self.valid_time)}, "
                             f"{len(self.lat)}, {len(self.lon)})")
        if self.issued_time is not None and len(self.issued_time) != len(self.valid_time):
            raise ValueError(f"{self.name}: issued_time must align with valid_time")
        if np.any(np.diff(self.valid_time) < 0):
            raise ValueError(f"{self.name}: valid_time must be ascending")


class LeakageError(RuntimeError):
    """A query asked for data that was not available at decision time."""


@dataclass
class FieldStore:
    fields: dict[str, Field3D] = field(default_factory=dict)

    def add(self, f: Field3D) -> None:
        self.fields[f.name] = f

    def sample(self, name: str, lat: np.ndarray, lon: np.ndarray,
               t: np.ndarray, decision_time: int | None = None) -> np.ndarray:
        """Bilinear-in-space, nearest-in-time values at (lat, lon, t).

        decision_time: REQUIRED for operational fields (those carrying
        issued_time); slices issued after it are excluded before the time
        lookup, so the caller can only ever condition on forecasts that
        existed at the window's origin. Passing None for an operational
        field raises — analysis-vs-operational must be an explicit choice,
        never a silent default.
        """
        f = self.fields[name]
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        t = np.asarray(t, dtype=np.int64)

        valid, values = f.valid_time, f.values
        if f.issued_time is not None:
            if decision_time is None:
                raise LeakageError(
                    f"{name} is an operational (forecast) field; sample() "
                    f"requires decision_time to enforce issuance ordering")
            avail = f.issued_time <= int(decision_time)
            if not avail.any():
                raise LeakageError(
                    f"{name}: no forecast issued at or before "
                    f"decision_time={decision_time}")
            valid, values = valid[avail], values[avail]
            iss = f.issued_time[avail]
            # several available runs may cover the same valid hour; the
            # freshest issuance wins (that is what an operator would use)
            order = np.lexsort((iss, valid))
            valid, values = valid[order], values[order]
            keep = np.append(valid[1:] != valid[:-1], True)
            valid, values = valid[keep], values[keep]

        ti = np.clip(np.searchsorted(valid, t), 1, len(valid) - 1)
        ti = np.where(np.abs(valid[ti - 1] - t) <= np.abs(valid[ti] - t),
                      ti - 1, ti)

        la, lo = f.lat, f.lon
        lat_asc = la[0] <= la[-1]
        la_s = la if lat_asc else la[::-1]
        yi = np.clip(np.searchsorted(la_s, lat), 1, len(la_s) - 1)
        y0 = yi - 1
        wy = np.clip((lat - la_s[y0]) / (la_s[yi] - la_s[y0]), 0.0, 1.0)
        xi = np.clip(np.searchsorted(lo, lon), 1, len(lo) - 1)
        x0 = xi - 1
        wx = np.clip((lon - lo[x0]) / (lo[xi] - lo[x0]), 0.0, 1.0)
        if not lat_asc:                       # map back to original row order
            y0, yi = len(la) - 1 - y0, len(la) - 1 - yi

        v = (values[ti, y0, x0] * (1 - wy) * (1 - wx)
             + values[ti, yi, x0] * wy * (1 - wx)
             + values[ti, y0, xi] * (1 - wy) * wx
             + values[ti, yi, xi] * wy * wx)
        return v.astype(np.float32)
