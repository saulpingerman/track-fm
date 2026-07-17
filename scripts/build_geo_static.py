"""Derive static geography rasters from the GEBCO subset.

Channels (float32, GEBCO-native ~460m grid, bbox 54-58.5N 7-16E):
  land          — 1.0 where elevation >= 0
  log_depth     — log1p(depth_m) over water, 0 on land
  sdist_coast   — SIGNED distance to coastline in km (+ at sea, - inland),
                  the single most useful channel for carving densities:
                  it is smooth across the boundary the Fourier head
                  cannot represent.
Written to ~/data/trackfm/context_static/geo.npz with grid metadata.
"""
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

ds = xr.open_dataset("/home/paul/data/context/static/gebco_dk.nc")
elev = ds["elevation"].values.astype(np.float32)          # (nlat, nlon), lat ascending
lat = ds["latitude"].values.astype(np.float64)
lon = ds["longitude"].values.astype(np.float64)

land = (elev >= 0).astype(np.float32)
depth = np.where(elev < 0, -elev, 0.0).astype(np.float32)
log_depth = np.log1p(depth)

# anisotropic EDT in km (lat cell ~0.463 km, lon cell ~0.463*cos(56.25) km)
dlat_km = float((lat[1] - lat[0]) * 111.32)
dlon_km = float((lon[1] - lon[0]) * 111.32 * np.cos(np.radians(56.25)))
d_water = distance_transform_edt(land == 0, sampling=(dlat_km, dlon_km))
d_land = distance_transform_edt(land == 1, sampling=(dlat_km, dlon_km))
sdist = np.where(land == 0, d_water, -d_land).astype(np.float32)

out = "/home/paul/data/trackfm/context_static/geo.npz"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, land=land, log_depth=log_depth, sdist_coast=sdist,
                    lat=lat.astype(np.float32), lon=lon.astype(np.float32))
print(f"written {out}: {land.shape}, land {land.mean()*100:.1f}%, "
      f"sdist range [{sdist.min():.0f}, {sdist.max():.0f}] km")
