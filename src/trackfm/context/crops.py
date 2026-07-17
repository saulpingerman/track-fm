"""Canvas-registered static-context crops + zero-init conditioning head.

The conditioning principle: inject fields WHERE THE MODEL USES THEM. The
density head predicts over a canvas (fixed ±grid_range°, or cone ±R(t)°)
centred on the window origin; these utilities sample global rasters on
exactly that canvas so cell (i, j) of a crop is cell (i, j) of the
predicted density. A small CNN then acts spatially on the prediction:

  logits = fourier_synthesis(coeffs(z)) + conv1x1(cnn(crops))     [bias]
  z      = z + zero_init_linear(pooled_cnn_features)              [FiLM-lite]

The additive per-cell bias path deliberately bypasses the Fourier band
limit: coastlines are high-frequency and the basis cannot represent
them, but a bias map can carve them. Both paths are ZERO-INIT so a
conditioned model starts exactly equal to its unconditioned baseline.

Axis convention (matches the loss grid): density/crop dim -2 is LAT
offset ascending, dim -1 is LON offset ascending; canvas offsets are
raw DEGREES on both axes (same convention as displacement targets).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATIC_DIR = Path("~/data/trackfm/context_static").expanduser()


class StaticContext:
    """Holds the static raster stack as one (C, H, W) tensor + affine.

    Channels (normalized to O(1) at load):
      0 land            {0, 1}
      1 log_depth       log1p(m) / 5
      2 sdist_coast     km / 50, clipped to [-2, 2]
      3 traffic         log1p(count_moving) / 10          (if available)
      4 flow_u          mean kn / 30                       (if available)
      5 flow_v          mean kn / 30                       (if available)
    """

    def __init__(self, static_dir: Path | str = STATIC_DIR,
                 with_traffic: bool = True):
        d = Path(static_dir).expanduser()
        geo = np.load(d / "geo.npz")
        lat, lon = geo["lat"], geo["lon"]
        chans = [geo["land"],
                 geo["log_depth"] / 5.0,
                 np.clip(geo["sdist_coast"] / 50.0, -2.0, 2.0)]
        self.channel_names = ["land", "log_depth", "sdist_coast"]
        if with_traffic and (d / "traffic_prior.npz").exists():
            tp = np.load(d / "traffic_prior.npz")
            # traffic grid differs from GEBCO grid — resample to GEBCO grid
            chans += [self._regrid(np.log1p(tp["count_moving"].astype(np.float32)) / 10.0, tp, lat, lon),
                      self._regrid(tp["flow_u"] / 30.0, tp, lat, lon),
                      self._regrid(tp["flow_v"] / 30.0, tp, lat, lon)]
            self.channel_names += ["traffic", "flow_u", "flow_v"]
        self.stack = torch.from_numpy(np.stack(chans).astype(np.float32))
        self.lat0, self.lat1 = float(lat[0]), float(lat[-1])
        self.lon0, self.lon1 = float(lon[0]), float(lon[-1])

    @staticmethod
    def _regrid(arr: np.ndarray, tp, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Nearest-neighbour regrid of a traffic raster onto the GEBCO grid."""
        src_lat = tp["lat0"] + (np.arange(arr.shape[0]) + 0.5) * tp["dlat"]
        src_lon = tp["lon0"] + (np.arange(arr.shape[1]) + 0.5) * tp["dlon"]
        iy = np.clip(np.searchsorted(src_lat, lat), 0, arr.shape[0] - 1)
        ix = np.clip(np.searchsorted(src_lon, lon), 0, arr.shape[1] - 1)
        return arr[np.ix_(iy, ix)]

    @property
    def num_channels(self) -> int:
        return self.stack.shape[0]

    def to(self, device) -> "StaticContext":
        self.stack = self.stack.to(device)
        return self

    def crops(self, origin_lat: torch.Tensor, origin_lon: torch.Tensor,
              R_deg: torch.Tensor | float, G: int) -> torch.Tensor:
        """Sample (B, C, G, G) crops on the ±R_deg canvas around each origin.

        origin_lat/lon: (B,) absolute degrees. R_deg: scalar or (B,).
        Positions outside the raster clamp to the border (open North Sea
        edges read as their boundary values).
        """
        B = origin_lat.shape[0]
        device = self.stack.device
        if not torch.is_tensor(R_deg):
            R_deg = torch.full((B,), float(R_deg), device=device)
        off = torch.linspace(-1.0, 1.0, G, device=device)          # canvas frac
        # absolute coords of crop cells
        lat_c = origin_lat.view(B, 1) + R_deg.view(B, 1) * off      # (B, G)
        lon_c = origin_lon.view(B, 1) + R_deg.view(B, 1) * off      # (B, G)
        # normalize to raster [-1, 1]
        y = (lat_c - self.lat0) / (self.lat1 - self.lat0) * 2 - 1
        x = (lon_c - self.lon0) / (self.lon1 - self.lon0) * 2 - 1
        grid = torch.stack([x.view(B, 1, G).expand(B, G, G),        # x varies along dim -1
                            y.view(B, G, 1).expand(B, G, G)], dim=-1)
        src = self.stack.unsqueeze(0).expand(B, -1, -1, -1)
        return F.grid_sample(src, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)


class ContextBias(nn.Module):
    """CNN over canvas crops -> per-cell logit bias + pooled FiLM vector.

    Both outputs zero-init: at step 0 the conditioned model IS the
    unconditioned model (warm-start friendly, attribution clean).
    """

    def __init__(self, in_channels: int, d_model: int, hidden: int = 16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.bias_proj = nn.Conv2d(hidden, 1, 1)
        self.film = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.bias_proj.weight); nn.init.zeros_(self.bias_proj.bias)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """crops (B, C, G, G) -> (bias (B, G, G), film (B, d_model))."""
        h = self.cnn(crops)
        bias = self.bias_proj(h).squeeze(1)
        film = self.film(h.mean(dim=(-2, -1)))
        return bias, film
