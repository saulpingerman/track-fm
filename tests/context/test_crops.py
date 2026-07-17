"""Crop sampler + conditioning head invariants.

The tests pin what the conditioning design depends on:
- geometric alignment: a feature at absolute position P lands in the
  crop cell whose canvas offset corresponds to P, for any R (cone or
  fixed scaling);
- axis convention: crop dim -2 is LAT, dim -1 is LON (same as density);
- zero-init: conditioned == unconditioned at step 0;
- border clamp: off-raster crops don't NaN.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from trackfm.context.crops import ContextBias, StaticContext


def _toy_context(bump_lat=56.0, bump_lon=11.0):
    """Synthetic 1-channel raster with a single bright cell."""
    ctx = StaticContext.__new__(StaticContext)
    nlat, nlon = 451, 901                       # 54-58.5 x 7-16 at 0.01
    lat = np.linspace(54.0, 58.5, nlat)
    lon = np.linspace(7.0, 16.0, nlon)
    a = np.zeros((nlat, nlon), dtype=np.float32)
    iy = int(np.argmin(np.abs(lat - bump_lat)))
    ix = int(np.argmin(np.abs(lon - bump_lon)))
    a[iy, ix] = 1.0
    ctx.stack = torch.from_numpy(a).unsqueeze(0)
    ctx.lat0, ctx.lat1 = float(lat[0]), float(lat[-1])
    ctx.lon0, ctx.lon1 = float(lon[0]), float(lon[-1])
    ctx.channel_names = ["bump"]
    return ctx


@pytest.mark.parametrize("R", [0.3, 0.174, 1.25])
def test_bump_lands_in_expected_cell(R):
    """Bump at origin + (dlat, dlon) must appear at the canvas cell for
    that offset, for fixed-like and cone-like R."""
    dlat, dlon = 0.4 * R, -0.6 * R              # inside canvas
    origin_lat, origin_lon = 56.0 - dlat, 11.0 - dlon
    ctx = _toy_context()
    G = 65
    crops = ctx.crops(torch.tensor([origin_lat]), torch.tensor([origin_lon]), R, G)
    b = crops[0, 0]
    iy, ix = np.unravel_index(torch.argmax(b).item(), (G, G))
    # expected cell: offset frac (dlat/R) in [-1,1] -> index
    ey = int(round((0.4 + 1) / 2 * (G - 1)))
    ex = int(round((-0.6 + 1) / 2 * (G - 1)))
    assert abs(iy - ey) <= 1 and abs(ix - ex) <= 1, (iy, ix, ey, ex)


def test_lat_is_dim_minus2():
    """Two bumps differing only in LAT must move along dim -2 only."""
    G = 33
    c1 = _toy_context(bump_lat=56.1, bump_lon=11.0)
    c2 = _toy_context(bump_lat=56.2, bump_lon=11.0)
    o = dict(origin_lat=torch.tensor([56.0]), origin_lon=torch.tensor([11.0]))
    b1 = c1.crops(o["origin_lat"], o["origin_lon"], 0.3, G)[0, 0]
    b2 = c2.crops(o["origin_lat"], o["origin_lon"], 0.3, G)[0, 0]
    y1, x1 = np.unravel_index(torch.argmax(b1).item(), (G, G))
    y2, x2 = np.unravel_index(torch.argmax(b2).item(), (G, G))
    assert y2 > y1 and abs(int(x2) - int(x1)) <= 1


def test_per_sample_R():
    """R as a (B,) tensor scales each sample's canvas independently."""
    ctx = _toy_context()
    G = 33
    lat = torch.tensor([55.9, 55.9]); lon = torch.tensor([11.0, 11.0])
    R = torch.tensor([0.2, 0.8])                # bump at dlat=+0.1
    crops = ctx.crops(lat, lon, R, G)
    y0, _ = np.unravel_index(torch.argmax(crops[0, 0]).item(), (G, G))
    y1, _ = np.unravel_index(torch.argmax(crops[1, 0]).item(), (G, G))
    # same physical offset is a LARGER canvas fraction under smaller R
    assert y0 > y1 > G // 2


def test_border_clamp_no_nan():
    ctx = _toy_context()
    crops = ctx.crops(torch.tensor([54.05]), torch.tensor([7.05]), 1.0, 33)
    assert torch.isfinite(crops).all()


def test_zero_init_identity():
    """bias == 0 and film == 0 at init regardless of input."""
    head = ContextBias(in_channels=3, d_model=128)
    crops = torch.randn(4, 3, 64, 64)
    bias, film = head(crops)
    assert bias.shape == (4, 64, 64) and film.shape == (4, 128)
    assert bias.abs().max().item() == 0.0
    assert film.abs().max().item() == 0.0


def test_real_static_stack_loads():
    """Integration: the built geo.npz loads and crops sanely (skips if
    the rasters have not been built on this machine)."""
    import pathlib
    if not (pathlib.Path.home() / "data/trackfm/context_static/geo.npz").exists():
        pytest.skip("geo.npz not built")
    ctx = StaticContext(with_traffic=False)
    assert ctx.num_channels == 3
    # Copenhagen-area origin: land must appear in the crop (Zealand)
    crops = ctx.crops(torch.tensor([55.7]), torch.tensor([12.6]), 0.3, 64)
    land = crops[0, 0]
    assert 0.05 < land.mean().item() < 0.95    # mixed land/water
    assert torch.isfinite(crops).all()
