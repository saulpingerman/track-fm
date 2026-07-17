"""Decode point predictions from density heads for ADE/FDE comparison.

The model's product is the density (km²@90 is the pinned metric), but
external benchmarks (e.g. EnvShip-Bench on the same DMA data) report
ADE/FDE in meters. These decoders make the comparison possible without
changing the model:

  argmax      — center of the highest-probability cell. Right decode for
                multimodal densities scored by displacement error only
                when one mode dominates; robust default.
  expectation — probability-weighted mean position. Lower ADE for
                unimodal densities; badly wrong between modes (decodes
                into the water between two route branches).

Both return displacement in CANVAS-degree units converted to physical
degrees via the per-pair half-range R (grid_range for fixed, R(t) for
cone), so one code path serves both geometries.
"""
from __future__ import annotations

import math

import torch

KM_PER_DEG = 111.32


def decode_points(log_density: torch.Tensor, R_deg: torch.Tensor | float,
                  mode: str = "argmax") -> torch.Tensor:
    """(..., G, G) log-density -> (..., 2) displacement in PHYSICAL degrees.

    R_deg: scalar or broadcastable to log_density.shape[:-2]. Cell-center
    convention matches the loss grid: linspace(-1, 1, G) inclusive.
    """
    *lead, G, G2 = log_density.shape
    assert G == G2
    device = log_density.device
    centers = torch.linspace(-1.0, 1.0, G, device=device)
    if mode == "argmax":
        flat = log_density.reshape(*lead, G * G)
        idx = flat.argmax(dim=-1)
        iy, ix = idx // G, idx % G
        d = torch.stack([centers[iy], centers[ix]], dim=-1)
    elif mode == "expectation":
        p = log_density.exp()
        p = p / p.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        dy = (p.sum(dim=-1) * centers).sum(dim=-1)          # lat axis (dim -2)
        dx = (p.sum(dim=-2) * centers).sum(dim=-1)          # lon axis (dim -1)
        d = torch.stack([dy, dx], dim=-1)
    else:
        raise ValueError(mode)
    if not torch.is_tensor(R_deg):
        R_deg = torch.tensor(float(R_deg), device=device)
    return d * R_deg[..., None] if R_deg.dim() else d * R_deg


def displacement_error_m(pred_deg: torch.Tensor, tgt_deg: torch.Tensor,
                         lat0: float = 56.25) -> torch.Tensor:
    """Great-circle-approx displacement error in METERS.

    pred/tgt: (..., 2) [dlat, dlon] physical degrees relative to the same
    origin; equirectangular at lat0 (fine at <100 km scales).
    """
    dlat = (pred_deg[..., 0] - tgt_deg[..., 0]) * KM_PER_DEG
    dlon = (pred_deg[..., 1] - tgt_deg[..., 1]) * KM_PER_DEG * math.cos(math.radians(lat0))
    return torch.hypot(dlat, dlon) * 1000.0
