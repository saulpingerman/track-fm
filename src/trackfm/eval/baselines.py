"""Strong analytic forecasting baselines.

The paper's dead-reckoning baseline is a point prediction scored as a
Gaussian with one globally-tuned sigma — necessarily overconfident at long
horizons and underconfident at short ones. This module provides the honest
versions, so model-vs-baseline ratios cannot be dismissed as strawmen:

  * per-horizon sigma tuning for any point predictor (LP, DR): the best
    fixed-width Gaussian at EACH horizon, fit on validation data
  * a constant-velocity Kalman filter: velocity estimated over the whole
    input window (not the last two points) and a covariance that grows
    analytically with horizon; process/measurement noise tuned on val

All baselines are scored with the same soft-target loss as the model.
"""
from __future__ import annotations

import logging
import math

import torch

from trackfm.config import NormalizationConfig
from trackfm.training.losses import compute_soft_target_loss

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ scoring
def gaussian_density_loss_per_sample(
    pred: torch.Tensor,          # (B, H, 2) predicted displacement (degrees)
    sigma: torch.Tensor,         # (B, H, 2) or broadcastable — Gaussian widths
    targets: torch.Tensor,       # (B, H, 2) true displacement
    grid_range: float, grid_size: int, target_sigma: float,
) -> torch.Tensor:
    """Score a Gaussian point-forecast with per-sample, per-axis widths."""
    device = pred.device
    ax = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")

    pred = torch.clamp(pred, -grid_range * 0.99, grid_range * 0.99)
    sig = torch.clamp(sigma, min=1e-4)
    zx = (xx - pred[..., 0:1, None]) / sig[..., 0:1, None]
    zy = (yy - pred[..., 1:2, None]) / sig[..., 1:2, None]
    # Analytic log-density (metrics v2, audit F2): no exp-log floor.
    logit = -0.5 * (zx ** 2 + zy ** 2)
    log_density = logit - torch.logsumexp(
        logit.reshape(*logit.shape[:-2], -1), dim=-1)[..., None, None]
    return compute_soft_target_loss(log_density, targets, grid_range,
                                    grid_size, target_sigma)


# ------------------------------------------------------- per-horizon sigma
def fit_sigma_per_horizon(
    preds: torch.Tensor,      # (N, H, 2) point predictions on val
    targets: torch.Tensor,    # (N, H, 2)
    grid_range: float, grid_size: int, target_sigma: float,
    candidates: tuple = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2),
) -> torch.Tensor:
    """Best fixed sigma per horizon (grid search on val). Returns (H,)."""
    H = preds.shape[1]
    best = torch.full((H,), float("nan"))
    for h in range(H):
        losses = []
        for s in candidates:
            sig = torch.full_like(preds[:, h:h+1], s)
            losses.append(gaussian_density_loss_per_sample(
                preds[:, h:h+1], sig, targets[:, h:h+1],
                grid_range, grid_size, target_sigma).item())
        best[h] = candidates[int(torch.tensor(losses).argmin())]
    return best


# ------------------------------------------------------------ Kalman filter
def kalman_cv_forecast(
    features: torch.Tensor,        # (B, L+, 6) normalized model input
    horizon_indices: torch.Tensor, # (H,)
    seq_len: int,
    norm: NormalizationConfig,
    q: float = 1e-9,               # process noise (deg^2/s^3, white accel)
    r: float = 1e-6,               # measurement noise (deg^2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant-velocity Kalman filter per axis, filtered over the window,
    projected to each horizon.

    Returns (pred (B,H,2) displacement from last position,
             sigma (B,H,2) predicted std in degrees).
    """
    device = features.device
    B = features.shape[0]
    lat = features[:, :seq_len, 0] * norm.lat_scale + norm.lat_center
    lon = features[:, :seq_len, 1] * norm.lon_scale + norm.lon_center
    dt_all = features[:, :, 5] * norm.dt_scale                   # seconds
    z = torch.stack([lat, lon], dim=-1)                          # (B, L, 2)

    # state per axis: [position, velocity]; vectorized over batch and axis
    x = torch.zeros(B, 2, 2, device=device)                      # (B, axis, [p,v])
    x[:, :, 0] = z[:, 0]
    P = torch.zeros(B, 2, 2, 2, device=device)                   # (B, axis, 2, 2)
    P[:, :, 0, 0] = r
    P[:, :, 1, 1] = 1e-6

    for k in range(1, seq_len):
        dt = dt_all[:, k].clamp(min=1e-3)[:, None]               # (B, 1)
        # predict
        xp = torch.stack([x[..., 0] + dt * x[..., 1], x[..., 1]], dim=-1)
        dt_ = dt[..., None]
        Q00 = q * dt_ ** 3 / 3
        Q01 = q * dt_ ** 2 / 2
        Q11 = q * dt_
        Pp = P.clone()
        Pp[..., 0, 0] = P[..., 0, 0] + dt_[..., 0] * (P[..., 0, 1] + P[..., 1, 0]) \
            + dt_[..., 0] ** 2 * P[..., 1, 1] + Q00[..., 0]
        Pp[..., 0, 1] = P[..., 0, 1] + dt_[..., 0] * P[..., 1, 1] + Q01[..., 0]
        Pp[..., 1, 0] = Pp[..., 0, 1]
        Pp[..., 1, 1] = P[..., 1, 1] + Q11[..., 0]
        # update with measurement z[:, k]
        S = Pp[..., 0, 0] + r
        K0 = Pp[..., 0, 0] / S
        K1 = Pp[..., 1, 0] / S
        innov = z[:, k] - xp[..., 0]
        x = torch.stack([xp[..., 0] + K0 * innov, xp[..., 1] + K1 * innov], dim=-1)
        P = Pp.clone()
        P[..., 0, 0] = (1 - K0) * Pp[..., 0, 0]
        P[..., 0, 1] = (1 - K0) * Pp[..., 0, 1]
        P[..., 1, 0] = Pp[..., 1, 0] - K1 * Pp[..., 0, 0]
        P[..., 1, 1] = Pp[..., 1, 1] - K1 * Pp[..., 0, 1]

    # project to horizons: elapsed time from last input to each horizon
    cumsum = torch.cumsum(dt_all, dim=1)
    last = cumsum[:, seq_len - 1]
    preds, sigmas = [], []
    for h in horizon_indices:
        T = (cumsum[:, seq_len + int(h) - 1] - last).clamp(min=1.0)[:, None]  # (B,1)
        disp = x[..., 1] * T                                     # (B, 2 axes)
        var = P[..., 0, 0] + 2 * T * P[..., 0, 1] + T ** 2 * P[..., 1, 1] \
            + q * T ** 3 / 3
        preds.append(disp)
        sigmas.append(var.clamp(min=1e-10).sqrt())
    return torch.stack(preds, dim=1), torch.stack(sigmas, dim=1)


def tune_kalman(features_val: torch.Tensor, targets_val: torch.Tensor,
                horizon_indices: torch.Tensor, seq_len: int,
                norm: NormalizationConfig, grid_range: float, grid_size: int,
                target_sigma: float) -> tuple[float, float]:
    """Small grid search for (q, r) minimizing val loss. Returns best (q, r)."""
    best = (None, math.inf)
    for q in (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        for r in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
            pred, sig = kalman_cv_forecast(features_val, horizon_indices,
                                           seq_len, norm, q, r)
            loss = gaussian_density_loss_per_sample(
                pred, sig, targets_val, grid_range, grid_size, target_sigma).item()
            if loss < best[1]:
                best = ((q, r), loss)
    logger.info(f"kalman tuned: q={best[0][0]:.0e}, r={best[0][1]:.0e} "
                f"(val loss {best[1]:.3f})")
    return best[0]
