"""Pretraining loss and analytic baselines.

Ported verbatim from experiment 11 (`run_experiment.py`): soft-target KL
divergence over the Fourier density grid, plus the dead-reckoning baseline
used for the paper's headline "Nx lower loss than DR" metric.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trackfm.config import NormalizationConfig


def compute_soft_target_loss(log_density: torch.Tensor, target: torch.Tensor,
                             grid_range: float, grid_size: int, sigma: float,
                             chunk_size: int = 512) -> torch.Tensor:
    """Soft target KL divergence loss with chunked computation for memory efficiency."""
    batch_size, num_pairs, gs, _ = log_density.shape
    device = log_density.device

    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Clip targets to grid range
    target_clipped = torch.clamp(target, -grid_range * 0.99, grid_range * 0.99)

    total_loss = 0.0
    total_count = 0

    for i in range(0, batch_size, chunk_size):
        end_i = min(i + chunk_size, batch_size)
        chunk_target = target_clipped[i:end_i]
        chunk_log_density = log_density[i:end_i]

        target_x = chunk_target[:, :, 0:1, None]
        target_y = chunk_target[:, :, 1:2, None]

        dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
        soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
        soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

        chunk_loss = F.kl_div(chunk_log_density, soft_target, reduction='none').sum(dim=(-2, -1))
        total_loss = total_loss + chunk_loss.sum()
        total_count += chunk_loss.numel()

    return total_loss / total_count


def gaussian_log_density_loss(pred_displacement: torch.Tensor, targets: torch.Tensor,
                              grid_range: float, grid_size: int, sigma: float,
                              dr_sigma: float) -> torch.Tensor:
    """Score an analytic point prediction the same way the model is scored.

    Builds a Gaussian density (width dr_sigma) centred on the predicted
    displacement, discretised on the same grid, and evaluates the soft-target
    KL loss against the true displacement. Used by baselines so their loss is
    directly comparable to the model's.
    """
    device = pred_displacement.device
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    pred = torch.clamp(pred_displacement, -grid_range * 0.99, grid_range * 0.99)
    px = pred[..., 0:1, None]
    py = pred[..., 1:2, None]
    dist_sq = (xx - px) ** 2 + (yy - py) ** 2
    density = torch.exp(-dist_sq / (2 * dr_sigma ** 2))
    density = density / (density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    log_density = torch.log(density + 1e-10)

    return compute_soft_target_loss(
        log_density.unsqueeze(0) if log_density.dim() == 3 else log_density,
        targets.unsqueeze(0) if targets.dim() == 2 else targets,
        grid_range, grid_size, sigma,
    )


def dead_reckoning_displacement(features: torch.Tensor, horizon_indices: torch.Tensor,
                                seq_len: int, norm: NormalizationConfig) -> torch.Tensor:
    """Predict displacement via constant velocity from the last input step.

    Returns (batch, num_horizons, 2) displacement in degrees, comparable to
    `CausalAISModel.forward_train(causal=False)` targets.
    """
    lat = features[:, :, 0] * norm.lat_scale + norm.lat_center
    lon = features[:, :, 1] * norm.lon_scale + norm.lon_center
    dt = features[:, :, 5] * norm.dt_scale

    last_dlat = lat[:, seq_len - 1] - lat[:, seq_len - 2]
    last_dlon = lon[:, seq_len - 1] - lon[:, seq_len - 2]
    last_dt = dt[:, seq_len - 1].clamp(min=1.0)

    vlat = last_dlat / last_dt   # deg / s
    vlon = last_dlon / last_dt

    cumsum_dt = torch.cumsum(dt, dim=1)
    last_cumsum = cumsum_dt[:, seq_len - 1]

    preds = []
    for h in horizon_indices:
        h_val = int(h)
        elapsed = cumsum_dt[:, seq_len + h_val - 1] - last_cumsum
        preds.append(torch.stack([vlat * elapsed, vlon * elapsed], dim=-1))
    return torch.stack(preds, dim=1)
