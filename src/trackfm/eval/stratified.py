"""Difficulty-stratified evaluation.

Straight-line transits dominate the data; the operationally interesting
trajectories (maneuvers, port approaches) are the hard tail. Difficulty is
scored per sample-horizon by the DEAD-RECKONING RESIDUAL — how far the
constant-velocity extrapolation missed — a domain-native hardness signal
that needs no reference model. Metrics reported per difficulty decile
answer: is the model coasting on the easy majority, or does it actually
beat baselines where kinematics fail?

Caveat carried from the hard-vs-noisy literature: the top decile includes
some unlearnable noise (GPS glitches); interpret its absolute level
accordingly — the model-vs-baseline RATIO per decile is the honest signal.
"""
from __future__ import annotations

import torch


def per_sample_soft_ce(log_density: torch.Tensor, targets: torch.Tensor,
                       grid_range: float, grid_size: int,
                       sigma: float) -> torch.Tensor:
    """Per-sample soft-target CE: (B, H, G, G) x (B, H, 2) -> (B, H).

    Same math as training's compute_soft_target_loss without the mean
    reduction (verified equal in expectation by test).
    """
    device = log_density.device
    ax = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    t = torch.clamp(targets, -grid_range * 0.99, grid_range * 0.99)
    dist_sq = (xx - t[..., 0:1, None]) ** 2 + (yy - t[..., 1:2, None]) ** 2
    soft = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft = soft / (soft.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    # KL(p||q) summed over the grid, per (sample, horizon)
    return (soft * (torch.log(soft + 1e-10) - log_density)).sum(dim=(-2, -1))


def difficulty_score(dr_pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """DR residual magnitude (degrees): (B, H, 2) x2 -> (B, H)."""
    return (dr_pred - targets).norm(dim=-1)


def stratified_table(difficulty: torch.Tensor, metrics: dict[str, torch.Tensor],
                     n_bins: int = 10) -> dict:
    """Decile report. difficulty/metrics: aligned 1-D tensors (one horizon).

    Returns {decile: {"difficulty_range": [lo, hi], "n": int, <metric>: mean
    or median as appropriate}}. Rank-type metrics (integer-ish, heavy-tailed)
    get medians; loss-type get means.
    """
    edges = torch.quantile(difficulty, torch.linspace(0, 1, n_bins + 1))
    out = {}
    for d in range(n_bins):
        lo, hi = edges[d], edges[d + 1]
        mask = (difficulty >= lo) & (difficulty <= hi if d == n_bins - 1
                                     else difficulty < hi)
        if mask.sum() == 0:
            continue
        row: dict = {"difficulty_range_deg": [float(lo), float(hi)],
                     "n": int(mask.sum())}
        for name, vals in metrics.items():
            v = vals[mask].float()
            if name.startswith("rank"):
                valid = v[v > 0]
                row[f"{name}_median"] = float(valid.median()) if len(valid) else None
                row[f"capture10"] = float(((v > 0) & (v <= 10)).float().mean())
                row[f"censored"] = float((v < 0).float().mean())
            else:
                row[f"{name}_mean"] = float(v.mean())
        out[d] = row
    return out
