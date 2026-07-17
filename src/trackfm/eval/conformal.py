"""Split-conformal HDR calibration for grid density heads.

CE-optimal density heads are systematically overconfident in the 2-3
sigma tails — exactly where km2@90 lives — so raw HDR (highest-density-
region) areas under-cover. Split-conformal calibration fixes coverage
with distribution-free guarantees at the cost of a held-out calibration
set and minutes of compute:

  score_i = predicted mass of all cells at least as dense as the truth
            cell (the "HDR mass at truth"; Uniform(0,1) iff calibrated)
  tau     = ceil((1-alpha)(n+1))/n smallest-score quantile
  region  = the HDR containing tau mass; empirical coverage of that
            region is >= 1-alpha by the standard conformal argument.

Calibrated km2@90 = area of the tau-mass HDR, reported per horizon
bucket (stratify by calling once per bucket — thresholds differ).
Disconnected HDR components are allowed by construction (crucial at
route junctions where a connected region would inflate area).
"""
from __future__ import annotations

import math

import torch


def hdr_mass_at_truth(log_density: torch.Tensor,
                      truth_idx: torch.Tensor) -> torch.Tensor:
    """Nonconformity scores: HDR mass at the truth cell.

    log_density: (B, G, G) normalized log-densities.
    truth_idx: (B, 2) integer [row, col] of the truth cell.
    Returns (B,) scores in (0, 1]: the total mass of cells with density
    >= the truth cell's (small = truth in a high-density region).
    """
    B, G, _ = log_density.shape
    flat = log_density.reshape(B, -1)
    p = flat.softmax(dim=-1) if not torch.allclose(
        flat.exp().sum(-1), torch.ones(B), atol=1e-3) else flat.exp()
    t_flat = truth_idx[:, 0] * G + truth_idx[:, 1]
    p_truth = torch.gather(p, 1, t_flat.unsqueeze(-1))
    return (p * (p >= p_truth)).sum(dim=-1)


def conformal_threshold(scores: torch.Tensor, alpha: float = 0.1) -> float:
    """Split-conformal quantile: the ceil((1-alpha)(n+1))-th smallest score.

    Guarantees P(score_new <= tau) >= 1-alpha for exchangeable data.
    """
    n = scores.numel()
    k = math.ceil((1 - alpha) * (n + 1))
    if k > n:                                   # too few calibration points
        return 1.0
    return float(scores.sort().values[k - 1])


def hdr_region_cells(log_density: torch.Tensor, tau: float) -> torch.Tensor:
    """Cells in the tau-mass HDR per sample.

    Returns (B,) int counts: the number of highest-density cells whose
    cumulative mass first reaches tau. Multiply by cell area for the
    calibrated search-area metric.
    """
    B = log_density.shape[0]
    flat = log_density.reshape(B, -1)
    p = flat.exp()
    p = p / p.sum(dim=-1, keepdim=True)
    p_sorted = p.sort(dim=-1, descending=True).values
    cum = p_sorted.cumsum(dim=-1)
    return (cum < tau).sum(dim=-1) + 1


def calibrated_coverage(scores: torch.Tensor, tau: float) -> float:
    """Empirical coverage of the tau-region on a held-out score set."""
    return float((scores <= tau).float().mean())
