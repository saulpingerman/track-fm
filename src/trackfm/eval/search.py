"""Search-effort metric: how many cells must be tasked to capture the target?

Operational framing (satellite imaging): rank cells by predicted
probability; the metric is the rank of the cell containing the true
position — the number of images tasked before one captures the vessel.
Reported as mean/median/P90 rank and the capture-vs-k curve per horizon.

Properties:
  * evaluates the whole density (multimodal hedging earns credit)
  * fair to Gaussian baselines: they rank cells natively (elliptical
    shells); a point predictor's best strategy — concentric rings around
    its point — is ordering-equivalent to an isotropic Gaussian centred
    there, so tuned-DR *is* the honest "point + rings" competitor
  * targets outside the grid are CENSORED (rank = -1), reported as the
    uncaptured fraction — cells the grid cannot express cannot be tasked

`footprint_cells` aggregates the grid into square blocks approximating an
imaging footprint (e.g. 4 -> 4x4 grid cells per image). H3 hexagons are the
production version of the same idea; block aggregation is metric-identical
in structure and dependency-free.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def search_ranks(
    log_density: torch.Tensor,   # (B, H, G, G)
    targets: torch.Tensor,       # (B, H, 2) displacement in degrees
    grid_range: float,
    footprint_cells: int = 1,
) -> torch.Tensor:
    """Rank (1-based) of the footprint containing the truth; -1 if off-grid."""
    B, H, G0, _ = log_density.shape
    probs = log_density.float().exp()
    G = G0
    if footprint_cells > 1:
        probs = F.avg_pool2d(
            probs.reshape(B * H, 1, G0, G0), footprint_cells).reshape(
            B, H, G0 // footprint_cells, G0 // footprint_cells)
        probs = probs * footprint_cells ** 2      # sum-pool: block probability
        G = G0 // footprint_cells

    # truth -> NEAREST density node, then block. Density values live at
    # lattice nodes linspace(-R, R, G0) with pitch 2R/(G0-1); the old
    # floor-index with width 2R/G0 assigned 24.6% of targets (measured,
    # G=64) to a non-nearest node, biased away from canvas center
    # (metrics v2, audit F7).
    node = ((targets + grid_range) / (2 * grid_range / (G0 - 1))).round().long()
    idx = (node // footprint_cells) if footprint_cells > 1 else node  # (B, H, 2)
    inside = ((targets > -grid_range) & (targets < grid_range)).all(dim=-1)
    idx = idx.clamp(0, G - 1)

    flat = probs.reshape(B, H, G * G)
    truth_flat = (idx[..., 0] * G + idx[..., 1])                  # (B, H)
    p_truth = torch.gather(flat, 2, truth_flat.unsqueeze(-1)).squeeze(-1)
    ranks = (flat > p_truth.unsqueeze(-1)).sum(dim=-1) + 1        # (B, H)
    return torch.where(inside, ranks, torch.full_like(ranks, -1))


def summarize_ranks(ranks: torch.Tensor, ks: tuple = (1, 3, 10, 30, 100)) -> dict:
    """Mean/median/P90 rank + capture rate at k images + censored fraction."""
    total = ranks.numel()
    valid = ranks[ranks > 0].float()
    out = {"censored_frac": float((ranks < 0).sum()) / max(total, 1)}
    if len(valid) == 0:
        return out
    out.update({
        "mean_rank": float(valid.mean()),
        "median_rank": float(valid.median()),
        "p90_rank": float(valid.quantile(0.9)),
    })
    for k in ks:
        # capture@k counts censored samples as never captured
        captured = ((ranks > 0) & (ranks <= k)).sum()
        out[f"capture@{k}"] = float(captured) / total
    return out


def capture_curve(ranks: torch.Tensor, n_cells: int) -> dict:
    """Full containment curve: capture probability vs image budget k.

    ROC-like: the skill-less null (uniform density) is the diagonal k/n_cells;
    the model's curve bows above it. Returns the curve plus:
      k@90  -- images needed for 90% containment (None if ceiling < 0.9)
      auc   -- area under the capture curve, normalized to [0, 1]
      ceiling -- max achievable capture (1 - censored fraction)
    """
    total = ranks.numel()
    valid = ranks[ranks > 0]
    ceiling = len(valid) / max(total, 1)

    ks = torch.arange(1, n_cells + 1)
    if len(valid) == 0:
        capture = torch.zeros(n_cells)
    else:
        counts = torch.bincount(valid.reshape(-1).long(), minlength=n_cells + 1)[1:]
        capture = counts.cumsum(0).float() / total

    at90 = (capture >= 0.9).nonzero()
    return {
        "k": ks.tolist(),
        "capture": capture.tolist(),
        "ceiling": ceiling,
        "k@90": int(at90[0]) + 1 if len(at90) else None,
        "auc": float(capture.mean()),          # normalized: mean over k in [1, N]
        "null_auc": float((ks.float() / n_cells).mean()),
    }


def gaussian_grid_log_density(pred: torch.Tensor, sigma: torch.Tensor,
                              grid_range: float, grid_size: int) -> torch.Tensor:
    """Rasterize a Gaussian point-forecast onto the grid for ranking.

    pred, sigma: (B, H, 2). Returns (B, H, G, G) log-density.
    """
    device = pred.device
    ax = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    sig = torch.clamp(sigma, min=1e-4)
    zx = (xx - pred[..., 0:1, None]) / sig[..., 0:1, None]
    zy = (yy - pred[..., 1:2, None]) / sig[..., 1:2, None]
    # Analytic log-density (metrics v2, audit F2): exp-then-log(+1e-10)
    # floored bad misses at ~23 nats and created rank-tie plateaus that
    # flattered badly-missed baselines; log-space has no floor.
    logit = -0.5 * (zx ** 2 + zy ** 2)
    return logit - torch.logsumexp(logit.reshape(*logit.shape[:-2], -1),
                                   dim=-1)[..., None, None]
