"""Split-conformal HDR invariants: score distribution under a perfect
model, the conformal coverage guarantee, region-size monotonicity."""
from __future__ import annotations

import torch

from trackfm.eval.conformal import (calibrated_coverage, conformal_threshold,
                                     hdr_mass_at_truth, hdr_region_cells)


def _gaussian_grid(B, G=32, sigma_cells=3.0, seed=0):
    """Gaussian densities + truths SAMPLED FROM the density (perfect model)."""
    g = torch.Generator().manual_seed(seed)
    c = torch.arange(G, dtype=torch.float)
    xx, yy = torch.meshgrid(c, c, indexing="ij")
    mu = torch.randint(8, G - 8, (B, 2), generator=g).float()
    d2 = (xx - mu[:, 0, None, None]) ** 2 + (yy - mu[:, 1, None, None]) ** 2
    logit = -d2 / (2 * sigma_cells ** 2)
    ld = logit.reshape(B, -1).log_softmax(-1).reshape(B, G, G)
    # sample truth cells from each density
    idx_flat = torch.multinomial(ld.reshape(B, -1).exp(), 1, generator=g).squeeze(-1)
    truth = torch.stack([idx_flat // G, idx_flat % G], dim=-1)
    return ld, truth


def test_scores_near_uniform_for_perfect_model():
    ld, truth = _gaussian_grid(4000)
    s = hdr_mass_at_truth(ld, truth)
    assert 0.0 < s.min() and s.max() <= 1.0 + 1e-6
    # mean of U(0,1) is 0.5; discretization inflates slightly
    assert 0.40 < s.mean().item() < 0.65


def test_conformal_coverage_guarantee():
    """Coverage on held-out data must be >= 1-alpha (the whole point)."""
    ld_cal, tr_cal = _gaussian_grid(3000, seed=1)
    ld_test, tr_test = _gaussian_grid(3000, seed=2)
    for alpha in (0.1, 0.25):
        tau = conformal_threshold(hdr_mass_at_truth(ld_cal, tr_cal), alpha)
        cov = calibrated_coverage(hdr_mass_at_truth(ld_test, tr_test), tau)
        assert cov >= 1 - alpha - 0.02, (alpha, cov)     # small slack for finite n


def test_region_grows_with_tau():
    ld, _ = _gaussian_grid(16, seed=3)
    a = hdr_region_cells(ld, 0.5)
    b = hdr_region_cells(ld, 0.9)
    assert (b >= a).all() and b.float().mean() > a.float().mean()


def test_sharper_density_smaller_region():
    ld_sharp, _ = _gaussian_grid(64, sigma_cells=1.5, seed=4)
    ld_blur, _ = _gaussian_grid(64, sigma_cells=5.0, seed=4)
    assert hdr_region_cells(ld_sharp, 0.9).float().mean() < \
           hdr_region_cells(ld_blur, 0.9).float().mean()


def test_threshold_saturates_with_tiny_calibration_set():
    tau = conformal_threshold(torch.tensor([0.5, 0.6]), alpha=0.1)
    assert tau == 1.0                                   # can't certify; full region
