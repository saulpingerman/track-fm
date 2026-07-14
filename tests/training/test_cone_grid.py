"""Cone-grid canvas transform: equivalence and containment properties."""
from __future__ import annotations

import torch

from trackfm.training.losses import compute_soft_target_loss, cone_ranges


def test_degenerate_cone_equals_fixed_grid():
    """r0 = grid_range, v = 0 must reproduce the fixed-grid loss exactly:
    canvas scoring (targets/R, range 1, sigma/R) is a pure rescaling."""
    torch.manual_seed(0)
    B, P, G = 4, 3, 32
    ld = torch.log_softmax(torch.randn(B, P, G, G).reshape(B, P, -1), -1).reshape(B, P, G, G)
    tgt = torch.empty(B, P, 2).uniform_(-0.25, 0.25)
    h = torch.tensor([10, 100, 800])
    fixed = compute_soft_target_loss(ld, tgt, 0.3, G, 0.003)
    R = cone_ranges(h, r0=0.3, v=0.0)
    cone = compute_soft_target_loss(ld, tgt / R, 1.0, G, 0.003 / 0.3)
    torch.testing.assert_close(fixed, cone, rtol=1e-5, atol=1e-6)


def test_cone_ranges_shapes_and_growth():
    hP = torch.tensor([1, 400, 800])
    R = cone_ranges(hP, 0.05, 0.0015)
    assert R.shape == (3, 1)
    assert torch.allclose(R.squeeze(-1),
                          torch.tensor([0.0515, 0.65, 1.25]))
    hBK = torch.arange(6).reshape(2, 3)
    assert cone_ranges(hBK, 0.05, 0.0015).shape == (2, 3, 1)


def test_maneuver_containment_vs_fixed_grid():
    """A vessel doing a U-turn stays NEAR the origin: canvas coordinate
    shrinks with horizon, while a max-speed straight-liner sits at the
    canvas edge. The fixed grid clamps the straight-liner (|t| > 0.3)."""
    h = torch.tensor([800])
    R = cone_ranges(h, 0.05, 0.0015)          # 1.25 at h=800
    straight = torch.tensor([[[1.2, 0.0]]])    # far outside fixed ±0.3
    uturn = torch.tensor([[[0.05, 0.02]]])     # ended near where it started
    assert (straight.abs() > 0.3).any()        # fixed grid: clamped/censored
    s_c, u_c = straight / R, uturn / R
    assert s_c.abs().max() <= 1.0              # cone: still on canvas
    assert u_c.abs().max() < 0.05              # maneuver = deep interior
