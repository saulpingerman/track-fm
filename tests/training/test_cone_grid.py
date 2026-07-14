"""Cone-grid canvas transform: TIME-based growth, equivalence, containment."""
from __future__ import annotations

import torch

from trackfm.training.losses import (compute_soft_target_loss,
                                     cone_elapsed_seconds, cone_ranges)


def test_degenerate_cone_equals_fixed_grid():
    """r0 = grid_range, v = 0 must reproduce the fixed-grid loss exactly:
    canvas scoring (targets/R, range 1, sigma/R) is a pure rescaling."""
    torch.manual_seed(0)
    B, P, G = 4, 3, 32
    ld = torch.log_softmax(torch.randn(B, P, G, G).reshape(B, P, -1), -1).reshape(B, P, G, G)
    tgt = torch.empty(B, P, 2).uniform_(-0.25, 0.25)
    elapsed = torch.tensor([[60.0, 1800.0, 7200.0]]).expand(B, P)
    fixed = compute_soft_target_loss(ld, tgt, 0.3, G, 0.003)
    R = cone_ranges(elapsed, r0=0.3, v=0.0)
    cone = compute_soft_target_loss(ld, tgt / R, 1.0, G, 0.003 / 0.3)
    torch.testing.assert_close(fixed, cone, rtol=1e-5, atol=1e-6)


def test_growth_is_per_time_not_per_step():
    """Two vessels at the same STEP horizon but different report cadence
    must get different windows — Paul's correction: AIS deltas are
    irregular, so the reachable set grows with wall-clock time."""
    seq, h = 4, torch.tensor([3])
    fast = torch.zeros(1, 8, 6); fast[..., 5] = 2.0 / 300.0    # 2 s cadence
    slow = torch.zeros(1, 8, 6); slow[..., 5] = 20.0 / 300.0   # 20 s cadence
    el_fast = cone_elapsed_seconds(fast, h, seq, 300.0, causal=False)
    el_slow = cone_elapsed_seconds(slow, h, seq, 300.0, causal=False)
    torch.testing.assert_close(el_fast, torch.tensor([[6.0]]))
    torch.testing.assert_close(el_slow, torch.tensor([[60.0]]))
    r_fast = cone_ranges(el_fast, 0.05, 0.00015)
    r_slow = cone_ranges(el_slow, 0.05, 0.00015)
    assert r_slow.max() > r_fast.max()


def test_causal_layout_matches_encoder_pairs():
    """Causal elapsed times: pairs [h0 x seq, h1 x seq], each pair's time =
    cumsum[p+h] - cumsum[p]."""
    seq = 3
    feats = torch.zeros(2, 8, 6)
    feats[0, :, 5] = 10.0 / 300.0                # constant 10 s
    feats[1, :, 5] = torch.tensor([5, 10, 20, 40, 5, 10, 20, 40]) / 300.0
    hz = torch.tensor([1, 4])
    el = cone_elapsed_seconds(feats, hz, seq, 300.0, causal=True)
    assert el.shape == (2, 2 * seq)
    torch.testing.assert_close(el[0], torch.tensor([10.0, 10, 10, 40, 40, 40]),
                               rtol=1e-5, atol=1e-4)
    # vessel 1, h=1, position 0: cumsum[1]-cumsum[0] = dt[1] = 10
    torch.testing.assert_close(el[1, 0], torch.tensor(10.0), rtol=1e-5, atol=1e-4)
    # vessel 1, h=4, position 2: dt[3]+dt[4]+dt[5]+dt[6] = 40+5+10+20 = 75
    torch.testing.assert_close(el[1, seq + 2], torch.tensor(75.0), rtol=1e-5, atol=1e-4)


def test_maneuver_containment_vs_fixed_grid():
    """A U-turning vessel stays near the origin (deep canvas interior);
    a max-speed straight-liner defines the canvas edge. The fixed grid
    clamps the straight-liner (|t| > 0.3)."""
    two_hours = torch.tensor([[7200.0]])
    R = cone_ranges(two_hours, 0.05, 0.00015)     # 1.13 deg at 2 h
    straight = torch.tensor([[[1.1, 0.0]]])       # far outside fixed ±0.3
    uturn = torch.tensor([[[0.05, 0.02]]])        # ended near its origin
    assert (straight.abs() > 0.3).any()           # fixed grid: clamped
    assert (straight / R).abs().max() <= 1.0      # cone: on canvas
    assert (uturn / R).abs().max() < 0.06         # maneuver = deep interior
