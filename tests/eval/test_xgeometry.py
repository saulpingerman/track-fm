"""Cross-geometry containment scoring — ranks_on_fine_grid properties.

These tests pin the semantics that the flagship comparison depends on:
- restrict_deg=None yields the model's full canvas (baseline).
- restrict_deg=R_deg is a no-op (fixed grid's own extent).
- restrict_deg < R_deg samples ONLY the restricted physical region AND
  censors targets outside that region (matching what a fixed-grid model
  of ±restrict_deg would evaluate).
- Cell size (cell_deg vs target_cell_km) doesn't leak into restriction.
- Bilinear interpolation is well-behaved at restricted-region boundaries.
"""
from __future__ import annotations

import math

import pytest
import torch

from trackfm.eval.xgeometry import (FIXED_NATIVE_CELL_DEG, KM_PER_DEG, LAT0,
                                     ranks_on_fine_grid)


def _random_logdensity(B=8, G=64, seed=0):
    torch.manual_seed(seed)
    return torch.log_softmax(torch.randn(B, G, G).reshape(B, -1), dim=-1
                              ).reshape(B, G, G)


# -------------------- no-op cases --------------------

def test_restrict_equals_R_is_noop_ranks():
    """restrict_deg == R_deg must produce IDENTICAL ranks to no restriction —
    same fine cells, same censoring, same physical extent."""
    ld = _random_logdensity()
    tgt = torch.empty(8, 2).uniform_(-0.5, 0.5)
    for R in (0.3, 1.0, 1.25):
        r_none, n_none, a_none = ranks_on_fine_grid(
            ld, tgt, R_deg=R, cell_deg=0.02)
        r_full, n_full, a_full = ranks_on_fine_grid(
            ld, tgt, R_deg=R, cell_deg=0.02, restrict_deg=R)
        torch.testing.assert_close(r_none, r_full, rtol=0, atol=0)
        assert n_none == n_full and a_none == pytest.approx(a_full)


def test_fixed_model_fixcell_equals_fixgrid():
    """For a fixed-grid model (R_deg=grid_range), restricting to grid_range
    is a no-op — the metric val_fixcell_p90rank == val_fixgrid_p90rank."""
    ld = _random_logdensity(B=32, seed=17)
    tgt = torch.empty(32, 2).uniform_(-0.9, 0.9)          # canvas coords
    r_fixcell, _, _ = ranks_on_fine_grid(ld, tgt, R_deg=0.3,
                                          cell_deg=FIXED_NATIVE_CELL_DEG)
    r_fixgrid, _, _ = ranks_on_fine_grid(ld, tgt, R_deg=0.3,
                                          cell_deg=FIXED_NATIVE_CELL_DEG,
                                          restrict_deg=0.3)
    torch.testing.assert_close(r_fixcell, r_fixgrid, rtol=0, atol=0)


# -------------------- restriction correctness --------------------

def test_restrict_shrinks_fine_grid_count():
    """restrict_deg < R_deg reduces the fine grid to the smaller physical
    region; cell count scales linearly with extent (cell_deg fixed)."""
    ld = _random_logdensity()
    tgt = torch.zeros(8, 2)
    _, n_full, _ = ranks_on_fine_grid(ld, tgt, R_deg=1.0, cell_deg=0.02)
    _, n_res, _ = ranks_on_fine_grid(ld, tgt, R_deg=1.0, cell_deg=0.02,
                                      restrict_deg=0.3)
    # extent shrinks 1.0 -> 0.3 (ratio 0.3), cells scale as ratio² for square grid
    assert n_full > n_res
    ratio = (n_res / n_full) ** 0.5
    assert 0.28 < ratio < 0.32                                    # ≈ 0.3


def test_outside_restrict_targets_are_censored():
    """A target physical |t| > restrict_deg must return rank=-1, exactly the
    censoring a fixed grid of ±restrict_deg would apply — this is the
    filtering that matches vessel populations."""
    ld = _random_logdensity()
    R_deg = 1.0
    restrict = 0.3
    # canvas coords = physical / R_deg. targets at physical (0.5, 0.0) and
    # (0.0, 0.5) are inside cone canvas (|c|=0.5<1) but outside restrict.
    tgt = torch.tensor([
        [0.10, 0.10],   # physical 0.10° → inside restrict
        [0.29, 0.00],   # physical 0.29° → inside restrict (edge)
        [0.31, 0.00],   # physical 0.31° → OUTSIDE restrict → censored
        [0.00, 0.50],   # physical 0.50° → OUTSIDE → censored
        [0.90, 0.00],   # inside cone canvas but way outside restrict → censored
    ])
    ranks, _, _ = ranks_on_fine_grid(ld[:5], tgt, R_deg=R_deg,
                                      cell_deg=FIXED_NATIVE_CELL_DEG,
                                      restrict_deg=restrict)
    assert ranks[0].item() > 0                                    # inside
    assert ranks[1].item() > 0                                    # inside (near edge)
    assert ranks[2].item() == -1                                  # outside
    assert ranks[3].item() == -1                                  # outside
    assert ranks[4].item() == -1                                  # outside


def test_restrict_censors_same_targets_as_true_fixed_grid():
    """The censoring set under restrict_deg must EXACTLY match the censoring
    a native fixed-grid model of that extent would apply."""
    ld = _random_logdensity(B=64, seed=42)
    R_deg = 1.5
    restrict = 0.3
    # random physical targets in [-1, 1] (many outside restrict)
    torch.manual_seed(1)
    tgt_phys = torch.empty(64, 2).uniform_(-0.8, 0.8)
    tgt_canvas = tgt_phys / R_deg
    # who WOULD a native fixed-±0.3° grid censor? |physical| > 0.3
    would_be_censored = (tgt_phys.abs() > restrict).any(dim=-1)
    ranks, _, _ = ranks_on_fine_grid(ld, tgt_canvas, R_deg=R_deg,
                                      cell_deg=FIXED_NATIVE_CELL_DEG,
                                      restrict_deg=restrict)
    censored_by_us = ranks == -1
    torch.testing.assert_close(censored_by_us, would_be_censored,
                                rtol=0, atol=0)


# -------------------- truth-index correctness --------------------

def test_truth_cell_is_argmax_when_density_is_peaked_there():
    """Place a strong peak at a known physical location; the truth cell
    (that location) must get rank 1 whether restricted or not — semantics
    of the restricted fine grid preserve peak-follows-truth."""
    # peaked density at native NODE (10, 40) on a 64×64 canvas [-1,1]².
    # metrics v2: density values live at lattice nodes linspace(-1,1,64)
    # (pitch 2/63), not edge-tiled cell centers.
    ld = torch.full((1, 64, 64), -20.0)
    ld[0, 10, 40] = 5.0
    ld = torch.log_softmax(ld.reshape(1, -1), -1).reshape(1, 64, 64)
    # node 10 canvas y = -1 + 10*2/63 = -0.6825; node 40 x = +0.2698
    tgt = torch.tensor([[-0.6825, 0.2698]])
    # R=1.0 for canvas. physical target = canvas here.
    r_full, _, _ = ranks_on_fine_grid(ld, tgt, R_deg=1.0, cell_deg=0.02)
    assert r_full.item() == 1                                     # peak == truth
    # restrict to ±0.7 — target still inside (|-0.672|<0.7, |0.266|<0.7)
    r_res, _, _ = ranks_on_fine_grid(ld, tgt, R_deg=1.0, cell_deg=0.02,
                                      restrict_deg=0.7)
    assert r_res.item() == 1


def test_cell_area_scales_correctly_when_restricted():
    """cell_area_km2 must reflect the actual cell size — cell_deg fixed →
    same area under restriction; km-targeted cells → area matches km target."""
    ld = _random_logdensity()
    tgt = torch.zeros(8, 2)
    _, _, area_full = ranks_on_fine_grid(ld, tgt, R_deg=1.0,
                                          cell_deg=FIXED_NATIVE_CELL_DEG)
    _, _, area_res = ranks_on_fine_grid(ld, tgt, R_deg=1.0,
                                         cell_deg=FIXED_NATIVE_CELL_DEG,
                                         restrict_deg=0.3)
    # cell_deg unchanged -> area unchanged regardless of restriction
    assert area_full == pytest.approx(area_res, rel=1e-9)
    # sanity: fixed native cell area = (0.009375°·111.32km) × (·cos(56.25°))
    expected = (FIXED_NATIVE_CELL_DEG * KM_PER_DEG) ** 2 * math.cos(math.radians(LAT0))
    assert area_full == pytest.approx(expected, rel=1e-6)


# -------------------- inside-population invariance --------------------

def test_restricted_rank_matches_unrestricted_when_target_is_inside():
    """A target strictly inside ±restrict is inside BOTH the full and
    restricted regions. The RANK differs (different cell sets) but both
    must give a defined (positive) rank. The important guarantee is that
    the restriction doesn't accidentally censor a target that is inside."""
    ld = _random_logdensity(B=16, seed=99)
    torch.manual_seed(3)
    tgt = torch.empty(16, 2).uniform_(-0.2, 0.2)                   # canvas coords, all inside ±0.3 if R=1
    r_full, _, _ = ranks_on_fine_grid(ld, tgt, R_deg=1.0,
                                       cell_deg=FIXED_NATIVE_CELL_DEG)
    r_res, _, _ = ranks_on_fine_grid(ld, tgt, R_deg=1.0,
                                      cell_deg=FIXED_NATIVE_CELL_DEG,
                                      restrict_deg=0.3)
    assert (r_full > 0).all() and (r_res > 0).all()
