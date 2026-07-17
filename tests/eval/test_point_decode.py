"""Point-decode invariants: peak recovery, expectation symmetry, units."""
from __future__ import annotations

import math

import torch

from trackfm.eval.point_decode import (KM_PER_DEG, decode_points,
                                        displacement_error_m)


def _peaked(G=64, iy=40, ix=10, B=2):
    ld = torch.full((B, G, G), -20.0)
    ld[:, iy, ix] = 5.0
    return torch.log_softmax(ld.reshape(B, -1), -1).reshape(B, G, G)


def test_argmax_recovers_peak_cell_center():
    G = 64
    ld = _peaked(G, iy=40, ix=10)
    d = decode_points(ld, R_deg=0.3, mode="argmax")
    exp_lat = (-1 + 2 * 40 / (G - 1)) * 0.3
    exp_lon = (-1 + 2 * 10 / (G - 1)) * 0.3
    assert torch.allclose(d[0], torch.tensor([exp_lat, exp_lon]), atol=1e-6)


def test_expectation_matches_argmax_for_sharp_peak():
    ld = _peaked()
    da = decode_points(ld, 0.3, "argmax")
    de = decode_points(ld, 0.3, "expectation")
    assert torch.allclose(da, de, atol=0.01)


def test_expectation_of_uniform_is_center():
    G = 32
    ld = torch.zeros(1, G, G).log_softmax(-1)  # uniform-ish after softmax over last dim only
    ld = torch.full((1, G, G), math.log(1.0 / (G * G)))
    d = decode_points(ld, 1.0, "expectation")
    assert d.abs().max().item() < 1e-5


def test_per_pair_R_scaling():
    """Same canvas density, different R -> proportionally scaled decode."""
    ld = _peaked(B=2)
    R = torch.tensor([0.2, 1.0])
    d = decode_points(ld, R, "argmax")
    assert torch.allclose(d[1], d[0] * 5.0, atol=1e-6)


def test_displacement_error_units():
    p = torch.tensor([[0.1, 0.0]])              # 0.1 deg lat = 11.132 km
    t = torch.zeros(1, 2)
    e = displacement_error_m(p, t)
    assert abs(e.item() - 0.1 * KM_PER_DEG * 1000) < 1.0
    p2 = torch.tensor([[0.0, 0.1]])             # lon shrinks by cos(lat0)
    e2 = displacement_error_m(p2, t)
    assert e2.item() < e.item()
