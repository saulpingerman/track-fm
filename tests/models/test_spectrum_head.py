"""SpectrumHead2D invariants.

The load-bearing test is ANALYTIC DILATION: coefficient slot j on a
canvas of half-range R queries phi at physical frequency j/(2R), so
slot 2j on a 2R canvas must return EXACTLY the same value — the
scale-equivariance the head exists to provide (a learned-dilation head
satisfies this only approximately, after training, if at all).
"""
from __future__ import annotations

import pytest
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.models.fourier_head import SpectrumHead2D

SEQ, MAX_H = 32, 64


def _head(seed=0, zero_tail=False):
    torch.manual_seed(seed)
    h = SpectrumHead2D(64, grid_size=16, num_freqs=4, grid_range=0.3,
                       mlp_hidden=64)
    if not zero_tail:
        torch.nn.init.normal_(h.tail[-1].weight, std=0.05)
        torch.nn.init.normal_(h.tail[-1].bias, std=0.05)
    return h


def test_zero_init_uniform_at_any_scale():
    h = _head(zero_tail=True)
    z = torch.randn(6, 64)
    for R in (0.05, 0.3, 1.25):
        ld = h(z, R_deg=torch.full((6,), R))
        assert torch.allclose(ld, torch.full_like(ld, ld[0, 0, 0].item()), atol=1e-6)
        assert torch.allclose(torch.exp(ld).sum(dim=(1, 2)),
                              torch.ones(6), atol=1e-4)


def _coeffs(h, z, R):
    hh = h.trunk(z)
    k = torch.stack([h.freq_x, h.freq_y], -1).unsqueeze(0) / (2.0 * R.view(-1, 1, 1))
    return h.tail(h.k_proj(h._gamma(k)) + h.h_proj(hh).unsqueeze(1))


def test_analytic_dilation_exact():
    """slot j at R == slot 2j at 2R, exactly, for a NON-zero phi."""
    h = _head()
    z = torch.randn(3, 64)
    R = torch.full((3,), 0.2)
    cA = _coeffs(h, z, R)             # (3, P, 2), F=4 lattice (9x9)
    cB = _coeffs(h, z, 2 * R)
    F_ = h.num_freqs
    side = 2 * F_ + 1

    def slot(jx, jy):
        return (jx + F_) * side + (jy + F_)

    for jx, jy in [(1, 0), (0, 1), (1, -1), (2, 2), (-2, 1)]:
        a = cA[:, slot(jx, jy)]
        b = cB[:, slot(2 * jx, 2 * jy)]
        assert torch.allclose(a, b, atol=1e-6), (jx, jy)


def test_fixed_geometry_default_R():
    h = _head()
    z = torch.randn(4, 64)
    ld_default = h(z)
    ld_explicit = h(z, R_deg=torch.full((4,), h.grid_range))
    assert torch.allclose(ld_default, ld_explicit)


def _cfg(**over):
    return ModelConfig(d_model=64, nhead=4, num_layers=2,
                       dim_feedforward=256, dropout=0.0, max_seq_len=SEQ,
                       grid_size=16, grid_range=0.3, num_freqs=4,
                       head_type="spectrum", head_mlp_hidden=64,
                       grid_mode="cone", **over)


def test_encoder_integration_and_grads():
    torch.manual_seed(0)
    model = CausalAISModel(_cfg(), NormalizationConfig(), max_horizon=MAX_H,
                           num_horizon_samples=2)
    g = torch.Generator().manual_seed(1)
    f = 0.1 * torch.randn(2, SEQ + MAX_H, 6, generator=g)
    f[..., 5] = 0.2 + 0.5 * torch.rand(2, SEQ + MAX_H, generator=g)
    ld, tgt, hz, fm = model.forward_train(f, causal=True)
    assert ld.shape[-2:] == (16, 16)
    ld.sum().backward()
    assert model.fourier_head.tail[-3].weight.grad is not None
    assert model.fourier_head.trunk[0].weight.grad is not None
    ld2, *_ = model.forward_train(
        f, horizon_indices=torch.tensor([2, 8]), causal=False)
    assert ld2.shape[1] == 2


def test_mup_spectrum_refused():
    with pytest.raises(ValueError, match="spectrum"):
        ModelConfig(d_model=64, nhead=4, dim_feedforward=256,
                    head_type="spectrum",
                    mup={"enabled": True, "d_base": 32, "d_head": 16})
