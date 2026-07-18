"""MDN head invariants: same contract as the other density heads."""
from __future__ import annotations

import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.models.fourier_head import MDNHead2D


def test_mdn_normalized_and_shaped():
    torch.manual_seed(0)
    head = MDNHead2D(32, grid_size=16, grid_range=0.3)
    ld = head(torch.randn(5, 32))
    assert ld.shape == (5, 16, 16)
    total = ld.reshape(5, -1).logsumexp(-1)
    assert torch.allclose(total, torch.zeros(5), atol=1e-5)


def test_mdn_bias_moves_output():
    torch.manual_seed(1)
    head = MDNHead2D(32, grid_size=16, grid_range=0.3)
    z = torch.randn(3, 32)
    bias = torch.zeros(3, 16, 16)
    bias[:, :4, :4] = 5.0
    assert not torch.allclose(head(z), head(z, bias=bias))
    assert torch.allclose(head(z), head(z, bias=torch.zeros(3, 16, 16)))


def test_mdn_gradient_flows():
    head = MDNHead2D(32, grid_size=16, grid_range=0.3, mlp_hidden=64)
    z = torch.randn(4, 32, requires_grad=True)
    head(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_mdn_sigma_floor_prevents_delta():
    """Even with extreme params the rendered density stays finite."""
    torch.manual_seed(2)
    head = MDNHead2D(32, grid_size=16, grid_range=0.3)
    with torch.no_grad():
        for p in head.params.parameters():
            p.mul_(100.0)
    ld = head(torch.randn(4, 32))
    assert torch.isfinite(ld).all()


def test_mdn_in_model_forward_train():
    torch.manual_seed(3)
    m = ModelConfig(d_model=32, nhead=2, num_layers=1, dim_feedforward=64,
                    grid_size=16, grid_range=0.3, grid_mode="cone",
                    head_type="mdn")
    model = CausalAISModel(m, NormalizationConfig(), max_horizon=40,
                          num_horizon_samples=2)
    f = torch.randn(2, 168, 6) * 0.1
    f[..., 5] = torch.rand(2, 168) * 0.5 + 0.1
    ld, tgt, hz, fm = model.forward_train(
        f, horizon_indices=torch.tensor([5, 20]), causal=True)
    assert ld.shape[2:] == (16, 16) and torch.isfinite(ld).all()
    ld.sum().backward()
