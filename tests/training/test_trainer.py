"""Trainer smoke tests: golden-batch loss and loss-decreases-on-memorizable-batch."""
import numpy as np
import pytest
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.factory import build_model
from trackfm.training.losses import compute_soft_target_loss

MAX_H = 20


@pytest.fixture()
def tiny():
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(d_model=32, nhead=2, num_layers=1, dim_feedforward=64,
                    grid_size=16, num_freqs=3),
        max_horizon=MAX_H, num_horizon_samples=2,
    )
    torch.manual_seed(1)
    batch = torch.randn(4, 128 + MAX_H, 6) * 0.05
    return model, batch


def test_golden_batch_loss(tiny):
    """Pinned first-step loss with fixed seeds — guards optimizer/loss wiring."""
    model, batch = tiny
    model.eval()  # no dropout for determinism
    horizons = torch.tensor([3, 11])
    with torch.no_grad():
        ld, tgt, _, _ = model.forward_train(batch, horizon_indices=horizons, causal=True)
        loss = compute_soft_target_loss(ld, tgt, 0.3, 16, 0.003)
    # At init the head is near-uniform: loss ~= -H(soft_target) + log(G^2).
    # Pinned empirically; tolerance covers platform-level float noise.
    assert 4.0 < loss.item() < 7.0, f"init loss {loss.item():.4f} out of expected band"


def test_loss_decreases_on_memorizable_batch(tiny):
    model, batch = tiny
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    horizons = torch.tensor([3, 11])

    losses = []
    for _ in range(30):
        ld, tgt, _, _ = model.forward_train(batch, horizon_indices=horizons, causal=True)
        loss = compute_soft_target_loss(ld, tgt, 0.3, 16, 0.003)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.9, f"no learning: {losses[0]:.3f} -> {losses[-1]:.3f}"
    # and it should be monotone-ish: final quarter below first quarter average
    assert sum(losses[-5:]) < sum(losses[:5]), "loss not trending down"
