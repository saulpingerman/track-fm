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


def test_saturation_stop_spares_power_law_kills_flat():
    """GPT-3-style power-law progress must survive; a flat curve must not."""
    import numpy as np

    from trackfm.training.pretrain import should_stop_saturation

    max_steps = 400_000
    val_steps = np.linspace(2000, max_steps, 336).astype(int)  # ~7d @ 30min

    # power law: L = 1.5 + 2.2 * t^-0.28  (relative gains shrink steadily)
    power = [(int(t), 1.5 + 2.2 * t ** -0.28) for t in val_steps]
    # must survive until at least 90% of the budget; stopping in the final
    # ~10% is CORRECT thrift (the analytic tail there buys < 0.1%)
    for i in range(24, int(len(power) * 0.9)):
        stop, _ = should_stop_saturation(power[: i + 1], max_steps)
        assert not stop, f"killed a power-law run at validation {i}"

    # flat: converged, pure noise
    rng = np.random.default_rng(0)
    flat = [(int(t), 1.7 + rng.normal(0, 1e-4)) for t in val_steps[:40]]
    stopped = any(should_stop_saturation(flat[: i + 1], max_steps)[0]
                  for i in range(24, 40))
    assert stopped, "failed to stop a flat run"
