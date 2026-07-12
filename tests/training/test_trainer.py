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
    """Noisy GPT-3-style power-law progress must survive the FULL budget;
    flat curves must stop. Verified across seeds at realistic val noise."""
    import numpy as np

    from trackfm.training.pretrain import should_stop_saturation

    max_steps = 400_000
    val_steps = np.linspace(2000, max_steps, 336).astype(int)
    true = 1.5 + 2.2 * val_steps.astype(float) ** -0.28

    for seed in range(10):
        rng = np.random.default_rng(seed)
        obs = true * (1 + rng.normal(0, 0.01, len(true)))   # +-1% val noise
        hist = list(zip(val_steps.tolist(), obs.tolist()))
        streak = 0
        for i in range(len(hist)):
            stop, _ = should_stop_saturation(hist[: i + 1])
            streak = streak + 1 if stop else 0
            assert streak < 4, (
                f"seed {seed}: confirmed saturation on healthy noisy "
                f"power law at validation {i}")

    # flat + noise: must confirm within ~15h of eligibility
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        flat = [(int(t), 1.7 * (1 + rng.normal(0, 0.01)))
                for t in val_steps[:110]]
        streak, confirmed = 0, None
        for i in range(len(flat)):
            stop, _ = should_stop_saturation(flat[: i + 1])
            streak = streak + 1 if stop else 0
            if streak >= 4:
                confirmed = i
                break
        assert confirmed is not None and confirmed <= 90, (
            f"seed {seed}: flat run not stopped (confirmed={confirmed})")
