"""Numerical equivalence between the ported CausalAISModel and the legacy
experiment-11 implementation.

The pretrained checkpoint from the paper no longer exists, so code-level
equivalence is the only regression guard for the port.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from reference.legacy_model import CausalAISModel as LegacyModel
from reference.legacy_model import Config as LegacyConfig
from reference.legacy_model import compute_soft_target_loss as legacy_loss

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.training.losses import compute_soft_target_loss

SEQ_LEN = 128
MAX_HORIZON = 50
BATCH = 4


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    legacy_cfg = LegacyConfig(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        max_seq_len=SEQ_LEN, max_horizon=MAX_HORIZON, num_horizon_samples=3,
        grid_size=32, grid_range=0.3, num_freqs=6,
    )
    legacy = LegacyModel(legacy_cfg).eval()

    ported = CausalAISModel(
        model=ModelConfig(
            d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
            max_seq_len=SEQ_LEN, grid_size=32, grid_range=0.3, num_freqs=6,
        ),
        norm=NormalizationConfig(),
        max_horizon=MAX_HORIZON,
        num_horizon_samples=3,
    ).eval()

    # Identical module names -> state dicts must be key-compatible
    missing, unexpected = ported.load_state_dict(legacy.state_dict(), strict=False)
    assert not missing, f"ported model missing keys: {missing}"
    assert not unexpected, f"legacy keys not consumed: {unexpected}"
    return legacy, ported


@pytest.fixture()
def batch():
    torch.manual_seed(1)
    return torch.randn(BATCH, SEQ_LEN + MAX_HORIZON, 6) * 0.1


def test_forward_train_causal_equivalent(models, batch):
    legacy, ported = models
    horizons = torch.tensor([1, 7, 25])

    with torch.no_grad():
        ld_l, tgt_l, h_l, fm_l = legacy.forward_train(batch, horizon_indices=horizons, causal=True)
        ld_p, tgt_p, h_p, fm_p = ported.forward_train(batch, horizon_indices=horizons, causal=True)

    torch.testing.assert_close(ld_p, ld_l)
    torch.testing.assert_close(tgt_p, tgt_l)
    assert torch.equal(fm_p, fm_l)


def test_forward_train_noncausal_equivalent(models, batch):
    legacy, ported = models
    horizons = torch.tensor([1, 10, 50])

    with torch.no_grad():
        ld_l, tgt_l, *_ = legacy.forward_train(batch, horizon_indices=horizons, causal=False)
        ld_p, tgt_p, *_ = ported.forward_train(batch, horizon_indices=horizons, causal=False)

    torch.testing.assert_close(ld_p, ld_l)
    torch.testing.assert_close(tgt_p, tgt_l)


def test_loss_equivalent(models, batch):
    legacy, ported = models
    horizons = torch.tensor([1, 7, 25])
    with torch.no_grad():
        ld, tgt, *_ = ported.forward_train(batch, horizon_indices=horizons, causal=True)
        loss_l = legacy_loss(ld, tgt, grid_range=0.3, grid_size=32, sigma=0.003)
        loss_p = compute_soft_target_loss(ld, tgt, grid_range=0.3, grid_size=32, sigma=0.003)
    torch.testing.assert_close(loss_p, loss_l)
