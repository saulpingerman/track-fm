"""Architecture invariants: shapes, parameter counts, causal masking."""
import pytest
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.factory import SCALES, build_model, count_parameters, model_config_for_scale

# Exact param counts, verified against HYPERPARAMETERS.md scaling table
# (~1M, ~5M, ~18M, ~116M) at port time
EXPECTED_PARAMS = {
    "small": 1_004_898,
    "medium": 5_259_234,
    "large": 18_273_378,
    "xlarge": 116_145_122,
}


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return build_model(
        ModelConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                    grid_size=32, num_freqs=6),
        max_horizon=50, num_horizon_samples=3,
    ).eval()


def test_forward_shapes(tiny_model):
    batch = torch.randn(2, 128 + 50, 6) * 0.1
    horizons = torch.tensor([1, 5, 20])
    ld, tgt, h, fm = tiny_model.forward_train(batch, horizon_indices=horizons, causal=True)
    num_pairs = 3 * 128
    assert ld.shape == (2, num_pairs, 32, 32)
    assert tgt.shape == (2, num_pairs, 2)
    assert fm.shape == (num_pairs,)
    # log-densities normalize to 1 in probability space
    probs = ld.exp().sum(dim=(-2, -1))
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-4, rtol=1e-4)


def test_encode_shape(tiny_model):
    x = torch.randn(3, 128, 6) * 0.1
    emb = tiny_model.encode(x)
    assert emb.shape == (3, 128, 64)


@pytest.mark.parametrize("scale", list(SCALES))
def test_param_counts_match_paper(scale):
    cfg = model_config_for_scale(scale)
    n = count_parameters(build_model(cfg))
    expected = EXPECTED_PARAMS[scale]
    assert n == expected, f"{scale}: {n:,} params vs pinned {expected:,}"


def test_causal_mask_no_future_leak(tiny_model):
    """Perturbing a future timestep must not change earlier embeddings."""
    torch.manual_seed(3)
    x = torch.randn(1, 128, 6) * 0.1
    with torch.no_grad():
        base = tiny_model.encode(x)
        x_perturbed = x.clone()
        x_perturbed[0, 100, :] += 10.0  # large perturbation at position 100
        pert = tiny_model.encode(x_perturbed)

    # Positions strictly before 100 are unchanged; position >= 100 changes
    torch.testing.assert_close(pert[0, :100], base[0, :100])
    assert not torch.allclose(pert[0, 100:], base[0, 100:])
