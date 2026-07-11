"""Direct-grid ablation head tests."""
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.factory import build_model, count_parameters


def _tiny(head_type):
    return build_model(
        ModelConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                    grid_size=32, num_freqs=6, head_type=head_type),
        max_horizon=50, num_horizon_samples=2)


def test_direct_head_forward_matches_interface():
    torch.manual_seed(0)
    model = _tiny("direct").eval()
    x = torch.randn(3, 178, 6) * 0.1
    with torch.no_grad():
        ld, tgt, h, fm = model.forward_train(x, horizon_indices=torch.tensor([3, 20]),
                                             causal=True)
    assert ld.shape == (3, 2 * 128, 32, 32)
    probs = ld.exp().sum(dim=(-2, -1))
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-4, rtol=1e-4)


def test_param_tradeoff_direct_vs_fourier():
    n_f = count_parameters(_tiny("fourier"))
    n_d = count_parameters(_tiny("direct"))
    # direct head: d*G^2 = 64*1024 = 65k+ params vs fourier's d*2(2F+1)^2
    assert n_d > n_f  # at this size direct is heavier
    # fourier coefficient count is grid-independent
    big = ModelConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                      grid_size=64, num_freqs=6, head_type="fourier")
    assert count_parameters(build_model(big)) == n_f  # same params, 4x grid
