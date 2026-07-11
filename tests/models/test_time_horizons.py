"""Per-sample horizon forward + time-bucket index tests."""
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.eval.horizons import time_bucket_indices
from trackfm.models.factory import build_model

NORM = NormalizationConfig()


def test_forward_at_indices_matches_forward_train_on_shared_steps():
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(d_model=32, nhead=2, num_layers=1, dim_feedforward=64,
                    grid_size=16, num_freqs=3),
        max_horizon=50, num_horizon_samples=3).eval()
    x = torch.randn(4, 178, 6) * 0.1
    steps = torch.tensor([2, 20, 45])
    with torch.no_grad():
        ld_ref, tgt_ref, _, _ = model.forward_train(x, horizon_indices=steps, causal=False)
        ld_new, tgt_new = model.forward_at_indices(x, steps.repeat(4, 1))
    torch.testing.assert_close(ld_new, ld_ref)
    torch.testing.assert_close(tgt_new, tgt_ref)


def test_forward_at_indices_per_sample_steps_differ():
    torch.manual_seed(1)
    model = build_model(
        ModelConfig(d_model=32, nhead=2, num_layers=1, dim_feedforward=64,
                    grid_size=16, num_freqs=3),
        max_horizon=50, num_horizon_samples=1).eval()
    x = torch.randn(2, 178, 6) * 0.1
    with torch.no_grad():
        ld_a, tgt_a = model.forward_at_indices(x, torch.tensor([[10], [40]]))
        ld_ref0, tgt_ref0 = model.forward_at_indices(x[:1], torch.tensor([[10]]))
        ld_ref1, tgt_ref1 = model.forward_at_indices(x[1:], torch.tensor([[40]]))
    torch.testing.assert_close(ld_a[0], ld_ref0[0])
    torch.testing.assert_close(tgt_a[1], tgt_ref1[0])


def test_time_bucket_indices():
    # constant dt = 60s -> 15m bucket = step 15, 2h = step 120
    f = torch.zeros(1, 128 + 200, 6)
    f[:, :, 5] = 60.0 / NORM.dt_scale
    idx, valid = time_bucket_indices(f, NORM)
    assert idx[0].tolist() == [15, 30, 60, 120]
    assert valid[0].all()

    # short window: only 200 future steps at 10s = 2000s -> 1h/2h unavailable
    f2 = torch.zeros(1, 128 + 200, 6)
    f2[:, :, 5] = 10.0 / NORM.dt_scale
    idx2, valid2 = time_bucket_indices(f2, NORM)
    assert valid2[0, 0] and valid2[0, 1]          # 15m (step 90), 30m (step 180)
    assert not valid2[0, 2] and not valid2[0, 3]  # 1h, 2h out of reach
