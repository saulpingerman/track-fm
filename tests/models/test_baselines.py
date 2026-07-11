"""Kalman-filter baseline sanity checks."""
import numpy as np
import torch

from trackfm.config import NormalizationConfig
from trackfm.eval.baselines import (
    fit_sigma_per_horizon, gaussian_density_loss_per_sample, kalman_cv_forecast,
)

NORM = NormalizationConfig()


def make_track(n=228, vlat=1e-5, vlon=2e-5, noise=0.0, seed=0):
    """Constant-velocity normalized feature tensor (1, n, 6), dt=30s."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    lat = 56.25 + vlat * 30 * t + rng.normal(0, noise, n)
    lon = 11.5 + vlon * 30 * t + rng.normal(0, noise, n)
    f = np.zeros((1, n, 6), dtype=np.float32)
    f[0, :, 0] = (lat - NORM.lat_center) / NORM.lat_scale
    f[0, :, 1] = (lon - NORM.lon_center) / NORM.lon_scale
    f[0, :, 5] = 30.0 / NORM.dt_scale
    return torch.from_numpy(f)


def test_kalman_predicts_constant_velocity_exactly():
    x = make_track()
    h = torch.tensor([10, 50, 100])
    pred, sig = kalman_cv_forecast(x, h, seq_len=128, norm=NORM)
    # true displacement after h steps of 30s at (vlat, vlon) deg/s
    for k, hh in enumerate([10, 50, 100]):
        want = np.array([1e-5 * 30 * hh, 2e-5 * 30 * hh])
        got = pred[0, k].numpy()
        np.testing.assert_allclose(got, want, rtol=0.02)


def test_kalman_uncertainty_grows_with_horizon():
    x = make_track(noise=1e-4, seed=1)
    h = torch.tensor([10, 50, 100])  # track has 228 pos = 128 input + 100 future
    _, sig = kalman_cv_forecast(x, h, seq_len=128, norm=NORM, q=1e-12, r=1e-8)
    s = sig[0, :, 0]
    assert s[0] < s[1] < s[2], f"sigma not monotone: {s.tolist()}"


def test_kalman_beats_two_point_dr_on_noisy_track():
    """KF's whole-window velocity beats last-two-points DR under GPS noise."""
    from trackfm.training.losses import dead_reckoning_displacement

    xs = torch.cat([make_track(noise=2e-4, seed=s) for s in range(16)])
    h = torch.tensor([50])
    seq = 128
    true_v = torch.tensor([1e-5 * 30 * 50, 2e-5 * 30 * 50])
    kf_pred, _ = kalman_cv_forecast(xs, h, seq, NORM, q=1e-14, r=4e-8)
    dr_pred = dead_reckoning_displacement(xs, h, seq, NORM)
    kf_err = (kf_pred[:, 0] - true_v).norm(dim=-1).mean()
    dr_err = (dr_pred[:, 0] - true_v).norm(dim=-1).mean()
    assert kf_err < dr_err * 0.5, f"KF {kf_err:.5f} vs DR {dr_err:.5f}"


def test_sigma_fit_picks_reasonable_width():
    torch.manual_seed(0)
    preds = torch.zeros(64, 1, 2)
    targets = preds + torch.randn(64, 1, 2) * 0.05  # true spread 0.05
    s = fit_sigma_per_horizon(preds, targets, 0.3, 32, 0.003)
    assert 0.02 <= float(s[0]) <= 0.1
