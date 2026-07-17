"""End-to-end conditioning wiring invariants on CausalAISModel.

Pins the properties the S1/S2 ablations depend on:
- zero-init identity: context_mode='geo' produces BIT-IDENTICAL outputs
  to context_mode='none' at init (shared weights), on every forward path;
- the bias actually flows once the final conv is non-zero;
- cone and fixed both accept per-pair bias shapes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel


@pytest.fixture(scope="module")
def static_dir(tmp_path_factory):
    """Tiny synthetic geo.npz covering the norm bbox."""
    d = tmp_path_factory.mktemp("static")
    nlat, nlon = 91, 181
    lat = np.linspace(54.0, 58.5, nlat).astype(np.float32)
    lon = np.linspace(7.0, 16.0, nlon).astype(np.float32)
    rng = np.random.default_rng(0)
    np.savez(d / "geo.npz",
             land=(rng.random((nlat, nlon)) > 0.7).astype(np.float32),
             log_depth=rng.random((nlat, nlon)).astype(np.float32),
             sdist_coast=rng.normal(size=(nlat, nlon)).astype(np.float32),
             lat=lat, lon=lon)
    return str(d)


def _mk(context_mode, static_dir, grid_mode="cone", seed=7):
    torch.manual_seed(seed)
    m = ModelConfig(d_model=32, nhead=2, num_layers=1, dim_feedforward=64,
                    grid_size=16, grid_range=0.3, num_freqs=4,
                    grid_mode=grid_mode, context_mode=context_mode,
                    context_static_dir=static_dir)
    return CausalAISModel(m, NormalizationConfig(), max_horizon=40,
                          num_horizon_samples=2)


def _batch(B=3, seed=1):
    torch.manual_seed(seed)
    f = torch.randn(B, 168, 6) * 0.1
    f[..., 5] = torch.rand(B, 168) * 0.5 + 0.1          # positive dt
    return f


def _sync_weights(dst, src):
    """Copy all weights src -> dst for keys present in both."""
    sd_src = src.state_dict()
    sd_dst = dst.state_dict()
    for k in sd_dst:
        if k in sd_src:
            sd_dst[k] = sd_src[k]
    dst.load_state_dict(sd_dst)


@pytest.mark.parametrize("grid_mode", ["fixed", "cone"])
def test_zero_init_identity_all_paths(static_dir, grid_mode):
    base = _mk("none", static_dir, grid_mode)
    cond = _mk("geo", static_dir, grid_mode)
    _sync_weights(cond, base)
    base.eval(); cond.eval()
    f = _batch()
    hz = torch.tensor([5, 20])
    with torch.no_grad():
        for causal in (True, False):
            ld_b, tgt_b, _, _ = base.forward_train(f, horizon_indices=hz, causal=causal)
            ld_c, tgt_c, _, _ = cond.forward_train(f, horizon_indices=hz, causal=causal)
            torch.testing.assert_close(ld_b, ld_c, rtol=0, atol=0)
            torch.testing.assert_close(tgt_b, tgt_c, rtol=0, atol=0)
        step_idx = torch.randint(1, 30, (3, 2))
        ld_b, _ = base.forward_at_indices(f, step_idx)
        ld_c, _ = cond.forward_at_indices(f, step_idx)
        torch.testing.assert_close(ld_b, ld_c, rtol=0, atol=0)


def test_bias_flows_when_nonzero(static_dir):
    cond = _mk("geo", static_dir, "cone")
    ref = _mk("geo", static_dir, "cone")
    _sync_weights(ref, cond)
    with torch.no_grad():
        cond.context_bias.cnn[-1].weight.add_(0.5)
        cond.context_bias.cnn[-1].bias.add_(0.1)
    cond.eval(); ref.eval()
    f = _batch()
    hz = torch.tensor([5, 20])
    with torch.no_grad():
        ld_c, _, _, _ = cond.forward_train(f, horizon_indices=hz, causal=True)
        ld_r, _, _, _ = ref.forward_train(f, horizon_indices=hz, causal=True)
    assert not torch.allclose(ld_c, ld_r)
    assert torch.isfinite(ld_c).all()


def test_context_gradients_reach_cnn(static_dir):
    cond = _mk("geo", static_dir, "cone")
    f = _batch()
    ld, tgt, _, _ = cond.forward_train(f, horizon_indices=torch.tensor([5, 20]),
                                        causal=True)
    ld.sum().backward()
    g = cond.context_bias.cnn[-1].weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_checkpoint_roundtrip_excludes_raster(static_dir):
    """Raster buffer is non-persistent: checkpoints stay small and a
    conditioned model reloads its raster from disk, not the state_dict."""
    cond = _mk("geo", static_dir, "cone")
    sd = cond.state_dict()
    assert not any("raster" in k for k in sd)
    cond2 = _mk("geo", static_dir, "cone", seed=99)
    cond2.load_state_dict(sd)                            # strict ok
