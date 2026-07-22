"""R-binned gamma/k_proj cache (spectrum_r_bins) — FUTURE-run speed knob.

Binned path snaps each pair's R to the nearest log-spaced bin center and
gathers a per-bin k_proj(gamma(k)) table instead of evaluating gamma at
every pair's continuous R. Default 0 keeps the exact path (all runs to
date; no RNG consumed by the buffers, so seeds stay identical). 256 bins
over the cone envelope keep max k rel-error <1% — but gamma features at
high k*scale products still move under that snap, so this is consistent
train/eval quantization, not an output-invariant optimization: never
toggle it mid-run.
"""
from __future__ import annotations

import numpy as np
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.models.fourier_head import SpectrumHead2D


def _head(r_bins=0, seed=0):
    torch.manual_seed(seed)
    h = SpectrumHead2D(64, grid_size=16, num_freqs=4, grid_range=0.3,
                       mlp_hidden=64, r_bins=r_bins)
    torch.nn.init.normal_(h.tail[-1].weight, std=0.05)
    torch.nn.init.normal_(h.tail[-1].bias, std=0.05)
    return h


def test_k_rel_error_under_1pct_at_256_bins():
    h = _head(r_bins=256)
    R = torch.exp(torch.linspace(np.log(h.R_BIN_MIN), np.log(h.R_BIN_MAX),
                                 5000).double()).float()
    snapped = h.r_bin_centers[h._bin_ids(R)]
    # k = j/(2R), so rel-error in k == rel-error in snapped R
    rel = (snapped / R - 1).abs().max().item()
    assert rel < 0.01, rel


def test_binned_equals_exact_at_bin_centers():
    hb = _head(r_bins=256, seed=3)
    he = _head(r_bins=0, seed=3)          # same seed -> identical weights
    z = torch.randn(8, 64)
    R = hb.r_bin_centers[torch.tensor([0, 17, 50, 99, 100, 200, 254, 255])]
    torch.testing.assert_close(hb(z, R_deg=R), he(z, R_deg=R),
                               atol=1e-5, rtol=1e-5)


def test_out_of_envelope_R_clamps_to_edge_bins():
    h = _head(r_bins=256)
    ids = h._bin_ids(torch.tensor([1e-6, 0.02, 2.0, 100.0]))
    assert ids.tolist() == [0, 0, 255, 255]


def test_grads_flow_through_binned_checkpointed_path():
    h = _head(r_bins=256)
    h.train()                             # exercise USE_CKPT branch
    ld = h(torch.randn(4, 64), R_deg=torch.tensor([0.05, 0.3, 0.9, 1.5]))
    ld.sum().backward()
    for p in (h.k_proj.weight, h.trunk[0].weight, h.tail[-3].weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_config_plumbing_and_default_off():
    assert ModelConfig(d_model=64, nhead=4,
                       dim_feedforward=256).spectrum_r_bins == 0
    cfg = ModelConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                      dropout=0.0, max_seq_len=32, grid_size=16,
                      grid_range=0.3, num_freqs=4, head_type="spectrum",
                      head_mlp_hidden=64, grid_mode="cone",
                      spectrum_r_bins=128)
    m = CausalAISModel(cfg, NormalizationConfig(), max_horizon=64,
                       num_horizon_samples=2)
    assert m.fourier_head.r_bins == 128
    assert m.fourier_head.r_bin_centers.shape == (128,)
