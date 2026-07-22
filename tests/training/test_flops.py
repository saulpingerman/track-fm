"""head_flops_per_sample must track head_type — the spectrum head's
per-harmonic phi MLP is encoder-scale work and can't be silently dropped
(that's the CHAIN12 MFU-reads-2x-low bug)."""
import pytest

from trackfm.config import ModelConfig
from trackfm.training.flops import head_flops_per_sample


def _cfg(**kw):
    base = dict(d_model=256, num_layers=2, nhead=4, dim_feedforward=512,
                grid_mode="cone")
    base.update(kw)
    return ModelConfig(**base)


def test_mlp_projector_counted():
    lin = _cfg(head_mlp_hidden=0)
    mlp = _cfg(head_mlp_hidden=384)
    d, coeffs = 256, 2 * 25 ** 2
    diff = (head_flops_per_sample(mlp, 1) - head_flops_per_sample(lin, 1))
    expect = 2.0 * (d * 384 + 384 * coeffs - d * coeffs)
    assert diff == pytest.approx(expect)


def test_spectrum_phi_dominates():
    mlp = _cfg(head_mlp_hidden=384)
    spec = _cfg(head_mlp_hidden=384, head_type="spectrum")
    ratio = head_flops_per_sample(spec, 1) / head_flops_per_sample(mlp, 1)
    # 625 harmonics x (k_proj 27x128 + tail 128x128+256) per pair swamps
    # the slot head's single 384x1250 linear
    assert ratio > 2.0


def test_spectrum_per_pair_formula():
    spec = _cfg(head_mlp_hidden=384, head_type="spectrum")
    d, phi, lattice, grid = 256, 128, 25 ** 2, 64 * 64
    per_freq = 10 * 6 + 27 * phi + phi * phi + phi * 2
    per_pair = (d + d * d + 2 * d * d + d * 384 + 384 * phi
                + lattice * per_freq + 2 * lattice * grid)
    assert head_flops_per_sample(spec, 7) == pytest.approx(2.0 * 7 * per_pair)


def test_scales_linearly_in_pairs():
    spec = _cfg(head_type="spectrum")
    assert head_flops_per_sample(spec, 10) == pytest.approx(
        10 * head_flops_per_sample(spec, 1))
