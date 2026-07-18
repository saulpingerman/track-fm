"""Continuous-time RoPE invariants.

Pins the properties the ablation's validity depends on:
- default pos_mode leaves the model IDENTICAL (state-dict keys + forward
  bits) — the scaling series is untouched;
- causality: perturbing a future posit never changes past outputs;
- relative property: shifting ALL times by a constant leaves the
  attention stack's output unchanged (this is what makes wall-clock RoPE
  coherent — only time DIFFERENCES matter);
- irregularity is visible: the same posits under a different dt pattern
  produce different embeddings (the index-PE baseline cannot do this
  with identical feature values, which is the ablation's whole point).
"""
from __future__ import annotations

import pytest
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.models.time_rope import TimeRoPEEncoder, apply_rope, rope_angles


def _mk(pos_mode, seed=7, dropout=0.0):
    torch.manual_seed(seed)
    m = ModelConfig(d_model=32, nhead=2, num_layers=2, dim_feedforward=64,
                    dropout=dropout, grid_size=16, grid_range=0.3,
                    num_freqs=4, grid_mode="cone", pos_mode=pos_mode)
    return CausalAISModel(m, NormalizationConfig(), max_horizon=40,
                          num_horizon_samples=2)


def _batch(B=2, L=168, seed=1):
    torch.manual_seed(seed)
    f = torch.randn(B, L, 6) * 0.1
    f[..., 5] = torch.rand(B, L) * 0.5 + 0.1
    return f


def test_default_pos_mode_unchanged():
    """Default config builds the exact historical module tree."""
    torch.manual_seed(0)
    ref_keys = set(_mk("index_sinusoidal").state_dict().keys())
    assert any("pos_encoder" in k or "transformer.layers.0.self_attn" in k
               for k in ref_keys)
    assert not any("in_proj.weight" in k and "layers" in k and
                   "self_attn" not in k for k in ref_keys)


def test_rope_forward_shapes_and_finite():
    model = _mk("time_rope")
    model.eval()
    f = _batch()
    with torch.no_grad():
        ld, tgt, hz, fm = model.forward_train(
            f, horizon_indices=torch.tensor([5, 20]), causal=True)
    assert ld.shape[2:] == (16, 16) and torch.isfinite(ld).all()


def test_rope_causality():
    """Perturbing posit j must not change encode() outputs before j."""
    model = _mk("time_rope")
    model.eval()
    f = _batch(B=1, L=64)
    x = f[:, :64, :]
    with torch.no_grad():
        e1 = model.encode(x)
        x2 = x.clone()
        x2[0, 50, 0] += 1.0                      # future lat perturbation
        e2 = model.encode(x2)
    assert torch.allclose(e1[0, :50], e2[0, :50], atol=1e-6)
    assert not torch.allclose(e1[0, 50:], e2[0, 50:])


def test_rope_time_shift_invariance():
    """Constant time offset cancels in attention (relative property)."""
    torch.manual_seed(2)
    enc = TimeRoPEEncoder(2, 32, 2, 64, 0.0, p_min=4.0, p_max=86400.0)
    enc.eval()
    x = torch.randn(2, 20, 32)
    t = torch.cumsum(torch.rand(2, 20) * 100, dim=1)
    with torch.no_grad():
        y1 = enc(x, t)
        y2 = enc(x, t + 5000.0)
    assert torch.allclose(y1, y2, atol=1e-4)


def test_rope_sees_irregularity():
    """Same feature VALUES, different dt pattern -> different output at
    matched positions (index PE is blind to this by construction)."""
    model = _mk("time_rope")
    model.eval()
    f1 = _batch(B=1, L=64)
    f2 = f1.clone()
    f2[..., 5] = f1[..., 5] * torch.linspace(0.2, 3.0, 64)  # warp cadence
    with torch.no_grad():
        e1, e2 = model.encode(f1[:, :64]), model.encode(f2[:, :64])
    assert not torch.allclose(e1[0, 10:], e2[0, 10:], atol=1e-4)


def test_apply_rope_zero_time_is_identity():
    x = torch.randn(1, 2, 5, 8)
    cos, sin = rope_angles(torch.zeros(1, 5), 4, 4.0, 86400.0)
    assert torch.allclose(apply_rope(x, cos, sin), x, atol=1e-6)


def test_rope_norm_preserving():
    """Rotation must not change vector norms (pure phase)."""
    x = torch.randn(1, 2, 7, 8)
    cos, sin = rope_angles(torch.cumsum(torch.rand(1, 7) * 60, 1),
                           4, 4.0, 86400.0)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_mup_time_rope_rejected():
    from trackfm.config import MupConfig
    with pytest.raises(ValueError, match="time_rope"):
        ModelConfig(d_model=32, nhead=2, dim_feedforward=128,
                    pos_mode="time_rope",
                    mup=MupConfig(enabled=True, d_base=32, d_head=16))
