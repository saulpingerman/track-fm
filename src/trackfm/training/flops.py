"""Analytic FLOPs accounting for MFU (Model FLOPs Utilization).

MFU = achieved model-FLOPs/s / peak FLOPs/s. The numerator uses the FLOPs
the architecture mathematically requires per sample (counted here), so
dataloader stalls, launch overhead, and memory-bound kernels all show up
as lost utilization. nvidia-smi's "GPU-Util" only reports whether ANY
kernel was resident and says nothing about this.
"""
from __future__ import annotations

from trackfm.config import ModelConfig


def encoder_flops_per_sample(m: ModelConfig, seq_len: int | None = None) -> float:
    """Forward-pass FLOPs for the causal transformer encoder (one sample)."""
    s = seq_len or m.max_seq_len
    d = m.d_model
    f = m.dim_feedforward
    # per layer MACs: QKV (3sd^2) + attn out (sd^2) + scores+apply (2 s^2 d) + FFN (2sdf)
    per_layer = 4 * s * d * d + 2 * s * s * d + 2 * s * d * f
    macs = m.num_layers * per_layer + s * m.input_features * d  # + input proj
    return 2.0 * macs


def head_flops_per_sample(m: ModelConfig, num_pairs: int) -> float:
    """Forward FLOPs for horizon conditioning + density head over num_pairs.

    Head-type aware: the spectrum head evaluates a per-frequency MLP
    (gamma -> k_proj -> tail) at every (pair, harmonic) — encoder-scale
    work; omitting it made CHAIN12's reported MFU read ~2x low. MFU
    convention, not HFU: gradient-checkpoint recompute is NOT counted,
    so a checkpointed spectrum run's hardware-FLOPs sit ~1 extra phi
    forward above this number.
    """
    d = m.d_model
    lattice = (2 * m.num_freqs + 1) ** 2      # harmonic lattice points
    coeffs = 2 * lattice                       # (cos, sin) per point
    grid = m.grid_size * m.grid_size
    cond = d + d * d + 2 * d * d   # time_proj MLP (1->d, d->d) + horizon_proj
    hidden = m.head_mlp_hidden
    if m.head_type == "spectrum":
        from trackfm.models.fourier_head import SpectrumHead2D
        s_scales = SpectrumHead2D.N_SCALES
        gamma_dim = 4 * s_scales + 3
        phi = 128                  # SpectrumHead2D phi_hidden (ctor default)
        trunk_hidden = hidden if hidden > 0 else d
        per_freq = (
            10 * s_scales          # gamma: ang mults + sin/cos + log terms
            + gamma_dim * phi      # k_proj
            + phi * phi + phi * 2  # tail
        )
        per_pair = (cond
                    + d * trunk_hidden      # trunk
                    + trunk_hidden * phi    # h_proj (once per pair)
                    + lattice * per_freq    # phi at every harmonic
                    + coeffs * grid)        # cos/sin synthesis -> logits
    else:
        if hidden > 0:             # mlp projector (CHAIN4 heads)
            predict = d * hidden + hidden * coeffs
        else:
            predict = d * coeffs
        per_pair = cond + predict + coeffs * grid
    return 2.0 * num_pairs * per_pair


def train_flops_per_sample(m: ModelConfig, num_horizon_samples: int,
                           seq_len: int | None = None) -> float:
    """Total training FLOPs per sample: forward + backward (~2x forward)."""
    s = seq_len or m.max_seq_len
    num_pairs = num_horizon_samples * s  # causal subwindow training
    fwd = encoder_flops_per_sample(m, s) + head_flops_per_sample(m, num_pairs)
    return 3.0 * fwd
