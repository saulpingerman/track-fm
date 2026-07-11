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
    """Forward FLOPs for horizon conditioning + Fourier head over num_pairs."""
    d = m.d_model
    coeffs = 2 * (2 * m.num_freqs + 1) ** 2
    grid = m.grid_size * m.grid_size
    per_pair = (
        d + d * d          # time_proj MLP (1->d, d->d)
        + 2 * d * d        # horizon_proj (2d -> d)
        + d * coeffs       # coefficient predictor
        + coeffs * grid    # basis matmul -> logits
    )
    return 2.0 * num_pairs * per_pair


def train_flops_per_sample(m: ModelConfig, num_horizon_samples: int,
                           seq_len: int | None = None) -> float:
    """Total training FLOPs per sample: forward + backward (~2x forward)."""
    s = seq_len or m.max_seq_len
    num_pairs = num_horizon_samples * s  # causal subwindow training
    fwd = encoder_flops_per_sample(m, s) + head_flops_per_sample(m, num_pairs)
    return 3.0 * fwd
