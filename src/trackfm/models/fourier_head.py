"""2D Fourier density head.

Predicts DCT/Fourier coefficients from a conditioning vector and combines
them with precomputed cos/sin bases over a grid_size x grid_size spatial
grid spanning +/- grid_range degrees, returning a log-probability density.

Ported verbatim from experiment 11 (`run_experiment.py`), with experiment
14's `.clone()` fix on the meshgrid buffers (meshgrid views break
state_dict round-trips) — see docs/MIGRATION.md.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectGridHead(nn.Module):
    """Ablation head: straight linear map to per-cell logits.

    No spectral basis — d_model -> G^2 logits -> log_softmax. Compared to
    FourierHead2D it is parameter-HEAVY (d*G^2 grows with the grid: 28.3M
    params at d=768/G=192 vs the Fourier head's grid-independent ~2.1M at
    F=18) but compute-competitive, and it has NO continuous density — it
    can only be evaluated at its own cells. Exists to answer: is the
    Fourier smoothness prior earning its place, or just compressing params?
    """

    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 0, grid_range: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.logits = nn.Linear(d_model, grid_size * grid_size)
        nn.init.normal_(self.logits.weight, std=0.01)
        nn.init.zeros_(self.logits.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        log_density = F.log_softmax(self.logits(z), dim=-1)
        return log_density.view(z.shape[0], self.grid_size, self.grid_size)


class FourierHead2D(nn.Module):
    """2D Fourier density head."""

    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 12, grid_range: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        num_freq_pairs = (2 * num_freqs + 1) ** 2
        self.coeff_predictor = nn.Linear(d_model, 2 * num_freq_pairs)

        # Precompute grid and basis
        x = torch.linspace(-grid_range, grid_range, grid_size)
        y = torch.linspace(-grid_range, grid_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', xx.clone())
        self.register_buffer('grid_y', yy.clone())

        freqs = torch.arange(-num_freqs, num_freqs + 1, dtype=torch.float)
        freq_x, freq_y = torch.meshgrid(freqs, freqs, indexing='ij')
        self.register_buffer('freq_x', freq_x.flatten())
        self.register_buffer('freq_y', freq_y.flatten())

        # Precompute phase matrix
        L = 2 * grid_range
        phase = (2 * np.pi / L) * (
            self.freq_x.view(1, 1, -1) * xx.flatten().view(1, -1, 1) +
            self.freq_y.view(1, 1, -1) * yy.flatten().view(1, -1, 1)
        )
        self.register_buffer('cos_basis', torch.cos(phase).squeeze(0))
        self.register_buffer('sin_basis', torch.sin(phase).squeeze(0))

        nn.init.normal_(self.coeff_predictor.weight, std=0.01)
        nn.init.zeros_(self.coeff_predictor.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        coeffs = self.coeff_predictor(z)

        num_freq_pairs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_freq_pairs]
        sin_coeffs = coeffs[:, num_freq_pairs:]

        logits = cos_coeffs @ self.cos_basis.T + sin_coeffs @ self.sin_basis.T
        log_density = F.log_softmax(logits, dim=-1)

        return log_density.view(batch_size, self.grid_size, self.grid_size)
