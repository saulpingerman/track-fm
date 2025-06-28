# four_head_2D_LR_opt.py
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations   # ← keeps “str | None” hints safe on 3.8/3.9
import math
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class FourierHead2DLR(nn.Module):
    r"""
    Continuous 2-D PDF on [-1,1]² with low-rank Fourier coefficients

        f(x,y) = Σ_{kx,ky} C[kx,ky] · e^{iπ(kx·x + ky·y)}

    C = Σ_{r=1..R} u^{(r)} ⊗ v^{(r)}, where each u/v is the autocorrelation of a
    learned complex (m+1)-vector a/b.   Rank-R ⇒  4·R·(m+1) parameters.
    """

    FFT_THRESHOLD = 256              # use FFT autocorr when L ≥ this

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        dim_input: int,
        num_frequencies: int,          # m
        rank: int = 1,
        regularisation_gamma: float = 0.0,   # (kept for compatibility)
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        compile_forward: Union[bool, str] = "auto",
    ):
        super().__init__()
        self.dim_input = dim_input
        self.m         = num_frequencies
        self.rank      = rank
        self.reg_gamma = regularisation_gamma
        self.dtype     = dtype
        self.dev       = device

        L       = self.m + 1
        out_dim = 4 * rank * L                       # Re/Im of a & b

        self.fc = nn.Linear(dim_input, out_dim)
        nn.init.normal_(self.fc.weight, 1e-2)
        nn.init.zeros_(self.fc.bias)

        # frequency grids for e^{iπ(kx x + ky y)}
        k   = torch.arange(-self.m, self.m + 1, dtype=dtype)
        kx, ky = torch.meshgrid(k, k, indexing="xy")
        self.register_buffer("kx_grid", kx)
        self.register_buffer("ky_grid", ky)

        self.to(device)

        # optional torch.compile
        if compile_forward and (
            compile_forward is True
            or (compile_forward == "auto" and torch.cuda.is_available())
        ):
            self.forward = torch.compile(self.forward, dynamic=True)

    # ------------------------------------------------------- 1-D autocorr
    @staticmethod
    def _autocorrelate1d(seq: torch.Tensor) -> torch.Tensor:
        """
        Autocorrelation along the last dim for a batch of complex sequences.
        seq: (B,1,L) complex  →  (B,1,2L-1) complex
        """
        B, _, L = seq.shape
        if L >= FourierHead2DLR.FFT_THRESHOLD:
            n  = 2 * L - 1
            f  = torch.fft.fft(seq, n=n)                 # complex FFT
            ac = torch.fft.ifft(f * f.conj(), n=n) / L   # correct scale
            return ac
        else:
            inp    = seq.transpose(0, 1)                 # (1,B,L)
            weight = seq.conj().resolve_conj()           # (B,1,L)
            ac     = F.conv1d(inp, weight, padding=L-1, groups=B)
            return ac.transpose(0, 1)                    # (B,1,2L-1)

    # ----------------------------------------------------- coeff builder
    def _fourier_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, dim_input)  →  C: (B, 2m+1, 2m+1) complex
        """
        B, R  = x.size(0), self.rank
        L     = self.m + 1
        K     = 2 * L - 1                                # length after acorr

        # fused linear layer → 4×(B,R,L)
        z  = self.fc(x).view(B, 4, R, L)
        ar, ai, br, bi = z[:, 0], z[:, 1], z[:, 2], z[:, 3]

        a = torch.complex(ar, ai).view(B * R, 1, L)
        b = torch.complex(br, bi).view(B * R, 1, L)

        ua = self._autocorrelate1d(a).view(B, R, K)      # (B,R,K)
        vb = self._autocorrelate1d(b).view(B, R, K)      # (B,R,K)

        C = torch.einsum("brk, brl -> bkl", ua, vb)      # Σ_r u⊗v  (B,K,K)

        # normalise so C[m,m] = 1/4
        eps         = 1e-2                       # <<—  was 1e-6  (raise the floor)
        centre_raw  = C[:, self.m, self.m].real        # (B,)
        scale       = centre_raw.abs().clamp_min(eps)  # (B,)
        sign        = centre_raw.sign().where(centre_raw != 0, torch.ones_like(centre_raw))
        C           = C * sign.view(B,1,1) / scale.view(B,1,1) / 4.0
        return C

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        inputs:  torch.Tensor,   # (..., dim_input)
        targets: torch.Tensor,   # (..., 2) in [-1,1]²
    ) -> torch.Tensor:
        leading = inputs.shape[:-1]
        x_flat  = inputs.reshape(-1, self.dim_input).to(self.dev, self.dtype)
        t_flat  = targets.reshape(-1, 2).to(self.dev, self.dtype)
        B       = x_flat.size(0)

        C = self._fourier_coeffs(x_flat)                 # (B,K,K)

        tx    = t_flat[:, 0].view(B, 1, 1)
        ty    = t_flat[:, 1].view(B, 1, 1)
        phase = self.kx_grid.unsqueeze(0) * tx + self.ky_grid.unsqueeze(0) * ty
        exps  = torch.exp(1j * math.pi * phase)          # (B,K,K)

        pdf = (C * exps).sum((-2, -1)).real
        return pdf.clamp_min(1e-8).view(*leading)        # positivity guard

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # NEW helper: analytic ∫∫ f(x,y) dx dy  (should equal 1)
    @torch.no_grad()
    def integral(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the exact integral of the PDF for each latent input.
        x: (B, dim_input)  →  (B,)
        """
        C = self._fourier_coeffs(x.to(self.dev, self.dtype))
        return 4.0 * C[:, self.m, self.m].real           # =1 ideally
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ------------------------------------------------ grid evaluation
    @torch.no_grad()
    def grid_pdf(
        self,
        inputs:    torch.Tensor,
        grid_size: int = 128,
        lim: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.dev, self.dtype)

        xs  = torch.linspace(*lim, grid_size, device=self.dev)
        ys  = torch.linspace(*lim, grid_size, device=self.dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")

        pts = torch.stack([gx, gy], -1).view(-1, 2)
        pdf = self.forward(inputs.expand(pts.size(0), -1), pts)
        return pdf.view(grid_size, grid_size).cpu()
