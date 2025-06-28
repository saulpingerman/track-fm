import torch
from torch import nn
from torch.nn.functional import conv2d
from typing import Tuple


class FourierHead2D(nn.Module):
    r"""
    Continuous 2-D PDF on [-1,1]² represented by a truncated Fourier series.

        f(x,y) = Σ_{kx=-m..m} Σ_{ky=-m..m}
                 C[kx,ky] · exp(iπ(kx·x + ky·y))

    A positive, Hermitian-symmetric, unit-mass coefficient table C is obtained
    by autocorrelating an unconstrained complex (m+1)×(m+1) table produced by
    a linear layer and scaling so C[0,0] = 1/4.
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        dim_input: int,
        num_frequencies: int,
        regularisation_gamma: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim_input = dim_input
        self.m = num_frequencies
        self.reg_gamma = regularisation_gamma
        self.dtype = dtype
        self.dev = device

        n_params = (self.m + 1) ** 2
        self.fc_autocorr = nn.Linear(dim_input, 2 * n_params)
        nn.init.normal_(self.fc_autocorr.weight, 1e-2)
        nn.init.zeros_(self.fc_autocorr.bias)

        # Integer frequency grids
        k = torch.arange(-self.m, self.m + 1, dtype=dtype)
        kx, ky = torch.meshgrid(k, k, indexing="xy")
        self.register_buffer("kx_grid", kx)          # (H,W)
        self.register_buffer("ky_grid", ky)

        self.to(device)

    # ------------------------------------------------------------ autocorrelation
    @staticmethod
    def _autocorrelate2d(seq: torch.Tensor) -> torch.Tensor:
        """
        seq : (B, 1, H, W) complex  →  (B, 1, 2H-1, 2W-1) complex
        """
        B, _, H, W = seq.shape
        inp    = seq.transpose(0, 1)                 # (1,B,H,W)
        weight = seq.conj().resolve_conj()           # (B,1,H,W)
        ac = conv2d(inp, weight,
                    padding=(H - 1, W - 1),
                    groups=B)                        # (1,B,2H-1,2W-1)
        return ac.transpose(0, 1)                    # (B,1, …)

    # ----------------------------------------------------------- coeff extraction
    def _fourier_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, dim_input)  →  (B, 2m+1, 2m+1) complex
        """
        p = self.fc_autocorr(x)                      # (B, 2*(m+1)²)
        n = (self.m + 1) ** 2
        real = p[..., :n].view(-1, 1, self.m + 1, self.m + 1)
        imag = p[..., n:].view(-1, 1, self.m + 1, self.m + 1)
        seq  = torch.complex(real, imag)             # (B,1,H,W)

        C = self._autocorrelate2d(seq)[:, 0]         # (B,2m+1,2m+1)

        centre = C[:, self.m, self.m].real           # (B,)
        C = C / centre.view(-1, 1, 1) / 4.0          # C[0,0] = 1/4
        return C

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        inputs:  torch.Tensor,       # (..., dim_input)
        targets: torch.Tensor,       # (..., 2) in [-1,1]²
    ) -> torch.Tensor:
        leading = inputs.shape[:-1]                      # original batch view

        x_flat = inputs.view(-1, self.dim_input).to(self.dev, self.dtype)
        t_flat = targets.view(-1, 2).to(self.dev, self.dtype)
        B = x_flat.size(0)

        C = self._fourier_coeffs(x_flat)                # (B,H,W)

        # ---- build exp(iπ(kx·x + ky·y))  →  shape (B,H,W)
        tx = t_flat[:, 0].unsqueeze(-1).unsqueeze(-1)   # (B,1,1)
        ty = t_flat[:, 1].unsqueeze(-1).unsqueeze(-1)
        phase = self.kx_grid.unsqueeze(0) * tx + self.ky_grid.unsqueeze(0) * ty
        exps  = torch.exp(1j * torch.pi * phase)        # (B,H,W)

        pdf = (C * exps).sum(dim=(-1, -2)).real         # (B,)
        return pdf.view(*leading)

    # -------------------------------------------------------------- grid helper
    @torch.no_grad()
    def grid_pdf(
        self,
        inputs: torch.Tensor,            # (d_in,) or (1,d_in)
        grid_size: int = 128,
        lim: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """
        Return a (grid_size × grid_size) tensor of f(x,y) for visualisation.
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.dev, self.dtype)

        xs = torch.linspace(*lim, grid_size, device=self.dev)
        ys = torch.linspace(*lim, grid_size, device=self.dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")      # (G,G)

        pts = torch.stack([gx, gy], dim=-1).view(-1, 2)     # (G²,2)
        pdf = self.forward(inputs.expand(pts.size(0), -1), pts)
        return pdf.view(grid_size, grid_size).cpu()
