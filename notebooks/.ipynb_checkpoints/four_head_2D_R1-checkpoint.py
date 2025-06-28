import torch
from torch import nn
from torch.nn.functional import conv1d, conv2d
from typing import Tuple

class FourierHead2DR1(nn.Module):
    r"""
    Continuous 2-D PDF on [-1,1]² represented by a truncated Fourier series:

        f(x,y) = Σ_{kx=-m..m} Σ_{ky=-m..m}
                 C[kx,ky] · exp(iπ(kx·x + ky·y))

    Here C[kx,ky] = u[kx]·v[ky] where
      u = autocorr(a),  v = autocorr(b)
    and a,b are unconstrained complex vectors of length (m+1).
    """

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

        # two small linear layers producing (m+1)-length latent vectors (real+imag)
        L = self.m + 1
        self.fc_a = nn.Linear(dim_input, 2 * L)
        self.fc_b = nn.Linear(dim_input, 2 * L)
        nn.init.normal_(self.fc_a.weight, 1e-2)
        nn.init.zeros_(self.fc_a.bias)
        nn.init.normal_(self.fc_b.weight, 1e-2)
        nn.init.zeros_(self.fc_b.bias)

        # prepare integer grids for the final Fourier exp
        k = torch.arange(-self.m, self.m + 1, dtype=dtype)
        kx, ky = torch.meshgrid(k, k, indexing="xy")
        self.register_buffer("kx_grid", kx)
        self.register_buffer("ky_grid", ky)

        self.to(device)

    @staticmethod
    def _autocorrelate1d(seq: torch.Tensor) -> torch.Tensor:
        """
        1D autocorrelation via grouped conv1d.
        seq: (B, 1, L) complex → (B, 1, 2L-1) complex
        """
        B, _, L = seq.shape
        # reshape for grouped conv: (1, B, L)
        inp    = seq.transpose(0, 1)
        # weight: conj seq → (B, 1, L)
        weight = seq.conj().resolve_conj()
        ac = conv1d(inp, weight, padding=L-1, groups=B)  # (1, B, 2L-1)
        return ac.transpose(0, 1)                        # (B, 1, 2L-1)

    def _fourier_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, dim_input) → (B, 2m+1, 2m+1) complex coefficients C
        via u = autocorr(a), v = autocorr(b), C = u⊗v.
        """
        B = x.shape[0]
        L = self.m + 1

        # produce latent vectors a and b
        pa = self.fc_a(x)  # (B, 2L)
        pb = self.fc_b(x)  # (B, 2L)

        # split real/imag, reshape to (B,1,L)
        ar = pa[:, :L].view(B, 1, L)
        ai = pa[:, L:].view(B, 1, L)
        br = pb[:, :L].view(B, 1, L)
        bi = pb[:, L:].view(B, 1, L)

        a = torch.complex(ar, ai)
        b = torch.complex(br, bi)

        # autocorrelate to get length 2L-1 = 2m+1
        ua = self._autocorrelate1d(a)[:, 0]  # (B, 2m+1)
        va = self._autocorrelate1d(b)[:, 0]  # (B, 2m+1)

        # form outer product u[kx]·v[ky] → (B, 2m+1, 2m+1)
        C = ua.unsqueeze(-1) * va.unsqueeze(-2)

        # normalize so C[0,0] = 1/4
        centre = C[:, self.m, self.m].real        # (B,)
        C = C / centre.view(B, 1, 1) / 4.0
        return C

    def forward(
        self,
        inputs:  torch.Tensor,       # (..., dim_input)
        targets: torch.Tensor,       # (..., 2) in [-1,1]²
    ) -> torch.Tensor:
        leading = inputs.shape[:-1]

        x_flat = inputs.view(-1, self.dim_input).to(self.dev, self.dtype)
        t_flat = targets.view(-1, 2).to(self.dev, self.dtype)
        B = x_flat.shape[0]

        C = self._fourier_coeffs(x_flat)                # (B, 2m+1, 2m+1)

        # build exp(iπ(kx x + ky y))
        tx = t_flat[:, 0].unsqueeze(-1).unsqueeze(-1)
        ty = t_flat[:, 1].unsqueeze(-1).unsqueeze(-1)
        phase = self.kx_grid.unsqueeze(0) * tx + self.ky_grid.unsqueeze(0) * ty
        exps  = torch.exp(1j * torch.pi * phase)        # (B,2m+1,2m+1)

        pdf = (C * exps).sum(dim=(-2, -1)).real         # (B,)
        return pdf.view(*leading)

    @torch.no_grad()
    def grid_pdf(
        self,
        inputs: torch.Tensor,            # (d_in,) or (1,d_in)
        grid_size: int = 128,
        lim: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.dev, self.dtype)

        xs = torch.linspace(*lim, grid_size, device=self.dev)
        ys = torch.linspace(*lim, grid_size, device=self.dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")      # (G,G)

        pts = torch.stack([gx, gy], dim=-1).view(-1, 2)     # (G²,2)
        pdf = self.forward(inputs.expand(pts.size(0), -1), pts)
        return pdf.view(grid_size, grid_size).cpu()
