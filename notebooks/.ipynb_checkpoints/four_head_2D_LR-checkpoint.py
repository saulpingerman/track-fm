import torch
from torch import nn
from torch.nn.functional import conv1d
from typing import Tuple

class FourierHead2DLR(nn.Module):
    r"""
    Continuous 2-D PDF on [-1,1]^2:
        f(x,y) = Σ_{kx,ky} C[kx,ky] · e^{iπ(kx·x + ky·y)},
    with 
        C = Σ_{r=1}^R u^{(r)} ⊗ v^{(r)}, 
    where u^{(r)},v^{(r)} are 1D autocorrelations of learned vectors a^{(r)},b^{(r)}.
    """

    def __init__(
        self,
        dim_input: int,
        num_frequencies: int,
        rank: int = 1,
        regularisation_gamma: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim_input = dim_input
        self.m = num_frequencies
        self.rank = rank
        self.reg_gamma = regularisation_gamma
        self.dtype = dtype
        self.dev = device

        L = self.m + 1
        # two linear layers, each outputs R complex vectors of length L
        self.fc_a = nn.Linear(dim_input, 2 * rank * L)
        self.fc_b = nn.Linear(dim_input, 2 * rank * L)
        nn.init.normal_(self.fc_a.weight, 1e-2)
        nn.init.zeros_(self.fc_a.bias)
        nn.init.normal_(self.fc_b.weight, 1e-2)
        nn.init.zeros_(self.fc_b.bias)

        # prepare frequency grids for the final Fourier phase
        k = torch.arange(-self.m, self.m + 1, dtype=dtype)
        kx, ky = torch.meshgrid(k, k, indexing="xy")
        self.register_buffer("kx_grid", kx)
        self.register_buffer("ky_grid", ky)

        self.to(device)

    @staticmethod
    def _autocorrelate1d(seq: torch.Tensor) -> torch.Tensor:
        """
        1D autocorrelation via grouped conv1d.
          seq: (B,1,L) complex
        → (B,1,2L-1) complex
        """
        B, _, L = seq.shape
        inp    = seq.transpose(0,1)              # (1,B,L)
        weight = seq.conj().resolve_conj()       # (B,1,L)
        ac = conv1d(inp, weight, padding=L-1, groups=B)  # (1,B,2L-1)
        return ac.transpose(0,1)                 # (B,1,2L-1)

    def _fourier_coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,dim_input) → C: (B,2m+1,2m+1) complex
        using rank-R sum of outer products of 1D autocorrs.
        """
        B = x.size(0)
        L = self.m + 1
        R = self.rank

        # --- produce R latent vectors a^(r), b^(r):
        pa = self.fc_a(x)  # (B, 2*R*L)
        pb = self.fc_b(x)

        # split real/imag and reshape to (B,R,L)
        pa = pa.view(B, R, 2, L)
        ar, ai = pa[:,:,0,:], pa[:,:,1,:]
        pb = pb.view(B, R, 2, L)
        br, bi = pb[:,:,0,:], pb[:,:,1,:]

        # form complex (B*R,1,L) for batched autocorr
        a = torch.complex(ar, ai).view(B*R,1,L)
        b = torch.complex(br, bi).view(B*R,1,L)

        # autocorrelate to (B*R,1,2L-1) → reshape to (B,R,2m+1)
        ua = self._autocorrelate1d(a).view(B, R, 2*L-1)[:, :, :]
        va = self._autocorrelate1d(b).view(B, R, 2*L-1)[:, :, :]

        # build C = Σ_r [u^(r) ⊗ v^(r)] → (B,2m+1,2m+1)
        # ua: (B,R,K), va: (B,R,K) with K=2m+1
        K = 2*L - 1
        C = (ua.unsqueeze(3) * va.unsqueeze(2)).sum(dim=1)  # (B,K,K)

        # normalize so C[0,0] = 1/4
        centre = C[:, self.m, self.m].real.view(B,1,1)
        C = C / centre / 4.0

        return C

    def forward(
        self,
        inputs:  torch.Tensor,       # (..., dim_input)
        targets: torch.Tensor,       # (...,2) ∈ [-1,1]^2
    ) -> torch.Tensor:
        leading = inputs.shape[:-1]
        x_flat = inputs.view(-1, self.dim_input).to(self.dev, self.dtype)
        t_flat = targets.view(-1, 2).to(self.dev, self.dtype)
        B = x_flat.size(0)

        C = self._fourier_coeffs(x_flat)   # (B,2m+1,2m+1)

        # build e^{iπ(kx x + ky y)}  →  (B,2m+1,2m+1)
        tx = t_flat[:,0].view(B,1,1)
        ty = t_flat[:,1].view(B,1,1)
        phase = self.kx_grid.unsqueeze(0)*tx + self.ky_grid.unsqueeze(0)*ty
        exps  = torch.exp(1j*torch.pi*phase)

        pdf = (C * exps).sum(dim=(-2,-1)).real  # (B,)
        return pdf.view(*leading)

    @torch.no_grad()
    def grid_pdf(
        self,
        inputs: torch.Tensor,
        grid_size: int = 128,
        lim: Tuple[float, float] = (-1.,1.),
    ) -> torch.Tensor:
        if inputs.dim()==1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.dev, self.dtype)

        xs = torch.linspace(*lim, grid_size, device=self.dev)
        ys = torch.linspace(*lim, grid_size, device=self.dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")

        pts = torch.stack([gx,gy],-1).view(-1,2)
        pdf = self.forward(inputs.expand(pts.size(0),-1), pts)
        return pdf.view(grid_size,grid_size).cpu()
