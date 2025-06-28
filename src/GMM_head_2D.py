import torch
from torch import nn
from typing import Tuple, Union

TensorOrFloat = Union[torch.Tensor, float]


class GMMHead2D(nn.Module):
    r"""
    Mixture of K full-covariance Gaussians on the square domain [-1, 1]².

    For each component *k* the network predicts
        • weight  wₖ   (via softmax)  
        • mean    μₖ = (μₖˣ, μₖʸ)  
        • scale   σₖ = (σₖˣ, σₖʸ)  (positive)  
        • correlation ρₖ  (tanh-squashed into (-1,1))

    PDF of one component:
        N₂(x; μ, Σ)  with
            Σ = [[σˣ², ρ σˣ σʸ],
                 [ρ σˣ σʸ, σʸ²]]

    Parameters
    ----------
    dim_input : int        – size of conditioning vector per sample
    num_components : int   – mixture size K
    dtype / device :       – usual torch settings
    """

    def __init__(
        self,
        dim_input: int,
        num_components: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        target_range: Tuple[float, float] = (-1.0, 1.0),   # kept for API parity
    ):
        super().__init__()
        self.d_in = dim_input
        self.K = num_components
        self.dtype, self.dev = dtype, device

        # store target range buffers (unused internally but keeps same signature)
        t_min, t_max = target_range
        self.register_buffer("t_min", torch.tensor(t_min, dtype=dtype))
        self.register_buffer("t_max", torch.tensor(t_max, dtype=dtype))

        # For each k: [logit_w, μx, μy, log_σx, log_σy, ρ_raw]  → 6K outputs
        self.fc_params = nn.Linear(dim_input, 6 * num_components)
        nn.init.normal_(self.fc_params.weight, 1e-2)
        nn.init.zeros_(self.fc_params.bias)

        self.to(device)

    # ---------------------------------------------------------------- forward --
    def forward(
        self,
        inputs: torch.Tensor,          # (..., d_in)
        targets: torch.Tensor,         # (..., 2) in [-1,1]²
        return_params: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the mixture PDF evaluated at each (x,y).

        Shapes:
            pdf   – (...,)          same leading dims as `inputs`
            means – (..., K, 2)     only if `return_params=True`
        """
        B = inputs.numel() // self.d_in
        x = inputs.view(B, self.d_in).to(self.dev, self.dtype)
        t = targets.view(B, 2).to(self.dev, self.dtype)

        # raw network output  → (B, K, 6)
        p = self.fc_params(x).view(B, self.K, 6)
        logit_w, mu_x, mu_y, log_sig_x, log_sig_y, rho_raw = p.unbind(-1)

        w = torch.softmax(logit_w, dim=1)                 # (B,K)
        sig_x = torch.exp(log_sig_x)                      # (B,K)
        sig_y = torch.exp(log_sig_y)                      # (B,K)
        rho = torch.tanh(rho_raw).clamp(-0.999, 0.999)    # (B,K) keep numerically safe

        # ---------------------------------------------------------------- 2-D Gaussian pdf per component
        # centred & scaled coordinates
        z_x = (t[:, 0:1] - mu_x) / sig_x                  # (B,K)
        z_y = (t[:, 1:2] - mu_y) / sig_y                  # (B,K)

        inv_rho = 1.0 / (1.0 - rho ** 2)                  # (B,K)
        exponent = -0.5 * inv_rho * (z_x**2 - 2*rho*z_x*z_y + z_y**2)  # (B,K)

        norm_const = (
            1.0
            / (2.0 * torch.pi * sig_x * sig_y * torch.sqrt(1.0 - rho ** 2))
        )  # (B,K)

        comp_pdf = norm_const * torch.exp(exponent)       # (B,K)

        pdf = (w * comp_pdf).sum(dim=1)                   # (B,)
        pdf = pdf.view(*inputs.shape[:-1])                # (...)

        if return_params:
            means = torch.stack([mu_x, mu_y], dim=-1)     # (B,K,2)
            means = means.view(*inputs.shape[:-1], self.K, 2)
            return pdf, means
        return pdf

    # ---------------------------------------------------------- visualisation --
    @torch.no_grad()
    def grid_pdf(
        self,
        inputs: torch.Tensor,            # (d_in,) or (1,d_in)
        grid_size: int = 128,
        lim: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """
        Convenience helper: returns a (grid_size, grid_size) tensor of p(x,y).

        Call like:
            img = model.grid_pdf(context_vec, 200)
            plt.imshow(img, extent=[-1,1,-1,1]); plt.show()
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.dev, self.dtype)

        xs = torch.linspace(*lim, grid_size, device=self.dev)
        ys = torch.linspace(*lim, grid_size, device=self.dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")          # (G,G)

        pts = torch.stack([gx, gy], dim=-1).view(-1, 2)         # (G²,2)
        pdf = self.forward(inputs.expand(pts.size(0), -1), pts) # (G²,)
        return pdf.view(grid_size, grid_size).cpu()
