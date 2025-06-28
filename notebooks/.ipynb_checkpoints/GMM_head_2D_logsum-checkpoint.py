# GMM_head_2D_logsum.py  ───── numerically stable 2-D GMM head
import math
from typing import Tuple, Union

import torch
from torch import nn

TensorOrFloat = Union[torch.Tensor, float]


class GMMHead2D(nn.Module):
    r"""
    Mixture of *K* full-covariance Gaussians on the square domain [-1, 1]².

    Network predicts per component k:
        • mixture weight   w_k   (via softmax)
        • mean             μ_k = (μx, μy)
        • std-devs         σ_k = (σx, σy)      – constrained σ ≥ 1e-3
        • correlation      ρ_k                 – squashed to (-0.95, 0.95)

    Forward returns **log-pdf** for stability.
    """

    # ------------------------------------------------------------ constructor
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

        # Dummy buffers (unused) so the API matches earlier heads
        self.register_buffer("t_min", torch.tensor(target_range[0], dtype=dtype))
        self.register_buffer("t_max", torch.tensor(target_range[1], dtype=dtype))

        # fc outputs: [logit_w, μx, μy, log_σx, log_σy, ρ_raw]  → 6K per sample
        self.fc_params = nn.Linear(dim_input, 6 * num_components)
        nn.init.normal_(self.fc_params.weight, 1e-2)
        nn.init.zeros_(self.fc_params.bias)

        self.to(device)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        inputs: torch.Tensor,          # (..., dim_input)
        targets: torch.Tensor,         # (..., 2)
        return_params: bool = False,
    ):
        """
        Returns:
            log_pdf   (...,)            – same leading dims as `inputs`
            means     (..., K, 2)       – only if `return_params=True`
        """
        leading = inputs.shape[:-1]                       # save batch dims
        B = inputs.numel() // self.d_in

        x = inputs.view(B, self.d_in).to(self.dev, self.dtype)
        t = targets.view(B, 2).to(self.dev, self.dtype)

        # (B, K, 6)
        p = self.fc_params(x).view(B, self.K, 6)
        logit_w, mu_x, mu_y, log_sig_x, log_sig_y, rho_raw = p.unbind(-1)

        # ---------- stable parameterisations
        sig_x = torch.exp(log_sig_x).clamp_min(1e-3)      # σ ≥ 1e-3
        sig_y = torch.exp(log_sig_y).clamp_min(1e-3)
        rho   = torch.tanh(rho_raw) * 0.95                # |ρ| ≤ 0.95

        # centred & scaled coordinates
        z_x = (t[:, 0:1] - mu_x) / sig_x                  # (B,K)
        z_y = (t[:, 1:2] - mu_y) / sig_y

        inv_rho2 = 1.0 / (1.0 - rho**2)

        # ---------- log-pdf per component
        log_2pi = math.log(2.0 * math.pi)

        log_norm = (
            -log_2pi
            - torch.log(sig_x)
            - torch.log(sig_y)
            - 0.5 * torch.log1p(-rho**2)                 # log √(1-ρ²)
        )

        quad = -0.5 * inv_rho2 * (z_x**2 - 2 * rho * z_x * z_y + z_y**2)
        log_comp = log_norm + quad                       # (B,K)

        # ---------- mixture via log-sum-exp
        log_w   = torch.log_softmax(logit_w, dim=1)      # (B,K)
        log_pdf = torch.logsumexp(log_w + log_comp, dim=1)   # (B,)

        log_pdf = log_pdf.view(*leading)                 # restore batch dims

        if return_params:
            means = torch.stack([mu_x, mu_y], dim=-1).view(*leading, self.K, 2)
            return log_pdf, means
        return log_pdf


# --------------------------------------------------------------------------- sanity-check
if __name__ == "__main__":
    torch.manual_seed(0)
    head = GMMHead2D(dim_input=32, num_components=5, device="cpu")
    dummy_x = torch.randn(4, 32)               # batch of 4 context vectors
    dummy_y = torch.rand(4, 2) * 2 - 1         # targets in [-1,1]²
    lp = head(dummy_x, dummy_y)                # log-pdf
    print("log-p:", lp)
