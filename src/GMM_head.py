import torch
from torch import nn
from typing import Tuple, Union

TensorOrFloat = Union[torch.Tensor, float]

class GMMHead(nn.Module):
    """
    Gaussian Mixture Model head. Given an input vector and per-sample targets,
    predicts a mixture of K Gaussians over the original target domain and returns
    either just the PDF p(T) or (PDF, means) if you need the component means.
    """

    def __init__(
        self,
        dim_input: int,
        num_components: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        target_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.dim_input = dim_input
        self.num_components = num_components
        self.dtype = dtype
        self.device = device

        # target_range buffer (for API parity)
        t_min, t_max = target_range
        self.register_buffer('t_min', torch.tensor(t_min, dtype=dtype))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=dtype))

        # predict [logit_w, mean, logvar] for each of K components
        self.fc_params = nn.Linear(dim_input, 3 * num_components)
        nn.init.normal_(self.fc_params.weight, std=1e-2)
        nn.init.zeros_(self.fc_params.bias)
        self.to(device)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs:  (..., dim_input) real
        targets: (...,)           real in [t_min, t_max]
        return_params: if True, returns (pdf, means); else just pdf

        - pdf:   shape (...) giving p(T)
        - means: shape (..., K) giving each component mean μ_k
        """
        B = inputs.numel() // self.dim_input
        x = inputs.view(B, self.dim_input).to(self.device).to(self.dtype)
        t = targets.view(B).to(self.device).to(self.dtype)

        # raw params → (B, K, 3)
        p = self.fc_params(x).view(B, self.num_components, 3)
        logit_w = p[..., 0]      # (B, K)
        means   = p[..., 1]      # (B, K)
        logvars = p[..., 2]      # (B, K)

        # mixture weights and variances
        weights = torch.softmax(logit_w, dim=1)   # (B, K)
        vars    = torch.exp(logvars)              # (B, K)

        # compute each component’s Gaussian PDF at t
        t_exp = t.unsqueeze(1)                    # (B, 1)
        coeff    = 1.0 / torch.sqrt(2*torch.pi*vars)
        exponent = -0.5 * ( (t_exp - means)**2 / vars )
        comp_pdf = coeff * torch.exp(exponent)    # (B, K)

        # mix
        pdf = (weights * comp_pdf).sum(dim=1)      # (B,)

        pdf = pdf.view(*inputs.shape[:-1])        # reshape to (...)
        if return_params:
            # also reshape means to (..., K)
            means = means.view(*inputs.shape[:-1], self.num_components)
            return pdf, means
        else:
            return pdf