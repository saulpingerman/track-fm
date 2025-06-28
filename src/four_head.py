import torch
from torch import nn
from torch.nn.functional import conv1d
from typing import Tuple, Union


class FourierHead(nn.Module):
    """
    Continuous Fourier-based PDF evaluator. Given an input vector and per-sample targets,
    returns the probability density f(T) at each target T in the original domain.
    Internally, targets are scaled to [-1,1] for the Fourier series and then
    the resulting PDF is mapped back to the original target scale.
    """

    def __init__(
        self,
        dim_input: int,
        num_frequencies: int,
        regularization_gamma: float = 0.0,
        dtype=torch.float32,
        device: str = "cuda",
        # target_range: (min, max) of the original target domain
        target_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.dim_input = dim_input
        self.num_frequencies = num_frequencies
        self.regularization_gamma = regularization_gamma
        self.dtype = dtype
        self.device = device

        # Register target normalization/unnormalization range
        t_min, t_max = target_range
        self.register_buffer('t_min', torch.tensor(t_min, dtype=dtype))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=dtype))

        # Linear layer to predict real & imag parts of autocorrelation parameters
        self.fc_autocorr = nn.Linear(
            dim_input,
            2 * (num_frequencies + 1)
        )
        nn.init.normal_(self.fc_autocorr.weight, std=1e-2)
        nn.init.zeros_(self.fc_autocorr.bias)
        self.fc_autocorr.to(device)

        # Precompute and register frequencies buffer: shape (m,), complex
        freqs = torch.arange(1, num_frequencies + 1, dtype=dtype) * torch.pi
        freqs = (1j * freqs)
        self.register_buffer('frequencies', freqs)

        # Move entire module to device
        self.to(self.device)

    def autocorrelate(self, seq: torch.Tensor) -> torch.Tensor:
        """
        This function computes the autocorrelation of a complex sequence for a batch of sequences via grouped conv1d. 
        
        Q: You may wonder, why do we even have an autocorrelation function????
        A: It's easy to let the model output two values for each complex number as long as these values do not have
        restrictions. One value is used for the real part and one for the imaginary part. Unfortunately, there ARE 
        restictions on what the Fourier coefficients can be if we want the function defined by the Fourier series to be a PDF.
        These restrictions require Fourier coefficents to be 
            - Hermitian symmetric (look it up)
                - Why? You want the final function to NOT have imaginary components & this guarantees that.
            - Positive defintie (look it up)
                - Why? You want the final function to only give non-negative values (because PDF)
                & this guarantees that.
            - Normalization control
                - Can normalize the function to integrate to 1 by dividing by the real part of the 0th fourier 
                coefficient. That doesn't work if this first term has imaginary components OR is negative OR is zero.
        By taking the autocorrelation of ANY set of complex numbers the resulting same size set of numbers
        have all of the above properties. Thus, when used as the coefficients of a fourier series we get an output PDF.
                
        seq: (B, m+1) complex
        returns: (B, m+1) complex autocorr (only non-negative lags)
        """
        B, L = seq.shape
        inp = seq[None, :, :]
        weight = seq[:, None, :].conj().resolve_conj()
        ac = conv1d(inp, weight, padding=L-1, groups=B)
        return ac[0, :, L-1:]

    def compute_fourier_coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, dim_input) real (unchanged domain)
        returns: (B, m+1) complex autocorrelated Fourier coefficients
        """
        x = x.to(self.device).to(self.dtype)
        p = self.fc_autocorr(x)  # (B, 2*(m+1)) real
        a = torch.complex(
            p[..., :self.num_frequencies+1].to(self.dtype),
            p[..., self.num_frequencies+1:].to(self.dtype)
        )  # (B, m+1)
        return self.autocorrelate(a)

    def evaluate_pdf_at_one(
        self,
        fourier_coeffs: torch.Tensor,
        t_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute f_norm(t_norm_i) for each sample i on the normalized [-1,1] domain:
        fourier_coeffs: (B, m) normalized coefficients (a1..am)
        t_norm:         (B,) values in [-1,1]
        return:         (B,) real PDF values on normalized domain
        """
        exps = torch.exp(
            t_norm.unsqueeze(1) * self.frequencies.unsqueeze(0)
        )  # (B, m)
        pdf = 0.5 + (fourier_coeffs * exps).sum(dim=1)
        return pdf.real

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        inputs:  (..., dim_input) real — passed through unchanged
        targets: (...,)           real in original [t_min, t_max]
        returns: (...) of PDF(inputs_i at targets_i) in original target scale
        """
        # flatten inputs and targets
        B = inputs.numel() // self.dim_input
        x_flat = inputs.view(B, self.dim_input)
        t_flat = targets.view(B)

        # compute full coeffs
        coeffs_full = self.compute_fourier_coefficients(x_flat)  # (B, m+1)
        # split DC and harmonics
        a0 = coeffs_full[:, :1].real   # (B,1)
        ak = coeffs_full[:, 1:]        # (B, m)
        coeffs_norm = ak / a0          # (B, m), ensures ∫PDF=1 on [-1,1]

        # Normalize targets to [-1,1]
        t_norm = 2 * (t_flat - self.t_min) / (self.t_max - self.t_min) - 1

        # evaluate on normalized domain
        pdf_norm = self.evaluate_pdf_at_one(coeffs_norm, t_norm)  # (B,)

        # Unnormalize PDF back to original target scale:
        # if T ∈ [t_min, t_max], then t_norm = (2/(t_max-t_min))*(T - t_min) -1
        # so dT = (t_max - t_min)/2 * dt_norm => pdf_original = pdf_norm * dt_norm/dT
        scale = 2.0 / (self.t_max - self.t_min)
        pdf_flat = pdf_norm * scale

        return pdf_flat.view(*inputs.shape[:-1])
