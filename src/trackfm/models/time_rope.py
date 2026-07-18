"""Continuous-time rotary position embedding for irregular AIS series.

The stock encoder adds an INDEX-based sinusoidal PE, so attention sees
"5 posits apart" identically whether that is 10 seconds or 20 minutes of
elapsed time — the irregular sampling is visible only through the dt
input feature. Time-RoPE instead rotates each attention query/key by
angles proportional to the posit's cumulative elapsed time, making every
attention logit a function of the TRUE time difference between positions
(RoPE's relative property, Su et al. 2021, applied to wall-clock time
rather than token index).

Rotation frequencies span geometrically spaced periods [p_min, p_max]
seconds, covering AIS reporting cadence (~2-10 s) through multi-hour
window spans. Angles depend only on time differences after the dot
product, so the layer is invariant to shifting all times by a constant
(pinned by test).

The layer mirrors nn.TransformerEncoderLayer (pre-norm, ReLU FF, same
dropout placement and init family) so the ablation isolates the
positional mechanism, not incidental layer differences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(times_s: torch.Tensor, n_freqs: int,
                p_min: float, p_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for per-position rotation.

    times_s: (B, L) elapsed seconds. Returns cos, sin of shape
    (B, 1, L, n_freqs) broadcastable over heads.
    """
    steps = torch.arange(n_freqs, device=times_s.device, dtype=torch.float32)
    periods = p_min * (p_max / p_min) ** (steps / max(n_freqs - 1, 1))
    ang = 2 * torch.pi * times_s.unsqueeze(-1).float() / periods
    return (ang.cos().unsqueeze(1).to(times_s.dtype),
            ang.sin().unsqueeze(1).to(times_s.dtype))


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """Rotate feature pairs. x: (B, H, L, Dh), cos/sin: (B, 1, L, Dh/2)."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        [even * cos - odd * sin, even * sin + odd * cos], dim=-1
    ).flatten(-2)


class TimeRoPEEncoderLayer(nn.Module):
    """Pre-norm transformer layer with time-rotary self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, p_min: float, p_max: float):
        super().__init__()
        assert d_model % nhead == 0 and (d_model // nhead) % 2 == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.p_min, self.p_max = p_min, p_max
        self.dropout_p = dropout

        self.in_proj = nn.Linear(d_model, 3 * d_model)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.bias)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = self.in_proj(h).chunk(3, dim=-1)
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        cos, sin = rope_angles(times_s, self.d_head // 2, self.p_min, self.p_max)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout1(self.out_proj(attn))
        h = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(h)))))
        return x


class TimeRoPEEncoder(nn.Module):
    """Stack of TimeRoPEEncoderLayer. No final norm — matches the
    historical pre-norm stack, so the ablation changes ONLY the
    positional mechanism."""

    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int, dropout: float,
                 p_min: float, p_max: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TimeRoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                 p_min, p_max)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, times_s)
        return x
