"""Legacy experiment-11 model classes, extracted verbatim for equivalence testing.

Source: track-fm archive/2026-pre-overhaul,
experiments/11_long_horizon_69_days/run_experiment.py L505-L788.
Do not edit — this is the reference implementation the port is tested against.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128
    max_horizon: int = 400
    num_horizon_samples: int = 4
    grid_size: int = 64
    grid_range: float = 0.3
    num_freqs: int = 12
    lat_mean: float = 56.25
    lat_std: float = 1.0
    lon_mean: float = 11.5
    lon_std: float = 2.0
    dt_max: float = 300.0


class SinusoidalEncoding(nn.Module):
    """Sinusoidal encoding for continuous values."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


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
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)

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


class CausalAISModel(nn.Module):
    """Causal transformer for AIS trajectory prediction."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Input projection: 6 features -> d_model
        self.input_proj = nn.Linear(6, config.d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalEncoding(config.d_model)

        # Time encoding for horizon conditioning
        self.time_proj = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Horizon conditioning
        self.horizon_proj = nn.Linear(config.d_model * 2, config.d_model)

        # Fourier head
        self.fourier_head = FourierHead2D(
            config.d_model, config.grid_size, config.num_freqs, config.grid_range
        )

        # Causal mask cache
        self._causal_masks = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device):
        if seq_len not in self._causal_masks:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self._causal_masks[seq_len] = mask
        return self._causal_masks[seq_len].to(device)

    def forward_train(self, features: torch.Tensor, horizon_indices: torch.Tensor = None,
                      causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with random horizon sampling.

        Args:
            features: (batch, seq_len + max_horizon, 6)
                      [lat, lon, sog, cog_sin, cog_cos, dt]
            horizon_indices: (num_horizon_samples,) - which horizons to predict
                            If None, randomly samples num_horizon_samples horizons
            causal: If True, use all valid positions for each horizon (causal subwindow training)
                   If False, only use last position (for validation/baseline comparison)

        Returns:
            log_densities: (batch, num_pairs, grid_size, grid_size)
            targets: (batch, num_pairs, 2) - relative displacement in lat/lon
            sampled_horizons: (num_horizon_samples,) - the horizon indices used
            future_mask: (num_pairs,) - boolean mask, True for future predictions (from last position)
        """
        batch_size = features.shape[0]
        seq_len = self.config.max_seq_len
        max_horizon = self.config.max_horizon
        num_samples = self.config.num_horizon_samples
        device = features.device

        # Sample random horizons if not provided
        if horizon_indices is None:
            # Uniform random sampling from 1 to max_horizon
            horizon_indices = torch.randint(1, max_horizon + 1, (num_samples,), device=device)
            horizon_indices = torch.sort(horizon_indices)[0]  # Sort for easier debugging

        num_samples = len(horizon_indices)

        # Split into input sequence and future positions
        input_seq = features[:, :seq_len, :]  # (batch, seq_len, 6)

        # Project and encode
        x = self.input_proj(input_seq)
        x = self.pos_encoder(x)

        # Causal transformer
        mask = self._get_causal_mask(seq_len, device)
        embeddings = self.transformer(x, mask=mask)

        # Compute cumulative dt for horizon conditioning
        dt = features[:, :, 5] * self.config.dt_max  # Denormalize all dt
        cumsum_dt = torch.cumsum(dt, dim=1)

        # Get all positions (denormalized) for causal training
        positions = torch.stack([
            features[:, :seq_len, 0] * self.config.lat_std + self.config.lat_mean,
            features[:, :seq_len, 1] * self.config.lon_std + self.config.lon_mean
        ], dim=-1)  # (batch, seq_len, 2)

        # Future positions (beyond input sequence)
        future_pos = torch.stack([
            features[:, seq_len:seq_len+max_horizon, 0] * self.config.lat_std + self.config.lat_mean,
            features[:, seq_len:seq_len+max_horizon, 1] * self.config.lon_std + self.config.lon_mean
        ], dim=-1)  # (batch, max_horizon, 2)

        if causal:
            # Causal subwindow training: predict from ALL positions for each horizon
            # This ensures equal training signal across all horizons (seq_len predictions each)

            # Combine input and future positions for easier indexing
            all_positions = torch.cat([positions, future_pos], dim=1)  # (batch, seq_len + max_horizon, 2)

            all_embeddings = []
            all_cum_dt = []
            all_targets = []
            future_mask_list = []

            for h in horizon_indices:
                h_val = h.item()

                # ALL seq_len positions predict h steps ahead
                # Source: positions 0 to seq_len-1
                # Target: positions h to seq_len-1+h (may be within-sequence or future)
                src_emb = embeddings  # (batch, seq_len, d_model)
                src_cumsum = cumsum_dt[:, :seq_len]  # (batch, seq_len)
                tgt_cumsum = cumsum_dt[:, h_val:seq_len+h_val]  # (batch, seq_len)
                cum_dt_h = tgt_cumsum - src_cumsum  # (batch, seq_len)

                src_pos = positions  # (batch, seq_len, 2)
                tgt_pos = all_positions[:, h_val:seq_len+h_val, :]  # (batch, seq_len, 2)
                tgt_rel = tgt_pos - src_pos  # (batch, seq_len, 2)

                all_embeddings.append(src_emb)
                all_cum_dt.append(cum_dt_h)
                all_targets.append(tgt_rel)

                # Mark which predictions are future (target index >= seq_len)
                # Position i predicts position i+h, which is future if i+h >= seq_len
                if h_val >= seq_len:
                    # All predictions are future
                    future_mask_list.extend([True] * seq_len)
                else:
                    # Positions 0 to (seq_len-h_val-1) predict within-sequence
                    # Positions (seq_len-h_val) to (seq_len-1) predict future
                    num_within = seq_len - h_val
                    num_future = h_val
                    future_mask_list.extend([False] * num_within + [True] * num_future)

            # Concatenate all predictions
            all_embeddings = torch.cat(all_embeddings, dim=1)  # (batch, num_horizons * seq_len, d_model)
            all_cum_dt = torch.cat(all_cum_dt, dim=1)  # (batch, num_horizons * seq_len)
            all_targets = torch.cat(all_targets, dim=1)  # (batch, num_horizons * seq_len, 2)
            future_mask = torch.tensor(future_mask_list, dtype=torch.bool, device=device)

            num_pairs = all_embeddings.shape[1]

            # Time encoding
            time_input = all_cum_dt.unsqueeze(-1) / 300.0  # Normalize by 5 minutes
            time_enc = self.time_proj(time_input)  # (batch, num_pairs, d_model)

            # Combine and project
            combined = torch.cat([all_embeddings, time_enc], dim=-1)
            conditioned = self.horizon_proj(combined)  # (batch, num_pairs, d_model)

            # Fourier head
            cond_flat = conditioned.view(batch_size * num_pairs, -1)
            log_dens_flat = self.fourier_head(cond_flat)
            log_densities = log_dens_flat.view(batch_size, num_pairs, self.config.grid_size, self.config.grid_size)

            return log_densities, all_targets, horizon_indices, future_mask

        else:
            # Non-causal: only use last position (for validation/baseline comparison)
            last_emb = embeddings[:, -1, :]  # (batch, d_model)
            last_cumsum = cumsum_dt[:, seq_len - 1]  # (batch,)
            src_pos = positions[:, -1, :]  # (batch, 2)

            # Get future positions for sampled horizons
            targets = []
            cum_times = []

            for h in horizon_indices:
                h_val = h.item()
                # Future prediction - use cumsum_dt directly (covers all positions)
                cum_time = cumsum_dt[:, seq_len + h_val - 1] - last_cumsum
                tgt_rel = future_pos[:, h_val-1, :] - src_pos

                targets.append(tgt_rel)
                cum_times.append(cum_time)

            targets = torch.stack(targets, dim=1)  # (batch, num_samples, 2)
            cum_times = torch.stack(cum_times, dim=1)  # (batch, num_samples)

            # Time encoding
            time_input = cum_times.unsqueeze(-1) / 300.0  # Normalize by 5 minutes
            time_enc = self.time_proj(time_input)  # (batch, num_samples, d_model)

            # Expand last embedding for all horizons
            last_emb_expanded = last_emb.unsqueeze(1).expand(-1, num_samples, -1)

            # Combine and project
            combined = torch.cat([last_emb_expanded, time_enc], dim=-1)
            conditioned = self.horizon_proj(combined)  # (batch, num_samples, d_model)

            # Fourier head
            cond_flat = conditioned.view(batch_size * num_samples, -1)
            log_dens_flat = self.fourier_head(cond_flat)
            log_densities = log_dens_flat.view(batch_size, num_samples, self.config.grid_size, self.config.grid_size)

            # All predictions are future predictions when causal=False
            future_mask = torch.ones(num_samples, dtype=torch.bool, device=device)

            return log_densities, targets, horizon_indices, future_mask



# ============================================================================
# Loss Function
# ============================================================================

def compute_soft_target_loss(log_density: torch.Tensor, target: torch.Tensor,
                             grid_range: float, grid_size: int, sigma: float,
                             chunk_size: int = 512) -> torch.Tensor:
    """Soft target KL divergence loss with chunked computation for memory efficiency."""
    batch_size, num_pairs, gs, _ = log_density.shape
    device = log_density.device

    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Clip targets to grid range
    target_clipped = torch.clamp(target, -grid_range * 0.99, grid_range * 0.99)

    # Process in chunks to avoid memory explosion
    total_loss = 0.0
    total_count = 0

    for i in range(0, batch_size, chunk_size):
        end_i = min(i + chunk_size, batch_size)
        chunk_target = target_clipped[i:end_i]
        chunk_log_density = log_density[i:end_i]

        target_x = chunk_target[:, :, 0:1, None]
        target_y = chunk_target[:, :, 1:2, None]

        dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
        soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
        soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

        chunk_loss = F.kl_div(chunk_log_density, soft_target, reduction='none').sum(dim=(-2, -1))
        total_loss = total_loss + chunk_loss.sum()
        total_count += chunk_loss.numel()

    return total_loss / total_count

