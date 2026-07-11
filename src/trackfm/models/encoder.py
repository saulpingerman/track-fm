"""TrackFM causal transformer encoder and pretraining model.

`CausalAISModel` is the SINGLE canonical definition of the architecture
(the old repo had three hand-synced copies). Ported numerically-verbatim
from experiment 11 (`run_experiment.py` L505-L788); the legacy-equivalence
test in tests/models/ guards the port.

Downstream tasks reuse `encode()` (the causal backbone) and attach their
own heads from trackfm.models.heads.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.fourier_head import FourierHead2D


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


class CausalAISModel(nn.Module):
    """Causal transformer for AIS trajectory prediction.

    Input features per position: [lat, lon, sog, cog_sin, cog_cos, dt],
    normalized per `NormalizationConfig`.
    """

    def __init__(
        self,
        model: ModelConfig,
        norm: NormalizationConfig,
        max_horizon: int = 800,
        num_horizon_samples: int = 4,
    ):
        super().__init__()
        self.model_cfg = model
        self.norm = norm
        self.d_model = model.d_model
        self.max_seq_len = model.max_seq_len
        self.max_horizon = max_horizon
        self.num_horizon_samples = num_horizon_samples

        # Input projection: 6 features -> d_model
        self.input_proj = nn.Linear(model.input_features, model.d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalEncoding(model.d_model)

        # Time encoding for horizon conditioning
        self.time_proj = nn.Sequential(
            nn.Linear(1, model.d_model),
            nn.GELU(),
            nn.Linear(model.d_model, model.d_model)
        )

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model.d_model,
            nhead=model.nhead,
            dim_feedforward=model.dim_feedforward,
            dropout=model.dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model.num_layers)

        # Horizon conditioning
        self.horizon_proj = nn.Linear(model.d_model * 2, model.d_model)

        # Fourier head
        self.fourier_head = FourierHead2D(
            model.d_model, model.grid_size, model.num_freqs, model.grid_range
        )

        # Causal mask cache
        self._causal_masks = {}

    # -------------------------------------------------------------- helpers
    def _get_causal_mask(self, seq_len: int, device: torch.device):
        if seq_len not in self._causal_masks:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self._causal_masks[seq_len] = mask
        return self._causal_masks[seq_len].to(device)

    def _denorm_positions(self, features: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Denormalize lat/lon slice to degrees. Returns (batch, end-start, 2)."""
        return torch.stack([
            features[:, start:end, 0] * self.norm.lat_scale + self.norm.lat_center,
            features[:, start:end, 1] * self.norm.lon_scale + self.norm.lon_center,
        ], dim=-1)

    def encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Causal backbone: (batch, seq_len, 6) -> (batch, seq_len, d_model).

        This is the reusable representation for downstream tasks.
        """
        x = self.input_proj(input_seq)
        x = self.pos_encoder(x)
        mask = self._get_causal_mask(input_seq.shape[1], input_seq.device)
        return self.transformer(x, mask=mask)

    # ------------------------------------------------------------- training
    def forward_train(
        self,
        features: torch.Tensor,
        horizon_indices: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass with random horizon sampling.

        Args:
            features: (batch, seq_len + max_horizon, 6)
                      [lat, lon, sog, cog_sin, cog_cos, dt]
            horizon_indices: (num_horizon_samples,) horizons to predict;
                             randomly sampled when None
            causal: True  -> causal subwindow training over ALL positions
                    False -> predict only from the last position (validation)

        Returns:
            log_densities: (batch, num_pairs, grid_size, grid_size)
            targets: (batch, num_pairs, 2) relative displacement (degrees)
            sampled_horizons: the horizon indices used
            future_mask: (num_pairs,) True where the target lies beyond the
                         input sequence
        """
        batch_size = features.shape[0]
        seq_len = self.max_seq_len
        max_horizon = self.max_horizon
        num_samples = self.num_horizon_samples
        device = features.device
        grid_size = self.model_cfg.grid_size

        if horizon_indices is None:
            horizon_indices = torch.randint(1, max_horizon + 1, (num_samples,), device=device)
            horizon_indices = torch.sort(horizon_indices)[0]

        num_samples = len(horizon_indices)

        input_seq = features[:, :seq_len, :]
        embeddings = self.encode(input_seq)

        # Cumulative dt (denormalized seconds) for horizon conditioning
        dt = features[:, :, 5] * self.norm.dt_scale
        cumsum_dt = torch.cumsum(dt, dim=1)

        positions = self._denorm_positions(features, 0, seq_len)
        future_pos = self._denorm_positions(features, seq_len, seq_len + max_horizon)

        if causal:
            # Causal subwindow training: every position predicts each sampled
            # horizon, giving equal training signal across horizons.
            all_positions = torch.cat([positions, future_pos], dim=1)

            all_embeddings = []
            all_cum_dt = []
            all_targets = []
            future_mask_list = []

            for h in horizon_indices:
                h_val = h.item()

                src_cumsum = cumsum_dt[:, :seq_len]
                tgt_cumsum = cumsum_dt[:, h_val:seq_len + h_val]
                cum_dt_h = tgt_cumsum - src_cumsum

                tgt_pos = all_positions[:, h_val:seq_len + h_val, :]
                tgt_rel = tgt_pos - positions

                all_embeddings.append(embeddings)
                all_cum_dt.append(cum_dt_h)
                all_targets.append(tgt_rel)

                if h_val >= seq_len:
                    future_mask_list.extend([True] * seq_len)
                else:
                    future_mask_list.extend(
                        [False] * (seq_len - h_val) + [True] * h_val
                    )

            all_embeddings = torch.cat(all_embeddings, dim=1)
            all_cum_dt = torch.cat(all_cum_dt, dim=1)
            all_targets = torch.cat(all_targets, dim=1)
            future_mask = torch.tensor(future_mask_list, dtype=torch.bool, device=device)

            num_pairs = all_embeddings.shape[1]

            time_input = all_cum_dt.unsqueeze(-1) / 300.0  # Normalize by 5 minutes
            time_enc = self.time_proj(time_input)

            combined = torch.cat([all_embeddings, time_enc], dim=-1)
            conditioned = self.horizon_proj(combined)

            cond_flat = conditioned.view(batch_size * num_pairs, -1)
            log_dens_flat = self.fourier_head(cond_flat)
            log_densities = log_dens_flat.view(batch_size, num_pairs, grid_size, grid_size)

            return log_densities, all_targets, horizon_indices, future_mask

        else:
            # Predict only from the last input position
            last_emb = embeddings[:, -1, :]
            last_cumsum = cumsum_dt[:, seq_len - 1]
            src_pos = positions[:, -1, :]

            targets = []
            cum_times = []

            for h in horizon_indices:
                h_val = h.item()
                cum_time = cumsum_dt[:, seq_len + h_val - 1] - last_cumsum
                tgt_rel = future_pos[:, h_val - 1, :] - src_pos

                targets.append(tgt_rel)
                cum_times.append(cum_time)

            targets = torch.stack(targets, dim=1)
            cum_times = torch.stack(cum_times, dim=1)

            time_input = cum_times.unsqueeze(-1) / 300.0
            time_enc = self.time_proj(time_input)

            last_emb_expanded = last_emb.unsqueeze(1).expand(-1, num_samples, -1)

            combined = torch.cat([last_emb_expanded, time_enc], dim=-1)
            conditioned = self.horizon_proj(combined)

            cond_flat = conditioned.view(batch_size * num_samples, -1)
            log_dens_flat = self.fourier_head(cond_flat)
            log_densities = log_dens_flat.view(batch_size, num_samples, grid_size, grid_size)

            future_mask = torch.ones(num_samples, dtype=torch.bool, device=device)

            return log_densities, targets, horizon_indices, future_mask
