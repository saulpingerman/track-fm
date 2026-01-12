#!/usr/bin/env python3
"""
Experiment 8: Causal Multi-Horizon Prediction with Variable Time Spacing

This experiment implements a decoder transformer that:
1. Processes variable-length tracks with irregular time spacing
2. At each position, predicts multiple future horizons (1, 2, ..., k positions ahead)
3. Conditions predictions on cumulative Δt to target
4. Outputs 2D density via Fourier head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import json
from pathlib import Path
import sys
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}", flush=True)


# ============================================================================
# Part 1: Data Generation
# ============================================================================

class VariableSpacedDataset(Dataset):
    """Dataset with variable time spacing between positions."""
    def __init__(self, num_samples: int, seq_len: int,
                 velocity_range: Tuple[float, float] = (0.5, 4.5),
                 dt_mean: float = 1.0, dt_std: float = 0.3,
                 dt_min: float = 0.2, dt_max: float = 3.0):

        print(f"  Generating {num_samples} samples...", flush=True)

        # Pre-generate all data as tensors for efficiency
        velocities = np.random.uniform(velocity_range[0], velocity_range[1], num_samples)
        angles = np.random.uniform(0, 2 * np.pi, num_samples)

        vx = velocities * np.cos(angles)
        vy = velocities * np.sin(angles)

        # Generate all dt values at once
        dt_values = np.clip(
            np.random.normal(dt_mean, dt_std, (num_samples, seq_len)),
            dt_min, dt_max
        ).astype(np.float32)
        dt_values[:, 0] = 0.0  # First position has no previous

        # Cumulative time
        times = np.cumsum(dt_values, axis=1)

        # Positions
        x = vx[:, None] * times
        y = vy[:, None] * times
        positions = np.stack([x, y], axis=-1).astype(np.float32)

        self.positions = torch.from_numpy(positions)
        self.dt_values = torch.from_numpy(dt_values)

        print(f"  Done generating samples", flush=True)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.dt_values[idx]


# ============================================================================
# Part 2: Model Architecture
# ============================================================================

class SinusoidalTimeEncoding(nn.Module):
    """Encode time values using sinusoidal functions."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: any shape -> same shape + (d_model,)"""
        t = t.unsqueeze(-1)
        angles = t * self.div_term
        encoding = torch.zeros(*t.shape[:-1], self.d_model, device=t.device, dtype=t.dtype)
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles)
        return encoding


class FourierHead2D(nn.Module):
    """2D Fourier density head - optimized."""
    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 8, grid_range: float = 5.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        num_freq_pairs = (2 * num_freqs + 1) ** 2
        self.coeff_predictor = nn.Linear(d_model, 2 * num_freq_pairs)

        # Precompute and register all buffers
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
        self.register_buffer('cos_basis', torch.cos(phase).squeeze(0))  # (grid^2, num_freqs)
        self.register_buffer('sin_basis', torch.sin(phase).squeeze(0))

        # Use small init, NOT zeros (zeros blocks gradient flow to input)
        nn.init.normal_(self.coeff_predictor.weight, std=0.01)
        nn.init.zeros_(self.coeff_predictor.bias)

    def forward(self, z):
        """z: (batch, d_model) -> (batch, grid_size, grid_size)"""
        batch_size = z.shape[0]
        coeffs = self.coeff_predictor(z)

        num_freq_pairs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_freq_pairs]
        sin_coeffs = coeffs[:, num_freq_pairs:]

        # Efficient matmul instead of einsum
        logits = cos_coeffs @ self.cos_basis.T + sin_coeffs @ self.sin_basis.T

        log_density = F.log_softmax(logits, dim=-1)
        return log_density.view(batch_size, self.grid_size, self.grid_size)


class CausalMultiHorizonModel(nn.Module):
    """Causal decoder with multi-horizon Fourier prediction."""
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 max_horizon: int = 5, grid_size: int = 64, num_freqs: int = 8,
                 grid_range: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.max_horizon = max_horizon
        self.grid_size = grid_size
        self.grid_range = grid_range

        # Input: (x, y, dt) -> d_model
        self.input_proj = nn.Linear(3, d_model)

        # Positional encoding
        max_len = 200
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Time encoding for horizon conditioning
        self.time_encoder = SinusoidalTimeEncoding(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Horizon conditioning
        self.horizon_proj = nn.Linear(d_model * 2, d_model)

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

        # Precompute causal masks for different sequence lengths
        self._causal_masks = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device):
        if seq_len not in self._causal_masks:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self._causal_masks[seq_len] = mask
        return self._causal_masks[seq_len].to(device)

    def forward(self, positions: torch.Tensor, dt_values: torch.Tensor,
                query_dt: Optional[torch.Tensor] = None):
        """
        Forward for inference with specific query time.

        Args:
            positions: (batch, seq_len, 2)
            dt_values: (batch, seq_len)
            query_dt: (batch,) time horizon to predict
        """
        batch_size, seq_len, _ = positions.shape

        # Build input: (x, y, dt) - absolute coordinates
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x + self.pe[:, :seq_len]

        # Causal transformer
        mask = self._get_causal_mask(seq_len, x.device)
        x = self.transformer(x, mask=mask)

        # Get last embedding
        last_emb = x[:, -1, :]  # (batch, d_model)

        if query_dt is None:
            return last_emb

        # Condition on query time
        time_enc = self.time_encoder(query_dt)  # (batch, d_model)
        combined = torch.cat([last_emb, time_enc], dim=-1)
        conditioned = self.horizon_proj(combined)

        return self.fourier_head(conditioned)

    def forward_train(self, positions: torch.Tensor, dt_values: torch.Tensor,
                      max_horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient forward for training - vectorized multi-horizon.

        Returns:
            log_densities: (batch, num_valid_pairs, grid_size, grid_size)
            targets_relative: (batch, num_valid_pairs, 2)
        """
        batch_size, seq_len, _ = positions.shape
        device = positions.device

        # Build input and run transformer
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = x + self.pe[:, :seq_len]

        mask = self._get_causal_mask(seq_len, device)
        embeddings = self.transformer(x, mask=mask)  # (batch, seq_len, d_model)

        # Compute cumulative dt using cumsum (vectorized)
        # cumsum_dt[i] = sum of dt[0:i+1]
        cumsum_dt = torch.cumsum(dt_values, dim=1)  # (batch, seq_len)

        # Build all (position, horizon) pairs efficiently
        # For position i, we predict i+1, i+2, ..., min(i+max_horizon, seq_len-1)

        all_embeddings = []
        all_cum_dt = []
        all_targets = []

        for h in range(1, max_horizon + 1):
            # Positions that can predict horizon h
            valid_positions = seq_len - h  # positions 0 to seq_len-h-1
            if valid_positions <= 0:
                continue

            # Source embeddings: positions 0 to valid_positions-1
            src_emb = embeddings[:, :valid_positions, :]  # (batch, valid_pos, d_model)

            # Cumulative dt from source to target
            # For source i, target is i+h, cum_dt = cumsum[i+h] - cumsum[i]
            src_cumsum = cumsum_dt[:, :valid_positions]  # (batch, valid_pos)
            tgt_cumsum = cumsum_dt[:, h:h+valid_positions]  # (batch, valid_pos)
            cum_dt = tgt_cumsum - src_cumsum  # (batch, valid_pos)

            # Target positions relative to source
            src_pos = positions[:, :valid_positions, :]  # (batch, valid_pos, 2)
            tgt_pos = positions[:, h:h+valid_positions, :]  # (batch, valid_pos, 2)
            tgt_relative = tgt_pos - src_pos  # (batch, valid_pos, 2)

            all_embeddings.append(src_emb)
            all_cum_dt.append(cum_dt)
            all_targets.append(tgt_relative)

        # Concatenate all pairs
        all_embeddings = torch.cat(all_embeddings, dim=1)  # (batch, total_pairs, d_model)
        all_cum_dt = torch.cat(all_cum_dt, dim=1)  # (batch, total_pairs)
        all_targets = torch.cat(all_targets, dim=1)  # (batch, total_pairs, 2)

        num_pairs = all_embeddings.shape[1]

        # Encode cumulative dt
        time_enc = self.time_encoder(all_cum_dt)  # (batch, total_pairs, d_model)

        # Combine and project
        combined = torch.cat([all_embeddings, time_enc], dim=-1)  # (batch, total_pairs, d_model*2)
        conditioned = self.horizon_proj(combined)  # (batch, total_pairs, d_model)

        # Fourier head (reshape for batch processing)
        conditioned_flat = conditioned.view(batch_size * num_pairs, -1)
        log_densities_flat = self.fourier_head(conditioned_flat)
        log_densities = log_densities_flat.view(batch_size, num_pairs, self.grid_size, self.grid_size)

        return log_densities, all_targets


# ============================================================================
# Part 3: Loss Function
# ============================================================================

def compute_soft_target_loss(log_density, target, grid_range, grid_size, sigma=0.5):
    """
    Vectorized soft target loss.

    Args:
        log_density: (batch, num_pairs, grid_size, grid_size)
        target: (batch, num_pairs, 2)
    """
    batch_size, num_pairs, gs, _ = log_density.shape
    device = log_density.device

    # Create grid once
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')  # (gs, gs)

    # Reshape for broadcasting
    # target: (batch, num_pairs, 2) -> (batch, num_pairs, 1, 1)
    target_x = target[:, :, 0:1, None]  # (batch, num_pairs, 1, 1)
    target_y = target[:, :, 1:2, None]

    # Compute distances: (batch, num_pairs, gs, gs)
    dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2

    # Soft target
    soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    # KL divergence
    loss = F.kl_div(log_density, soft_target, reduction='none').sum(dim=(-2, -1))  # (batch, num_pairs)

    return loss.mean()


# ============================================================================
# Part 4: Training
# ============================================================================

def train_model(model, train_loader, val_loader, config):
    """Train with proper logging."""
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']
    )

    num_batches = len(train_loader)
    total_steps = config['num_epochs'] * num_batches

    print(f"  Total batches per epoch: {num_batches}", flush=True)
    print(f"  Total training steps: {total_steps}", flush=True)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, (positions, dt_values) in enumerate(train_loader):
            positions = positions.to(DEVICE)
            dt_values = dt_values.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            log_densities, targets = model.forward_train(
                positions, dt_values, config['max_horizon']
            )

            # Loss
            loss = compute_soft_target_loss(
                log_densities, targets,
                config['grid_range'], config['grid_size'], config['sigma']
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Progress update every 20 batches
            if (batch_idx + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{num_batches}, "
                      f"Loss: {loss.item():.4f}", flush=True)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for positions, dt_values in val_loader:
                positions = positions.to(DEVICE)
                dt_values = dt_values.to(DEVICE)

                log_densities, targets = model.forward_train(
                    positions, dt_values, config['max_horizon']
                )
                loss = compute_soft_target_loss(
                    log_densities, targets,
                    config['grid_range'], config['grid_size'], config['sigma']
                )
                val_loss += loss.item()

        avg_train = epoch_loss / num_batches
        avg_val = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        print(f"  Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train={avg_train:.4f}, Val={avg_val:.4f}, "
              f"Time={epoch_time:.1f}s", flush=True)

    return history


# ============================================================================
# Part 5: Evaluation
# ============================================================================

def evaluate_model(model, val_loader, config):
    """Evaluate prediction errors by horizon."""
    model.eval()

    horizon_errors = {h: [] for h in range(1, config['max_horizon'] + 1)}

    grid_x = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)
    grid_y = grid_x.clone()

    with torch.no_grad():
        for positions, dt_values in val_loader:
            positions = positions.to(DEVICE)
            dt_values = dt_values.to(DEVICE)

            batch_size, seq_len, _ = positions.shape

            # Get embeddings
            x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
            x = model.input_proj(x)
            x = x + model.pe[:, :seq_len]
            mask = model._get_causal_mask(seq_len, DEVICE)
            embeddings = model.transformer(x, mask=mask)

            cumsum_dt = torch.cumsum(dt_values, dim=1)

            for h in range(1, config['max_horizon'] + 1):
                valid_pos = seq_len - h
                if valid_pos <= 0:
                    continue

                # Get predictions for this horizon
                src_emb = embeddings[:, :valid_pos, :]
                src_cumsum = cumsum_dt[:, :valid_pos]
                tgt_cumsum = cumsum_dt[:, h:h+valid_pos]
                cum_dt = tgt_cumsum - src_cumsum

                time_enc = model.time_encoder(cum_dt)
                combined = torch.cat([src_emb, time_enc], dim=-1)
                conditioned = model.horizon_proj(combined)

                # Get densities
                cond_flat = conditioned.view(-1, model.d_model)
                log_dens = model.fourier_head(cond_flat)
                densities = torch.exp(log_dens)  # (batch*valid_pos, gs, gs)

                # Expected value
                marginal_x = densities.sum(dim=2)  # (batch*valid_pos, gs)
                marginal_y = densities.sum(dim=1)
                exp_x = (marginal_x * grid_x).sum(dim=1)
                exp_y = (marginal_y * grid_y).sum(dim=1)

                # Target
                src_pos = positions[:, :valid_pos, :].reshape(-1, 2)
                tgt_pos = positions[:, h:h+valid_pos, :].reshape(-1, 2)
                tgt_rel = tgt_pos - src_pos

                # Error
                error = torch.sqrt((exp_x - tgt_rel[:, 0])**2 + (exp_y - tgt_rel[:, 1])**2)
                horizon_errors[h].extend(error.cpu().numpy().tolist())

    metrics = {h: np.mean(errs) for h, errs in horizon_errors.items() if errs}
    return metrics


# ============================================================================
# Part 6: Visualization
# ============================================================================

def visualize_predictions(model, val_loader, config, save_path):
    """Visualize predictions from positions with sufficient history."""
    model.eval()

    positions, dt_values = next(iter(val_loader))
    positions = positions[:1].to(DEVICE)
    dt_values = dt_values[:1].to(DEVICE)

    batch_size, seq_len, _ = positions.shape
    gr = config['grid_range']
    grid = torch.linspace(-gr, gr, config['grid_size'], device=DEVICE)

    # Get transformer embeddings
    with torch.no_grad():
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = model.input_proj(x)
        x = x + model.pe[:, :seq_len]
        mask = model._get_causal_mask(seq_len, DEVICE)
        embeddings = model.transformer(x, mask=mask)

        cumsum_dt = torch.cumsum(dt_values, dim=1)

    # Show predictions from different source positions (with enough history)
    # Start from position 5 so model has seen some trajectory
    source_positions = [5, 7, 9, 11]
    horizon = 3  # Fixed horizon for visualization

    n_show = len(source_positions)
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))

    positions_np = positions[0].cpu().numpy()

    for col, src_idx in enumerate(source_positions):
        tgt_idx = src_idx + horizon
        if tgt_idx >= seq_len:
            continue

        # Get prediction
        with torch.no_grad():
            src_emb = embeddings[0, src_idx:src_idx+1, :]  # (1, d_model)
            cum_dt = cumsum_dt[0, tgt_idx] - cumsum_dt[0, src_idx]
            time_enc = model.time_encoder(cum_dt.unsqueeze(0))
            combined = torch.cat([src_emb, time_enc], dim=-1)
            conditioned = model.horizon_proj(combined)
            log_dens = model.fourier_head(conditioned)
            dens = torch.exp(log_dens[0]).cpu().numpy()

            # Expected value
            dens_t = torch.exp(log_dens[0])
            exp_x = (dens_t.sum(dim=1) * grid).sum().item()
            exp_y = (dens_t.sum(dim=0) * grid).sum().item()

        # Target (relative displacement)
        target_rel = positions_np[tgt_idx] - positions_np[src_idx]

        # Top: trajectory with source and target marked
        ax = axes[0, col]
        ax.plot(positions_np[:, 0], positions_np[:, 1], 'b.-', alpha=0.5, label='track')
        ax.scatter([positions_np[src_idx, 0]], [positions_np[src_idx, 1]],
                   c='green', s=150, marker='o', zorder=5, label='source')
        ax.scatter([positions_np[tgt_idx, 0]], [positions_np[tgt_idx, 1]],
                   c='red', s=150, marker='x', zorder=5, label='target')
        ax.set_xlim(-gr, gr)
        ax.set_ylim(-gr, gr)
        ax.set_aspect('equal')
        ax.set_title(f'src={src_idx}, h={horizon}')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

        # Bottom: density with target and prediction marked
        ax = axes[1, col]
        im = ax.imshow(dens.T, origin='lower', extent=[-gr, gr, -gr, gr], cmap='hot')
        ax.scatter([target_rel[0]], [target_rel[1]], c='cyan', s=150, marker='x',
                   zorder=5, label=f'target ({target_rel[0]:.1f},{target_rel[1]:.1f})')
        ax.scatter([exp_x], [exp_y], c='lime', s=150, marker='+',
                   zorder=5, label=f'pred ({exp_x:.1f},{exp_y:.1f})')
        ax.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax)

        error = np.sqrt((exp_x - target_rel[0])**2 + (exp_y - target_rel[1])**2)
        ax.set_title(f'error={error:.2f}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_history(history, save_path):
    """Plot training history."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


# ============================================================================
# Part 7: Main
# ============================================================================

def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 8: CAUSAL MULTI-HORIZON PREDICTION", flush=True)
    print("=" * 70, flush=True)

    # Calculate required grid range:
    # max_velocity = 4.5, max_dt = 3.0, max_horizon = 5
    # max cumulative dt ~ 5 * 3.0 = 15
    # max displacement ~ 4.5 * 15 = 67.5
    # Use grid_range = 40 to cover most cases with some margin
    # sigma should cover 1-2 grid cells: grid_cell = 80/64 = 1.25, so sigma ~ 1.5

    config = {
        'num_train': 5000,
        'num_val': 500,
        'seq_len': 15,
        'velocity_range': (0.5, 4.5),
        'dt_mean': 1.0,
        'dt_std': 0.3,
        'dt_min': 0.2,
        'dt_max': 3.0,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'max_horizon': 5,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 40.0,  # Must cover max displacement!
        'batch_size': 64,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'num_epochs': 30,
        'sigma': 1.5,  # ~1-2 grid cells
    }

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    print("\nConfig:", flush=True)
    for k, v in config.items():
        print(f"  {k}: {v}", flush=True)

    # Data
    print("\n" + "=" * 70, flush=True)
    print("GENERATING DATA", flush=True)
    print("=" * 70, flush=True)

    train_dataset = VariableSpacedDataset(
        config['num_train'], config['seq_len'], config['velocity_range'],
        config['dt_mean'], config['dt_std'], config['dt_min'], config['dt_max']
    )
    val_dataset = VariableSpacedDataset(
        config['num_val'], config['seq_len'], config['velocity_range'],
        config['dt_mean'], config['dt_std'], config['dt_min'], config['dt_max']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    print(f"  Batches per epoch: {len(train_loader)}", flush=True)

    # Model
    print("\n" + "=" * 70, flush=True)
    print("CREATING MODEL", flush=True)
    print("=" * 70, flush=True)

    model = CausalMultiHorizonModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_horizon=config['max_horizon'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}", flush=True)

    # Train
    print("\n" + "=" * 70, flush=True)
    print("TRAINING", flush=True)
    print("=" * 70, flush=True)

    history = train_model(model, train_loader, val_loader, config)

    # Evaluate
    print("\n" + "=" * 70, flush=True)
    print("EVALUATION", flush=True)
    print("=" * 70, flush=True)

    metrics = evaluate_model(model, val_loader, config)
    print("  Prediction errors by horizon:", flush=True)
    for h, err in sorted(metrics.items()):
        print(f"    Horizon {h}: {err:.4f}", flush=True)

    # Visualize
    print("\n" + "=" * 70, flush=True)
    print("SAVING RESULTS", flush=True)
    print("=" * 70, flush=True)

    visualize_predictions(model, val_loader, config, output_dir / 'predictions.png')
    plot_history(history, output_dir / 'training_history.png')

    # Save results
    results = {
        'config': config,
        'metrics': metrics,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir}", flush=True)
    print("=" * 70, flush=True)
    print("DONE", flush=True)
    print("=" * 70, flush=True)

    return results


if __name__ == '__main__':
    main()
