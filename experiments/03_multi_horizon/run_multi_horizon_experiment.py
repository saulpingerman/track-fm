#!/usr/bin/env python3
"""
Multi-Horizon Prediction Experiment

Tests whether we can:
1. Pass trajectory through transformer ONCE to get embedding
2. Duplicate embedding N times
3. Concatenate each with different time horizons
4. Predict N densities simultaneously

This is more efficient than running transformer N times.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import json
from pathlib import Path

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# Data Generation for Multi-Horizon
# ============================================================================

class MultiHorizonDataset(Dataset):
    """Dataset that provides multiple prediction horizons per trajectory."""

    def __init__(self, num_samples: int, seq_len: int, num_horizons: int = 5,
                 velocity_range: Tuple[float, float] = (0.1, 4.5),
                 dt_range: Tuple[float, float] = (0.5, 4.0),
                 noise_std: float = 0.0):
        self.samples = []
        self.num_horizons = num_horizons

        for _ in range(num_samples):
            velocity = np.random.uniform(*velocity_range)
            angle = np.random.uniform(0, 2 * np.pi)
            vx = velocity * np.cos(angle)
            vy = velocity * np.sin(angle)

            # Generate input trajectory
            times = np.arange(seq_len)
            positions = np.stack([vx * times, vy * times], axis=-1)

            # Relative to last position
            last_pos = positions[-1]
            input_seq = positions - last_pos

            # Generate N different time horizons
            dts = np.sort(np.random.uniform(*dt_range, size=num_horizons))

            # Targets for each horizon: displacement = velocity * dt
            targets = np.stack([
                np.array([vx * dt, vy * dt]) for dt in dts
            ], axis=0)  # (num_horizons, 2)

            if noise_std > 0:
                input_seq += np.random.normal(0, noise_std, input_seq.shape)

            self.samples.append((
                input_seq.astype(np.float32),
                dts.astype(np.float32),
                targets.astype(np.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Model Components
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FourierHead2D(nn.Module):
    """2D Fourier head for density prediction."""

    def __init__(self, d_input: int, grid_size: int = 64,
                 num_freqs: int = 8, grid_range: float = 20.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        num_freq_pairs = (2 * num_freqs + 1) ** 2
        self.coeff_predictor = nn.Linear(d_input, 2 * num_freq_pairs)

        # Precompute grid
        x = torch.linspace(-grid_range, grid_range, grid_size)
        y = torch.linspace(-grid_range, grid_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', xx.flatten())
        self.register_buffer('grid_y', yy.flatten())

        # Precompute frequencies
        freqs = torch.arange(-num_freqs, num_freqs + 1, dtype=torch.float)
        freq_x, freq_y = torch.meshgrid(freqs, freqs, indexing='ij')
        self.register_buffer('freq_x', freq_x.flatten())
        self.register_buffer('freq_y', freq_y.flatten())

        self._init_uniform()

    def _init_uniform(self):
        nn.init.zeros_(self.coeff_predictor.weight)
        nn.init.zeros_(self.coeff_predictor.bias)

    def forward(self, z):
        batch_size = z.shape[0]

        coeffs = self.coeff_predictor(z)
        num_freq_pairs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_freq_pairs]
        sin_coeffs = coeffs[:, num_freq_pairs:]

        L = 2 * self.grid_range
        phase = (2 * np.pi / L) * (
            self.freq_x.unsqueeze(0) * self.grid_x.unsqueeze(1) +
            self.freq_y.unsqueeze(0) * self.grid_y.unsqueeze(1)
        )

        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        density = torch.einsum('bf,gf->bg', cos_coeffs, cos_basis) + \
                  torch.einsum('bf,gf->bg', sin_coeffs, sin_basis)

        density = density.view(batch_size, self.grid_size, self.grid_size)
        density = F.softmax(density.view(batch_size, -1), dim=-1)
        density = density.view(batch_size, self.grid_size, self.grid_size)

        return density


class MultiHorizonModel(nn.Module):
    """
    Multi-horizon prediction model:
    1. Transformer encodes trajectory ONCE
    2. Embedding is duplicated for each horizon
    3. Each copy concatenated with its Δt
    4. MLP + Fourier head predicts density for each horizon
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=20.0,
                 dt_embed_dim=16):
        super().__init__()
        self.d_model = d_model
        self.dt_embed_dim = dt_embed_dim

        # Trajectory encoder (run ONCE)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Δt embedding
        self.dt_embed = nn.Sequential(
            nn.Linear(1, dt_embed_dim),
            nn.ReLU(),
            nn.Linear(dt_embed_dim, dt_embed_dim)
        )

        # Fusion MLP: combines embedding + Δt
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model + dt_embed_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def encode_trajectory(self, x):
        """Encode trajectory ONCE, return embedding."""
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]  # Last token embedding
        return z

    def predict_horizon(self, embedding, dt):
        """Predict density for a single horizon given pre-computed embedding."""
        dt_emb = self.dt_embed(dt)
        fused = self.fusion_mlp(torch.cat([embedding, dt_emb], dim=-1))
        density = self.fourier_head(fused)
        return density

    def forward(self, x, dts):
        """
        Forward pass for multiple horizons.

        Args:
            x: (batch, seq_len, 2) trajectory
            dts: (batch, num_horizons) time horizons

        Returns:
            densities: (batch, num_horizons, grid_size, grid_size)
        """
        batch_size, num_horizons = dts.shape

        # Encode trajectory ONCE
        embedding = self.encode_trajectory(x)  # (batch, d_model)

        # Predict for each horizon
        densities = []
        for h in range(num_horizons):
            dt_h = dts[:, h:h+1]  # (batch, 1)
            density_h = self.predict_horizon(embedding, dt_h)
            densities.append(density_h)

        densities = torch.stack(densities, dim=1)  # (batch, num_horizons, grid, grid)
        return densities

    def forward_batched(self, x, dts):
        """
        More efficient batched forward for all horizons at once.
        """
        batch_size, num_horizons = dts.shape

        # Encode trajectory ONCE
        embedding = self.encode_trajectory(x)  # (batch, d_model)

        # Expand embedding for all horizons
        embedding_expanded = embedding.unsqueeze(1).expand(-1, num_horizons, -1)
        embedding_flat = embedding_expanded.reshape(batch_size * num_horizons, -1)

        # Flatten dts
        dts_flat = dts.reshape(batch_size * num_horizons, 1)

        # Embed all dts
        dt_emb = self.dt_embed(dts_flat)

        # Fuse and predict
        fused = self.fusion_mlp(torch.cat([embedding_flat, dt_emb], dim=-1))
        densities_flat = self.fourier_head(fused)

        # Reshape back
        densities = densities_flat.view(batch_size, num_horizons, self.grid_size, self.grid_size)
        return densities


class SingleHorizonBaseline(nn.Module):
    """
    Baseline: Run full model for EACH horizon separately.
    Less efficient but simpler.
    """

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=20.0):
        super().__init__()
        self.d_model = d_model

        # Full model per prediction (inefficient)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dt_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model + 16, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dts):
        """Run transformer for EACH horizon (inefficient baseline)."""
        batch_size, num_horizons = dts.shape
        densities = []

        for h in range(num_horizons):
            # Run full transformer each time
            z = self.input_proj(x)
            z = self.pos_encoding(z)
            z = self.transformer(z)
            z = z[:, -1, :]

            dt_h = dts[:, h:h+1]
            dt_emb = self.dt_embed(dt_h)
            fused = self.fusion_mlp(torch.cat([z, dt_emb], dim=-1))
            density = self.fourier_head(fused)
            densities.append(density)

        return torch.stack(densities, dim=1)


class RegressionBaseline(nn.Module):
    """Direct (x, y) regression for each horizon."""

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dt_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model + 16, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, x, dts):
        """Predict (x, y) for each horizon."""
        batch_size, num_horizons = dts.shape

        # Encode once
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]

        predictions = []
        for h in range(num_horizons):
            dt_h = dts[:, h:h+1]
            dt_emb = self.dt_embed(dt_h)
            pred = self.output_head(torch.cat([z, dt_emb], dim=-1))
            predictions.append(pred)

        return torch.stack(predictions, dim=1)  # (batch, num_horizons, 2)


# ============================================================================
# Training
# ============================================================================

def create_soft_target(target_xy, grid_size, grid_range, sigma=0.5):
    """Create Gaussian soft target centered at target position."""
    device = target_xy.device
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    target_x = target_xy[..., 0:1, None]
    target_y = target_xy[..., 1:2, None]

    dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    return soft_target


def train_fourier_model(model, train_loader, val_loader, config, use_batched=True):
    """Train a Fourier-based model."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    grid_size = config['grid_size']
    grid_range = config['grid_range']

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            x, dts, targets = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()

            if use_batched and hasattr(model, 'forward_batched'):
                densities = model.forward_batched(x, dts)
            else:
                densities = model(x, dts)

            # Create soft targets for all horizons
            soft_targets = create_soft_target(targets, grid_size, grid_range)

            # NLL loss
            log_densities = torch.log(densities + 1e-10)
            loss = -(soft_targets * log_densities).sum(dim=(-2, -1)).mean()

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % 5 == 0:
            val_error = evaluate_fourier_model(model, val_loader, config, use_batched)
            print(f"  Epoch {epoch+1}/{config['num_epochs']}: Loss={avg_loss:.4f}, Val Error={val_error:.4f}")

    return model


def train_regression_model(model, train_loader, val_loader, config):
    """Train regression baseline."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            x, dts, targets = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            predictions = model(x, dts)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 5 == 0:
            val_error = evaluate_regression_model(model, val_loader)
            print(f"  Epoch {epoch+1}/{config['num_epochs']}: Loss={total_loss/num_batches:.4f}, Val Error={val_error:.4f}")

    return model


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_fourier_model(model, val_loader, config, use_batched=True):
    """Evaluate Fourier model using expected position error."""
    model.eval()
    grid_size = config['grid_size']
    grid_range = config['grid_range']

    x_coords = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    y_coords = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            x, dts, targets = [b.to(DEVICE) for b in batch]
            batch_size, num_horizons = dts.shape

            if use_batched and hasattr(model, 'forward_batched'):
                densities = model.forward_batched(x, dts)
            else:
                densities = model(x, dts)

            # Expected position for each horizon
            for h in range(num_horizons):
                density_h = densities[:, h]
                exp_x = (density_h * xx).sum(dim=(-2, -1))
                exp_y = (density_h * yy).sum(dim=(-2, -1))

                target_h = targets[:, h]
                error = torch.sqrt((exp_x - target_h[:, 0])**2 + (exp_y - target_h[:, 1])**2)
                total_error += error.sum().item()

            total_samples += batch_size * num_horizons

    return total_error / total_samples


def evaluate_regression_model(model, val_loader):
    """Evaluate regression model."""
    model.eval()
    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            x, dts, targets = [b.to(DEVICE) for b in batch]
            predictions = model(x, dts)

            error = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1))
            total_error += error.sum().item()
            total_samples += error.numel()

    return total_error / total_samples


def measure_inference_time(model, x, dts, num_runs=100, use_batched=True):
    """Measure average inference time."""
    import time

    model.eval()
    x = x.to(DEVICE)
    dts = dts.to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if use_batched and hasattr(model, 'forward_batched'):
                _ = model.forward_batched(x, dts)
            else:
                _ = model(x, dts)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            if use_batched and hasattr(model, 'forward_batched'):
                _ = model.forward_batched(x, dts)
            else:
                _ = model(x, dts)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    return (end - start) / num_runs * 1000  # ms


# ============================================================================
# Visualization
# ============================================================================

def find_good_sample(val_loader, min_velocity=3.5):
    """Find a sample with high velocity for better visualization."""
    for batch in val_loader:
        x, dts, targets = batch
        for i in range(x.shape[0]):
            traj = x[i].numpy()
            displacement = np.sqrt((traj[-1, 0] - traj[0, 0])**2 + (traj[-1, 1] - traj[0, 1])**2)
            velocity_est = displacement / (len(traj) - 1)
            if velocity_est >= min_velocity:
                return x[i:i+1], dts[i:i+1], targets[i:i+1]
    x, dts, targets = next(iter(val_loader))
    return x[:1], dts[:1], targets[:1]


def visualize_multi_horizon(model, val_loader, config, save_path, use_batched=True):
    """Simple, clean visualization of multi-horizon prediction."""
    model.eval()
    grid_range = config['grid_range']
    num_horizons = config['num_horizons']
    grid_size = config['grid_size']

    # Find a fast-moving sample
    x, dts, targets = find_good_sample(val_loader, min_velocity=3.5)
    x_np = x[0].numpy()
    x = x.to(DEVICE)
    dts = dts.to(DEVICE)

    with torch.no_grad():
        if use_batched and hasattr(model, 'forward_batched'):
            densities = model.forward_batched(x, dts)
        else:
            densities = model(x, dts)

    densities = densities[0].cpu().numpy()
    targets_np = targets[0].numpy()
    dts_np = dts[0].cpu().numpy()

    # Compute expected positions
    x_coords = np.linspace(-grid_range, grid_range, grid_size)
    y_coords = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    fig, ax = plt.subplots(figsize=(10, 10))

    # Input trajectory - thick blue line
    ax.plot(x_np[:, 0], x_np[:, 1], 'b-', linewidth=3, solid_capstyle='round')
    ax.scatter(x_np[:-1, 0], x_np[:-1, 1], c='blue', s=60, zorder=3)
    ax.scatter(x_np[-1, 0], x_np[-1, 1], c='blue', s=150, marker='s',
               edgecolors='white', linewidths=2, zorder=5, label='Last observed')

    # Predictions for each horizon
    for h in range(num_horizons):
        target_x, target_y = targets_np[h]
        exp_x = (densities[h] * xx).sum()
        exp_y = (densities[h] * yy).sum()

        # Target: green X
        ax.scatter(target_x, target_y, c='green', s=200, marker='x',
                   linewidths=4, zorder=10)

        # Prediction: red circle
        ax.scatter(exp_x, exp_y, c='red', s=120, marker='o',
                   edgecolors='white', linewidths=2, zorder=9)

        # Label
        ax.annotate(f't+{dts_np[h]:.1f}', (target_x, target_y),
                    fontsize=11, fontweight='bold', ha='left', va='bottom',
                    xytext=(6, 6), textcoords='offset points')

    # Clean bounds
    all_x = np.concatenate([x_np[:, 0], targets_np[:, 0]])
    all_y = np.concatenate([x_np[:, 1], targets_np[:, 1]])
    margin = 3
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=3, marker='o', markersize=8, label='Input trajectory'),
        Line2D([0], [0], color='w', marker='s', markersize=12, markerfacecolor='blue',
               markeredgecolor='white', markeredgewidth=2, linestyle='', label='Last position'),
        Line2D([0], [0], color='w', marker='x', markersize=12, markerfacecolor='green',
               markeredgecolor='green', markeredgewidth=4, linestyle='', label='Target'),
        Line2D([0], [0], color='w', marker='o', markersize=10, markerfacecolor='red',
               markeredgecolor='white', markeredgewidth=2, linestyle='', label='Predicted'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_trajectory_with_horizons(model, val_loader, config, save_path, use_batched=True):
    """Show density heatmaps for each horizon."""
    model.eval()
    grid_range = config['grid_range']
    num_horizons = config['num_horizons']
    grid_size = config['grid_size']

    x, dts, targets = find_good_sample(val_loader, min_velocity=3.5)
    x_np = x[0].numpy()
    x = x.to(DEVICE)
    dts = dts.to(DEVICE)

    with torch.no_grad():
        if use_batched and hasattr(model, 'forward_batched'):
            densities = model.forward_batched(x, dts)
        else:
            densities = model(x, dts)

    densities = densities[0].cpu().numpy()
    targets_np = targets[0].numpy()
    dts_np = dts[0].cpu().numpy()

    # Compute bounds for zoom
    all_x = np.concatenate([x_np[:, 0], targets_np[:, 0]])
    all_y = np.concatenate([x_np[:, 1], targets_np[:, 1]])
    margin = 3
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    fig, axes = plt.subplots(1, num_horizons, figsize=(3.5 * num_horizons, 4))

    for h in range(num_horizons):
        ax = axes[h]
        target_x, target_y = targets_np[h]

        # Density heatmap
        ax.imshow(densities[h].T, origin='lower',
                  extent=[-grid_range, grid_range, -grid_range, grid_range],
                  cmap='hot', aspect='equal')

        # Input trajectory
        ax.plot(x_np[:, 0], x_np[:, 1], 'c-', linewidth=2)
        ax.scatter(x_np[-1, 0], x_np[-1, 1], c='cyan', s=80, marker='s',
                   edgecolors='white', linewidths=1.5, zorder=5)

        # Target
        ax.scatter(target_x, target_y, c='lime', s=150, marker='x', linewidths=3, zorder=10)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f't + {dts_np[h]:.1f}s', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("MULTI-HORIZON PREDICTION EXPERIMENT")
    print("=" * 80)

    # Configuration
    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'num_horizons': 5,
        'velocity_range': (0.1, 4.5),
        'dt_range': (0.5, 4.0),
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 20.0,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'num_epochs': 30,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MultiHorizonDataset(
        config['num_train'], config['seq_len'], config['num_horizons'],
        config['velocity_range'], config['dt_range']
    )
    val_dataset = MultiHorizonDataset(
        config['num_val'], config['seq_len'], config['num_horizons'],
        config['velocity_range'], config['dt_range']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    results = {}

    # =========================================================================
    # Method 1: Multi-Horizon Model (encode once, predict all)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: Multi-Horizon Model (encode trajectory ONCE)")
    print("=" * 80)

    model_multi = MultiHorizonModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    print(f"Parameters: {sum(p.numel() for p in model_multi.parameters()):,}")
    model_multi = train_fourier_model(model_multi, train_loader, val_loader, config, use_batched=True)

    error_multi = evaluate_fourier_model(model_multi, val_loader, config, use_batched=True)
    print(f"\nFinal Error: {error_multi:.4f}")

    # Measure inference time
    sample_x = torch.randn(1, config['seq_len'], 2)
    sample_dts = torch.sort(torch.rand(1, config['num_horizons']) * 3.5 + 0.5)[0]
    time_multi = measure_inference_time(model_multi, sample_x, sample_dts, use_batched=True)
    print(f"Inference time (batched): {time_multi:.2f} ms")

    results['multi_horizon'] = {
        'error': error_multi,
        'inference_time_ms': time_multi,
        'description': 'Encode trajectory once, predict all horizons'
    }

    # Visualize
    visualize_multi_horizon(model_multi, val_loader, config,
                           results_dir / "multi_horizon_predictions.png")
    visualize_trajectory_with_horizons(model_multi, val_loader, config,
                                       results_dir / "multi_horizon_trajectory.png")

    # =========================================================================
    # Method 2: Single-Horizon Baseline (encode N times)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: Single-Horizon Baseline (encode trajectory N times)")
    print("=" * 80)

    model_single = SingleHorizonBaseline(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    print(f"Parameters: {sum(p.numel() for p in model_single.parameters()):,}")
    model_single = train_fourier_model(model_single, train_loader, val_loader, config, use_batched=False)

    error_single = evaluate_fourier_model(model_single, val_loader, config, use_batched=False)
    print(f"\nFinal Error: {error_single:.4f}")

    time_single = measure_inference_time(model_single, sample_x, sample_dts, use_batched=False)
    print(f"Inference time: {time_single:.2f} ms")

    results['single_horizon'] = {
        'error': error_single,
        'inference_time_ms': time_single,
        'description': 'Encode trajectory N times (once per horizon)'
    }

    visualize_multi_horizon(model_single, val_loader, config,
                           results_dir / "single_horizon_predictions.png")

    # =========================================================================
    # Method 3: Regression Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 3: Regression Baseline")
    print("=" * 80)

    model_reg = RegressionBaseline(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )

    print(f"Parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
    model_reg = train_regression_model(model_reg, train_loader, val_loader, config)

    error_reg = evaluate_regression_model(model_reg, val_loader)
    print(f"\nFinal Error: {error_reg:.4f}")

    results['regression'] = {
        'error': error_reg,
        'description': 'Direct (x, y) regression'
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n| Method | Error | Inference Time |")
    print("|--------|-------|----------------|")
    print(f"| Multi-Horizon (encode once) | {error_multi:.4f} | {time_multi:.2f} ms |")
    print(f"| Single-Horizon (encode N times) | {error_single:.4f} | {time_single:.2f} ms |")
    print(f"| Regression Baseline | {error_reg:.4f} | - |")

    speedup = time_single / time_multi
    print(f"\nSpeedup from encoding once: {speedup:.1f}x faster")

    # Save results
    with open(results_dir / "experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
