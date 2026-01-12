#!/usr/bin/env python3
"""
Experiment 4: Pointwise NLL vs Soft Target NLL

Compares two loss functions:
1. Pointwise NLL: Interpolate density at exact target location, compute -log(p)
2. Soft Target NLL: Create Gaussian blob at target, compute KL divergence

The pointwise approach is the "true" NLL but may have optimization challenges.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# Data Generation (same as experiment 1)
# ============================================================================

class StraightLineDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, pred_horizon: int,
                 velocity_range: Tuple[float, float] = (0.1, 4.5)):
        self.samples = []

        for _ in range(num_samples):
            velocity = np.random.uniform(*velocity_range)
            angle = np.random.uniform(0, 2 * np.pi)

            vx = velocity * np.cos(angle)
            vy = velocity * np.sin(angle)

            times = np.arange(seq_len + pred_horizon)
            x = vx * times
            y = vy * times
            traj = np.stack([x, y], axis=-1)

            history = traj[:seq_len]
            target_pos = traj[seq_len]

            last_pos = history[-1]
            input_seq = history - last_pos
            target = target_pos - last_pos

            self.samples.append((input_seq.astype(np.float32), target.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Model Architecture
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
    """
    2D Fourier head that can:
    1. Evaluate on a grid (for soft target loss)
    2. Evaluate at exact points (for true pointwise NLL)
    """
    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 8, grid_range: float = 5.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        num_freq_pairs = (2 * num_freqs + 1) ** 2
        self.coeff_predictor = nn.Linear(d_model, 2 * num_freq_pairs)

        x = torch.linspace(-grid_range, grid_range, grid_size)
        y = torch.linspace(-grid_range, grid_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)

        freqs = torch.arange(-num_freqs, num_freqs + 1, dtype=torch.float)
        freq_x, freq_y = torch.meshgrid(freqs, freqs, indexing='ij')
        self.register_buffer('freq_x', freq_x.flatten())
        self.register_buffer('freq_y', freq_y.flatten())

        nn.init.zeros_(self.coeff_predictor.weight)
        nn.init.zeros_(self.coeff_predictor.bias)

    def get_coeffs(self, z):
        """Get Fourier coefficients from embedding."""
        coeffs = self.coeff_predictor(z)
        num_freq_pairs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_freq_pairs]
        sin_coeffs = coeffs[:, num_freq_pairs:]
        return cos_coeffs, sin_coeffs

    def eval_at_points(self, cos_coeffs, sin_coeffs, x, y):
        """
        Evaluate Fourier series at arbitrary (x, y) points.

        Args:
            cos_coeffs: (batch, num_freqs) cosine coefficients
            sin_coeffs: (batch, num_freqs) sine coefficients
            x: (batch,) or (batch, n_points) x coordinates
            y: (batch,) or (batch, n_points) y coordinates

        Returns:
            logits at the specified points (unnormalized)
        """
        L = 2 * self.grid_range

        # Handle both single point and multiple points
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (batch, 1)
            y = y.unsqueeze(1)  # (batch, 1)

        # phase: (batch, n_points, num_freqs)
        phase = (2 * np.pi / L) * (
            self.freq_x.unsqueeze(0).unsqueeze(0) * x.unsqueeze(2) +
            self.freq_y.unsqueeze(0).unsqueeze(0) * y.unsqueeze(2)
        )

        cos_basis = torch.cos(phase)  # (batch, n_points, num_freqs)
        sin_basis = torch.sin(phase)

        # logits: (batch, n_points)
        logits = (
            torch.einsum('bf,bpf->bp', cos_coeffs, cos_basis) +
            torch.einsum('bf,bpf->bp', sin_coeffs, sin_basis)
        )

        return logits.squeeze(1) if logits.shape[1] == 1 else logits

    def forward(self, z):
        """Standard forward: evaluate on grid, return normalized density."""
        batch_size = z.shape[0]
        cos_coeffs, sin_coeffs = self.get_coeffs(z)

        L = 2 * self.grid_range
        grid_x_flat = self.grid_x.flatten()
        grid_y_flat = self.grid_y.flatten()

        phase = (2 * np.pi / L) * (
            self.freq_x.unsqueeze(0) * grid_x_flat.unsqueeze(1) +
            self.freq_y.unsqueeze(0) * grid_y_flat.unsqueeze(1)
        )

        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        logits = (
            torch.einsum('bf,gf->bg', cos_coeffs, cos_basis) +
            torch.einsum('bf,gf->bg', sin_coeffs, sin_basis)
        )

        logits = logits.view(batch_size, self.grid_size, self.grid_size)
        density = F.softmax(logits.view(batch_size, -1), dim=-1)
        density = density.view(batch_size, self.grid_size, self.grid_size)

        return density

    def forward_with_logits(self, z):
        """Return both density and logits (for pointwise NLL)."""
        batch_size = z.shape[0]
        cos_coeffs, sin_coeffs = self.get_coeffs(z)

        L = 2 * self.grid_range
        grid_x_flat = self.grid_x.flatten()
        grid_y_flat = self.grid_y.flatten()

        phase = (2 * np.pi / L) * (
            self.freq_x.unsqueeze(0) * grid_x_flat.unsqueeze(1) +
            self.freq_y.unsqueeze(0) * grid_y_flat.unsqueeze(1)
        )

        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        logits = (
            torch.einsum('bf,gf->bg', cos_coeffs, cos_basis) +
            torch.einsum('bf,gf->bg', sin_coeffs, sin_basis)
        )

        logits_grid = logits.view(batch_size, self.grid_size, self.grid_size)
        density = F.softmax(logits.view(batch_size, -1), dim=-1)
        density = density.view(batch_size, self.grid_size, self.grid_size)

        return density, logits_grid, cos_coeffs, sin_coeffs


class TrajectoryModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=5.0):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.grid_range = grid_range

        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

    def get_embedding(self, x):
        """Get transformer embedding."""
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]
        return z

    def forward(self, x):
        """Standard forward: return density grid."""
        z = self.get_embedding(x)
        density = self.fourier_head(z)
        return density

    def forward_for_pointwise(self, x):
        """Return everything needed for pointwise NLL."""
        z = self.get_embedding(x)
        density, logits_grid, cos_coeffs, sin_coeffs = self.fourier_head.forward_with_logits(z)
        return density, logits_grid, cos_coeffs, sin_coeffs


class RegressionModel(nn.Module):
    """Direct regression baseline."""
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]
        return self.head(z)


# ============================================================================
# Loss Functions
# ============================================================================

def true_pointwise_nll_loss(model, input_seq, target):
    """
    TRUE pointwise NLL: Evaluate Fourier series directly at target (x, y).

    No grid interpolation - we evaluate the continuous Fourier function
    at the exact target coordinates.

    NLL = -log(p(target)) = -log(exp(logit_target) / Z)
        = -logit_target + log(Z)

    where Z is computed from the grid for normalization.
    """
    density, logits_grid, cos_coeffs, sin_coeffs = model.forward_for_pointwise(input_seq)

    # Evaluate Fourier series at exact target point
    target_x = target[:, 0]
    target_y = target[:, 1]
    logit_at_target = model.fourier_head.eval_at_points(cos_coeffs, sin_coeffs, target_x, target_y)

    # Compute log(Z) from grid logits for normalization
    # Z = sum(exp(logits)) over grid
    # log(Z) = logsumexp(logits)
    batch_size = logits_grid.shape[0]
    logits_flat = logits_grid.view(batch_size, -1)
    log_Z = torch.logsumexp(logits_flat, dim=1)

    # NLL = -logit_target + log(Z)
    nll = -logit_at_target + log_Z

    return nll.mean()


def grid_interpolated_nll_loss(density, target, grid_range):
    """
    Grid-interpolated NLL: Bilinear interpolation from density grid.

    This still uses the grid but interpolates to get density at target.
    """
    batch_size = density.shape[0]

    target_normalized = target / grid_range
    grid = target_normalized.view(batch_size, 1, 1, 2)
    density_4d = density.unsqueeze(1)

    sampled = F.grid_sample(density_4d, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)

    density_at_target = sampled.view(batch_size)
    nll = -torch.log(density_at_target + 1e-10)

    return nll.mean()


def hard_target_nll_loss(density, target, grid_range):
    """
    Hard target: Put probability 1 on the nearest grid cell.
    Cross-entropy against a one-hot target.
    """
    batch_size = density.shape[0]
    grid_size = density.shape[1]
    device = density.device

    # Find nearest grid cell for each target
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)

    # Find closest indices
    target_x = target[:, 0]
    target_y = target[:, 1]

    x_idx = torch.argmin(torch.abs(x.unsqueeze(0) - target_x.unsqueeze(1)), dim=1)
    y_idx = torch.argmin(torch.abs(y.unsqueeze(0) - target_y.unsqueeze(1)), dim=1)

    # Clamp to valid range
    x_idx = torch.clamp(x_idx, 0, grid_size - 1)
    y_idx = torch.clamp(y_idx, 0, grid_size - 1)

    # Get log density at target cell
    log_density = torch.log(density + 1e-10)
    log_prob_at_target = log_density[torch.arange(batch_size, device=device), x_idx, y_idx]

    # NLL = -log(p) at target cell
    return -log_prob_at_target.mean()


def soft_target_nll_loss(density, target, grid_range, sigma=0.5):
    """
    Soft target loss: create Gaussian blob at target, compute cross-entropy.
    """
    batch_size = density.shape[0]
    grid_size = density.shape[1]
    device = density.device

    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    target_x = target[:, 0:1, None]
    target_y = target[:, 1:2, None]

    dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    log_density = torch.log(density + 1e-10)
    loss = -(soft_target * log_density).sum(dim=(-2, -1)).mean()

    return loss


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, config, loss_fn, is_pointwise=False):
    """Train model with specified loss function."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    history = {'train_loss': [], 'val_error': []}

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0

        for input_seq, target in train_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            if is_pointwise:
                # True pointwise needs model, not just density
                loss = loss_fn(model, input_seq, target)
            else:
                density = model(input_seq)
                loss = loss_fn(density, target, config['grid_range'])

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Validation
        val_error = evaluate_model(model, val_loader, config)
        history['train_loss'].append(total_loss / max(num_batches, 1))
        history['val_error'].append(val_error)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Loss={history['train_loss'][-1]:.4f}, Val Error={val_error:.4f}")

    return history


def train_regression(model, train_loader, val_loader, config):
    """Train regression baseline."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        model.train()
        for input_seq, target in train_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model(input_seq)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_error = evaluate_regression(model, val_loader)
            print(f"  Epoch {epoch+1}/{config['num_epochs']}: Val Error={val_error:.4f}")

    return evaluate_regression(model, val_loader)


def evaluate_model(model, val_loader, config):
    """Evaluate Fourier model using expected position error."""
    model.eval()
    grid_range = config['grid_range']
    grid_size = config['grid_size']

    x = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for input_seq, target in val_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
            density = model(input_seq)

            exp_x = (density * xx).sum(dim=(-2, -1))
            exp_y = (density * yy).sum(dim=(-2, -1))

            error = torch.sqrt((exp_x - target[:, 0])**2 + (exp_y - target[:, 1])**2)
            total_error += error.sum().item()
            total_samples += target.shape[0]

    return total_error / total_samples


def evaluate_regression(model, val_loader):
    """Evaluate regression model."""
    model.eval()
    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for input_seq, target in val_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
            pred = model(input_seq)
            error = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
            total_error += error.sum().item()
            total_samples += target.shape[0]

    return total_error / total_samples


# ============================================================================
# Visualization
# ============================================================================

def visualize_comparison(models, val_loader, config, save_path):
    """Compare predictions from different loss functions."""
    grid_range = config['grid_range']
    grid_size = config['grid_size']

    x = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Get a sample
    input_seq, target = next(iter(val_loader))
    input_seq, target = input_seq[:1].to(DEVICE), target[:1].to(DEVICE)
    input_np = input_seq[0].cpu().numpy()
    target_np = target[0].cpu().numpy()

    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 4))

    # Input trajectory
    ax = axes[0]
    ax.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8)
    ax.scatter([0], [0], c='blue', s=100, marker='s', zorder=5)
    ax.scatter(target_np[0], target_np[1], c='green', s=150, marker='x', linewidths=3, zorder=10)
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_title('Input + Target', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Each model's prediction
    for i, (name, model) in enumerate(models.items()):
        ax = axes[i + 1]
        model.eval()
        with torch.no_grad():
            density = model(input_seq)
            density_np = density[0].cpu().numpy()

            exp_x = (density[0] * xx).sum().item()
            exp_y = (density[0] * yy).sum().item()

        ax.imshow(density_np.T, origin='lower',
                  extent=[-grid_range, grid_range, -grid_range, grid_range],
                  cmap='hot', aspect='equal')
        ax.scatter(target_np[0], target_np[1], c='lime', s=150, marker='x', linewidths=3, zorder=10)
        ax.scatter(exp_x, exp_y, c='cyan', s=100, marker='o', edgecolors='white', linewidths=2, zorder=9)
        ax.set_title(name, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 4: POINTWISE NLL vs SOFT TARGET NLL")
    print("=" * 80)

    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'pred_horizon': 1,
        'velocity_range': (0.1, 4.5),
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 5.0,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'num_epochs': 50,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = StraightLineDataset(
        config['num_train'], config['seq_len'], config['pred_horizon'],
        config['velocity_range']
    )
    val_dataset = StraightLineDataset(
        config['num_val'], config['seq_len'], config['pred_horizon'],
        config['velocity_range']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    results = {}
    models = {}

    # =========================================================================
    # Regression Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("REGRESSION BASELINE")
    print("=" * 80)

    reg_model = RegressionModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    reg_error = train_regression(reg_model, train_loader, val_loader, config)
    print(f"\nFinal Error: {reg_error:.4f}")
    results['regression'] = {'error': reg_error}

    # =========================================================================
    # TRUE Pointwise NLL (evaluate Fourier at exact target)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRUE POINTWISE NLL (Evaluate Fourier series at exact target)")
    print("=" * 80)
    print("  NLL = -log(p(target)) where p is evaluated directly from Fourier coefficients")

    model_true_pointwise = TrajectoryModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    history_true_pointwise = train_model(
        model_true_pointwise, train_loader, val_loader, config,
        loss_fn=true_pointwise_nll_loss,
        is_pointwise=True
    )
    error_true_pointwise = evaluate_model(model_true_pointwise, val_loader, config)
    print(f"\nFinal Error: {error_true_pointwise:.4f}")
    results['true_pointwise_nll'] = {'error': error_true_pointwise}
    models['True Pointwise'] = model_true_pointwise

    # =========================================================================
    # Grid-Interpolated NLL (interpolate from density grid)
    # =========================================================================
    print("\n" + "=" * 80)
    print("GRID-INTERPOLATED NLL (Bilinear interpolation from grid)")
    print("=" * 80)

    model_grid_interp = TrajectoryModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    history_grid_interp = train_model(
        model_grid_interp, train_loader, val_loader, config,
        loss_fn=grid_interpolated_nll_loss
    )
    error_grid_interp = evaluate_model(model_grid_interp, val_loader, config)
    print(f"\nFinal Error: {error_grid_interp:.4f}")
    results['grid_interpolated_nll'] = {'error': error_grid_interp}
    models['Grid Interp'] = model_grid_interp

    # =========================================================================
    # Hard Target NLL (one-hot on nearest grid cell)
    # =========================================================================
    print("\n" + "=" * 80)
    print("HARD TARGET NLL (One-hot on nearest grid cell)")
    print("=" * 80)

    model_hard = TrajectoryModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    history_hard = train_model(
        model_hard, train_loader, val_loader, config,
        loss_fn=hard_target_nll_loss
    )
    error_hard = evaluate_model(model_hard, val_loader, config)
    print(f"\nFinal Error: {error_hard:.4f}")
    results['hard_target_nll'] = {'error': error_hard}
    models['Hard Target'] = model_hard

    # =========================================================================
    # Soft Target NLL (sigma=0.5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SOFT TARGET NLL (Gaussian blob, sigma=0.5)")
    print("=" * 80)

    model_soft_05 = TrajectoryModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    def soft_loss_05(density, target, grid_range):
        return soft_target_nll_loss(density, target, grid_range, sigma=0.5)

    history_soft_05 = train_model(
        model_soft_05, train_loader, val_loader, config,
        loss_fn=soft_loss_05
    )
    error_soft_05 = evaluate_model(model_soft_05, val_loader, config)
    print(f"\nFinal Error: {error_soft_05:.4f}")
    results['soft_nll_0.5'] = {'error': error_soft_05, 'history': history_soft_05}
    models['Soft (σ=0.5)'] = model_soft_05

    # =========================================================================
    # Soft Target NLL (sigma=0.2) - tighter
    # =========================================================================
    print("\n" + "=" * 80)
    print("SOFT TARGET NLL (Gaussian blob, sigma=0.2)")
    print("=" * 80)

    model_soft_02 = TrajectoryModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    def soft_loss_02(density, target, grid_range):
        return soft_target_nll_loss(density, target, grid_range, sigma=0.2)

    history_soft_02 = train_model(
        model_soft_02, train_loader, val_loader, config,
        loss_fn=soft_loss_02
    )
    error_soft_02 = evaluate_model(model_soft_02, val_loader, config)
    print(f"\nFinal Error: {error_soft_02:.4f}")
    results['soft_nll_0.2'] = {'error': error_soft_02, 'history': history_soft_02}
    models['Soft (σ=0.2)'] = model_soft_02

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n| Method | Error | Description |")
    print("|--------|-------|-------------|")
    print(f"| Regression Baseline | {reg_error:.4f} | Direct (x,y) prediction |")
    print(f"| True Pointwise NLL | {error_true_pointwise:.4f} | Evaluate Fourier at exact target |")
    print(f"| Grid-Interpolated NLL | {error_grid_interp:.4f} | Bilinear interp from grid |")
    print(f"| Hard Target (one-hot) | {error_hard:.4f} | One-hot on nearest grid cell |")
    print(f"| Soft Target (σ=0.5) | {error_soft_05:.4f} | Gaussian blob cross-entropy |")
    print(f"| Soft Target (σ=0.2) | {error_soft_02:.4f} | Tighter Gaussian |")

    # Visualize
    visualize_comparison(models, val_loader, config, results_dir / "loss_comparison.png")

    # Save results
    results_serializable = {
        k: {'error': v['error']} for k, v in results.items()
    }
    with open(results_dir / "experiment_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
