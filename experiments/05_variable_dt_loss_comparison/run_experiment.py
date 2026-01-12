#!/usr/bin/env python3
"""
Experiment 5: Loss Function Comparison with Variable Δt

Tests different loss functions on the variable time horizon task:
1. True Pointwise NLL - evaluate Fourier at exact target
2. Grid-Interpolated NLL - bilinear interpolation from grid
3. Hard Target - one-hot on nearest grid cell
4. Soft Target (σ=0.5) - Gaussian blob cross-entropy
5. Soft Target (σ=0.2) - tighter Gaussian
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# Data Generation (Variable Δt)
# ============================================================================

class VariableDtDataset(Dataset):
    """Dataset with variable prediction time horizon."""

    def __init__(self, num_samples: int, seq_len: int,
                 velocity_range: Tuple[float, float] = (0.1, 4.5),
                 dt_range: Tuple[float, float] = (0.5, 4.0)):
        self.samples = []

        for _ in range(num_samples):
            velocity = np.random.uniform(*velocity_range)
            angle = np.random.uniform(0, 2 * np.pi)
            dt = np.random.uniform(*dt_range)

            vx = velocity * np.cos(angle)
            vy = velocity * np.sin(angle)

            # Generate input trajectory
            times = np.arange(seq_len)
            positions = np.stack([vx * times, vy * times], axis=-1)

            # Relative to last position
            last_pos = positions[-1]
            input_seq = positions - last_pos

            # Target: displacement = velocity * dt
            target = np.array([vx * dt, vy * dt])

            self.samples.append((
                input_seq.astype(np.float32),
                np.array([dt], dtype=np.float32),
                target.astype(np.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Model Architecture (Concat to Input - best from exp 02)
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
    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 8, grid_range: float = 20.0):
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
        coeffs = self.coeff_predictor(z)
        num_freq_pairs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_freq_pairs]
        sin_coeffs = coeffs[:, num_freq_pairs:]
        return cos_coeffs, sin_coeffs

    def eval_at_points(self, cos_coeffs, sin_coeffs, x, y):
        """Evaluate Fourier series at arbitrary (x, y) points."""
        L = 2 * self.grid_range

        if x.dim() == 1:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

        phase = (2 * np.pi / L) * (
            self.freq_x.unsqueeze(0).unsqueeze(0) * x.unsqueeze(2) +
            self.freq_y.unsqueeze(0).unsqueeze(0) * y.unsqueeze(2)
        )

        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        logits = (
            torch.einsum('bf,bpf->bp', cos_coeffs, cos_basis) +
            torch.einsum('bf,bpf->bp', sin_coeffs, sin_basis)
        )

        return logits.squeeze(1) if logits.shape[1] == 1 else logits

    def forward(self, z):
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


class VariableDtModel(nn.Module):
    """Concat Δt to input (best method from experiment 02)."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=20.0):
        super().__init__()
        self.d_model = d_model

        # Input: (x, y, dt)
        self.input_proj = nn.Linear(3, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dt):
        batch_size, seq_len, _ = x.shape
        dt_expanded = dt.unsqueeze(1).expand(-1, seq_len, -1)
        x_with_dt = torch.cat([x, dt_expanded], dim=-1)

        z = self.input_proj(x_with_dt)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]

        density = self.fourier_head(z)
        return density

    def forward_for_pointwise(self, x, dt):
        batch_size, seq_len, _ = x.shape
        dt_expanded = dt.unsqueeze(1).expand(-1, seq_len, -1)
        x_with_dt = torch.cat([x, dt_expanded], dim=-1)

        z = self.input_proj(x_with_dt)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]

        density, logits_grid, cos_coeffs, sin_coeffs = self.fourier_head.forward_with_logits(z)
        return density, logits_grid, cos_coeffs, sin_coeffs


class RegressionModel(nn.Module):
    """Direct regression baseline."""

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x, dt):
        batch_size, seq_len, _ = x.shape
        dt_expanded = dt.unsqueeze(1).expand(-1, seq_len, -1)
        x_with_dt = torch.cat([x, dt_expanded], dim=-1)

        z = self.input_proj(x_with_dt)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]
        return self.head(z)


# ============================================================================
# Loss Functions
# ============================================================================

def true_pointwise_nll_loss(model, x, dt, target):
    """Evaluate Fourier at exact target, no grid interpolation."""
    density, logits_grid, cos_coeffs, sin_coeffs = model.forward_for_pointwise(x, dt)

    target_x = target[:, 0]
    target_y = target[:, 1]
    logit_at_target = model.fourier_head.eval_at_points(cos_coeffs, sin_coeffs, target_x, target_y)

    batch_size = logits_grid.shape[0]
    logits_flat = logits_grid.view(batch_size, -1)
    log_Z = torch.logsumexp(logits_flat, dim=1)

    nll = -logit_at_target + log_Z
    return nll.mean()


def grid_interpolated_nll_loss(density, target, grid_range):
    """Bilinear interpolation from density grid."""
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
    """One-hot on nearest grid cell."""
    batch_size = density.shape[0]
    grid_size = density.shape[1]
    device = density.device

    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)

    target_x = target[:, 0]
    target_y = target[:, 1]

    x_idx = torch.argmin(torch.abs(x.unsqueeze(0) - target_x.unsqueeze(1)), dim=1)
    y_idx = torch.argmin(torch.abs(y.unsqueeze(0) - target_y.unsqueeze(1)), dim=1)

    x_idx = torch.clamp(x_idx, 0, grid_size - 1)
    y_idx = torch.clamp(y_idx, 0, grid_size - 1)

    log_density = torch.log(density + 1e-10)
    log_prob_at_target = log_density[torch.arange(batch_size, device=device), x_idx, y_idx]

    return -log_prob_at_target.mean()


def soft_target_nll_loss(density, target, grid_range, sigma=0.5):
    """Gaussian blob cross-entropy."""
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

def train_fourier_model(model, train_loader, val_loader, config, loss_fn, is_pointwise=False):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0

        for x, dt, target in train_loader:
            x, dt, target = x.to(DEVICE), dt.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            if is_pointwise:
                loss = loss_fn(model, x, dt, target)
            else:
                density = model(x, dt)
                loss = loss_fn(density, target, config['grid_range'])

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            val_error = evaluate_model(model, val_loader, config)
            print(f"  Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Loss={total_loss/max(num_batches,1):.4f}, Val Error={val_error:.4f}")

    return model


def train_regression(model, train_loader, val_loader, config):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        model.train()
        for x, dt, target in train_loader:
            x, dt, target = x.to(DEVICE), dt.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, dt)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_error = evaluate_regression(model, val_loader)
            print(f"  Epoch {epoch+1}/{config['num_epochs']}: Val Error={val_error:.4f}")

    return model


def evaluate_model(model, val_loader, config):
    model.eval()
    grid_range = config['grid_range']
    grid_size = config['grid_size']

    x_coords = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    y_coords = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for x, dt, target in val_loader:
            x, dt, target = x.to(DEVICE), dt.to(DEVICE), target.to(DEVICE)
            density = model(x, dt)

            exp_x = (density * xx).sum(dim=(-2, -1))
            exp_y = (density * yy).sum(dim=(-2, -1))

            error = torch.sqrt((exp_x - target[:, 0])**2 + (exp_y - target[:, 1])**2)
            total_error += error.sum().item()
            total_samples += target.shape[0]

    return total_error / total_samples


def evaluate_regression(model, val_loader):
    model.eval()
    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for x, dt, target in val_loader:
            x, dt, target = x.to(DEVICE), dt.to(DEVICE), target.to(DEVICE)
            pred = model(x, dt)
            error = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
            total_error += error.sum().item()
            total_samples += target.shape[0]

    return total_error / total_samples


# ============================================================================
# Visualization
# ============================================================================

def visualize_comparison(models, val_loader, config, save_path):
    grid_range = config['grid_range']
    grid_size = config['grid_size']

    x_coords = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    y_coords = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # Get a sample with decent velocity
    for batch in val_loader:
        x, dt, target = batch
        for i in range(x.shape[0]):
            traj = x[i].numpy()
            disp = np.sqrt((traj[-1, 0] - traj[0, 0])**2 + (traj[-1, 1] - traj[0, 1])**2)
            if disp > 20:  # Good velocity
                x_sample = x[i:i+1].to(DEVICE)
                dt_sample = dt[i:i+1].to(DEVICE)
                target_sample = target[i:i+1]
                break
        else:
            continue
        break

    x_np = x_sample[0].cpu().numpy()
    target_np = target_sample[0].numpy()
    dt_val = dt_sample[0].item()

    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.5 * (len(models) + 1), 3.5))

    # Input
    ax = axes[0]
    ax.plot(x_np[:, 0], x_np[:, 1], 'b.-', linewidth=2, markersize=6)
    ax.scatter([0], [0], c='blue', s=80, marker='s', zorder=5)
    ax.scatter(target_np[0], target_np[1], c='green', s=120, marker='x', linewidths=3, zorder=10)
    ax.set_title(f'Input (Δt={dt_val:.1f})', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Compute bounds
    all_x = np.concatenate([x_np[:, 0], [target_np[0]]])
    all_y = np.concatenate([x_np[:, 1], [target_np[1]]])
    margin = 3
    xlim = (all_x.min() - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, all_y.max() + margin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for i, (name, model) in enumerate(models.items()):
        ax = axes[i + 1]
        model.eval()
        with torch.no_grad():
            density = model(x_sample, dt_sample)
            density_np = density[0].cpu().numpy()

            exp_x = (density[0] * xx).sum().item()
            exp_y = (density[0] * yy).sum().item()

        ax.imshow(density_np.T, origin='lower',
                  extent=[-grid_range, grid_range, -grid_range, grid_range],
                  cmap='hot', aspect='equal')
        ax.scatter(target_np[0], target_np[1], c='lime', s=120, marker='x', linewidths=3, zorder=10)
        ax.scatter(exp_x, exp_y, c='cyan', s=80, marker='o', edgecolors='white', linewidths=2, zorder=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(name, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 5: LOSS COMPARISON WITH VARIABLE Δt")
    print("=" * 80)

    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'velocity_range': (0.1, 4.5),
        'dt_range': (0.5, 4.0),
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 20.0,  # Must cover max displacement: 4.5 * 4.0 = 18
        'batch_size': 64,
        'learning_rate': 3e-4,
        'num_epochs': 30,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = VariableDtDataset(
        config['num_train'], config['seq_len'],
        config['velocity_range'], config['dt_range']
    )
    val_dataset = VariableDtDataset(
        config['num_val'], config['seq_len'],
        config['velocity_range'], config['dt_range']
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
    reg_model = train_regression(reg_model, train_loader, val_loader, config)
    reg_error = evaluate_regression(reg_model, val_loader)
    print(f"\nFinal Error: {reg_error:.4f}")
    results['regression'] = reg_error

    # =========================================================================
    # True Pointwise NLL
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRUE POINTWISE NLL")
    print("=" * 80)

    model_pointwise = VariableDtModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )
    model_pointwise = train_fourier_model(
        model_pointwise, train_loader, val_loader, config,
        loss_fn=true_pointwise_nll_loss, is_pointwise=True
    )
    error_pointwise = evaluate_model(model_pointwise, val_loader, config)
    print(f"\nFinal Error: {error_pointwise:.4f}")
    results['true_pointwise'] = error_pointwise
    models['True Pointwise'] = model_pointwise

    # =========================================================================
    # Grid-Interpolated NLL
    # =========================================================================
    print("\n" + "=" * 80)
    print("GRID-INTERPOLATED NLL")
    print("=" * 80)

    model_grid = VariableDtModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )
    model_grid = train_fourier_model(
        model_grid, train_loader, val_loader, config,
        loss_fn=grid_interpolated_nll_loss
    )
    error_grid = evaluate_model(model_grid, val_loader, config)
    print(f"\nFinal Error: {error_grid:.4f}")
    results['grid_interpolated'] = error_grid
    models['Grid Interp'] = model_grid

    # =========================================================================
    # Hard Target NLL
    # =========================================================================
    print("\n" + "=" * 80)
    print("HARD TARGET NLL (one-hot)")
    print("=" * 80)

    model_hard = VariableDtModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )
    model_hard = train_fourier_model(
        model_hard, train_loader, val_loader, config,
        loss_fn=hard_target_nll_loss
    )
    error_hard = evaluate_model(model_hard, val_loader, config)
    print(f"\nFinal Error: {error_hard:.4f}")
    results['hard_target'] = error_hard
    models['Hard Target'] = model_hard

    # =========================================================================
    # Soft Target (σ=0.5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SOFT TARGET NLL (σ=0.5)")
    print("=" * 80)

    model_soft_05 = VariableDtModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    def soft_loss_05(density, target, grid_range):
        return soft_target_nll_loss(density, target, grid_range, sigma=0.5)

    model_soft_05 = train_fourier_model(
        model_soft_05, train_loader, val_loader, config,
        loss_fn=soft_loss_05
    )
    error_soft_05 = evaluate_model(model_soft_05, val_loader, config)
    print(f"\nFinal Error: {error_soft_05:.4f}")
    results['soft_0.5'] = error_soft_05
    models['Soft σ=0.5'] = model_soft_05

    # =========================================================================
    # Soft Target (σ=0.2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SOFT TARGET NLL (σ=0.2)")
    print("=" * 80)

    model_soft_02 = VariableDtModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    def soft_loss_02(density, target, grid_range):
        return soft_target_nll_loss(density, target, grid_range, sigma=0.2)

    model_soft_02 = train_fourier_model(
        model_soft_02, train_loader, val_loader, config,
        loss_fn=soft_loss_02
    )
    error_soft_02 = evaluate_model(model_soft_02, val_loader, config)
    print(f"\nFinal Error: {error_soft_02:.4f}")
    results['soft_0.2'] = error_soft_02
    models['Soft σ=0.2'] = model_soft_02

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n| Method | Error |")
    print("|--------|-------|")
    print(f"| Regression Baseline | {reg_error:.4f} |")
    print(f"| True Pointwise NLL | {error_pointwise:.4f} |")
    print(f"| Grid-Interpolated NLL | {error_grid:.4f} |")
    print(f"| Hard Target (one-hot) | {error_hard:.4f} |")
    print(f"| Soft Target (σ=0.5) | {error_soft_05:.4f} |")
    print(f"| Soft Target (σ=0.2) | {error_soft_02:.4f} |")

    # Visualize
    visualize_comparison(models, val_loader, config, results_dir / "loss_comparison.png")

    # Save results
    with open(results_dir / "experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
