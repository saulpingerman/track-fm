#!/usr/bin/env python3
"""
Experiment 6: Relative Displacement Prediction with Normalized Grid

Key changes from previous experiments:
- Trajectories end at arbitrary positions (not always at origin)
- Model predicts RELATIVE displacement from last position
- Displacement is normalized to [-1, 1] × [-1, 1] grid
- Tests all loss functions

This tests if the model truly learns velocity extraction rather than
exploiting the fact that all tracks end at origin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# Data Generation - Relative Displacement
# =============================================================================

def generate_relative_displacement_data(num_samples, seq_len, velocity_range, dt,
                                         endpoint_range=10.0):
    """
    Generate trajectories ending at arbitrary positions.

    Returns:
        input_seq: (num_samples, seq_len, 2) - trajectory positions
        target: (num_samples, 2) - normalized displacement in [-1, 1]
        scale_factor: float - for denormalizing predictions
    """
    v_min, v_max = velocity_range

    # Random velocities and angles
    velocities = np.random.uniform(v_min, v_max, num_samples)
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    vx = velocities * np.cos(angles)
    vy = velocities * np.sin(angles)

    # Random endpoint for each trajectory (where the track ends)
    end_x = np.random.uniform(-endpoint_range, endpoint_range, num_samples)
    end_y = np.random.uniform(-endpoint_range, endpoint_range, num_samples)

    # Build trajectory backwards from endpoint
    # Time steps go from -(seq_len-1)*dt to 0 (ending at t=0)
    t = np.arange(-(seq_len - 1), 1) * dt  # e.g., [-9, -8, ..., 0] * dt

    # Positions: endpoint + velocity * time_offset
    # At t=0, position = endpoint
    # At t=-k*dt, position = endpoint - velocity * k * dt
    x = end_x[:, None] + vx[:, None] * t[None, :]
    y = end_y[:, None] + vy[:, None] * t[None, :]

    input_seq = np.stack([x, y], axis=-1).astype(np.float32)

    # Future position at time dt after the last position
    future_x = end_x + vx * dt
    future_y = end_y + vy * dt

    # Displacement from last position (endpoint)
    displacement_x = future_x - end_x  # = vx * dt
    displacement_y = future_y - end_y  # = vy * dt

    # Scale factor: max possible displacement
    scale_factor = v_max * dt

    # Normalize to [-1, 1]
    normalized_x = displacement_x / scale_factor
    normalized_y = displacement_y / scale_factor

    target = np.stack([normalized_x, normalized_y], axis=-1).astype(np.float32)

    return (torch.tensor(input_seq),
            torch.tensor(target),
            scale_factor)


# =============================================================================
# Model Architecture
# =============================================================================

class FourierHead2D(nn.Module):
    """2D Fourier series density head with pointwise evaluation support."""

    def __init__(self, input_dim, grid_size=64, num_freqs=8, grid_range=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        # Fourier coefficients predictor
        num_coeffs = (2 * num_freqs + 1) ** 2
        self.coeff_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * num_coeffs)  # cos and sin coefficients
        )

        # Precompute frequency indices
        freqs = torch.arange(-num_freqs, num_freqs + 1)
        freq_x, freq_y = torch.meshgrid(freqs, freqs, indexing='ij')
        self.register_buffer('freq_x', freq_x.flatten().float())
        self.register_buffer('freq_y', freq_y.flatten().float())

        # Precompute grid
        x = torch.linspace(-grid_range, grid_range, grid_size)
        y = torch.linspace(-grid_range, grid_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', xx)
        self.register_buffer('grid_y', yy)

    def forward(self, x):
        batch_size = x.shape[0]
        coeffs = self.coeff_net(x)
        num_coeffs = (2 * self.num_freqs + 1) ** 2
        cos_coeffs = coeffs[:, :num_coeffs]
        sin_coeffs = coeffs[:, num_coeffs:]

        # Evaluate on grid
        L = 2 * self.grid_range
        phase = (2 * np.pi / L) * (
            self.freq_x.view(1, 1, 1, -1) * self.grid_x.unsqueeze(0).unsqueeze(-1) +
            self.freq_y.view(1, 1, 1, -1) * self.grid_y.unsqueeze(0).unsqueeze(-1)
        )

        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        logits = (
            torch.einsum('bf,hwf->bhw', cos_coeffs, cos_basis[0]) +
            torch.einsum('bf,hwf->bhw', sin_coeffs, sin_basis[0])
        )

        density = F.softmax(logits.view(batch_size, -1), dim=-1)
        density = density.view(batch_size, self.grid_size, self.grid_size)

        return density, logits, cos_coeffs, sin_coeffs

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


class TransformerFourierModel(nn.Module):
    """Transformer encoder with 2D Fourier head."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=1.0):
        super().__init__()

        self.input_proj = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        embedding = h[:, -1, :]
        density, logits, cos_coeffs, sin_coeffs = self.fourier_head(embedding)
        return density, logits, cos_coeffs, sin_coeffs


class TransformerRegressionModel(nn.Module):
    """Transformer with regression head (baseline)."""

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        embedding = h[:, -1, :]
        return self.output_head(embedding)


# =============================================================================
# Loss Functions
# =============================================================================

def true_pointwise_nll_loss(model, density, logits, cos_coeffs, sin_coeffs, target):
    """TRUE pointwise NLL: Evaluate Fourier at exact target coordinates."""
    target_x = target[:, 0]
    target_y = target[:, 1]

    logit_at_target = model.fourier_head.eval_at_points(cos_coeffs, sin_coeffs, target_x, target_y)

    batch_size = logits.shape[0]
    logits_flat = logits.view(batch_size, -1)
    log_Z = torch.logsumexp(logits_flat, dim=1)

    nll = -logit_at_target + log_Z
    return nll.mean()


def grid_interpolated_nll_loss(density, target, grid_range):
    """Bilinear interpolation from density grid."""
    target_normalized = target / grid_range
    grid = target_normalized.unsqueeze(1).unsqueeze(1)
    grid = grid.flip(-1)

    density_unsqueezed = density.unsqueeze(1)
    interpolated = F.grid_sample(density_unsqueezed, grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)
    prob_at_target = interpolated.squeeze()

    nll = -torch.log(prob_at_target + 1e-10)
    return nll.mean()


def hard_target_nll_loss(density, target, grid_range, grid_size):
    """One-hot on nearest grid cell."""
    batch_size = density.shape[0]

    x = torch.linspace(-grid_range, grid_range, grid_size, device=density.device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=density.device)

    target_x = target[:, 0]
    target_y = target[:, 1]

    x_idx = torch.argmin(torch.abs(x.unsqueeze(0) - target_x.unsqueeze(1)), dim=1)
    y_idx = torch.argmin(torch.abs(y.unsqueeze(0) - target_y.unsqueeze(1)), dim=1)

    log_density = torch.log(density + 1e-10)
    log_prob_at_target = log_density[torch.arange(batch_size, device=density.device), x_idx, y_idx]

    return -log_prob_at_target.mean()


def soft_target_nll_loss(density, target, grid_range, grid_size, sigma=0.5):
    """Gaussian blob cross-entropy over whole grid."""
    batch_size = density.shape[0]

    x = torch.linspace(-grid_range, grid_range, grid_size, device=density.device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=density.device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    target_x = target[:, 0].view(-1, 1, 1)
    target_y = target[:, 1].view(-1, 1, 1)

    dist_sq = (xx.unsqueeze(0) - target_x) ** 2 + (yy.unsqueeze(0) - target_y) ** 2
    soft_target = torch.exp(-dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    log_density = torch.log(density + 1e-10)
    loss = -(soft_target * log_density).sum(dim=(-2, -1)).mean()

    return loss


def get_mode_prediction(density, grid_range):
    """Get mode (argmax) of density as prediction."""
    batch_size = density.shape[0]
    grid_size = density.shape[1]

    flat_idx = density.view(batch_size, -1).argmax(dim=1)
    x_idx = flat_idx // grid_size
    y_idx = flat_idx % grid_size

    x = torch.linspace(-grid_range, grid_range, grid_size, device=density.device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=density.device)

    pred_x = x[x_idx]
    pred_y = y[y_idx]

    return torch.stack([pred_x, pred_y], dim=1)


# =============================================================================
# Training Functions
# =============================================================================

def train_regression(model, train_loader, val_data, num_epochs, lr, scale_factor):
    """Train regression baseline."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    val_input, val_target = val_data
    val_input = val_input.to(device)
    val_target = val_target.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            pred = model(batch_input)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_input)
                # Denormalize for error calculation
                val_error = (val_pred - val_target).abs().mean().item() * scale_factor
            print(f"  Epoch {epoch+1}/{num_epochs}: Val Error={val_error:.4f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(val_input)
        final_error = (val_pred - val_target).abs().mean().item() * scale_factor

    return final_error


def train_fourier(model, train_loader, val_data, num_epochs, lr, loss_fn,
                  loss_name, grid_range, grid_size, scale_factor):
    """Train Fourier model with specified loss function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_input, val_target = val_data
    val_input = val_input.to(device)
    val_target = val_target.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            density, logits, cos_coeffs, sin_coeffs = model(batch_input)

            if loss_name == 'true_pointwise':
                loss = true_pointwise_nll_loss(model, density, logits, cos_coeffs, sin_coeffs, batch_target)
            elif loss_name == 'grid_interpolated':
                loss = grid_interpolated_nll_loss(density, batch_target, grid_range)
            elif loss_name == 'hard_target':
                loss = hard_target_nll_loss(density, batch_target, grid_range, grid_size)
            elif loss_name == 'soft_0.5':
                loss = soft_target_nll_loss(density, batch_target, grid_range, grid_size, sigma=0.5)
            elif loss_name == 'soft_0.2':
                loss = soft_target_nll_loss(density, batch_target, grid_range, grid_size, sigma=0.2)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                density, _, _, _ = model(val_input)
                val_pred = get_mode_prediction(density, grid_range)
                val_error = (val_pred - val_target).abs().mean().item() * scale_factor
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss/len(train_loader):.4f}, Val Error={val_error:.4f}")

    model.eval()
    with torch.no_grad():
        density, _, _, _ = model(val_input)
        val_pred = get_mode_prediction(density, grid_range)
        final_error = (val_pred - val_target).abs().mean().item() * scale_factor

    return final_error


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(model, val_input, val_target, grid_range, scale_factor,
                      save_path, title):
    """Visualize sample predictions."""
    model.eval()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    with torch.no_grad():
        density, _, _, _ = model(val_input[:6].to(device))
        density = density.cpu().numpy()

    val_target_np = val_target[:6].numpy()

    for i, ax in enumerate(axes.flat):
        ax.imshow(density[i].T, origin='lower', extent=[-grid_range, grid_range, -grid_range, grid_range],
                  cmap='hot', aspect='equal')
        ax.scatter(val_target_np[i, 0], val_target_np[i, 1], c='cyan', s=100, marker='x', linewidths=2)
        ax.set_xlim(-grid_range, grid_range)
        ax.set_ylim(-grid_range, grid_range)
        # Show normalized coordinates in title to match the plot
        ax.set_title(f'Target: ({val_target_np[i,0]:.2f}, {val_target_np[i,1]:.2f})')
        ax.set_xlabel('Δx (normalized)')
        ax.set_ylabel('Δy (normalized)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 6: RELATIVE DISPLACEMENT PREDICTION")
    print("=" * 80)

    # Configuration
    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'velocity_range': (0.1, 4.5),
        'dt': 1.0,
        'endpoint_range': 10.0,  # Trajectories end in [-10, 10] x [-10, 10]
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 1.0,  # Normalized grid [-1, 1]
        'batch_size': 64,
        'learning_rate': 3e-4,
        'num_epochs': 30,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Generate data
    print("\nCreating datasets...")
    train_input, train_target, scale_factor = generate_relative_displacement_data(
        config['num_train'], config['seq_len'], config['velocity_range'],
        config['dt'], config['endpoint_range']
    )
    val_input, val_target, _ = generate_relative_displacement_data(
        config['num_val'], config['seq_len'], config['velocity_range'],
        config['dt'], config['endpoint_range']
    )

    print(f"  Scale factor: {scale_factor:.2f} (max displacement)")
    print(f"  Train target range: [{train_target.min():.3f}, {train_target.max():.3f}]")

    train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )

    results = {}
    results_dir = Path(__file__).parent / 'results'

    # 1. Regression Baseline
    print("\n" + "=" * 80)
    print("REGRESSION BASELINE")
    print("=" * 80)
    model = TransformerRegressionModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers']
    ).to(device)
    error = train_regression(model, train_loader, (val_input, val_target),
                             config['num_epochs'], config['learning_rate'], scale_factor)
    print(f"\nFinal Error: {error:.4f}")
    results['regression'] = error

    # 2. True Pointwise NLL
    print("\n" + "=" * 80)
    print("TRUE POINTWISE NLL")
    print("=" * 80)
    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)
    error = train_fourier(model, train_loader, (val_input, val_target),
                          config['num_epochs'], config['learning_rate'], None,
                          'true_pointwise', config['grid_range'], config['grid_size'], scale_factor)
    print(f"\nFinal Error: {error:.4f}")
    results['true_pointwise'] = error
    visualize_results(model, val_input, val_target, config['grid_range'], scale_factor,
                      results_dir / 'true_pointwise.png', 'True Pointwise NLL')

    # 3. Grid-Interpolated NLL
    print("\n" + "=" * 80)
    print("GRID-INTERPOLATED NLL")
    print("=" * 80)
    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)
    error = train_fourier(model, train_loader, (val_input, val_target),
                          config['num_epochs'], config['learning_rate'], None,
                          'grid_interpolated', config['grid_range'], config['grid_size'], scale_factor)
    print(f"\nFinal Error: {error:.4f}")
    results['grid_interpolated'] = error

    # 4. Hard Target NLL
    print("\n" + "=" * 80)
    print("HARD TARGET NLL (one-hot)")
    print("=" * 80)
    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)
    error = train_fourier(model, train_loader, (val_input, val_target),
                          config['num_epochs'], config['learning_rate'], None,
                          'hard_target', config['grid_range'], config['grid_size'], scale_factor)
    print(f"\nFinal Error: {error:.4f}")
    results['hard_target'] = error
    visualize_results(model, val_input, val_target, config['grid_range'], scale_factor,
                      results_dir / 'hard_target.png', 'Hard Target NLL')

    # 5. Soft Target (σ=0.5)
    print("\n" + "=" * 80)
    print("SOFT TARGET NLL (σ=0.5)")
    print("=" * 80)
    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)
    error = train_fourier(model, train_loader, (val_input, val_target),
                          config['num_epochs'], config['learning_rate'], None,
                          'soft_0.5', config['grid_range'], config['grid_size'], scale_factor)
    print(f"\nFinal Error: {error:.4f}")
    results['soft_0.5'] = error
    visualize_results(model, val_input, val_target, config['grid_range'], scale_factor,
                      results_dir / 'soft_0.5.png', 'Soft Target NLL (σ=0.5)')

    # 6. Soft Target (σ=0.2)
    print("\n" + "=" * 80)
    print("SOFT TARGET NLL (σ=0.2)")
    print("=" * 80)
    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)
    error = train_fourier(model, train_loader, (val_input, val_target),
                          config['num_epochs'], config['learning_rate'], None,
                          'soft_0.2', config['grid_range'], config['grid_size'], scale_factor)
    print(f"\nFinal Error: {error:.4f}")
    results['soft_0.2'] = error
    visualize_results(model, val_input, val_target, config['grid_range'], scale_factor,
                      results_dir / 'soft_0.2.png', 'Soft Target NLL (σ=0.2)')

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n| Method | Error |")
    print("|--------|-------|")
    for method, error in results.items():
        print(f"| {method} | {error:.4f} |")

    # Save results
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump({
            'config': config,
            'scale_factor': scale_factor,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
