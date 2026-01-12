#!/usr/bin/env python3
"""
Fourier Head Trajectory Prediction: Full Diagnostic Experiment

This script implements the complete experiment from fourier_head_trajectory_test_design.md
to diagnose why a transformer model with 2D Fourier head fails to learn dead reckoning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# Part 1: Synthetic Data Generation
# ============================================================================

def generate_straight_line_trajectory(seq_len: int, pred_horizon: int,
                                      velocity: float, angle: float) -> np.ndarray:
    """Generate a straight-line trajectory with constant velocity."""
    vx = velocity * np.cos(angle)
    vy = velocity * np.sin(angle)

    times = np.arange(seq_len + pred_horizon)
    x = vx * times
    y = vy * times

    return np.stack([x, y], axis=-1)


def prepare_training_sample(trajectory: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert trajectory to model input/output format, relative to last position."""
    history = trajectory[:seq_len]
    target_pos = trajectory[seq_len]

    last_pos = history[-1]
    input_seq = history - last_pos
    target = target_pos - last_pos

    return input_seq, target


class StraightLineDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, pred_horizon: int,
                 velocity_range: Tuple[float, float] = (0.5, 2.0), noise_std: float = 0.0):
        self.samples = []

        for _ in range(num_samples):
            velocity = np.random.uniform(*velocity_range)
            angle = np.random.uniform(0, 2 * np.pi)

            traj = generate_straight_line_trajectory(seq_len, pred_horizon, velocity, angle)

            if noise_std > 0:
                traj += np.random.normal(0, noise_std, traj.shape)

            input_seq, target = prepare_training_sample(traj, seq_len)
            self.samples.append((input_seq.astype(np.float32), target.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Part 2: Model Architecture
# ============================================================================

class TrajectoryEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2, d_model)

    def forward(self, x):
        return self.proj(x)


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


class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128):
        super().__init__()
        self.embedding = TrajectoryEmbedding(d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x[:, -1, :]


class FourierHead2D(nn.Module):
    def __init__(self, d_model: int, grid_size: int = 64,
                 num_freqs: int = 8, grid_range: float = 5.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.grid_range = grid_range

        num_freq_pairs = (2 * num_freqs + 1) ** 2
        self.coeff_predictor = nn.Linear(d_model, 2 * num_freq_pairs)

        # Precompute grid coordinates
        x = torch.linspace(-grid_range, grid_range, grid_size)
        y = torch.linspace(-grid_range, grid_range, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', xx.flatten())
        self.register_buffer('grid_y', yy.flatten())

        # Precompute frequency indices
        freqs = torch.arange(-num_freqs, num_freqs + 1, dtype=torch.float)
        freq_x, freq_y = torch.meshgrid(freqs, freqs, indexing='ij')
        self.register_buffer('freq_x', freq_x.flatten())
        self.register_buffer('freq_y', freq_y.flatten())

        self.dx = 2 * grid_range / (grid_size - 1)
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

        logits = (
            torch.einsum('bf,gf->bg', cos_coeffs, cos_basis) +
            torch.einsum('bf,gf->bg', sin_coeffs, sin_basis)
        )

        log_density = F.log_softmax(logits, dim=-1)
        log_density = log_density.view(batch_size, self.grid_size, self.grid_size)

        return log_density

    def get_grid_coordinates(self):
        x = torch.linspace(-self.grid_range, self.grid_range, self.grid_size)
        y = torch.linspace(-self.grid_range, self.grid_range, self.grid_size)
        return x, y


class TrajectoryForecastModel(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 grid_size: int = 64, num_freqs: int = 8, grid_range: float = 5.0):
        super().__init__()
        self.transformer = TrajectoryTransformer(d_model, nhead, num_layers)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_range = grid_range
        self.grid_size = grid_size

    def forward(self, x):
        z = self.transformer(x)
        log_density = self.fourier_head(z)
        return log_density

    def compute_nll_loss(self, log_density, target):
        batch_size = target.shape[0]

        grid_x, grid_y = self.fourier_head.get_grid_coordinates()
        grid_x = grid_x.to(target.device)
        grid_y = grid_y.to(target.device)

        x_idx = torch.argmin(torch.abs(grid_x.unsqueeze(0) - target[:, 0:1]), dim=1)
        y_idx = torch.argmin(torch.abs(grid_y.unsqueeze(0) - target[:, 1:2]), dim=1)

        x_idx = torch.clamp(x_idx, 0, self.grid_size - 1)
        y_idx = torch.clamp(y_idx, 0, self.grid_size - 1)

        log_prob = log_density[torch.arange(batch_size, device=target.device), x_idx, y_idx]

        return -log_prob.mean()


class DirectRegressionModel(nn.Module):
    """Direct regression baseline (no density)."""
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.transformer = TrajectoryTransformer(d_model, nhead, num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        z = self.transformer(x)
        return self.head(z)


# ============================================================================
# Part 3: Loss Functions
# ============================================================================

def compute_soft_nll_loss(model, log_density, target, sigma: float = 0.5):
    """Soft target assignment with Gaussian kernel."""
    grid_x, grid_y = model.fourier_head.get_grid_coordinates()
    grid_x = grid_x.to(target.device)
    grid_y = grid_y.to(target.device)

    xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')

    dx = xx.unsqueeze(0) - target[:, 0:1, None]
    dy = yy.unsqueeze(0) - target[:, 1:2, None]

    soft_target = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    soft_target = soft_target / soft_target.sum(dim=(1, 2), keepdim=True)

    loss = F.kl_div(log_density, soft_target, reduction='batchmean')
    return loss


def compute_mixed_loss(model, log_density, target, alpha: float = 0.5):
    """Combine NLL with mode-targeting loss."""
    nll_loss = model.compute_nll_loss(log_density, target)

    densities = torch.exp(log_density)
    grid_x, grid_y = model.fourier_head.get_grid_coordinates()
    grid_x = grid_x.to(target.device)
    grid_y = grid_y.to(target.device)

    batch_size = log_density.shape[0]
    flat_density = densities.view(batch_size, -1)
    mode_idx = flat_density.argmax(dim=1)

    grid_size = model.grid_size
    mode_x_idx = mode_idx // grid_size
    mode_y_idx = mode_idx % grid_size

    mode_x = grid_x[mode_x_idx]
    mode_y = grid_y[mode_y_idx]
    mode_pos = torch.stack([mode_x, mode_y], dim=1)

    mode_loss = F.mse_loss(mode_pos, target)

    return alpha * nll_loss + (1 - alpha) * mode_loss


# ============================================================================
# Part 4: Training Functions
# ============================================================================

def train_regression_baseline(model, train_loader, val_loader,
                              num_epochs: int = 50, lr: float = 1e-3):
    """Train direct regression baseline with MSE loss."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for input_seq, target in train_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            pred = model(input_seq)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for input_seq, target in val_loader:
                input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
                pred = model(input_seq)
                val_loss += F.mse_loss(pred, target).item()
                val_mae += torch.abs(pred - target).mean().item()

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_mae'].append(val_mae / len(val_loader))

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Train MSE = {history['train_loss'][-1]:.6f}, "
                  f"Val MSE = {history['val_loss'][-1]:.6f}, Val MAE = {history['val_mae'][-1]:.6f}")

    return history


def train_fourier_model(model, train_loader, val_loader, config: Dict,
                        loss_type: str = 'nll'):
    """Train Fourier head model with diagnostics."""
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    diagnostics = {
        'train_loss': [],
        'val_loss': [],
        'grad_norms': [],
        'coeff_norms': [],
        'prediction_spread': [],
        'mean_predicted_displacement': [],
        'mean_target_displacement': [],
    }

    step = 0
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0

        for input_seq, target in train_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            log_density = model(input_seq)

            if loss_type == 'nll':
                loss = model.compute_nll_loss(log_density, target)
            elif loss_type == 'soft':
                loss = compute_soft_nll_loss(model, log_density, target, sigma=0.5)
            elif loss_type == 'mixed':
                loss = compute_mixed_loss(model, log_density, target, alpha=0.5)

            coeff_norm = model.fourier_head.coeff_predictor.weight.norm()
            loss = loss + config['fourier_coeff_reg'] * coeff_norm

            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norm += total_norm

            optimizer.step()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()

        # Validation and diagnostics
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_log_densities = []
            all_targets = []

            for input_seq, target in val_loader:
                input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
                log_density = model(input_seq)
                val_loss += model.compute_nll_loss(log_density, target).item()
                all_log_densities.append(log_density)
                all_targets.append(target)

            log_densities = torch.cat(all_log_densities, dim=0)
            targets = torch.cat(all_targets, dim=0)

            densities = torch.exp(log_densities)
            grid_x, grid_y = model.fourier_head.get_grid_coordinates()
            grid_x = grid_x.to(DEVICE)
            grid_y = grid_y.to(DEVICE)

            marginal_x = densities.sum(dim=2)
            marginal_y = densities.sum(dim=1)
            expected_x = (marginal_x * grid_x.unsqueeze(0)).sum(dim=1)
            expected_y = (marginal_y * grid_y.unsqueeze(0)).sum(dim=1)

            mean_pred_disp = torch.stack([expected_x.mean(), expected_y.mean()])
            mean_target_disp = targets.mean(dim=0)

            entropy = -(densities * log_densities.clamp(min=-100)).sum(dim=(1,2)).mean()

        diagnostics['train_loss'].append(epoch_loss / len(train_loader))
        diagnostics['val_loss'].append(val_loss / len(val_loader))
        diagnostics['grad_norms'].append(epoch_grad_norm / len(train_loader))
        diagnostics['coeff_norms'].append(coeff_norm.item())
        diagnostics['mean_predicted_displacement'].append(mean_pred_disp.cpu().tolist())
        diagnostics['mean_target_displacement'].append(mean_target_disp.cpu().tolist())
        diagnostics['prediction_spread'].append(entropy.item())

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss={diagnostics['train_loss'][-1]:.4f}/{diagnostics['val_loss'][-1]:.4f} "
                  f"GradNorm={diagnostics['grad_norms'][-1]:.2f} Entropy={entropy.item():.2f}")

    return diagnostics


def curriculum_training(model, config: Dict, loss_type: str = 'soft'):
    """Train with curriculum learning (increasing velocity)."""
    model.to(DEVICE)

    velocity_schedule = [
        (0.1, 0.5, 20),
        (0.1, 1.5, 20),
        (0.1, 3.0, 30),
        (0.1, 4.5, 30),
    ]

    all_diagnostics = []

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    for stage_idx, (v_min, v_max, num_epochs) in enumerate(velocity_schedule):
        print(f"\n  Curriculum Stage {stage_idx + 1}: velocity range [{v_min}, {v_max}]")

        train_dataset = StraightLineDataset(
            config['num_train'], config['seq_len'], config['pred_horizon'],
            velocity_range=(v_min, v_max)
        )
        val_dataset = StraightLineDataset(
            config['num_val'], config['seq_len'], config['pred_horizon'],
            velocity_range=(v_min, v_max)
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for input_seq, target in train_loader:
                input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                log_density = model(input_seq)

                if loss_type == 'soft':
                    loss = compute_soft_nll_loss(model, log_density, target, sigma=0.3)
                else:
                    loss = model.compute_nll_loss(log_density, target)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for input_seq, target in val_loader:
                        input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
                        log_density = model(input_seq)
                        val_loss += model.compute_nll_loss(log_density, target).item()

                print(f"    Epoch {epoch}: Train={epoch_loss/len(train_loader):.4f}, "
                      f"Val={val_loss/len(val_loader):.4f}")

    return model


# ============================================================================
# Part 5: Diagnostic Tests
# ============================================================================

def test_input_ablation(model, val_loader):
    """Check if model uses input by comparing with zeroed/random inputs."""
    model.eval()

    real_outputs = []
    zero_outputs = []
    random_outputs = []

    with torch.no_grad():
        for input_seq, target in val_loader:
            input_seq = input_seq.to(DEVICE)

            real_out = model(input_seq)
            real_outputs.append(real_out)

            zero_input = torch.zeros_like(input_seq)
            zero_out = model(zero_input)
            zero_outputs.append(zero_out)

            random_input = torch.randn_like(input_seq)
            random_out = model(random_input)
            random_outputs.append(random_out)

    real_outputs = torch.cat(real_outputs, dim=0)
    zero_outputs = torch.cat(zero_outputs, dim=0)
    random_outputs = torch.cat(random_outputs, dim=0)

    real_density = torch.exp(real_outputs).view(real_outputs.shape[0], -1)
    zero_density = torch.exp(zero_outputs).view(zero_outputs.shape[0], -1)
    random_density = torch.exp(random_outputs).view(random_outputs.shape[0], -1)

    # KL divergence
    kl_real_vs_zero = (real_density * (torch.log(real_density + 1e-10) -
                                        torch.log(zero_density + 1e-10))).sum(dim=1).mean()
    kl_real_vs_random = (real_density * (torch.log(real_density + 1e-10) -
                                          torch.log(random_density + 1e-10))).sum(dim=1).mean()

    # Also compute mean L1 difference
    l1_real_vs_zero = (real_density - zero_density).abs().mean()
    l1_real_vs_random = (real_density - random_density).abs().mean()

    print("  Input Ablation Results:")
    print(f"    KL(real || zero): {kl_real_vs_zero.item():.6f}")
    print(f"    KL(real || random): {kl_real_vs_random.item():.6f}")
    print(f"    L1(real, zero): {l1_real_vs_zero.item():.6f}")
    print(f"    L1(real, random): {l1_real_vs_random.item():.6f}")
    print("    If KL/L1 values are near 0, model is ignoring input!")

    return {
        'kl_real_vs_zero': kl_real_vs_zero.item(),
        'kl_real_vs_random': kl_real_vs_random.item(),
        'l1_real_vs_zero': l1_real_vs_zero.item(),
        'l1_real_vs_random': l1_real_vs_random.item(),
    }


def analyze_gradient_flow(model, sample_batch):
    """Check if gradients flow through all layers."""
    input_seq, target = sample_batch
    input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)

    model.zero_grad()
    log_density = model(input_seq)
    loss = model.compute_nll_loss(log_density, target)
    loss.backward()

    print("  Gradient Flow Analysis:")
    print("  " + "-" * 70)

    gradient_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            ratio = grad_norm / (param_norm + 1e-8)
            status = "OK" if grad_norm > 1e-7 else "WEAK" if grad_norm > 1e-10 else "DEAD"
            print(f"  {name:45s} | grad: {grad_norm:.2e} | param: {param_norm:.2e} | {status}")
            gradient_info[name] = {'grad_norm': grad_norm, 'param_norm': param_norm, 'status': status}
        else:
            print(f"  {name:45s} | NO GRADIENT")
            gradient_info[name] = {'grad_norm': 0, 'param_norm': 0, 'status': 'NO_GRAD'}

    return gradient_info


def compute_prediction_metrics(model, val_loader):
    """Compute detailed prediction metrics."""
    model.eval()

    all_pred_x = []
    all_pred_y = []
    all_target_x = []
    all_target_y = []
    all_mode_x = []
    all_mode_y = []

    with torch.no_grad():
        for input_seq, target in val_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
            log_density = model(input_seq)
            densities = torch.exp(log_density)

            grid_x, grid_y = model.fourier_head.get_grid_coordinates()
            grid_x = grid_x.to(DEVICE)
            grid_y = grid_y.to(DEVICE)

            # Expected value
            marginal_x = densities.sum(dim=2)
            marginal_y = densities.sum(dim=1)
            expected_x = (marginal_x * grid_x.unsqueeze(0)).sum(dim=1)
            expected_y = (marginal_y * grid_y.unsqueeze(0)).sum(dim=1)

            # Mode
            flat_density = densities.view(densities.shape[0], -1)
            mode_idx = flat_density.argmax(dim=1)
            mode_x_idx = mode_idx // model.grid_size
            mode_y_idx = mode_idx % model.grid_size
            mode_x = grid_x[mode_x_idx]
            mode_y = grid_y[mode_y_idx]

            all_pred_x.append(expected_x)
            all_pred_y.append(expected_y)
            all_target_x.append(target[:, 0])
            all_target_y.append(target[:, 1])
            all_mode_x.append(mode_x)
            all_mode_y.append(mode_y)

    pred_x = torch.cat(all_pred_x)
    pred_y = torch.cat(all_pred_y)
    target_x = torch.cat(all_target_x)
    target_y = torch.cat(all_target_y)
    mode_x = torch.cat(all_mode_x)
    mode_y = torch.cat(all_mode_y)

    # Compute errors
    expected_error = torch.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2).mean()
    mode_error = torch.sqrt((mode_x - target_x)**2 + (mode_y - target_y)**2).mean()

    # Correlation with target magnitude
    target_mag = torch.sqrt(target_x**2 + target_y**2)
    pred_mag = torch.sqrt(pred_x**2 + pred_y**2)

    metrics = {
        'expected_error': expected_error.item(),
        'mode_error': mode_error.item(),
        'mean_pred_x': pred_x.mean().item(),
        'mean_pred_y': pred_y.mean().item(),
        'mean_target_x': target_x.mean().item(),
        'mean_target_y': target_y.mean().item(),
        'std_pred_x': pred_x.std().item(),
        'std_pred_y': pred_y.std().item(),
        'std_target_x': target_x.std().item(),
        'std_target_y': target_y.std().item(),
        'pred_mag_mean': pred_mag.mean().item(),
        'target_mag_mean': target_mag.mean().item(),
    }

    return metrics


# ============================================================================
# Part 6: Visualization
# ============================================================================

def visualize_predictions(model, val_loader, save_path: str, num_samples: int = 6):
    """Visualize predicted densities vs ground truth."""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    with torch.no_grad():
        sample_iter = iter(val_loader)
        input_seq, target = next(sample_iter)
        input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)

        for idx in range(min(num_samples, input_seq.shape[0])):
            inp = input_seq[idx:idx+1]
            tgt = target[idx]

            log_density = model(inp)
            density = torch.exp(log_density[0]).cpu().numpy()

            grid_x, grid_y = model.fourier_head.get_grid_coordinates()

            # Plot 1: Input trajectory
            ax = axes[idx, 0]
            traj = inp[0].cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], 'b.-', linewidth=2, markersize=8, label='History')
            ax.scatter([0], [0], c='green', s=100, marker='o', label='Last pos', zorder=5)
            ax.scatter([tgt[0].item()], [tgt[1].item()], c='red', s=100, marker='x',
                      linewidths=3, label='Target', zorder=5)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.legend(loc='upper left')
            ax.set_title('Input Trajectory')
            ax.grid(True, alpha=0.3)

            # Plot 2: Predicted density
            ax = axes[idx, 1]
            im = ax.imshow(density.T, origin='lower',
                          extent=[-model.grid_range, model.grid_range,
                                  -model.grid_range, model.grid_range],
                          cmap='hot', aspect='equal')
            ax.scatter([tgt[0].item()], [tgt[1].item()], c='cyan', s=100, marker='x',
                      linewidths=3, label='Target', zorder=5)
            ax.scatter([0], [0], c='green', s=50, marker='o', label='Origin', zorder=5)
            plt.colorbar(im, ax=ax)
            ax.set_title('Predicted Density')
            ax.legend(loc='upper left')

            # Plot 3: Log density
            ax = axes[idx, 2]
            im = ax.imshow(log_density[0].cpu().numpy().T, origin='lower',
                          extent=[-model.grid_range, model.grid_range,
                                  -model.grid_range, model.grid_range],
                          cmap='viridis', aspect='equal')
            ax.scatter([tgt[0].item()], [tgt[1].item()], c='red', s=100, marker='x',
                      linewidths=3, zorder=5)
            plt.colorbar(im, ax=ax)
            ax.set_title('Log Density')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {save_path}")


def plot_training_diagnostics(diagnostics: Dict, save_path: str):
    """Plot training diagnostics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = range(len(diagnostics['train_loss']))

    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, diagnostics['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, diagnostics['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norms
    ax = axes[0, 1]
    ax.plot(epochs, diagnostics['grad_norms'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Over Training')
    ax.grid(True, alpha=0.3)

    # Coefficient norms
    ax = axes[0, 2]
    ax.plot(epochs, diagnostics['coeff_norms'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Coefficient Norm')
    ax.set_title('Fourier Coefficient Norm')
    ax.grid(True, alpha=0.3)

    # Prediction spread (entropy)
    ax = axes[1, 0]
    ax.plot(epochs, diagnostics['prediction_spread'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy')
    ax.set_title('Prediction Entropy (spread)')
    ax.grid(True, alpha=0.3)

    # Mean predicted displacement
    ax = axes[1, 1]
    pred_disp = np.array(diagnostics['mean_predicted_displacement'])
    target_disp = np.array(diagnostics['mean_target_displacement'])
    ax.plot(epochs, pred_disp[:, 0], label='Pred X', linewidth=2)
    ax.plot(epochs, pred_disp[:, 1], label='Pred Y', linewidth=2)
    ax.axhline(y=target_disp[-1, 0], color='r', linestyle='--', alpha=0.5, label='Target X')
    ax.axhline(y=target_disp[-1, 1], color='g', linestyle='--', alpha=0.5, label='Target Y')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Displacement')
    ax.set_title('Mean Predicted vs Target Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Displacement magnitude
    ax = axes[1, 2]
    pred_mag = np.sqrt(pred_disp[:, 0]**2 + pred_disp[:, 1]**2)
    target_mag = np.sqrt(target_disp[:, 0]**2 + target_disp[:, 1]**2)
    ax.plot(epochs, pred_mag, label='Predicted', linewidth=2)
    ax.axhline(y=target_mag[-1], color='r', linestyle='--', label='Target', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Magnitude')
    ax.set_title('Mean Displacement Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved diagnostics plot to {save_path}")


# ============================================================================
# Part 7: Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("FOURIER HEAD TRAJECTORY PREDICTION: FULL DIAGNOSTIC EXPERIMENT")
    print("=" * 80)

    # Configuration
    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'pred_horizon': 1,
        'velocity_range': (0.1, 4.5),  # Wide range: targets cover most of [-5,5] grid
        'noise_std': 0.0,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 5.0,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'warmup_steps': 500,
        'fourier_coeff_reg': 0.01,
    }

    output_dir = Path('/home/ec2-user/projects/trackfm-deadreckon/fourier_trajectory_test/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("=" * 80)

    train_dataset = StraightLineDataset(
        config['num_train'], config['seq_len'], config['pred_horizon'],
        velocity_range=config['velocity_range'], noise_std=config['noise_std']
    )
    val_dataset = StraightLineDataset(
        config['num_val'], config['seq_len'], config['pred_horizon'],
        velocity_range=config['velocity_range'], noise_std=config['noise_std']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Check data statistics
    sample_input, sample_target = train_dataset[0]
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Target shape: {sample_target.shape}")

    # Compute target statistics
    all_targets = np.array([train_dataset[i][1] for i in range(min(1000, len(train_dataset)))])
    target_magnitudes = np.sqrt(all_targets[:, 0]**2 + all_targets[:, 1]**2)
    print(f"  Target magnitude: mean={target_magnitudes.mean():.3f}, std={target_magnitudes.std():.3f}")
    print(f"  Target X: mean={all_targets[:, 0].mean():.3f}, std={all_targets[:, 0].std():.3f}")
    print(f"  Target Y: mean={all_targets[:, 1].mean():.3f}, std={all_targets[:, 1].std():.3f}")

    results = {'config': config}

    # =========================================================================
    # STEP 2: Train Regression Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING REGRESSION BASELINE (Direct MSE)")
    print("=" * 80)
    print("This tests whether the transformer can extract velocity at all.\n")

    regression_model = DirectRegressionModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )

    regression_history = train_regression_baseline(
        regression_model, train_loader, val_loader, num_epochs=50
    )

    # Evaluate regression baseline
    regression_model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for input_seq, target in val_loader:
            input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
            pred = regression_model(input_seq)
            val_preds.append(pred)
            val_targets.append(target)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        final_mse = F.mse_loss(val_preds, val_targets).item()
        final_mae = torch.abs(val_preds - val_targets).mean().item()
        pred_error = torch.sqrt((val_preds - val_targets).pow(2).sum(dim=1)).mean().item()

    print(f"\n  REGRESSION BASELINE RESULTS:")
    print(f"    Final Val MSE: {final_mse:.6f}")
    print(f"    Final Val MAE: {final_mae:.6f}")
    print(f"    Mean prediction error: {pred_error:.6f}")

    if pred_error < 0.1:
        print("    STATUS: PASS - Transformer can extract velocity!")
    else:
        print("    STATUS: FAIL - Transformer cannot extract velocity. Problem is in backbone.")

    results['regression_baseline'] = {
        'final_mse': final_mse,
        'final_mae': final_mae,
        'pred_error': pred_error,
        'history': regression_history,
    }

    # =========================================================================
    # STEP 3: Train Fourier Head Model with Standard NLL
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING FOURIER HEAD MODEL (Standard NLL)")
    print("=" * 80)

    fourier_model_nll = TrajectoryForecastModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    nll_diagnostics = train_fourier_model(
        fourier_model_nll, train_loader, val_loader, config, loss_type='nll'
    )

    # Run diagnostics
    print("\n  Running input ablation test...")
    nll_ablation = test_input_ablation(fourier_model_nll, val_loader)

    print("\n  Computing prediction metrics...")
    nll_metrics = compute_prediction_metrics(fourier_model_nll, val_loader)
    print(f"    Expected value error: {nll_metrics['expected_error']:.4f}")
    print(f"    Mode error: {nll_metrics['mode_error']:.4f}")
    print(f"    Pred magnitude mean: {nll_metrics['pred_mag_mean']:.4f}")
    print(f"    Target magnitude mean: {nll_metrics['target_mag_mean']:.4f}")

    print("\n  Running gradient flow analysis...")
    sample_batch = next(iter(train_loader))
    nll_gradients = analyze_gradient_flow(fourier_model_nll, sample_batch)

    # Visualize
    visualize_predictions(
        fourier_model_nll, val_loader,
        str(output_dir / 'nll_predictions.png')
    )
    plot_training_diagnostics(nll_diagnostics, str(output_dir / 'nll_diagnostics.png'))

    results['fourier_nll'] = {
        'diagnostics': nll_diagnostics,
        'ablation': nll_ablation,
        'metrics': nll_metrics,
    }

    # =========================================================================
    # STEP 4: Train with Soft Target Loss
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING FOURIER HEAD MODEL (Soft Target Loss)")
    print("=" * 80)

    fourier_model_soft = TrajectoryForecastModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    soft_diagnostics = train_fourier_model(
        fourier_model_soft, train_loader, val_loader, config, loss_type='soft'
    )

    print("\n  Running input ablation test...")
    soft_ablation = test_input_ablation(fourier_model_soft, val_loader)

    print("\n  Computing prediction metrics...")
    soft_metrics = compute_prediction_metrics(fourier_model_soft, val_loader)
    print(f"    Expected value error: {soft_metrics['expected_error']:.4f}")
    print(f"    Mode error: {soft_metrics['mode_error']:.4f}")
    print(f"    Pred magnitude mean: {soft_metrics['pred_mag_mean']:.4f}")
    print(f"    Target magnitude mean: {soft_metrics['target_mag_mean']:.4f}")

    visualize_predictions(
        fourier_model_soft, val_loader,
        str(output_dir / 'soft_predictions.png')
    )
    plot_training_diagnostics(soft_diagnostics, str(output_dir / 'soft_diagnostics.png'))

    results['fourier_soft'] = {
        'diagnostics': soft_diagnostics,
        'ablation': soft_ablation,
        'metrics': soft_metrics,
    }

    # =========================================================================
    # STEP 5: Train with Curriculum Learning + Soft Targets
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING WITH CURRICULUM LEARNING")
    print("=" * 80)

    fourier_model_curriculum = TrajectoryForecastModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    fourier_model_curriculum = curriculum_training(fourier_model_curriculum, config, loss_type='soft')

    print("\n  Running input ablation test...")
    curriculum_ablation = test_input_ablation(fourier_model_curriculum, val_loader)

    print("\n  Computing prediction metrics...")
    curriculum_metrics = compute_prediction_metrics(fourier_model_curriculum, val_loader)
    print(f"    Expected value error: {curriculum_metrics['expected_error']:.4f}")
    print(f"    Mode error: {curriculum_metrics['mode_error']:.4f}")
    print(f"    Pred magnitude mean: {curriculum_metrics['pred_mag_mean']:.4f}")
    print(f"    Target magnitude mean: {curriculum_metrics['target_mag_mean']:.4f}")

    visualize_predictions(
        fourier_model_curriculum, val_loader,
        str(output_dir / 'curriculum_predictions.png')
    )

    results['fourier_curriculum'] = {
        'ablation': curriculum_ablation,
        'metrics': curriculum_metrics,
    }

    # =========================================================================
    # STEP 6: Train with Mixed Loss
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING WITH MIXED LOSS (NLL + Mode Loss)")
    print("=" * 80)

    fourier_model_mixed = TrajectoryForecastModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        grid_size=config['grid_size'],
        num_freqs=config['num_freqs'],
        grid_range=config['grid_range']
    )

    mixed_diagnostics = train_fourier_model(
        fourier_model_mixed, train_loader, val_loader, config, loss_type='mixed'
    )

    print("\n  Running input ablation test...")
    mixed_ablation = test_input_ablation(fourier_model_mixed, val_loader)

    print("\n  Computing prediction metrics...")
    mixed_metrics = compute_prediction_metrics(fourier_model_mixed, val_loader)
    print(f"    Expected value error: {mixed_metrics['expected_error']:.4f}")
    print(f"    Mode error: {mixed_metrics['mode_error']:.4f}")
    print(f"    Pred magnitude mean: {mixed_metrics['pred_mag_mean']:.4f}")
    print(f"    Target magnitude mean: {mixed_metrics['target_mag_mean']:.4f}")

    visualize_predictions(
        fourier_model_mixed, val_loader,
        str(output_dir / 'mixed_predictions.png')
    )
    plot_training_diagnostics(mixed_diagnostics, str(output_dir / 'mixed_diagnostics.png'))

    results['fourier_mixed'] = {
        'diagnostics': mixed_diagnostics,
        'ablation': mixed_ablation,
        'metrics': mixed_metrics,
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print("\n  REGRESSION BASELINE (Direct MSE):")
    print(f"    Prediction Error: {results['regression_baseline']['pred_error']:.4f}")
    print(f"    Status: {'PASS' if results['regression_baseline']['pred_error'] < 0.1 else 'FAIL'}")

    print("\n  FOURIER HEAD MODELS (Expected Error | Mode Error | Uses Input?):")

    for name, key in [('Standard NLL', 'fourier_nll'),
                      ('Soft Targets', 'fourier_soft'),
                      ('Curriculum', 'fourier_curriculum'),
                      ('Mixed Loss', 'fourier_mixed')]:
        m = results[key]['metrics']
        a = results[key]['ablation']
        uses_input = "YES" if a['l1_real_vs_zero'] > 0.001 else "NO"
        print(f"    {name:15s}: {m['expected_error']:.4f} | {m['mode_error']:.4f} | {uses_input}")

    # Determine which approach works best
    best_approach = min(
        [('fourier_nll', results['fourier_nll']),
         ('fourier_soft', results['fourier_soft']),
         ('fourier_curriculum', results['fourier_curriculum']),
         ('fourier_mixed', results['fourier_mixed'])],
        key=lambda x: x[1]['metrics']['expected_error']
    )

    print(f"\n  BEST FOURIER APPROACH: {best_approach[0]}")
    print(f"    Expected Error: {best_approach[1]['metrics']['expected_error']:.4f}")

    # Diagnosis
    print("\n  DIAGNOSIS:")

    reg_works = results['regression_baseline']['pred_error'] < 0.1
    best_fourier_works = best_approach[1]['metrics']['expected_error'] < 0.5

    if not reg_works:
        print("    The transformer backbone cannot extract velocity.")
        print("    RECOMMENDATION: Debug embedding and positional encoding.")
    elif not best_fourier_works:
        print("    The transformer works but Fourier head fails to learn.")
        print("    LIKELY CAUSE: Optimization issues with Fourier parameterization.")
        print("    RECOMMENDATION: Try higher learning rate, different initialization,")
        print("                    or consider alternative density representations.")
    else:
        print("    The model successfully learns dead reckoning!")
        print(f"    Best approach: {best_approach[0]}")

    # Save results
    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    serializable_results = make_serializable(results)

    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_dir}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = main()
