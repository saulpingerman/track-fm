#!/usr/bin/env python3
"""
Systematic diagnosis of why multi-horizon causal model collapses to centered predictions.

Hypotheses to test:
A. Single horizon works, multi-horizon breaks
B. Encoder (bidirectional) works, decoder (causal) breaks
C. Time conditioning is the problem
D. Loss function encourages collapse
E. Gradient flow issues
F. Target distribution problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

from run_experiment import (
    VariableSpacedDataset,
    FourierHead2D,
    SinusoidalTimeEncoding,
    compute_soft_target_loss
)

RESULTS_DIR = Path(__file__).parent / 'diagnosis_results'
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Simplified models for testing
# ============================================================================

class SimpleEncoderModel(nn.Module):
    """Encoder (bidirectional) - no causal mask."""
    def __init__(self, d_model=64, nhead=4, num_layers=2, grid_size=64,
                 num_freqs=8, grid_range=40.0):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

    def forward(self, positions, dt_values):
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        x = self.transformer(x)  # No mask = bidirectional
        last_emb = x[:, -1, :]
        return self.fourier_head(last_emb)


class SimpleDecoderModel(nn.Module):
    """Decoder (causal) - with causal mask."""
    def __init__(self, d_model=64, nhead=4, num_layers=2, grid_size=64,
                 num_freqs=8, grid_range=40.0):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

    def forward(self, positions, dt_values):
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)  # Causal mask
        last_emb = x[:, -1, :]
        return self.fourier_head(last_emb)


class MultiHorizonDecoderModel(nn.Module):
    """Decoder with multi-horizon and time conditioning."""
    def __init__(self, d_model=64, nhead=4, num_layers=2, grid_size=64,
                 num_freqs=8, grid_range=40.0):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.time_encoder = SinusoidalTimeEncoding(d_model)
        self.horizon_proj = nn.Linear(d_model * 2, d_model)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

    def forward(self, positions, dt_values, query_dt):
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        last_emb = x[:, -1, :]

        # Time conditioning
        time_enc = self.time_encoder(query_dt)
        combined = torch.cat([last_emb, time_enc], dim=-1)
        conditioned = self.horizon_proj(combined)

        return self.fourier_head(conditioned)


class MultiHorizonNoTimeModel(nn.Module):
    """Decoder with multi-horizon but NO time conditioning (control)."""
    def __init__(self, d_model=64, nhead=4, num_layers=2, grid_size=64,
                 num_freqs=8, grid_range=40.0):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)

    def forward(self, positions, dt_values, query_dt=None):
        # query_dt is ignored - no time conditioning
        x = torch.cat([positions, dt_values.unsqueeze(-1)], dim=-1)
        x = self.input_proj(x)
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        last_emb = x[:, -1, :]
        return self.fourier_head(last_emb)


# ============================================================================
# Training utilities
# ============================================================================

def train_single_horizon(model, train_loader, val_loader, epochs=20, grid_range=40.0,
                         grid_size=64, sigma=1.5, horizon=1):
    """Train model to predict a fixed horizon ahead."""
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for positions, dt_values in train_loader:
            positions = positions.to(DEVICE)
            dt_values = dt_values.to(DEVICE)

            batch_size, seq_len, _ = positions.shape

            # Target: horizon steps ahead from last observed position
            # Using positions up to seq_len - horizon, predicting position at seq_len - horizon + horizon
            src_idx = seq_len - horizon - 1
            tgt_idx = seq_len - 1

            input_pos = positions[:, :src_idx+1, :]
            input_dt = dt_values[:, :src_idx+1]

            target_relative = positions[:, tgt_idx, :] - positions[:, src_idx, :]

            optimizer.zero_grad()

            # Check if model takes query_dt
            if hasattr(model, 'time_encoder'):
                # Compute cumulative dt from src to tgt
                cum_dt = dt_values[:, src_idx+1:tgt_idx+1].sum(dim=1)
                log_dens = model(input_pos, input_dt, cum_dt)
            else:
                log_dens = model(input_pos, input_dt)

            # Soft target loss
            log_dens = log_dens.unsqueeze(1)  # (batch, 1, gs, gs)
            target = target_relative.unsqueeze(1)  # (batch, 1, 2)
            loss = compute_soft_target_loss(log_dens, target, grid_range, grid_size, sigma)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        history.append(epoch_loss / len(train_loader))
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={history[-1]:.4f}", flush=True)

    return history


def train_multi_horizon(model, train_loader, val_loader, epochs=20, grid_range=40.0,
                        grid_size=64, sigma=1.5, max_horizon=5):
    """Train model with multiple horizons per position."""
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for positions, dt_values in train_loader:
            positions = positions.to(DEVICE)
            dt_values = dt_values.to(DEVICE)

            batch_size, seq_len, _ = positions.shape
            cumsum_dt = torch.cumsum(dt_values, dim=1)

            # Collect all (input, target, cum_dt) pairs
            all_log_dens = []
            all_targets = []

            for src_idx in range(seq_len - 1):
                input_pos = positions[:, :src_idx+1, :]
                input_dt = dt_values[:, :src_idx+1]

                for h in range(1, min(max_horizon + 1, seq_len - src_idx)):
                    tgt_idx = src_idx + h
                    target_rel = positions[:, tgt_idx, :] - positions[:, src_idx, :]
                    cum_dt = cumsum_dt[:, tgt_idx] - cumsum_dt[:, src_idx]

                    if hasattr(model, 'time_encoder'):
                        log_dens = model(input_pos, input_dt, cum_dt)
                    else:
                        log_dens = model(input_pos, input_dt, None)

                    all_log_dens.append(log_dens)
                    all_targets.append(target_rel)

            if not all_log_dens:
                continue

            # Stack and compute loss
            all_log_dens = torch.stack(all_log_dens, dim=1)  # (batch, num_pairs, gs, gs)
            all_targets = torch.stack(all_targets, dim=1)  # (batch, num_pairs, 2)

            optimizer.zero_grad()
            loss = compute_soft_target_loss(all_log_dens, all_targets, grid_range, grid_size, sigma)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        history.append(epoch_loss / max(num_batches, 1))
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={history[-1]:.4f}", flush=True)

    return history


def evaluate_model(model, val_loader, grid_range=40.0, grid_size=64, horizon=1):
    """Evaluate model and return mean error."""
    model.eval()
    grid = torch.linspace(-grid_range, grid_range, grid_size, device=DEVICE)

    errors = []
    predictions = []
    targets_list = []

    with torch.no_grad():
        for positions, dt_values in val_loader:
            positions = positions.to(DEVICE)
            dt_values = dt_values.to(DEVICE)

            batch_size, seq_len, _ = positions.shape

            src_idx = seq_len - horizon - 1
            tgt_idx = seq_len - 1

            input_pos = positions[:, :src_idx+1, :]
            input_dt = dt_values[:, :src_idx+1]
            target_rel = positions[:, tgt_idx, :] - positions[:, src_idx, :]

            if hasattr(model, 'time_encoder'):
                cum_dt = dt_values[:, src_idx+1:tgt_idx+1].sum(dim=1)
                log_dens = model(input_pos, input_dt, cum_dt)
            else:
                log_dens = model(input_pos, input_dt)

            dens = torch.exp(log_dens)

            # Expected value
            exp_x = (dens.sum(dim=2) * grid).sum(dim=1)
            exp_y = (dens.sum(dim=1) * grid).sum(dim=1)

            error = torch.sqrt((exp_x - target_rel[:, 0])**2 + (exp_y - target_rel[:, 1])**2)
            errors.extend(error.cpu().numpy().tolist())
            predictions.extend(list(zip(exp_x.cpu().numpy(), exp_y.cpu().numpy())))
            targets_list.extend(target_rel.cpu().numpy().tolist())

    return np.mean(errors), predictions, targets_list


# ============================================================================
# Diagnostic tests
# ============================================================================

def test_A_single_vs_multi_horizon():
    """Test if single horizon works but multi-horizon breaks."""
    print("\n" + "="*70, flush=True)
    print("TEST A: Single Horizon vs Multi-Horizon", flush=True)
    print("="*70, flush=True)

    train_data = VariableSpacedDataset(1000, seq_len=15, velocity_range=(0.5, 4.5))
    val_data = VariableSpacedDataset(200, seq_len=15, velocity_range=(0.5, 4.5))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    results = {}

    # Single horizon (h=1)
    print("\nA1: Decoder with single horizon (h=1):", flush=True)
    model_single = SimpleDecoderModel(grid_range=40.0)
    train_single_horizon(model_single, train_loader, val_loader, epochs=15, horizon=1)
    error, preds, tgts = evaluate_model(model_single, val_loader, horizon=1)
    results['single_h1'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error: {error:.4f}", flush=True)
    print(f"  Sample predictions: {preds[:3]}", flush=True)
    print(f"  Sample targets: {tgts[:3]}", flush=True)

    # Single horizon (h=3)
    print("\nA2: Decoder with single horizon (h=3):", flush=True)
    model_single3 = SimpleDecoderModel(grid_range=40.0)
    train_single_horizon(model_single3, train_loader, val_loader, epochs=15, horizon=3)
    error, preds, tgts = evaluate_model(model_single3, val_loader, horizon=3)
    results['single_h3'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error: {error:.4f}", flush=True)
    print(f"  Sample predictions: {preds[:3]}", flush=True)
    print(f"  Sample targets: {tgts[:3]}", flush=True)

    # Multi-horizon with time conditioning
    print("\nA3: Decoder with multi-horizon + time conditioning:", flush=True)
    model_multi = MultiHorizonDecoderModel(grid_range=40.0)
    train_multi_horizon(model_multi, train_loader, val_loader, epochs=15, max_horizon=3)
    error, preds, tgts = evaluate_model(model_multi, val_loader, horizon=1)
    results['multi_h1'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error (h=1): {error:.4f}", flush=True)
    error3, preds3, tgts3 = evaluate_model(model_multi, val_loader, horizon=3)
    results['multi_h3'] = {'error': error3, 'preds': preds3[:10], 'tgts': tgts3[:10]}
    print(f"  Error (h=3): {error3:.4f}", flush=True)
    print(f"  Sample predictions (h=1): {preds[:3]}", flush=True)
    print(f"  Sample targets (h=1): {tgts[:3]}", flush=True)

    return results


def test_B_encoder_vs_decoder():
    """Test if encoder works but decoder breaks."""
    print("\n" + "="*70, flush=True)
    print("TEST B: Encoder (Bidirectional) vs Decoder (Causal)", flush=True)
    print("="*70, flush=True)

    train_data = VariableSpacedDataset(1000, seq_len=15, velocity_range=(0.5, 4.5))
    val_data = VariableSpacedDataset(200, seq_len=15, velocity_range=(0.5, 4.5))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    results = {}

    # Encoder
    print("\nB1: Encoder (bidirectional attention):", flush=True)
    model_enc = SimpleEncoderModel(grid_range=40.0)
    train_single_horizon(model_enc, train_loader, val_loader, epochs=15, horizon=1)
    error, preds, tgts = evaluate_model(model_enc, val_loader, horizon=1)
    results['encoder'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error: {error:.4f}", flush=True)
    print(f"  Sample predictions: {preds[:3]}", flush=True)

    # Decoder
    print("\nB2: Decoder (causal attention):", flush=True)
    model_dec = SimpleDecoderModel(grid_range=40.0)
    train_single_horizon(model_dec, train_loader, val_loader, epochs=15, horizon=1)
    error, preds, tgts = evaluate_model(model_dec, val_loader, horizon=1)
    results['decoder'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error: {error:.4f}", flush=True)
    print(f"  Sample predictions: {preds[:3]}", flush=True)

    return results


def test_C_time_conditioning():
    """Test if time conditioning is the problem."""
    print("\n" + "="*70, flush=True)
    print("TEST C: With vs Without Time Conditioning", flush=True)
    print("="*70, flush=True)

    train_data = VariableSpacedDataset(1000, seq_len=15, velocity_range=(0.5, 4.5))
    val_data = VariableSpacedDataset(200, seq_len=15, velocity_range=(0.5, 4.5))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    results = {}

    # With time conditioning
    print("\nC1: Multi-horizon WITH time conditioning:", flush=True)
    model_with = MultiHorizonDecoderModel(grid_range=40.0)
    train_multi_horizon(model_with, train_loader, val_loader, epochs=15, max_horizon=3)
    error, preds, tgts = evaluate_model(model_with, val_loader, horizon=1)
    results['with_time'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error: {error:.4f}", flush=True)
    print(f"  Sample predictions: {preds[:3]}", flush=True)

    # Without time conditioning
    print("\nC2: Multi-horizon WITHOUT time conditioning:", flush=True)
    model_without = MultiHorizonNoTimeModel(grid_range=40.0)
    train_multi_horizon(model_without, train_loader, val_loader, epochs=15, max_horizon=3)
    error, preds, tgts = evaluate_model(model_without, val_loader, horizon=1)
    results['without_time'] = {'error': error, 'preds': preds[:10], 'tgts': tgts[:10]}
    print(f"  Error: {error:.4f}", flush=True)
    print(f"  Sample predictions: {preds[:3]}", flush=True)

    return results


def test_D_target_distribution():
    """Analyze the distribution of targets."""
    print("\n" + "="*70, flush=True)
    print("TEST D: Target Distribution Analysis", flush=True)
    print("="*70, flush=True)

    train_data = VariableSpacedDataset(1000, seq_len=15, velocity_range=(0.5, 4.5))

    all_targets = []
    for positions, dt_values in DataLoader(train_data, batch_size=len(train_data)):
        seq_len = positions.shape[1]
        for src_idx in range(seq_len - 1):
            for h in range(1, min(6, seq_len - src_idx)):
                tgt_idx = src_idx + h
                target_rel = positions[:, tgt_idx, :] - positions[:, src_idx, :]
                all_targets.append(target_rel.numpy())

    all_targets = np.concatenate(all_targets, axis=0)

    print(f"\nTarget statistics:", flush=True)
    print(f"  Mean: ({all_targets[:, 0].mean():.2f}, {all_targets[:, 1].mean():.2f})", flush=True)
    print(f"  Std: ({all_targets[:, 0].std():.2f}, {all_targets[:, 1].std():.2f})", flush=True)
    print(f"  Min: ({all_targets[:, 0].min():.2f}, {all_targets[:, 1].min():.2f})", flush=True)
    print(f"  Max: ({all_targets[:, 0].max():.2f}, {all_targets[:, 1].max():.2f})", flush=True)

    # What fraction of targets are near origin?
    distances = np.sqrt(all_targets[:, 0]**2 + all_targets[:, 1]**2)
    print(f"\n  Distance from origin:", flush=True)
    print(f"    Mean: {distances.mean():.2f}", flush=True)
    print(f"    < 1.0: {(distances < 1.0).mean()*100:.1f}%", flush=True)
    print(f"    < 5.0: {(distances < 5.0).mean()*100:.1f}%", flush=True)
    print(f"    < 10.0: {(distances < 10.0).mean()*100:.1f}%", flush=True)

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist2d(all_targets[:, 0], all_targets[:, 1], bins=50, cmap='hot')
    axes[0].set_xlabel('Target X')
    axes[0].set_ylabel('Target Y')
    axes[0].set_title('Target Distribution (all horizons)')

    axes[1].hist(distances, bins=50)
    axes[1].set_xlabel('Distance from Origin')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Target Distance Distribution')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'target_distribution.png', dpi=150)
    print(f"\n  Saved: {RESULTS_DIR / 'target_distribution.png'}", flush=True)

    return {'mean': all_targets.mean(axis=0), 'std': all_targets.std(axis=0)}


def test_E_gradient_analysis():
    """Analyze gradients during training."""
    print("\n" + "="*70, flush=True)
    print("TEST E: Gradient Flow Analysis", flush=True)
    print("="*70, flush=True)

    train_data = VariableSpacedDataset(100, seq_len=15, velocity_range=(0.5, 4.5))
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model = MultiHorizonDecoderModel(grid_range=40.0).to(DEVICE)

    positions, dt_values = next(iter(train_loader))
    positions = positions.to(DEVICE)
    dt_values = dt_values.to(DEVICE)

    # Forward pass
    batch_size, seq_len, _ = positions.shape
    cumsum_dt = torch.cumsum(dt_values, dim=1)

    all_log_dens = []
    all_targets = []

    for src_idx in range(seq_len - 1):
        input_pos = positions[:, :src_idx+1, :]
        input_dt = dt_values[:, :src_idx+1]

        for h in range(1, min(4, seq_len - src_idx)):
            tgt_idx = src_idx + h
            target_rel = positions[:, tgt_idx, :] - positions[:, src_idx, :]
            cum_dt = cumsum_dt[:, tgt_idx] - cumsum_dt[:, src_idx]

            log_dens = model(input_pos, input_dt, cum_dt)
            all_log_dens.append(log_dens)
            all_targets.append(target_rel)

    all_log_dens = torch.stack(all_log_dens, dim=1)
    all_targets = torch.stack(all_targets, dim=1)

    loss = compute_soft_target_loss(all_log_dens, all_targets, 40.0, 64, 1.5)
    loss.backward()

    print("\nGradient norms by component:", flush=True)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: {grad_norm:.6f}", flush=True)

    # Check if time encoder gradients are flowing
    time_grad = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if 'time_encoder' in n and p.grad is not None)
    horizon_grad = sum(p.grad.norm().item() for n, p in model.named_parameters()
                       if 'horizon_proj' in n and p.grad is not None)

    print(f"\n  Total time_encoder gradient: {time_grad:.6f}", flush=True)
    print(f"  Total horizon_proj gradient: {horizon_grad:.6f}", flush=True)

    return {'time_grad': time_grad, 'horizon_grad': horizon_grad}


def test_F_prediction_variance():
    """Check if model outputs vary with input or are constant."""
    print("\n" + "="*70, flush=True)
    print("TEST F: Prediction Variance Analysis", flush=True)
    print("="*70, flush=True)

    model = MultiHorizonDecoderModel(grid_range=40.0).to(DEVICE)

    # Don't train - check untrained model first
    print("\nF1: Untrained model predictions:", flush=True)

    grid = torch.linspace(-40, 40, 64, device=DEVICE)

    preds = []
    for _ in range(10):
        positions = torch.randn(1, 10, 2, device=DEVICE) * 10
        dt_values = torch.abs(torch.randn(1, 10, device=DEVICE)) + 0.5
        dt_values[0, 0] = 0
        query_dt = torch.tensor([2.0], device=DEVICE)

        with torch.no_grad():
            log_dens = model(positions, dt_values, query_dt)
            dens = torch.exp(log_dens[0])
            exp_x = (dens.sum(dim=1) * grid).sum().item()
            exp_y = (dens.sum(dim=0) * grid).sum().item()
            preds.append((exp_x, exp_y))

    preds = np.array(preds)
    print(f"  Prediction variance: x={preds[:, 0].var():.4f}, y={preds[:, 1].var():.4f}", flush=True)
    print(f"  Prediction range: x=[{preds[:, 0].min():.2f}, {preds[:, 0].max():.2f}]", flush=True)

    # Now train and check again
    print("\nF2: After 10 epochs of training:", flush=True)
    train_data = VariableSpacedDataset(500, seq_len=15, velocity_range=(0.5, 4.5))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(10):
        model.train()
        for positions, dt_values in train_loader:
            positions = positions.to(DEVICE)
            dt_values = dt_values.to(DEVICE)

            batch_size, seq_len, _ = positions.shape
            cumsum_dt = torch.cumsum(dt_values, dim=1)

            all_log_dens = []
            all_targets = []

            for src_idx in range(seq_len - 1):
                input_pos = positions[:, :src_idx+1, :]
                input_dt = dt_values[:, :src_idx+1]

                for h in range(1, min(4, seq_len - src_idx)):
                    tgt_idx = src_idx + h
                    target_rel = positions[:, tgt_idx, :] - positions[:, src_idx, :]
                    cum_dt = cumsum_dt[:, tgt_idx] - cumsum_dt[:, src_idx]

                    log_dens = model(input_pos, input_dt, cum_dt)
                    all_log_dens.append(log_dens)
                    all_targets.append(target_rel)

            if not all_log_dens:
                continue

            all_log_dens = torch.stack(all_log_dens, dim=1)
            all_targets = torch.stack(all_targets, dim=1)

            optimizer.zero_grad()
            loss = compute_soft_target_loss(all_log_dens, all_targets, 40.0, 64, 1.5)
            loss.backward()
            optimizer.step()

    model.eval()
    preds_trained = []
    for _ in range(10):
        positions = torch.randn(1, 10, 2, device=DEVICE) * 10
        dt_values = torch.abs(torch.randn(1, 10, device=DEVICE)) + 0.5
        dt_values[0, 0] = 0
        query_dt = torch.tensor([2.0], device=DEVICE)

        with torch.no_grad():
            log_dens = model(positions, dt_values, query_dt)
            dens = torch.exp(log_dens[0])
            exp_x = (dens.sum(dim=1) * grid).sum().item()
            exp_y = (dens.sum(dim=0) * grid).sum().item()
            preds_trained.append((exp_x, exp_y))

    preds_trained = np.array(preds_trained)
    print(f"  Prediction variance: x={preds_trained[:, 0].var():.4f}, y={preds_trained[:, 1].var():.4f}", flush=True)
    print(f"  Prediction range: x=[{preds_trained[:, 0].min():.2f}, {preds_trained[:, 0].max():.2f}]", flush=True)
    print(f"  Mean prediction: ({preds_trained[:, 0].mean():.2f}, {preds_trained[:, 1].mean():.2f})", flush=True)

    return {'untrained_var': preds.var(), 'trained_var': preds_trained.var()}


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70, flush=True)
    print("DIAGNOSIS: Multi-Horizon Causal Prediction Collapse", flush=True)
    print("="*70, flush=True)

    all_results = {}

    # Run all tests
    all_results['A'] = test_A_single_vs_multi_horizon()
    all_results['B'] = test_B_encoder_vs_decoder()
    all_results['C'] = test_C_time_conditioning()
    all_results['D'] = test_D_target_distribution()
    all_results['E'] = test_E_gradient_analysis()
    all_results['F'] = test_F_prediction_variance()

    # Summary
    print("\n" + "="*70, flush=True)
    print("SUMMARY", flush=True)
    print("="*70, flush=True)

    print("\nTest A (Single vs Multi-Horizon):", flush=True)
    print(f"  Single h=1 error: {all_results['A']['single_h1']['error']:.4f}", flush=True)
    print(f"  Single h=3 error: {all_results['A']['single_h3']['error']:.4f}", flush=True)
    print(f"  Multi h=1 error: {all_results['A']['multi_h1']['error']:.4f}", flush=True)
    print(f"  Multi h=3 error: {all_results['A']['multi_h3']['error']:.4f}", flush=True)

    print("\nTest B (Encoder vs Decoder):", flush=True)
    print(f"  Encoder error: {all_results['B']['encoder']['error']:.4f}", flush=True)
    print(f"  Decoder error: {all_results['B']['decoder']['error']:.4f}", flush=True)

    print("\nTest C (Time Conditioning):", flush=True)
    print(f"  With time error: {all_results['C']['with_time']['error']:.4f}", flush=True)
    print(f"  Without time error: {all_results['C']['without_time']['error']:.4f}", flush=True)

    print("\nTest E (Gradient Flow):", flush=True)
    print(f"  Time encoder grad: {all_results['E']['time_grad']:.6f}", flush=True)
    print(f"  Horizon proj grad: {all_results['E']['horizon_grad']:.6f}", flush=True)

    print("\nTest F (Prediction Variance):", flush=True)
    print(f"  Untrained variance: {all_results['F']['untrained_var']:.4f}", flush=True)
    print(f"  Trained variance: {all_results['F']['trained_var']:.4f}", flush=True)

    print("\n" + "="*70, flush=True)
    print("DIAGNOSIS COMPLETE - Results saved to:", RESULTS_DIR, flush=True)
    print("="*70, flush=True)

    return all_results


if __name__ == '__main__':
    main()
