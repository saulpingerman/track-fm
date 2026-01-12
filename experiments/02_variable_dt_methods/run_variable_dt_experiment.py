#!/usr/bin/env python3
"""
Variable Time Horizon Experiment

Tests different methods of incorporating Δt (prediction time horizon) into
a transformer + 2D Fourier head model for trajectory prediction.
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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# Data Generation with Variable Δt
# ============================================================================

def generate_straight_line_trajectory(seq_len: int, max_pred_horizon: float,
                                      velocity: float, angle: float) -> np.ndarray:
    """Generate positions at integer timesteps plus extended range for variable Δt."""
    vx = velocity * np.cos(angle)
    vy = velocity * np.sin(angle)

    # Generate more points than needed for interpolation
    times = np.arange(seq_len + int(max_pred_horizon) + 1)
    x = vx * times
    y = vy * times

    return np.stack([x, y], axis=-1), (vx, vy)


class VariableDtDataset(Dataset):
    """Dataset with variable prediction time horizon."""

    def __init__(self, num_samples: int, seq_len: int,
                 velocity_range: Tuple[float, float] = (0.1, 4.5),
                 dt_range: Tuple[float, float] = (0.5, 4.0),
                 noise_std: float = 0.0):
        self.samples = []

        for _ in range(num_samples):
            velocity = np.random.uniform(*velocity_range)
            angle = np.random.uniform(0, 2 * np.pi)
            dt = np.random.uniform(*dt_range)

            traj, (vx, vy) = generate_straight_line_trajectory(
                seq_len, dt_range[1], velocity, angle
            )

            # Input: first seq_len positions, relative to last
            history = traj[:seq_len]
            last_pos = history[-1]
            input_seq = history - last_pos

            # Target: position at time (seq_len - 1) + dt, relative to last
            # For constant velocity: target = (vx * dt, vy * dt)
            target = np.array([vx * dt, vy * dt])

            if noise_std > 0:
                input_seq += np.random.normal(0, noise_std, input_seq.shape)

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
                 num_freqs: int = 8, grid_range: float = 5.0):
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

        logits = (
            torch.einsum('bf,gf->bg', cos_coeffs, cos_basis) +
            torch.einsum('bf,gf->bg', sin_coeffs, sin_basis)
        )

        log_density = F.log_softmax(logits, dim=-1)
        return log_density.view(batch_size, self.grid_size, self.grid_size)

    def get_grid_coordinates(self):
        x = torch.linspace(-self.grid_range, self.grid_range, self.grid_size)
        y = torch.linspace(-self.grid_range, self.grid_range, self.grid_size)
        return x, y


# ============================================================================
# Method 1: Concat Δt to Transformer Output (Your Current Approach)
# ============================================================================

class Method1_ConcatToOutput(nn.Module):
    """Embed Δt, concatenate to transformer output, MLP, then Fourier head."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=5.0):
        super().__init__()
        self.d_model = d_model

        # Trajectory encoder
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Δt embedding
        self.dt_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Fusion MLP (transformer output + dt embedding)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dt):
        # x: (batch, seq_len, 2), dt: (batch, 1)
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]  # Last token

        dt_emb = self.dt_embed(dt)

        fused = torch.cat([z, dt_emb], dim=-1)
        fused = self.fusion_mlp(fused)

        return self.fourier_head(fused)


# ============================================================================
# Method 2: FiLM Conditioning
# ============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation."""
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.scale = nn.Linear(d_cond, d_model)
        self.shift = nn.Linear(d_cond, d_model)

    def forward(self, x, cond):
        return x * (1 + self.scale(cond)) + self.shift(cond)


class Method2_FiLM(nn.Module):
    """Use Δt to modulate transformer features via FiLM."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=5.0):
        super().__init__()
        self.d_model = d_model

        # Trajectory encoder
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Δt embedding
        self.dt_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # FiLM modulation
        self.film = FiLMLayer(d_model, d_model)

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dt):
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]

        dt_emb = self.dt_embed(dt)
        z = self.film(z, dt_emb)

        return self.fourier_head(z)


# ============================================================================
# Method 3: Concat Δt to Each Input Position
# ============================================================================

class Method3_ConcatToInput(nn.Module):
    """Concatenate Δt to each (x,y) position in input sequence."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=5.0):
        super().__init__()
        self.d_model = d_model

        # Input now has 3 dimensions: (x, y, dt)
        self.input_proj = nn.Linear(3, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dt):
        # x: (batch, seq_len, 2), dt: (batch, 1)
        batch_size, seq_len, _ = x.shape

        # Expand dt to all positions
        dt_expanded = dt.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, 1)
        x_with_dt = torch.cat([x, dt_expanded], dim=-1)  # (batch, seq_len, 3)

        z = self.input_proj(x_with_dt)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]

        return self.fourier_head(z)


# ============================================================================
# Method 4: Δt as Additional Token
# ============================================================================

class Method4_DtToken(nn.Module):
    """Append Δt as a special query token to the sequence."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=5.0):
        super().__init__()
        self.d_model = d_model

        # Position encoder
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=120)

        # Δt token embedding
        self.dt_token_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dt):
        # x: (batch, seq_len, 2), dt: (batch, 1)
        batch_size, seq_len, _ = x.shape

        # Embed positions
        pos_emb = self.input_proj(x)  # (batch, seq_len, d_model)

        # Embed dt as a token
        dt_token = self.dt_token_proj(dt).unsqueeze(1)  # (batch, 1, d_model)

        # Concatenate: [pos_tokens, dt_token]
        tokens = torch.cat([pos_emb, dt_token], dim=1)  # (batch, seq_len+1, d_model)

        tokens = self.pos_encoding(tokens)
        tokens = self.transformer(tokens)

        # Take the last token (dt query result)
        z = tokens[:, -1, :]

        return self.fourier_head(z)


# ============================================================================
# Method 5: Cross-Attention
# ============================================================================

class Method5_CrossAttention(nn.Module):
    """Trajectory encoder output attends to Δt via cross-attention."""

    def __init__(self, d_model=64, nhead=4, num_layers=2,
                 grid_size=64, num_freqs=8, grid_range=5.0):
        super().__init__()
        self.d_model = d_model

        # Trajectory encoder
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Δt embedding (key/value)
        self.dt_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Cross-attention: trajectory queries, dt is key/value
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        # Fourier head
        self.fourier_head = FourierHead2D(d_model, grid_size, num_freqs, grid_range)
        self.grid_size = grid_size
        self.grid_range = grid_range

    def forward(self, x, dt):
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1:, :]  # Keep as (batch, 1, d_model) for attention

        dt_emb = self.dt_embed(dt).unsqueeze(1)  # (batch, 1, d_model)

        # Cross attention: trajectory queries dt
        attn_out, _ = self.cross_attn(z, dt_emb, dt_emb)
        z = self.norm(z + attn_out)
        z = z.squeeze(1)

        return self.fourier_head(z)


# ============================================================================
# Direct Regression Baseline
# ============================================================================

class DirectRegressionWithDt(nn.Module):
    """Direct regression baseline for comparison."""

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dt_embed = nn.Linear(1, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, x, dt):
        z = self.input_proj(x)
        z = self.pos_encoding(z)
        z = self.transformer(z)
        z = z[:, -1, :]

        dt_emb = self.dt_embed(dt)
        combined = torch.cat([z, dt_emb], dim=-1)
        return self.head(combined)


# ============================================================================
# Training and Evaluation
# ============================================================================

def compute_nll_loss(log_density, target, grid_size, grid_range):
    """Compute NLL loss."""
    batch_size = target.shape[0]

    grid_x = torch.linspace(-grid_range, grid_range, grid_size, device=target.device)
    grid_y = torch.linspace(-grid_range, grid_range, grid_size, device=target.device)

    x_idx = torch.argmin(torch.abs(grid_x.unsqueeze(0) - target[:, 0:1]), dim=1)
    y_idx = torch.argmin(torch.abs(grid_y.unsqueeze(0) - target[:, 1:2]), dim=1)

    x_idx = torch.clamp(x_idx, 0, grid_size - 1)
    y_idx = torch.clamp(y_idx, 0, grid_size - 1)

    log_prob = log_density[torch.arange(batch_size, device=target.device), x_idx, y_idx]
    return -log_prob.mean()


def compute_soft_nll_loss(log_density, target, grid_size, grid_range, sigma=0.5):
    """Soft target NLL with Gaussian kernel."""
    grid_x = torch.linspace(-grid_range, grid_range, grid_size, device=target.device)
    grid_y = torch.linspace(-grid_range, grid_range, grid_size, device=target.device)
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')

    dx = xx.unsqueeze(0) - target[:, 0:1, None]
    dy = yy.unsqueeze(0) - target[:, 1:2, None]

    soft_target = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    soft_target = soft_target / soft_target.sum(dim=(1, 2), keepdim=True)

    return F.kl_div(log_density, soft_target, reduction='batchmean')


def train_model(model, train_loader, val_loader, config, use_soft_loss=True):
    """Train a model and return diagnostics."""
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])

    diagnostics = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for input_seq, dt, target in train_loader:
            input_seq = input_seq.to(DEVICE)
            dt = dt.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            log_density = model(input_seq, dt)

            if use_soft_loss:
                loss = compute_soft_nll_loss(log_density, target,
                                             config['grid_size'], config['grid_range'], sigma=0.5)
            else:
                loss = compute_nll_loss(log_density, target,
                                        config['grid_size'], config['grid_range'])

            # Skip if NaN
            if torch.isnan(loss):
                continue

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, dt, target in val_loader:
                input_seq = input_seq.to(DEVICE)
                dt = dt.to(DEVICE)
                target = target.to(DEVICE)
                log_density = model(input_seq, dt)
                vl = compute_nll_loss(log_density, target, config['grid_size'], config['grid_range'])
                if not torch.isnan(vl):
                    val_loss += vl.item()

        train_loss_avg = epoch_loss / max(num_batches, 1)
        val_loss_avg = val_loss / len(val_loader)
        diagnostics['train_loss'].append(train_loss_avg)
        diagnostics['val_loss'].append(val_loss_avg)

        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: Train={train_loss_avg:.4f}, Val={val_loss_avg:.4f}")

    return diagnostics


def train_regression_baseline(model, train_loader, val_loader, num_epochs=50):
    """Train regression baseline."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for input_seq, dt, target in train_loader:
            input_seq = input_seq.to(DEVICE)
            dt = dt.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            pred = model(input_seq, dt)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for input_seq, dt, target in val_loader:
                    input_seq = input_seq.to(DEVICE)
                    dt = dt.to(DEVICE)
                    target = target.to(DEVICE)
                    pred = model(input_seq, dt)
                    val_loss += F.mse_loss(pred, target).item()
            print(f"    Epoch {epoch}: Train MSE={train_loss/len(train_loader):.6f}, "
                  f"Val MSE={val_loss/len(val_loader):.6f}")

    return model


def evaluate_model(model, val_loader, config):
    """Compute prediction metrics."""
    model.eval()

    all_expected_x, all_expected_y = [], []
    all_mode_x, all_mode_y = [], []
    all_target_x, all_target_y = [], []
    all_dt = []

    grid_x = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)
    grid_y = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)

    with torch.no_grad():
        for input_seq, dt, target in val_loader:
            input_seq = input_seq.to(DEVICE)
            dt = dt.to(DEVICE)
            target = target.to(DEVICE)

            log_density = model(input_seq, dt)
            density = torch.exp(log_density)

            # Expected value
            marginal_x = density.sum(dim=2)
            marginal_y = density.sum(dim=1)
            expected_x = (marginal_x * grid_x.unsqueeze(0)).sum(dim=1)
            expected_y = (marginal_y * grid_y.unsqueeze(0)).sum(dim=1)

            # Mode
            flat_density = density.view(density.shape[0], -1)
            mode_idx = flat_density.argmax(dim=1)
            mode_x = grid_x[mode_idx // config['grid_size']]
            mode_y = grid_y[mode_idx % config['grid_size']]

            all_expected_x.append(expected_x)
            all_expected_y.append(expected_y)
            all_mode_x.append(mode_x)
            all_mode_y.append(mode_y)
            all_target_x.append(target[:, 0])
            all_target_y.append(target[:, 1])
            all_dt.append(dt.squeeze())

    expected_x = torch.cat(all_expected_x)
    expected_y = torch.cat(all_expected_y)
    mode_x = torch.cat(all_mode_x)
    mode_y = torch.cat(all_mode_y)
    target_x = torch.cat(all_target_x)
    target_y = torch.cat(all_target_y)
    dt_all = torch.cat(all_dt)

    expected_error = torch.sqrt((expected_x - target_x)**2 + (expected_y - target_y)**2).mean()
    mode_error = torch.sqrt((mode_x - target_x)**2 + (mode_y - target_y)**2).mean()

    return {
        'expected_error': expected_error.item(),
        'mode_error': mode_error.item(),
    }


def test_dt_sensitivity(model, val_loader, config):
    """
    Test if model output changes with different Δt values.
    FIXED: Compare mode/expected value locations, not L1 of full density.
    """
    model.eval()

    # Get a batch
    input_seq, dt_orig, target = next(iter(val_loader))
    input_seq = input_seq.to(DEVICE)

    grid_x = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)
    grid_y = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)

    test_dts = [0.5, 1.0, 2.0, 3.0, 4.0]
    expected_positions = []
    mode_positions = []

    with torch.no_grad():
        for test_dt in test_dts:
            dt = torch.full((input_seq.shape[0], 1), test_dt, device=DEVICE)
            log_density = model(input_seq, dt)
            density = torch.exp(log_density)

            # Expected value
            marginal_x = density.sum(dim=2)
            marginal_y = density.sum(dim=1)
            expected_x = (marginal_x * grid_x.unsqueeze(0)).sum(dim=1)
            expected_y = (marginal_y * grid_y.unsqueeze(0)).sum(dim=1)

            # Mode
            flat_density = density.view(density.shape[0], -1)
            mode_idx = flat_density.argmax(dim=1)
            mode_x = grid_x[mode_idx // config['grid_size']]
            mode_y = grid_y[mode_idx % config['grid_size']]

            expected_positions.append((expected_x.mean().item(), expected_y.mean().item()))
            mode_positions.append((mode_x.mean().item(), mode_y.mean().item()))

    # Compute how much predictions scale with dt
    expected_mags = [np.sqrt(p[0]**2 + p[1]**2) for p in expected_positions]
    mode_mags = [np.sqrt(p[0]**2 + p[1]**2) for p in mode_positions]

    # Ideal: magnitude should scale linearly with dt
    # Compute correlation
    dt_array = np.array(test_dts)
    expected_correlation = np.corrcoef(dt_array, expected_mags)[0, 1]
    mode_correlation = np.corrcoef(dt_array, mode_mags)[0, 1]

    # Also check variance in magnitude across different dts
    expected_mag_std = np.std(expected_mags)
    mode_mag_std = np.std(mode_mags)

    return {
        'test_dts': test_dts,
        'expected_mags': expected_mags,
        'mode_mags': mode_mags,
        'expected_dt_correlation': expected_correlation,
        'mode_dt_correlation': mode_correlation,
        'expected_mag_std': expected_mag_std,
        'mode_mag_std': mode_mag_std,
        'uses_dt': expected_mag_std > 0.5  # If magnitudes vary significantly with dt
    }


def test_input_sensitivity(model, val_loader, config):
    """
    FIXED: Test if model uses input trajectory by comparing expected values,
    not raw density L1.
    """
    model.eval()

    input_seq, dt, target = next(iter(val_loader))
    input_seq = input_seq.to(DEVICE)
    dt = dt.to(DEVICE)

    grid_x = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)
    grid_y = torch.linspace(-config['grid_range'], config['grid_range'],
                            config['grid_size'], device=DEVICE)

    with torch.no_grad():
        # Real input
        log_density_real = model(input_seq, dt)
        density_real = torch.exp(log_density_real)
        marginal_x = density_real.sum(dim=2)
        marginal_y = density_real.sum(dim=1)
        real_expected_x = (marginal_x * grid_x.unsqueeze(0)).sum(dim=1)
        real_expected_y = (marginal_y * grid_y.unsqueeze(0)).sum(dim=1)

        # Zeroed input
        zero_input = torch.zeros_like(input_seq)
        log_density_zero = model(zero_input, dt)
        density_zero = torch.exp(log_density_zero)
        marginal_x = density_zero.sum(dim=2)
        marginal_y = density_zero.sum(dim=1)
        zero_expected_x = (marginal_x * grid_x.unsqueeze(0)).sum(dim=1)
        zero_expected_y = (marginal_y * grid_y.unsqueeze(0)).sum(dim=1)

    # Compute distance between expected positions
    expected_diff = torch.sqrt((real_expected_x - zero_expected_x)**2 +
                               (real_expected_y - zero_expected_y)**2).mean()

    return {
        'expected_position_diff': expected_diff.item(),
        'uses_input': expected_diff.item() > 0.5
    }


def visualize_dt_effect(model, val_loader, config, save_path):
    """Visualize how predictions change with Δt."""
    model.eval()

    # Get one sample
    input_seq, _, target = next(iter(val_loader))
    input_seq = input_seq[0:1].to(DEVICE)

    test_dts = [0.5, 1.0, 2.0, 3.0, 4.0]

    fig, axes = plt.subplots(1, len(test_dts), figsize=(4*len(test_dts), 4))

    with torch.no_grad():
        for i, test_dt in enumerate(test_dts):
            dt = torch.tensor([[test_dt]], device=DEVICE)
            log_density = model(input_seq, dt)
            density = torch.exp(log_density[0]).cpu().numpy()

            ax = axes[i]
            im = ax.imshow(density.T, origin='lower',
                          extent=[-config['grid_range'], config['grid_range'],
                                  -config['grid_range'], config['grid_range']],
                          cmap='hot')

            # Draw trajectory
            traj = input_seq[0].cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], 'c.-', linewidth=2, markersize=5)
            ax.scatter([0], [0], c='green', s=100, marker='o', zorder=5)

            ax.set_title(f'Δt = {test_dt}')
            ax.set_xlim(-config['grid_range'], config['grid_range'])
            ax.set_ylim(-config['grid_range'], config['grid_range'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_predictions(model, val_loader, config, save_path, num_samples=4):
    """Visualize predictions for different trajectories."""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    with torch.no_grad():
        input_seq, dt, target = next(iter(val_loader))
        input_seq = input_seq.to(DEVICE)
        dt = dt.to(DEVICE)
        target = target.to(DEVICE)

        log_density = model(input_seq, dt)

        for idx in range(min(num_samples, input_seq.shape[0])):
            traj = input_seq[idx].cpu().numpy()
            tgt = target[idx].cpu().numpy()
            dt_val = dt[idx].item()
            density = torch.exp(log_density[idx]).cpu().numpy()

            # Input trajectory
            ax = axes[idx, 0]
            ax.plot(traj[:, 0], traj[:, 1], 'b.-', linewidth=2, markersize=8)
            ax.scatter([0], [0], c='green', s=100, marker='o', zorder=5)
            ax.scatter([tgt[0]], [tgt[1]], c='red', s=100, marker='x', linewidths=3, zorder=5)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_title(f'Input (Δt={dt_val:.2f})')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # Density
            ax = axes[idx, 1]
            im = ax.imshow(density.T, origin='lower',
                          extent=[-config['grid_range'], config['grid_range'],
                                  -config['grid_range'], config['grid_range']],
                          cmap='hot')
            ax.scatter([tgt[0]], [tgt[1]], c='cyan', s=100, marker='x', linewidths=3, zorder=5)
            ax.set_title('Predicted Density')
            plt.colorbar(im, ax=ax)

            # Log density
            ax = axes[idx, 2]
            im = ax.imshow(log_density[idx].cpu().numpy().T, origin='lower',
                          extent=[-config['grid_range'], config['grid_range'],
                                  -config['grid_range'], config['grid_range']],
                          cmap='viridis')
            ax.scatter([tgt[0]], [tgt[1]], c='red', s=100, marker='x', linewidths=3, zorder=5)
            ax.set_title('Log Density')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("VARIABLE TIME HORIZON (Δt) EXPERIMENT")
    print("=" * 80)

    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'velocity_range': (0.1, 4.5),
        'dt_range': (0.5, 4.0),
        'noise_std': 0.0,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 20.0,  # Must cover max displacement: 4.5 velocity × 4.0 dt = 18
        'batch_size': 64,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'num_epochs': 30,
    }

    output_dir = Path('/home/ec2-user/projects/trackfm-deadreckon/fourier_trajectory_test/experiments/02_variable_dt_methods/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("\n" + "=" * 80)
    print("GENERATING DATA WITH VARIABLE Δt")
    print("=" * 80)

    train_dataset = VariableDtDataset(
        config['num_train'], config['seq_len'],
        config['velocity_range'], config['dt_range'], config['noise_std']
    )
    val_dataset = VariableDtDataset(
        config['num_val'], config['seq_len'],
        config['velocity_range'], config['dt_range'], config['noise_std']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Check data
    sample_input, sample_dt, sample_target = train_dataset[0]
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Δt shape: {sample_dt.shape}")
    print(f"  Target shape: {sample_target.shape}")
    print(f"  Δt range: {config['dt_range']}")
    print(f"  Velocity range: {config['velocity_range']}")

    results = {'config': config}

    # =========================================================================
    # Regression Baseline
    # =========================================================================
    print("\n" + "=" * 80)
    print("REGRESSION BASELINE")
    print("=" * 80)

    reg_model = DirectRegressionWithDt(config['d_model'], config['nhead'], config['num_layers'])
    train_regression_baseline(reg_model, train_loader, val_loader, num_epochs=20)

    reg_model.eval()
    with torch.no_grad():
        val_preds, val_targets = [], []
        for input_seq, dt, target in val_loader:
            input_seq = input_seq.to(DEVICE)
            dt = dt.to(DEVICE)
            target = target.to(DEVICE)
            pred = reg_model(input_seq, dt)
            val_preds.append(pred)
            val_targets.append(target)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        reg_error = torch.sqrt((val_preds - val_targets).pow(2).sum(dim=1)).mean().item()

    print(f"\n  Regression baseline error: {reg_error:.4f}")
    results['regression_baseline'] = {'error': reg_error}

    # =========================================================================
    # Test Each Method
    # =========================================================================
    methods = [
        ("Method 1: Concat to Output", Method1_ConcatToOutput),
        ("Method 2: FiLM Conditioning", Method2_FiLM),
        ("Method 3: Concat to Input", Method3_ConcatToInput),
        ("Method 4: Δt as Token", Method4_DtToken),
        ("Method 5: Cross-Attention", Method5_CrossAttention),
    ]

    for method_name, ModelClass in methods:
        print("\n" + "=" * 80)
        print(f"TRAINING: {method_name}")
        print("=" * 80)

        model = ModelClass(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            grid_size=config['grid_size'],
            num_freqs=config['num_freqs'],
            grid_range=config['grid_range']
        )

        diagnostics = train_model(model, train_loader, val_loader, config, use_soft_loss=True)

        # Evaluate
        print("\n  Evaluating...")
        metrics = evaluate_model(model, val_loader, config)
        dt_sensitivity = test_dt_sensitivity(model, val_loader, config)
        input_sensitivity = test_input_sensitivity(model, val_loader, config)

        print(f"  Expected Error: {metrics['expected_error']:.4f}")
        print(f"  Mode Error: {metrics['mode_error']:.4f}")
        print(f"  Uses Δt: {dt_sensitivity['uses_dt']} (correlation: {dt_sensitivity['expected_dt_correlation']:.3f})")
        print(f"  Uses Input: {input_sensitivity['uses_input']} (diff: {input_sensitivity['expected_position_diff']:.3f})")

        # Save visualizations
        safe_name = method_name.replace(" ", "_").replace(":", "").replace("Δ", "dt")
        visualize_predictions(model, val_loader, config,
                            str(output_dir / f'{safe_name}_predictions.png'))
        visualize_dt_effect(model, val_loader, config,
                          str(output_dir / f'{safe_name}_dt_effect.png'))

        results[method_name] = {
            'expected_error': metrics['expected_error'],
            'mode_error': metrics['mode_error'],
            'uses_dt': dt_sensitivity['uses_dt'],
            'dt_correlation': dt_sensitivity['expected_dt_correlation'],
            'uses_input': input_sensitivity['uses_input'],
            'input_diff': input_sensitivity['expected_position_diff'],
            'final_train_loss': diagnostics['train_loss'][-1],
            'final_val_loss': diagnostics['val_loss'][-1],
        }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\n  Regression Baseline Error: {results['regression_baseline']['error']:.4f}")
    print("\n  Fourier Head Methods:")
    print(f"  {'Method':<35} {'Error':>8} {'Uses Δt':>10} {'Uses Input':>12} {'Δt Corr':>10}")
    print("  " + "-" * 75)

    for method_name, _ in methods:
        r = results[method_name]
        print(f"  {method_name:<35} {r['expected_error']:>8.4f} {str(r['uses_dt']):>10} "
              f"{str(r['uses_input']):>12} {r['dt_correlation']:>10.3f}")

    # Find best method
    best_method = min(methods, key=lambda m: results[m[0]]['expected_error'])
    print(f"\n  BEST METHOD: {best_method[0]}")
    print(f"  Best Error: {results[best_method[0]]['expected_error']:.4f}")

    # Save results
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_dir}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = main()
