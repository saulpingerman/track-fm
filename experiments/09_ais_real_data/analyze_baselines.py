#!/usr/bin/env python3
"""
Comprehensive baseline analysis for fair comparison.

Tests:
1. Dead Reckoning with multiple sigma values
2. Last Position baseline (predict zero displacement)
3. Random (untrained) model
4. Trained model

All baselines use the EXACT same loss function on the SAME validation data.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
import sys

# Import from main experiment
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    Config, AISTrackDataset, CausalAISModel, load_ais_data,
    compute_soft_target_loss, DEVICE
)


def compute_baseline_losses(features: torch.Tensor, targets: torch.Tensor,
                            config: Config, sigma: float) -> dict:
    """
    Compute all baseline losses using specified sigma.

    Returns dict with:
    - dead_reckoning: velocity * horizon extrapolation
    - last_position: predict zero displacement
    """
    device = features.device
    batch_size = features.shape[0]
    seq_len = config.max_seq_len
    max_horizon = config.max_horizon
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Get positions (denormalized)
    lat = features[:, :, 0] * config.lat_std + config.lat_mean
    lon = features[:, :, 1] * config.lon_std + config.lon_mean

    # Velocity from last two positions
    vlat = lat[:, seq_len-1] - lat[:, seq_len-2]
    vlon = lon[:, seq_len-1] - lon[:, seq_len-2]

    # Future targets only (for fair comparison)
    future_targets = targets[:, -max_horizon:, :]

    # Create grid
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Horizons
    horizons = torch.arange(1, max_horizon + 1, device=device).float()

    # === DEAD RECKONING ===
    dr_pred_lat = vlat.unsqueeze(1) * horizons.unsqueeze(0)
    dr_pred_lon = vlon.unsqueeze(1) * horizons.unsqueeze(0)
    dr_pred_lat = torch.clamp(dr_pred_lat.unsqueeze(-1).unsqueeze(-1), -grid_range * 0.99, grid_range * 0.99)
    dr_pred_lon = torch.clamp(dr_pred_lon.unsqueeze(-1).unsqueeze(-1), -grid_range * 0.99, grid_range * 0.99)

    dist_sq = (xx - dr_pred_lat) ** 2 + (yy - dr_pred_lon) ** 2
    dr_density = torch.exp(-dist_sq / (2 * sigma ** 2))
    dr_density = dr_density / (dr_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    dr_log_density = torch.log(dr_density + 1e-10)

    # === LAST POSITION (zero displacement) ===
    # Predict center of grid (zero displacement)
    lp_pred_lat = torch.zeros(batch_size, max_horizon, 1, 1, device=device)
    lp_pred_lon = torch.zeros(batch_size, max_horizon, 1, 1, device=device)

    dist_sq_lp = (xx - lp_pred_lat) ** 2 + (yy - lp_pred_lon) ** 2
    lp_density = torch.exp(-dist_sq_lp / (2 * sigma ** 2))
    lp_density = lp_density / (lp_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
    lp_log_density = torch.log(lp_density + 1e-10)

    # === TARGET DISTRIBUTION ===
    target_x = future_targets[:, :, 0:1, None]
    target_y = future_targets[:, :, 1:2, None]
    target_dist_sq = (xx - target_x) ** 2 + (yy - target_y) ** 2
    soft_target = torch.exp(-target_dist_sq / (2 * sigma ** 2))
    soft_target = soft_target / (soft_target.sum(dim=(-2, -1), keepdim=True) + 1e-10)

    # === COMPUTE LOSSES ===
    dr_loss = F.kl_div(dr_log_density, soft_target, reduction='none').sum(dim=(-2, -1)).mean()
    lp_loss = F.kl_div(lp_log_density, soft_target, reduction='none').sum(dim=(-2, -1)).mean()

    return {
        'dead_reckoning': dr_loss.item(),
        'last_position': lp_loss.item()
    }


def evaluate_all_baselines(model: torch.nn.Module, val_loader: DataLoader,
                           config: Config, max_batches: int = 50) -> dict:
    """Evaluate model and all baselines on validation set."""

    model.eval()
    max_horizon = config.max_horizon

    # Test multiple sigma values
    sigma_values = [0.0005, 0.001, 0.002, 0.005, 0.01]

    results = {
        'model': [],
        'random_model': [],
    }
    for sigma in sigma_values:
        results[f'dr_sigma_{sigma}'] = []
        results[f'lp_sigma_{sigma}'] = []

    # Create random (untrained) model for comparison
    random_model = CausalAISModel(config).to(DEVICE)
    random_model.eval()

    with torch.no_grad():
        for batch_idx, features in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            features = features.to(DEVICE, non_blocking=True)

            with autocast('cuda', enabled=config.use_amp):
                # Trained model
                log_densities, targets = model.forward_train(features)
                future_log_densities = log_densities[:, -max_horizon:, :, :]
                future_targets = targets[:, -max_horizon:, :]

                model_loss = compute_soft_target_loss(
                    future_log_densities, future_targets,
                    config.grid_range, config.grid_size, config.sigma
                )
                results['model'].append(model_loss.item())

                # Random model
                random_log_densities, _ = random_model.forward_train(features)
                random_future = random_log_densities[:, -max_horizon:, :, :]
                random_loss = compute_soft_target_loss(
                    random_future, future_targets,
                    config.grid_range, config.grid_size, config.sigma
                )
                results['random_model'].append(random_loss.item())

                # Test baselines with different sigma values
                for sigma in sigma_values:
                    baseline_losses = compute_baseline_losses(features, targets, config, sigma)
                    results[f'dr_sigma_{sigma}'].append(baseline_losses['dead_reckoning'])
                    results[f'lp_sigma_{sigma}'].append(baseline_losses['last_position'])

    # Compute means
    return {k: np.mean(v) for k, v in results.items()}


def main():
    print("=" * 70)
    print("COMPREHENSIVE BASELINE ANALYSIS")
    print("=" * 70)

    config = Config()

    # Load data
    print("\nLoading data...")
    train_data, val_data = load_ais_data(config, max_tracks=2000)

    val_dataset = AISTrackDataset(val_data, config.max_seq_len, config.max_horizon, config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)

    # Load trained model
    print("\nLoading trained model...")
    model = CausalAISModel(config).to(DEVICE)

    checkpoint_path = Path(__file__).parent / 'results' / 'best_model.pt'
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"  Loaded: {checkpoint_path}")
    else:
        print("  WARNING: No trained model found, using random initialization")

    # Evaluate
    print("\nEvaluating all baselines (50 batches)...")
    results = evaluate_all_baselines(model, val_loader, config, max_batches=50)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<30} {'Loss':<12} {'vs DR (default)':<15}")
    print("-" * 57)

    dr_default = results['dr_sigma_0.001']  # Default sigma

    # Model results
    model_loss = results['model']
    improvement = (dr_default - model_loss) / dr_default * 100
    print(f"{'Trained Model':<30} {model_loss:<12.4f} {improvement:+.1f}%")

    random_loss = results['random_model']
    improvement = (dr_default - random_loss) / dr_default * 100
    print(f"{'Random Model (untrained)':<30} {random_loss:<12.4f} {improvement:+.1f}%")

    print()

    # Dead reckoning with different sigma
    print("Dead Reckoning (velocity extrapolation):")
    for sigma in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        loss = results[f'dr_sigma_{sigma}']
        print(f"  sigma={sigma:<8} {loss:<12.4f}")

    print()

    # Last position with different sigma
    print("Last Position (zero displacement):")
    for sigma in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        loss = results[f'lp_sigma_{sigma}']
        print(f"  sigma={sigma:<8} {loss:<12.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    best_dr = min(results[f'dr_sigma_{s}'] for s in [0.0005, 0.001, 0.002, 0.005, 0.01])
    best_lp = min(results[f'lp_sigma_{s}'] for s in [0.0005, 0.001, 0.002, 0.005, 0.01])

    print(f"\nBest Dead Reckoning (any sigma): {best_dr:.4f}")
    print(f"Best Last Position (any sigma): {best_lp:.4f}")
    print(f"Trained Model: {model_loss:.4f}")

    if model_loss < best_dr:
        improvement = (best_dr - model_loss) / best_dr * 100
        print(f"\nModel beats BEST Dead Reckoning by {improvement:.1f}%")
    else:
        print(f"\nModel does NOT beat best Dead Reckoning")

    if model_loss < best_lp:
        improvement = (best_lp - model_loss) / best_lp * 100
        print(f"Model beats BEST Last Position by {improvement:.1f}%")


if __name__ == '__main__':
    main()
