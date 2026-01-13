#!/usr/bin/env python3
"""
Optimize the dead reckoning Gaussian sigma using ONLY training data.

Finds the sigma that minimizes KL divergence loss for dead reckoning predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import Config, AISTrackDataset, load_ais_data, DEVICE

def compute_dr_errors(train_dataset, config, num_samples=50000):
    """
    Compute dead reckoning errors on training data.

    Returns arrays of (dr_pred_lat, dr_pred_lon, actual_lat, actual_lon) displacements.
    """
    seq_len = config.max_seq_len
    max_horizon = config.max_horizon

    # Sample random indices and horizons
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_dataset), min(num_samples, len(train_dataset)), replace=False)

    # Sample horizons similar to training (evenly spaced for coverage)
    horizons_to_sample = np.linspace(1, max_horizon, 40).astype(int)

    dr_errors_lat = []
    dr_errors_lon = []
    actual_displacements_lat = []
    actual_displacements_lon = []
    horizon_list = []

    print(f"Computing DR errors on {len(sample_indices)} training samples...")

    for i, idx in enumerate(sample_indices):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(sample_indices)}")

        features = train_dataset[idx].numpy()

        # Denormalize positions
        lat = features[:, 0] * config.lat_std + config.lat_mean
        lon = features[:, 1] * config.lon_std + config.lon_mean
        dt = features[:, 5] * config.dt_max

        # Source position
        src_lat = lat[seq_len - 1]
        src_lon = lon[seq_len - 1]

        # Velocity at source
        dlat = lat[seq_len - 1] - lat[seq_len - 2]
        dlon = lon[seq_len - 1] - lon[seq_len - 2]
        last_dt = max(dt[seq_len - 1], 1.0)
        vlat = dlat / last_dt
        vlon = dlon / last_dt

        # Cumulative time
        cumsum_dt = np.cumsum(dt)
        last_cumsum = cumsum_dt[seq_len - 1]

        for h in horizons_to_sample:
            tgt_idx = seq_len + h - 1
            if tgt_idx >= len(features):
                continue

            # Cumulative time to target
            cum_time = cumsum_dt[tgt_idx] - last_cumsum

            # DR prediction (displacement from source)
            dr_lat = vlat * cum_time
            dr_lon = vlon * cum_time

            # Actual displacement
            actual_lat = lat[tgt_idx] - src_lat
            actual_lon = lon[tgt_idx] - src_lon

            dr_errors_lat.append(dr_lat - actual_lat)
            dr_errors_lon.append(dr_lon - actual_lon)
            actual_displacements_lat.append(actual_lat)
            actual_displacements_lon.append(actual_lon)
            horizon_list.append(h)

    return (np.array(dr_errors_lat), np.array(dr_errors_lon),
            np.array(actual_displacements_lat), np.array(actual_displacements_lon),
            np.array(horizon_list))


def compute_kl_loss_for_sigma(dr_errors_lat, dr_errors_lon, actual_lat, actual_lon,
                               sigma, grid_range, grid_size):
    """
    Compute KL divergence loss for DR predictions with given sigma.
    """
    device = torch.device('cpu')  # Use CPU to avoid GPU memory issues

    # Create grid
    x = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    y = torch.linspace(-grid_range, grid_range, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Process in batches
    batch_size = 10000
    total_loss = 0.0
    total_count = 0

    for i in range(0, len(dr_errors_lat), batch_size):
        end_i = min(i + batch_size, len(dr_errors_lat))

        # DR predictions are at (0, 0) since we're computing in displacement space
        # But DR error is (dr_pred - actual), so dr_pred = actual + error
        # We want to compare dr_pred distribution to actual position

        # DR predicts: dr_displacement = velocity * time
        # Actual: actual_displacement
        # Error: dr_displacement - actual_displacement

        # For the loss, we center Gaussian at DR prediction and evaluate at actual
        # DR prediction in displacement space = actual_displacement + dr_error

        batch_actual_lat = torch.tensor(actual_lat[i:end_i], device=device).float()
        batch_actual_lon = torch.tensor(actual_lon[i:end_i], device=device).float()
        batch_dr_err_lat = torch.tensor(dr_errors_lat[i:end_i], device=device).float()
        batch_dr_err_lon = torch.tensor(dr_errors_lon[i:end_i], device=device).float()

        # DR prediction = actual + error (since error = dr - actual, so dr = actual + error)
        dr_pred_lat = batch_actual_lat + batch_dr_err_lat
        dr_pred_lon = batch_actual_lon + batch_dr_err_lon

        # Clip to grid range
        dr_pred_lat = torch.clamp(dr_pred_lat, -grid_range * 0.99, grid_range * 0.99)
        dr_pred_lon = torch.clamp(dr_pred_lon, -grid_range * 0.99, grid_range * 0.99)
        batch_actual_lat = torch.clamp(batch_actual_lat, -grid_range * 0.99, grid_range * 0.99)
        batch_actual_lon = torch.clamp(batch_actual_lon, -grid_range * 0.99, grid_range * 0.99)

        # Create DR distribution (Gaussian centered at DR prediction)
        dr_pred_lat = dr_pred_lat.view(-1, 1, 1)
        dr_pred_lon = dr_pred_lon.view(-1, 1, 1)

        dist_sq = (xx - dr_pred_lat) ** 2 + (yy - dr_pred_lon) ** 2
        dr_density = torch.exp(-dist_sq / (2 * sigma ** 2))
        dr_density = dr_density / (dr_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)
        dr_log_density = torch.log(dr_density + 1e-10)

        # Target distribution (Gaussian centered at actual position)
        target_lat = batch_actual_lat.view(-1, 1, 1)
        target_lon = batch_actual_lon.view(-1, 1, 1)

        # Use same sigma for target as the model does
        target_sigma = 0.003
        target_dist_sq = (xx - target_lat) ** 2 + (yy - target_lon) ** 2
        target_density = torch.exp(-target_dist_sq / (2 * target_sigma ** 2))
        target_density = target_density / (target_density.sum(dim=(-2, -1), keepdim=True) + 1e-10)

        # KL divergence
        kl = F.kl_div(dr_log_density, target_density, reduction='none').sum(dim=(-2, -1))
        total_loss += kl.sum().item()
        total_count += len(kl)

    return total_loss / total_count


def main():
    print("=" * 70)
    print("OPTIMIZING DEAD RECKONING SIGMA (Training Data Only)")
    print("=" * 70)

    config = Config()

    # Load training data only
    print("\nLoading training data...")
    train_data, val_data = load_ais_data(config, max_tracks=None)

    # Create training dataset
    train_dataset = AISTrackDataset(train_data, config.max_seq_len, config.max_horizon, config)
    print(f"Training samples: {len(train_dataset)}")

    # Compute DR errors on training data
    print("\n" + "=" * 70)
    print("COMPUTING DR ERRORS ON TRAINING DATA")
    print("=" * 70)

    dr_err_lat, dr_err_lon, actual_lat, actual_lon, horizons = compute_dr_errors(
        train_dataset, config, num_samples=50000
    )

    print(f"\nTotal prediction pairs: {len(dr_err_lat)}")

    # Compute error statistics
    dr_errors = np.sqrt(dr_err_lat**2 + dr_err_lon**2)
    print(f"\nDR Error Statistics (degrees):")
    print(f"  Mean: {dr_errors.mean():.6f}")
    print(f"  Std:  {dr_errors.std():.6f}")
    print(f"  Median: {np.median(dr_errors):.6f}")
    print(f"  95th percentile: {np.percentile(dr_errors, 95):.6f}")
    print(f"  99th percentile: {np.percentile(dr_errors, 99):.6f}")

    # Convert to km for intuition
    print(f"\nDR Error Statistics (km):")
    print(f"  Mean: {dr_errors.mean() * 111:.2f}")
    print(f"  Std:  {dr_errors.std() * 111:.2f}")
    print(f"  Median: {np.median(dr_errors) * 111:.2f}")
    print(f"  95th percentile: {np.percentile(dr_errors, 95) * 111:.2f}")

    # Component-wise stats
    print(f"\nComponent-wise Error Std (degrees):")
    print(f"  Lat: {dr_err_lat.std():.6f}")
    print(f"  Lon: {dr_err_lon.std():.6f}")

    # The optimal sigma for a Gaussian is approximately the std of the errors
    optimal_sigma_approx = dr_errors.std()
    print(f"\nApproximate optimal sigma (error std): {optimal_sigma_approx:.6f}")

    # Grid search for optimal sigma
    print("\n" + "=" * 70)
    print("GRID SEARCH FOR OPTIMAL SIGMA")
    print("=" * 70)

    # Test range of sigmas
    sigmas = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]

    # Also include sigmas around the approximate optimal
    sigmas.extend([optimal_sigma_approx * 0.5, optimal_sigma_approx, optimal_sigma_approx * 1.5, optimal_sigma_approx * 2])
    sigmas = sorted(set(sigmas))

    print(f"Testing sigmas: {[f'{s:.4f}' for s in sigmas]}")
    print()

    results = []
    for sigma in sigmas:
        loss = compute_kl_loss_for_sigma(
            dr_err_lat, dr_err_lon, actual_lat, actual_lon,
            sigma, config.grid_range, config.grid_size
        )
        results.append((sigma, loss))
        print(f"  sigma={sigma:.4f}: loss={loss:.4f}")

    # Find best
    best_sigma, best_loss = min(results, key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOptimal sigma for dead reckoning: {best_sigma:.6f} degrees")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\nCurrent sigma in config: {config.sigma:.6f}")
    print(f"Ratio (optimal/current): {best_sigma/config.sigma:.1f}x")

    # Show by horizon
    print("\n" + "=" * 70)
    print("ERROR BY HORIZON")
    print("=" * 70)

    unique_horizons = sorted(set(horizons))
    for h in unique_horizons[::5]:  # Every 5th horizon
        mask = horizons == h
        h_errors = dr_errors[mask]
        if len(h_errors) > 0:
            print(f"  Horizon {h:3d}: mean={h_errors.mean()*111:.2f}km, std={h_errors.std()*111:.2f}km")

    return best_sigma


if __name__ == '__main__':
    best_sigma = main()
