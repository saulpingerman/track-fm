#!/usr/bin/env python3
"""
Visualize model predictions on validation tracks.

Shows:
- Actual trajectory
- Model's predicted probability distribution (heatmap)
- Dead reckoning prediction
- Comparison across horizons
- Training history (loss curves)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    Config, AISTrackDataset, CausalAISModel, load_ais_data, DEVICE
)


def plot_training_history(results_dir: Path):
    """
    Generate training history plot by parsing training.log directly.
    Can be run independently after training (even if stopped early).
    """
    import re

    log_path = results_dir / 'training.log'
    if not log_path.exists():
        print(f"  No training.log found in {results_dir}")
        return None

    with open(log_path) as f:
        log = f.read()

    # Parse validation entries from log
    pattern = r'VALIDATION at step (\d+).*?Train Loss:\s+([\d.]+).*?Model:\s+([\d.]+).*?Dead Reckoning:\s+([\d.]+).*?Last Position:\s+([\d.]+).*?Random Model:\s+([\d.]+)'
    matches = re.findall(pattern, log, re.DOTALL)

    if not matches:
        print(f"  No validation entries found in training.log")
        return None

    steps, train_loss, val_loss, dr_loss, lp_loss, random_loss = [], [], [], [], [], []
    for m in matches:
        steps.append(int(m[0]))
        train_loss.append(float(m[1]))
        val_loss.append(float(m[2]))
        dr_loss.append(float(m[3]))
        lp_loss.append(float(m[4]))
        random_loss.append(float(m[5]))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out step=0 for log scale
    valid_idx = [i for i, s in enumerate(steps) if s > 0]
    if not valid_idx:
        print(f"  No valid training steps found")
        return None

    steps_log = [steps[i] for i in valid_idx]

    ax.plot(steps_log, [train_loss[i] for i in valid_idx], 'c-d', label='Train', markersize=4, alpha=0.7)
    ax.plot(steps_log, [val_loss[i] for i in valid_idx], 'b-o', label='Val (Model)', markersize=4)
    ax.plot(steps_log, [dr_loss[i] for i in valid_idx], 'r--s', label='Dead Reckoning', markersize=4)
    ax.plot(steps_log, [lp_loss[i] for i in valid_idx], 'g--^', label='Last Position', markersize=4)
    ax.plot(steps_log, [random_loss[i] for i in valid_idx], 'gray', linestyle=':', label='Random Model', alpha=0.7)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (KL Divergence)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Training Progress vs Baselines (log-log)')

    plt.tight_layout()
    return fig


def visualize_single_prediction(model, features, config, sample_idx=0):
    """
    Visualize model prediction for a single sample.

    Shows the predicted probability distribution overlaid on the actual trajectory.
    """
    model.eval()

    seq_len = config.max_seq_len
    max_horizon = config.max_horizon
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Get single sample
    sample = features[sample_idx:sample_idx+1].to(DEVICE)

    # Select which horizons to visualize (5 horizons spread across the range)
    if max_horizon <= 5:
        horizons_to_show = list(range(1, max_horizon + 1))
    else:
        horizons_to_show = [1] + [int(i * max_horizon / 4) for i in range(1, 5)]
        horizons_to_show = sorted(set(horizons_to_show))[:5]

    # Request these specific horizons from the model
    viz_horizons = torch.tensor(horizons_to_show, device=DEVICE)

    with torch.no_grad():
        with autocast('cuda', enabled=config.use_amp):
            log_densities, targets, _ = model.forward_train(sample, viz_horizons)

    # Get the predictions for the requested horizons
    future_log_densities = log_densities[0, :, :, :].cpu()  # Shape: (num_viz_horizons, grid, grid)
    future_targets = targets[0, :, :].cpu()

    # Denormalize positions for plotting
    lat = (features[sample_idx, :, 0] * config.lat_std + config.lat_mean).numpy()
    lon = (features[sample_idx, :, 1] * config.lon_std + config.lon_mean).numpy()

    # Source position (end of input sequence)
    src_lat = lat[seq_len - 1]
    src_lon = lon[seq_len - 1]

    # Actual future positions
    future_lat = lat[seq_len:seq_len + max_horizon]
    future_lon = lon[seq_len:seq_len + max_horizon]

    # Get actual time deltas from features (dt is column 5, normalized by dt_max)
    dt_norm = features[sample_idx, :, 5].numpy()
    dt_seconds = dt_norm * config.dt_max
    future_dt = dt_seconds[seq_len:seq_len + max_horizon]
    cumulative_time = np.cumsum(future_dt)  # Time to reach each horizon

    # Dead reckoning prediction - use actual time delta
    vlat = lat[seq_len - 1] - lat[seq_len - 2]
    vlon = lon[seq_len - 1] - lon[seq_len - 2]
    last_dt = dt_seconds[seq_len - 1]  # Time of last step
    velocity_lat = vlat / max(last_dt, 1.0)  # degrees per second
    velocity_lon = vlon / max(last_dt, 1.0)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot for each requested horizon
    for h_idx, h in enumerate(horizons_to_show):
        if h_idx >= 5:
            break
        ax = axes[h_idx // 3, h_idx % 3]

        # Get predicted density - h_idx is the index into our requested horizons
        density = torch.exp(future_log_densities[h_idx]).numpy()

        # Grid coordinates (relative to source)
        x = np.linspace(-grid_range, grid_range, grid_size)
        y = np.linspace(-grid_range, grid_range, grid_size)

        # Plot heatmap (probability distribution)
        # Note: density[i,j] = prob at (delta_lat[i], delta_lon[j])
        # imshow: first dim (rows) = y-axis = latitude, second dim (cols) = x-axis = longitude
        # So NO transpose needed - density's axes match (lat, lon) -> (y, x)
        extent = [src_lon - grid_range, src_lon + grid_range,
                  src_lat - grid_range, src_lat + grid_range]
        im = ax.imshow(density, origin='lower', extent=extent,
                       cmap='hot', alpha=0.8, aspect='auto')

        # Plot input trajectory
        ax.plot(lon[:seq_len], lat[:seq_len], 'b-', linewidth=1, alpha=0.5, label='History')
        ax.plot(lon[seq_len-1], lat[seq_len-1], 'bo', markersize=8, label='Source')

        # Plot actual future position - use h-1 for horizon-indexed arrays
        if h-1 < len(future_lat):
            ax.plot(future_lon[h-1], future_lat[h-1], 'g*', markersize=15,
                    label='Actual', markeredgecolor='white', markeredgewidth=1)

        # Plot dead reckoning prediction using actual cumulative time
        cum_t = cumulative_time[h-1] if h-1 < len(cumulative_time) else h * 10
        dr_lat = src_lat + velocity_lat * cum_t
        dr_lon = src_lon + velocity_lon * cum_t
        ax.plot(dr_lon, dr_lat, 'rx', markersize=12, markeredgewidth=2, label='Dead Reckon')

        # Compute expected value from model
        xx, yy = np.meshgrid(x, y, indexing='ij')
        exp_dlat = (density * xx).sum()
        exp_dlon = (density * yy).sum()
        model_lat = src_lat + exp_dlat
        model_lon = src_lon + exp_dlon
        ax.plot(model_lon, model_lat, 'c^', markersize=10, label='Model Mean')

        ax.set_xlim(src_lon - grid_range * 1.2, src_lon + grid_range * 1.2)
        ax.set_ylim(src_lat - grid_range * 1.2, src_lat + grid_range * 1.2)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # Use actual cumulative time from data
        time_ahead = cumulative_time[h-1] if h-1 < len(cumulative_time) else 0
        ax.set_title(f'Horizon {h} ({time_ahead:.0f}s ahead)')

        if h_idx == 0:
            ax.legend(loc='upper left', fontsize=8)

        plt.colorbar(im, ax=ax, label='Probability')

    # Use last subplot for legend/info
    ax = axes[1, 2]
    ax.axis('off')

    # Add info text
    info_text = f"""
    Sample visualization

    Blue line: Input trajectory (history)
    Blue dot: Source position (prediction point)
    Green star: Actual future position
    Red X: Dead reckoning prediction
    Cyan triangle: Model's expected position
    Heatmap: Model's probability distribution

    Grid range: {grid_range}° ({grid_range * 111:.1f} km)
    """
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    return fig


def visualize_track_with_context(model, val_dataset, config, sample_idx=0):
    """
    Visualize a single track with both full context and zoomed prediction view.
    Two panels: left shows full trajectory, right shows zoomed prediction with heatmap.
    """
    model.eval()

    seq_len = config.max_seq_len
    max_horizon = config.max_horizon
    grid_range = config.grid_range
    grid_size = config.grid_size

    features = val_dataset[sample_idx].unsqueeze(0).to(DEVICE)

    # Use horizon 3 for visualization - pass specific horizon to model
    h = 3
    horizon_indices = torch.tensor([h], device=DEVICE)

    with torch.no_grad():
        with autocast('cuda', enabled=config.use_amp):
            log_densities, targets, _ = model.forward_train(features, horizon_indices)

    future_log_density = log_densities[0, 0, :, :].cpu()  # Index 0 since we only requested 1 horizon
    density = torch.exp(future_log_density).numpy()

    # Denormalize positions
    feat_np = features[0].cpu().numpy()
    lat = feat_np[:, 0] * config.lat_std + config.lat_mean
    lon = feat_np[:, 1] * config.lon_std + config.lon_mean

    # Split into input and future
    input_lat = lat[:seq_len]
    input_lon = lon[:seq_len]
    future_lat = lat[seq_len:]
    future_lon = lon[seq_len:]

    src_lat = lat[seq_len - 1]
    src_lon = lon[seq_len - 1]

    # Dead reckoning
    vlat = lat[seq_len - 1] - lat[seq_len - 2]
    vlon = lon[seq_len - 1] - lon[seq_len - 2]
    dr_lat = src_lat + vlat * h
    dr_lon = src_lon + vlon * h

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT PANEL: Full trajectory context
    ax1.set_title(f'Full Trajectory Context ({seq_len} input + {max_horizon} future positions)', fontsize=12)

    # Plot individual input positions with markers
    ax1.scatter(input_lon[:-1], input_lat[:-1], c=np.arange(seq_len-1), cmap='Blues',
                s=15, alpha=0.7, label='Input history')
    ax1.plot(input_lon, input_lat, 'b-', linewidth=0.5, alpha=0.3)

    # Source position (last input)
    ax1.plot(src_lon, src_lat, 'ko', markersize=12, markerfacecolor='yellow',
             markeredgewidth=2, label='Source (prediction point)', zorder=10)

    # Future positions
    ax1.scatter(future_lon, future_lat, c='green', s=80, marker='*',
                edgecolors='white', linewidths=1, label='Future (actual)', zorder=9)
    ax1.plot([src_lon] + list(future_lon), [src_lat] + list(future_lat),
             'g--', linewidth=1, alpha=0.5)

    # Draw prediction grid box
    rect = patches.Rectangle((src_lon - grid_range, src_lat - grid_range),
                              2 * grid_range, 2 * grid_range,
                              linewidth=2, edgecolor='red', facecolor='none',
                              linestyle='--', label='Prediction grid')
    ax1.add_patch(rect)

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc='best', fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Add scale info
    lat_range = input_lat.max() - input_lat.min()
    lon_range = input_lon.max() - input_lon.min()
    ax1.text(0.02, 0.98, f'Track extent: {lat_range*111:.1f}km x {lon_range*111*0.55:.1f}km\nGrid: {grid_range*111:.1f}km',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # RIGHT PANEL: Zoomed prediction view with heatmap
    ax2.set_title(f'Prediction Grid (Horizon {h}, zoomed to {grid_range*111:.1f}km)', fontsize=12)

    extent = [src_lon - grid_range, src_lon + grid_range,
              src_lat - grid_range, src_lat + grid_range]
    im = ax2.imshow(density, origin='lower', extent=extent,
                    cmap='hot', alpha=0.8, aspect='auto')

    # Plot what's visible in this zoom
    mask_lat = (input_lat >= src_lat - grid_range) & (input_lat <= src_lat + grid_range)
    mask_lon = (input_lon >= src_lon - grid_range) & (input_lon <= src_lon + grid_range)
    mask = mask_lat & mask_lon
    visible_count = mask.sum()

    ax2.scatter(input_lon[mask], input_lat[mask], c='blue', s=30, alpha=0.7)
    ax2.plot(input_lon[mask], input_lat[mask], 'b-', linewidth=1, alpha=0.5)

    # Source
    ax2.plot(src_lon, src_lat, 'ko', markersize=10, markerfacecolor='yellow', markeredgewidth=2)

    # Future positions
    for i, (flon, flat) in enumerate(zip(future_lon, future_lat)):
        ax2.plot(flon, flat, 'g*', markersize=15 if i == h-1 else 10,
                markeredgecolor='white', markeredgewidth=1)

    # Dead reckoning
    ax2.plot(dr_lon, dr_lat, 'rx', markersize=12, markeredgewidth=2, label='Dead Reckon')

    # Model expected value
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    exp_dlat = (density * xx).sum()
    exp_dlon = (density * yy).sum()
    ax2.plot(src_lon + exp_dlon, src_lat + exp_dlat, 'c^', markersize=10, label='Model Mean')

    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    plt.colorbar(im, ax=ax2, label='Probability')

    ax2.text(0.02, 0.98, f'{visible_count}/{seq_len} input positions visible',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='b', markersize=6, linestyle='', label='Input (visible)'),
        plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='yellow', markersize=10, linestyle='', markeredgewidth=2, label='Source'),
        plt.Line2D([0], [0], marker='*', color='g', markersize=12, linestyle='', label='Future (actual)'),
        plt.Line2D([0], [0], marker='x', color='r', markersize=10, linestyle='', markeredgewidth=2, label='Dead Reckon'),
        plt.Line2D([0], [0], marker='^', color='c', markersize=10, linestyle='', label='Model Mean'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def visualize_multiple_tracks(model, val_dataset, config, num_tracks=6):
    """
    Visualize predictions for multiple validation tracks.
    """
    model.eval()

    seq_len = config.max_seq_len
    max_horizon = config.max_horizon
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Sample random tracks
    np.random.seed(42)
    indices = np.random.choice(len(val_dataset), min(num_tracks, len(val_dataset)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for plot_idx, sample_idx in enumerate(indices):
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        features = val_dataset[sample_idx].unsqueeze(0).to(DEVICE)

        # Use a middle horizon for visualization (horizon 50 for long-range model)
        h = min(50, max_horizon)
        viz_horizons = torch.tensor([h], device=DEVICE)

        with torch.no_grad():
            with autocast('cuda', enabled=config.use_amp):
                log_densities, targets, _ = model.forward_train(features, viz_horizons)

        future_log_density = log_densities[0, 0, :, :].cpu()
        density = torch.exp(future_log_density).numpy()

        # Denormalize positions
        feat_np = features[0].cpu().numpy()
        lat = feat_np[:, 0] * config.lat_std + config.lat_mean
        lon = feat_np[:, 1] * config.lon_std + config.lon_mean

        src_lat = lat[seq_len - 1]
        src_lon = lon[seq_len - 1]

        # Actual future
        actual_lat = lat[seq_len + h - 1]
        actual_lon = lon[seq_len + h - 1]

        # Dead reckoning
        vlat = lat[seq_len - 1] - lat[seq_len - 2]
        vlon = lon[seq_len - 1] - lon[seq_len - 2]
        dr_lat = src_lat + vlat * h
        dr_lon = src_lon + vlon * h

        # Plot heatmap (no transpose - density axes are (lat, lon) matching (y, x))
        extent = [src_lon - grid_range, src_lon + grid_range,
                  src_lat - grid_range, src_lat + grid_range]
        im = ax.imshow(density, origin='lower', extent=extent,
                       cmap='hot', alpha=0.8, aspect='auto')

        # Plot trajectory
        ax.plot(lon[:seq_len], lat[:seq_len], 'b-', linewidth=1.5, alpha=0.7)
        ax.plot(lon[seq_len-1], lat[seq_len-1], 'bo', markersize=8)

        # Actual and predictions
        ax.plot(actual_lon, actual_lat, 'g*', markersize=15,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot(dr_lon, dr_lat, 'rx', markersize=12, markeredgewidth=2)

        # Model expected value
        x = np.linspace(-grid_range, grid_range, grid_size)
        y = np.linspace(-grid_range, grid_range, grid_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        exp_dlat = (density * xx).sum()
        exp_dlon = (density * yy).sum()
        ax.plot(src_lon + exp_dlon, src_lat + exp_dlat, 'c^', markersize=10)

        # Compute errors
        model_error = np.sqrt((exp_dlat - (actual_lat - src_lat))**2 +
                              (exp_dlon - (actual_lon - src_lon))**2) * 111 * 1000
        dr_error = np.sqrt((dr_lat - actual_lat)**2 + (dr_lon - actual_lon)**2) * 111 * 1000

        ax.set_xlim(src_lon - grid_range * 1.2, src_lon + grid_range * 1.2)
        ax.set_ylim(src_lat - grid_range * 1.2, src_lat + grid_range * 1.2)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Track {plot_idx+1} (h=3): Model={model_error:.0f}m, DR={dr_error:.0f}m')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='b', linewidth=2, label='History'),
        plt.Line2D([0], [0], marker='o', color='b', markersize=8, linestyle='', label='Source'),
        plt.Line2D([0], [0], marker='*', color='g', markersize=15, linestyle='', label='Actual'),
        plt.Line2D([0], [0], marker='x', color='r', markersize=12, linestyle='', markeredgewidth=2, label='Dead Reckon'),
        plt.Line2D([0], [0], marker='^', color='c', markersize=10, linestyle='', label='Model Mean'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Model Predictions on Validation Tracks (Horizon 3)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def visualize_horizon_comparison(model, val_dataset, config, sample_idx=0):
    """
    Compare predictions across all horizons for a single track.
    """
    model.eval()

    seq_len = config.max_seq_len
    max_horizon = config.max_horizon
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Need a batch for forward_train
    batch_indices = list(range(sample_idx, min(sample_idx + 10, len(val_dataset))))
    features_batch = torch.stack([val_dataset[i] for i in batch_indices]).to(DEVICE)
    features = features_batch

    # Request evenly spaced horizons for visualization
    num_samples = min(config.num_horizon_samples, max_horizon)
    viz_horizons = torch.linspace(1, max_horizon, num_samples).long().to(DEVICE)

    with torch.no_grad():
        with autocast('cuda', enabled=config.use_amp):
            log_densities, targets, _ = model.forward_train(features, viz_horizons)

    # Use first sample from batch
    log_densities = log_densities[0:1]

    # Denormalize (use first sample)
    feat_np = features[0].cpu().numpy()
    lat = feat_np[:, 0] * config.lat_std + config.lat_mean
    lon = feat_np[:, 1] * config.lon_std + config.lon_mean

    src_lat = lat[seq_len - 1]
    src_lon = lon[seq_len - 1]

    # Velocity for DR
    vlat = lat[seq_len - 1] - lat[seq_len - 2]
    vlon = lon[seq_len - 1] - lon[seq_len - 2]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot full trajectory
    ax.plot(lon[:seq_len], lat[:seq_len], 'b-', linewidth=2, alpha=0.7, label='History')
    ax.plot(lon[seq_len:], lat[seq_len:], 'g--', linewidth=2, alpha=0.7, label='Future (actual)')
    ax.plot(lon[seq_len-1], lat[seq_len-1], 'ko', markersize=12, label='Source', zorder=5)

    # Mark actual future positions
    for h in range(max_horizon):
        ax.plot(lon[seq_len + h], lat[seq_len + h], 'go', markersize=8, alpha=0.7)
        ax.annotate(f't+{h+1}', (lon[seq_len + h], lat[seq_len + h]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Plot model predictions for each horizon
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    model_lats = []
    model_lons = []
    dr_lats = []
    dr_lons = []

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, max_horizon))

    for h_idx, h in enumerate(range(1, max_horizon + 1)):
        density = torch.exp(log_densities[0, -max_horizon + h_idx, :, :]).cpu().numpy()

        # Model expected position
        exp_dlat = (density * xx).sum()
        exp_dlon = (density * yy).sum()
        model_lat = src_lat + exp_dlat
        model_lon = src_lon + exp_dlon
        model_lats.append(model_lat)
        model_lons.append(model_lon)

        # DR prediction
        dr_lat = src_lat + vlat * h
        dr_lon = src_lon + vlon * h
        dr_lats.append(dr_lat)
        dr_lons.append(dr_lon)

        # Plot contours for this horizon
        levels = [0.001, 0.01, 0.05]
        cs = ax.contour(src_lon + y, src_lat + x, density, levels=levels,
                        colors=[colors[h_idx]], alpha=0.5, linewidths=0.5)

    # Plot model trajectory
    ax.plot(model_lons, model_lats, 'c-o', markersize=6, linewidth=2,
            label='Model predictions', markeredgecolor='white')

    # Plot DR trajectory
    ax.plot(dr_lons, dr_lats, 'r--x', markersize=8, linewidth=2,
            label='Dead reckoning', markeredgewidth=2)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Model vs Dead Reckoning: Multi-Horizon Prediction', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def visualize_max_horizon_grid(model, val_dataset, config, num_tracks=16):
    """
    Visualize predictions at the maximum horizon (horizon 20) for many tracks.
    Shows a grid of tracks with the furthest-out predictions.
    """
    model.eval()

    seq_len = config.max_seq_len
    max_horizon = config.max_horizon
    grid_range = config.grid_range
    grid_size = config.grid_size

    # Use horizon = max_horizon (the furthest prediction)
    h = max_horizon

    # Sample random tracks
    np.random.seed(456)
    indices = np.random.choice(len(val_dataset), min(num_tracks, len(val_dataset)), replace=False)

    # Create 4x4 grid
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    model_errors = []
    dr_errors = []

    for plot_idx, sample_idx in enumerate(indices):
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]

        features = val_dataset[sample_idx].unsqueeze(0).to(DEVICE)

        # Request only the max horizon
        horizon_indices = torch.tensor([max_horizon], device=DEVICE)

        with torch.no_grad():
            with autocast('cuda', enabled=config.use_amp):
                log_densities, targets, _ = model.forward_train(features, horizon_indices)

        # Get density for max horizon (index 0 since we only requested one)
        future_log_density = log_densities[0, 0, :, :].cpu()
        density = torch.exp(future_log_density).numpy()

        # Denormalize positions
        feat_np = features[0].cpu().numpy()
        lat = feat_np[:, 0] * config.lat_std + config.lat_mean
        lon = feat_np[:, 1] * config.lon_std + config.lon_mean

        # Get time deltas for proper dead reckoning
        dt_norm = feat_np[:, 5]
        dt_seconds = dt_norm * config.dt_max
        future_dt = dt_seconds[seq_len:seq_len + max_horizon]
        cumulative_time = np.cumsum(future_dt)

        src_lat = lat[seq_len - 1]
        src_lon = lon[seq_len - 1]

        # Actual future at max horizon
        actual_lat = lat[seq_len + h - 1]
        actual_lon = lon[seq_len + h - 1]

        # Dead reckoning with proper velocity
        vlat = lat[seq_len - 1] - lat[seq_len - 2]
        vlon = lon[seq_len - 1] - lon[seq_len - 2]
        last_dt = max(dt_seconds[seq_len - 1], 1.0)
        velocity_lat = vlat / last_dt
        velocity_lon = vlon / last_dt
        cum_t = cumulative_time[h - 1] if h - 1 < len(cumulative_time) else h * 10
        dr_lat = src_lat + velocity_lat * cum_t
        dr_lon = src_lon + velocity_lon * cum_t

        # Plot heatmap
        extent = [src_lon - grid_range, src_lon + grid_range,
                  src_lat - grid_range, src_lat + grid_range]
        im = ax.imshow(density, origin='lower', extent=extent,
                       cmap='hot', alpha=0.8, aspect='auto')

        # Plot trajectory (last portion of history)
        history_show = min(20, seq_len)  # Show last 20 points of history
        ax.plot(lon[seq_len-history_show:seq_len], lat[seq_len-history_show:seq_len],
                'b-', linewidth=1.5, alpha=0.7)
        ax.plot(lon[seq_len-1], lat[seq_len-1], 'bo', markersize=8)

        # Plot actual future position
        ax.plot(actual_lon, actual_lat, 'g*', markersize=18,
                markeredgecolor='white', markeredgewidth=1.5)

        # Plot dead reckoning
        ax.plot(dr_lon, dr_lat, 'rx', markersize=14, markeredgewidth=2.5)

        # Model expected value
        x = np.linspace(-grid_range, grid_range, grid_size)
        y = np.linspace(-grid_range, grid_range, grid_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        exp_dlat = (density * xx).sum()
        exp_dlon = (density * yy).sum()
        model_lat = src_lat + exp_dlat
        model_lon = src_lon + exp_dlon
        ax.plot(model_lon, model_lat, 'c^', markersize=12)

        # Compute errors in meters
        model_error = np.sqrt((exp_dlat - (actual_lat - src_lat))**2 +
                              (exp_dlon - (actual_lon - src_lon))**2) * 111 * 1000
        dr_error = np.sqrt((dr_lat - actual_lat)**2 + (dr_lon - actual_lon)**2) * 111 * 1000
        model_errors.append(model_error)
        dr_errors.append(dr_error)

        # Check if actual is in grid
        in_grid = (abs(actual_lat - src_lat) <= grid_range and
                   abs(actual_lon - src_lon) <= grid_range)
        grid_status = "" if in_grid else " [OUT]"

        ax.set_xlim(src_lon - grid_range * 1.2, src_lon + grid_range * 1.2)
        ax.set_ylim(src_lat - grid_range * 1.2, src_lat + grid_range * 1.2)
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        ax.set_title(f'Track {plot_idx+1}: M={model_error:.0f}m DR={dr_error:.0f}m ({cum_t:.0f}s){grid_status}',
                     fontsize=10)
        ax.tick_params(labelsize=8)

    # Add overall legend and stats
    legend_elements = [
        plt.Line2D([0], [0], color='b', linewidth=2, label='History'),
        plt.Line2D([0], [0], marker='o', color='b', markersize=8, linestyle='', label='Source'),
        plt.Line2D([0], [0], marker='*', color='g', markersize=15, linestyle='', label=f'Actual (h={h})'),
        plt.Line2D([0], [0], marker='x', color='r', markersize=12, linestyle='', markeredgewidth=2, label='Dead Reckon'),
        plt.Line2D([0], [0], marker='^', color='c', markersize=10, linestyle='', label='Model Mean'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    # Compute summary stats
    mean_model = np.mean(model_errors)
    mean_dr = np.mean(dr_errors)
    improvement = (mean_dr - mean_model) / mean_dr * 100

    plt.suptitle(f'Maximum Horizon Predictions (Horizon {h})\n'
                 f'Mean Model Error: {mean_model:.0f}m | Mean DR Error: {mean_dr:.0f}m | '
                 f'Improvement: {improvement:+.1f}%',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def find_latest_checkpoint(exp_dir: Path):
    """Find the latest checkpoint file in the experiment directory."""
    # Check in checkpoints subfolder first
    checkpoints_dir = exp_dir / 'checkpoints'
    if checkpoints_dir.exists():
        search_dir = checkpoints_dir
    else:
        search_dir = exp_dir  # Fall back to root for old experiments

    # First try best_model.pt
    best_model = search_dir / 'best_model.pt'
    if best_model.exists():
        return best_model, "best_model"

    # Otherwise find latest checkpoint_step_N.pt
    checkpoints = list(search_dir.glob('checkpoint_step_*.pt'))
    if not checkpoints:
        return None, None

    # Extract step numbers and find max
    def get_step(p):
        try:
            return int(p.stem.split('_')[-1])
        except:
            return 0

    latest = max(checkpoints, key=get_step)
    step = get_step(latest)
    return latest, f"checkpoint_step_{step}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate training history plot')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name (loads from experiments/<exp-name>/)')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING TRAINING HISTORY PLOT")
    print("=" * 70)

    # Set up paths
    exp_dir = Path(__file__).parent / 'experiments' / args.exp_name
    if not exp_dir.exists():
        print(f"  ERROR: Experiment directory not found: {exp_dir}")
        print(f"  Available experiments:")
        experiments_dir = Path(__file__).parent / 'experiments'
        if experiments_dir.exists():
            for d in sorted(experiments_dir.iterdir()):
                if d.is_dir():
                    print(f"    - {d.name}")
        return

    print(f"\nExperiment: {args.exp_name}")
    print(f"Directory: {exp_dir}")

    # Use results subfolder if it exists
    results_dir = exp_dir / 'results'
    if results_dir.exists():
        output_dir = results_dir
    else:
        output_dir = exp_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        output_dir = results_dir

    # Generate training history plot
    print("\nGenerating training history plot...")
    fig_history = plot_training_history(output_dir)
    if fig_history is not None:
        fig_history.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close(fig_history)
        print(f"  Saved: training_history.png")
    else:
        print("  Skipped (no training.log found)")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_dir / 'training_history.png'}")
    print("Run make_horizon_videos.py for animated prediction videos")


if __name__ == '__main__':
    main()
