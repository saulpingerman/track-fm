#!/usr/bin/env python3
"""
Generate visualizations showing tracks and predictions for Experiment 6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import from main experiment
from run_experiment import (
    generate_relative_displacement_data,
    TransformerFourierModel,
    train_fourier,
    get_mode_prediction
)


def visualize_track_and_prediction(input_seq, target, density, pred,
                                    scale_factor, grid_range, idx, save_dir):
    """
    Visualize a single track with its prediction.

    Left: The input track in absolute coordinates
    Right: The predicted density in relative displacement space
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Denormalize target for display
    target_denorm = target.numpy() * scale_factor
    pred_denorm = pred.numpy() * scale_factor

    # Left plot: Input track in absolute coordinates
    ax1 = axes[0]
    track = input_seq.numpy()

    # Plot track with markers for each point
    ax1.plot(track[:, 0], track[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(track[:, 0], track[:, 1], c='blue', s=50, zorder=5)
    ax1.scatter(track[0, 0], track[0, 1], c='green', s=150, marker='o', zorder=6, label='Start')
    ax1.scatter(track[-1, 0], track[-1, 1], c='red', s=150, marker='s', zorder=6, label='End (last known)')

    # Show true future position in absolute coords
    last_pos = track[-1]
    future_pos = last_pos + target_denorm
    ax1.scatter(future_pos[0], future_pos[1], c='cyan', s=150, marker='^', zorder=6, label='True future')

    # Draw arrow from end to future
    ax1.annotate('', xy=future_pos, xytext=last_pos,
                 arrowprops=dict(arrowstyle='->', color='cyan', lw=2))

    ax1.set_xlabel('X position', fontsize=12)
    ax1.set_ylabel('Y position', fontsize=12)
    ax1.set_title(f'Input Track\n(10 positions, constant velocity)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right plot: Predicted density in relative displacement space
    ax2 = axes[1]
    density_np = density.numpy()
    target_np = target.numpy()  # normalized
    pred_np = pred.numpy()  # normalized

    im = ax2.imshow(density_np.T, origin='lower',
                    extent=[-grid_range, grid_range, -grid_range, grid_range],
                    cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax2, label='Density')

    # Mark true target (normalized)
    ax2.scatter(target_np[0], target_np[1], c='cyan', s=200, marker='x',
                linewidths=3, label=f'True: ({target_np[0]:.2f}, {target_np[1]:.2f})')

    # Mark predicted mode (normalized)
    ax2.scatter(pred_np[0], pred_np[1], c='lime', s=200, marker='+',
                linewidths=3, label=f'Pred: ({pred_np[0]:.2f}, {pred_np[1]:.2f})')

    # Add grid lines at origin
    ax2.axhline(0, color='white', alpha=0.3, linestyle='--')
    ax2.axvline(0, color='white', alpha=0.3, linestyle='--')

    ax2.set_xlim(-grid_range, grid_range)
    ax2.set_ylim(-grid_range, grid_range)
    ax2.set_xlabel('Δx (normalized, range [-1,1])', fontsize=12)
    ax2.set_ylabel('Δy (normalized, range [-1,1])', fontsize=12)

    # Compute error in actual units
    error = np.sqrt(((pred_denorm - target_denorm) ** 2).sum())
    ax2.set_title(f'Predicted Displacement Density\nActual Δ: ({target_denorm[0]:.2f}, {target_denorm[1]:.2f}), Error: {error:.2f}', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_dir / f'example_{idx:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("Generating visualizations...")

    # Configuration (same as main experiment)
    config = {
        'num_train': 10000,
        'num_val': 1000,
        'seq_len': 10,
        'velocity_range': (0.1, 4.5),
        'dt': 1.0,
        'endpoint_range': 10.0,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'grid_size': 64,
        'num_freqs': 8,
        'grid_range': 1.0,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'num_epochs': 30,
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Generate data
    print("Generating data...")
    train_input, train_target, scale_factor = generate_relative_displacement_data(
        config['num_train'], config['seq_len'], config['velocity_range'],
        config['dt'], config['endpoint_range']
    )
    val_input, val_target, _ = generate_relative_displacement_data(
        config['num_val'], config['seq_len'], config['velocity_range'],
        config['dt'], config['endpoint_range']
    )

    print(f"Scale factor: {scale_factor} (max displacement)")

    train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )

    # Train model (true pointwise since it worked well)
    print("Training model (true pointwise)...")
    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)

    error = train_fourier(model, train_loader, (val_input, val_target),
                          config['num_epochs'], config['learning_rate'], None,
                          'true_pointwise', config['grid_range'], config['grid_size'], scale_factor)
    print(f"Final error: {error:.4f}")

    # Generate visualizations for 12 examples
    print("\nGenerating example visualizations...")
    model.eval()

    num_examples = 12
    with torch.no_grad():
        sample_input = val_input[:num_examples].to(device)
        density, _, _, _ = model(sample_input)
        density = density.cpu()
        predictions = get_mode_prediction(density, config['grid_range'])

    for i in range(num_examples):
        visualize_track_and_prediction(
            val_input[i], val_target[i], density[i], predictions[i],
            scale_factor, config['grid_range'], i, results_dir
        )
        print(f"  Saved example_{i:02d}.png")

    print(f"\nDone! {num_examples} visualizations saved to {results_dir}")


if __name__ == '__main__':
    main()
