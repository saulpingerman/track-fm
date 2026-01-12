#!/usr/bin/env python3
"""
Test if the model is truly translation-invariant.

If the model learned relative displacement correctly, shifting the entire
track by a constant offset should produce the same prediction.
"""

import torch
import numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from run_experiment import (
    generate_relative_displacement_data,
    TransformerFourierModel,
    train_fourier,
    get_mode_prediction
)


def test_translation_invariance():
    print("Testing translation invariance...")
    print("=" * 60)

    # Configuration
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

    # Generate data and train model
    print("Training model...")
    train_input, train_target, scale_factor = generate_relative_displacement_data(
        config['num_train'], config['seq_len'], config['velocity_range'],
        config['dt'], config['endpoint_range']
    )
    val_input, val_target, _ = generate_relative_displacement_data(
        config['num_val'], config['seq_len'], config['velocity_range'],
        config['dt'], config['endpoint_range']
    )

    train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )

    model = TransformerFourierModel(
        d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'],
        grid_size=config['grid_size'], num_freqs=config['num_freqs'], grid_range=config['grid_range']
    ).to(device)

    train_fourier(model, train_loader, (val_input, val_target),
                  config['num_epochs'], config['learning_rate'], None,
                  'true_pointwise', config['grid_range'], config['grid_size'], scale_factor)

    # Test translation invariance
    print("\n" + "=" * 60)
    print("TRANSLATION INVARIANCE TEST")
    print("=" * 60)

    model.eval()

    # Take 10 sample tracks
    test_tracks = val_input[:10].clone()

    # Different translation offsets to test
    offsets = [
        (0, 0),      # No offset (baseline)
        (100, 0),    # Shift right by 100
        (0, 100),    # Shift up by 100
        (100, 100),  # Shift diagonally
        (-50, -50),  # Shift to negative
        (1000, 1000) # Large shift
    ]

    print("\nPredictions for same track at different absolute positions:")
    print("-" * 60)

    with torch.no_grad():
        for i in range(3):  # Test first 3 tracks
            print(f"\nTrack {i}:")
            print(f"  Original endpoint: ({test_tracks[i, -1, 0]:.2f}, {test_tracks[i, -1, 1]:.2f})")
            print(f"  True displacement (normalized): ({val_target[i, 0]:.3f}, {val_target[i, 1]:.3f})")
            print()

            predictions = []
            for dx, dy in offsets:
                # Shift the track
                shifted_track = test_tracks[i:i+1].clone()
                shifted_track[:, :, 0] += dx
                shifted_track[:, :, 1] += dy

                # Get prediction
                density, _, _, _ = model(shifted_track.to(device))
                pred = get_mode_prediction(density, config['grid_range'])
                pred = pred.cpu()

                predictions.append((dx, dy, pred[0, 0].item(), pred[0, 1].item()))
                print(f"  Offset ({dx:5d}, {dy:5d}): Pred = ({pred[0, 0]:.3f}, {pred[0, 1]:.3f})")

            # Check consistency
            preds = np.array([(p[2], p[3]) for p in predictions])
            std_x = preds[:, 0].std()
            std_y = preds[:, 1].std()
            print(f"  Std across offsets: x={std_x:.4f}, y={std_y:.4f}")

            if std_x < 0.05 and std_y < 0.05:
                print("  ✓ TRANSLATION INVARIANT")
            else:
                print("  ✗ NOT TRANSLATION INVARIANT")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
If predictions are consistent across offsets, the model learned
to extract velocity from position differences (relative motion).

If predictions vary with offset, the model may be exploiting
absolute position information.
""")


if __name__ == '__main__':
    test_translation_invariance()
