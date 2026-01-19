#!/usr/bin/env python3
"""
Combine validation loss curves from v3 (18M) and v4 (116M) experiments.

Data extracted from training logs - validation occurs at geometrically
increasing step intervals (not every epoch).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Validation losses and their corresponding step numbers (extracted from training logs)
# v3: 18M params (d_model=384, nhead=16, num_layers=8, dim_ff=2048)
v3_steps = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 28998]
v3_losses = [25.0439, 19.3476, 11.2766, 9.0508, 6.0911, 4.5810, 3.3784, 3.4763,
             2.9224, 2.7982, 2.3226, 2.3994, 2.2526, 1.9605, 1.8127]

# v4: 116M params (d_model=768, nhead=16, num_layers=16, dim_ff=3072)
v4_steps = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383,
            28998, 41613, 54228, 66843, 79458]
v4_losses = [73.4967, 73.4967, 61.5975, 79.5290, 66.7052, 45.3716, 24.9223, 9.3025,
             5.6760, 3.1153, 2.9466, 2.2419, 2.1214, 1.9822, 1.8377, 1.7820,
             1.7575, 1.7501, 1.7467]

# Baselines (same for both since same validation set)
dead_reckoning = 5.0121
last_position = 7.3674

# Convert to numpy arrays
v3_steps = np.array(v3_steps)
v3_losses = np.array(v3_losses)
v4_steps = np.array(v4_steps)
v4_losses = np.array(v4_losses)

# Create figure with log scale x-axis (steps grow geometrically)
fig, ax = plt.subplots(figsize=(12, 7))

# Plot validation losses
ax.plot(v3_steps, v3_losses, 'b-o', linewidth=2, markersize=6,
        label=f'v3 Large (18M params) - Best: {min(v3_losses):.4f}')
ax.plot(v4_steps, v4_losses, 'r-s', linewidth=2, markersize=6,
        label=f'v4 XLarge (116M params) - Best: {min(v4_losses):.4f}')

# Plot baselines
max_step = max(v3_steps[-1], v4_steps[-1])
ax.axhline(y=dead_reckoning, color='orange', linestyle='--', linewidth=2,
           label=f'Dead Reckoning Baseline: {dead_reckoning:.4f}')
ax.axhline(y=last_position, color='green', linestyle='--', linewidth=2,
           label=f'Last Position Baseline: {last_position:.4f}')

# Formatting - use log scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Training Step (log scale)', fontsize=12)
ax.set_ylabel('Validation Loss (NLL, log scale)', fontsize=12)
ax.set_title('Validation Loss: 18M vs 116M Parameter Models\n(69 Days AIS Data, Causal Subwindow Training)',
             fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Set y-axis range
ax.set_ylim(1, 100)

# Add annotations for final values
ax.annotate(f'{v3_losses[-1]:.4f}',
            xy=(v3_steps[-1], v3_losses[-1]),
            xytext=(v3_steps[-1] * 1.1, v3_losses[-1] + 1),
            fontsize=10, color='blue')
ax.annotate(f'{v4_losses[-1]:.4f}',
            xy=(v4_steps[-1], v4_losses[-1]),
            xytext=(v4_steps[-1] * 1.1, v4_losses[-1] + 0.5),
            fontsize=10, color='red')

# Save figure
output_dir = Path(__file__).parent.parent / 'results'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'combined_validation_loss.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved combined validation loss plot to {output_path}")

# Also create a zoomed version focusing on convergence
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.plot(v3_steps, v3_losses, 'b-o', linewidth=2, markersize=6,
         label=f'v3 Large (18M params)')
ax2.plot(v4_steps, v4_losses, 'r-s', linewidth=2, markersize=6,
         label=f'v4 XLarge (116M params)')
ax2.axhline(y=dead_reckoning, color='orange', linestyle='--', linewidth=2,
            label=f'Dead Reckoning: {dead_reckoning:.4f}')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Training Step (log scale)', fontsize=12)
ax2.set_ylabel('Validation Loss (NLL, log scale)', fontsize=12)
ax2.set_title('Validation Loss (Zoomed): 18M vs 116M Parameter Models', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_ylim(1, 10)

output_path2 = output_dir / 'combined_validation_loss_zoomed.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Saved zoomed validation loss plot to {output_path2}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

best_v3_idx = np.argmin(v3_losses)
best_v4_idx = np.argmin(v4_losses)

print(f"\nv3 Large (18M params):")
print(f"  Validation points: {len(v3_losses)}")
print(f"  Training steps: {v3_steps[-1]:,}")
print(f"  Best loss: {min(v3_losses):.4f} (at step {v3_steps[best_v3_idx]:,})")
print(f"  Final loss: {v3_losses[-1]:.4f}")
print(f"  vs Dead Reckoning: {100*(1 - min(v3_losses)/dead_reckoning):.1f}% better")
print(f"  vs Last Position: {100*(1 - min(v3_losses)/last_position):.1f}% better")

print(f"\nv4 XLarge (116M params):")
print(f"  Validation points: {len(v4_losses)}")
print(f"  Training steps: {v4_steps[-1]:,}")
print(f"  Best loss: {min(v4_losses):.4f} (at step {v4_steps[best_v4_idx]:,})")
print(f"  Final loss: {v4_losses[-1]:.4f}")
print(f"  vs Dead Reckoning: {100*(1 - min(v4_losses)/dead_reckoning):.1f}% better")
print(f"  vs Last Position: {100*(1 - min(v4_losses)/last_position):.1f}% better")

print(f"\nImprovement of 116M over 18M:")
print(f"  Loss reduction: {min(v3_losses) - min(v4_losses):.4f}")
print(f"  Relative improvement: {100*(min(v3_losses) - min(v4_losses))/min(v3_losses):.1f}%")

plt.show()
