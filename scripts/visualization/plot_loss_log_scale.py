#!/usr/bin/env python
"""Plot training loss with log scale and running average"""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

def plot_loss_from_log(log_file, output_path=None, window=50):
    """Extract and plot loss values from training log"""
    
    # Read log file
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract step numbers and loss values
    # Pattern: "Batch X | Step Y | Loss: Z | LR: ..."
    pattern = r'Batch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+Loss:\s+([-\d.]+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        print("No loss values found in log file")
        return
    
    # Convert to lists
    steps = [int(match[1]) for match in matches]
    losses = [float(match[2]) for match in matches]
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot raw loss with log scale in blue
    ax.semilogx(steps, losses, 'b', linewidth=1.0, label='Raw loss')
    
    # Add running average in red
    if len(losses) > window:
        running_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        avg_steps = steps[window-1:]
        ax.semilogx(avg_steps, running_avg, 'r-', linewidth=2.5, alpha=0.9, 
                   label=f'Running average (window={window})')
    
    # Configure plot
    ax.set_xlabel('Step (log scale)', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(f'Training Loss - {Path(log_file).stem}', fontsize=16)
    ax.grid(True, alpha=0.3, which="both", linestyle='-', linewidth=0.5)
    ax.legend(loc='best', fontsize=12)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Let matplotlib automatically set the y-axis limits
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\nTraining Summary for {Path(log_file).name}:")
    print(f"Total steps: {len(steps)}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average loss: {np.mean(losses):.4f}")
    print(f"Min loss: {min(losses):.4f}")
    print(f"Max loss: {max(losses):.4f}")
    
    # Calculate improvement
    if losses[0] != 0:
        improvement = (losses[0] - losses[-1]) / abs(losses[0]) * 100
        print(f"Improvement: {improvement:.1f}%")
    
    # Print running average stats if computed
    if len(losses) > window:
        print(f"\nRunning Average (window={window}):")
        print(f"Final avg: {running_avg[-1]:.4f}")
        print(f"Min avg: {min(running_avg):.4f}")
        print(f"Max avg: {max(running_avg):.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_loss_log_scale.py <log_file> [output_path]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_loss_from_log(log_file, output_path)