#!/usr/bin/env python3
"""Generate validation loss plots for trained models."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def parse_training_log(log_path):
    """Parse training log to extract validation losses."""
    steps = []
    model_losses = []
    dr_losses = []

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all validation blocks
    val_pattern = r'VALIDATION at step (\d+).*?Model:\s+([\d.]+).*?Dead Reckoning:\s+([\d.]+)'
    matches = re.findall(val_pattern, content, re.DOTALL)

    for match in matches:
        step = int(match[0])
        model_loss = float(match[1])
        dr_loss = float(match[2])
        steps.append(step)
        model_losses.append(model_loss)
        dr_losses.append(dr_loss)

    return steps, model_losses, dr_losses

def plot_single_model(log_path, output_path, title):
    """Plot validation loss for a single model."""
    steps, model_losses, dr_losses = parse_training_log(log_path)

    if not steps:
        print(f"No validation data found in {log_path}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, model_losses, 'b-', linewidth=2, label='Model', marker='o', markersize=3)
    ax.axhline(y=dr_losses[0], color='r', linestyle='--', linewidth=2, label=f'Dead Reckoning ({dr_losses[0]:.2f})')

    # Find best
    best_idx = model_losses.index(min(model_losses))
    best_step = steps[best_idx]
    best_loss = model_losses[best_idx]

    ax.axvline(x=best_step, color='g', linestyle=':', alpha=0.7)
    ax.scatter([best_step], [best_loss], color='g', s=100, zorder=5, label=f'Best: {best_loss:.2f} @ step {best_step}')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation Loss (NLL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return best_step, best_loss

def plot_comparison(logs_info, output_path, title):
    """Plot comparison of multiple models."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['blue', 'green', 'orange', 'purple']
    dr_loss = None

    for i, (log_path, label) in enumerate(logs_info):
        steps, model_losses, dr_losses = parse_training_log(log_path)
        if not steps:
            print(f"No data for {label}")
            continue

        color = colors[i % len(colors)]
        ax.plot(steps, model_losses, color=color, linewidth=2, label=label, marker='o', markersize=2, alpha=0.8)

        # Mark best point
        best_idx = model_losses.index(min(model_losses))
        best_step = steps[best_idx]
        best_loss = model_losses[best_idx]
        ax.scatter([best_step], [best_loss], color=color, s=80, zorder=5,
                   edgecolors='white', linewidths=1)
        ax.annotate(f'{best_loss:.2f}', (best_step, best_loss),
                    textcoords="offset points", xytext=(5, 5), fontsize=9, color=color)

        if dr_loss is None:
            dr_loss = dr_losses[0]

    if dr_loss:
        ax.axhline(y=dr_loss, color='red', linestyle='--', linewidth=2,
                   label=f'Dead Reckoning ({dr_loss:.2f})')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation Loss (NLL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Define log paths
    log_18m = base_dir / 'experiments' / 'run_18M_h800' / 'results' / 'training.log'
    log_100m = base_dir / 'experiments' / 'run_100M_h800' / 'results' / 'training.log'
    log_grid12_100m = base_dir / 'experiments' / 'run_grid1.2_100M' / 'training.log'

    # Check which logs exist
    print("=" * 60)
    print("GENERATING VALIDATION PLOTS")
    print("=" * 60)

    # Plot 18M model (grid_range=0.6)
    if log_18m.exists():
        print(f"\n18M Model (grid_range=0.6):")
        result = plot_single_model(log_18m, results_dir / '18M_validation_loss.png',
                          '18M Model Validation Loss (grid_range=0.6)')
        if result:
            print(f"  Best loss: {result[1]:.2f} at step {result[0]}")
    else:
        print(f"Log not found: {log_18m}")

    # Plot 100M model (grid_range=0.6)
    if log_100m.exists():
        print(f"\n100M Model (grid_range=0.6):")
        result = plot_single_model(log_100m, results_dir / '100M_validation_loss.png',
                          '100M Model Validation Loss (grid_range=0.6)')
        if result:
            print(f"  Best loss: {result[1]:.2f} at step {result[0]}")
    else:
        print(f"Log not found: {log_100m}")

    # Plot 100M model (grid_range=1.2) if it exists
    if log_grid12_100m.exists():
        print(f"\n100M Model (grid_range=1.2):")
        result = plot_single_model(log_grid12_100m, results_dir / '100M_grid1.2_validation_loss.png',
                          '100M Model Validation Loss (grid_range=1.2)')
        if result:
            print(f"  Best loss: {result[1]:.2f} at step {result[0]}")

    # Comparison plot: 18M vs 100M (grid_range=0.6)
    print("\n" + "=" * 60)
    print("COMPARISON PLOTS")
    print("=" * 60)

    comparison_logs = []
    if log_18m.exists():
        comparison_logs.append((log_18m, '18M Model (grid=0.6)'))
    if log_100m.exists():
        comparison_logs.append((log_100m, '100M Model (grid=0.6)'))

    if len(comparison_logs) >= 2:
        print("\n18M vs 100M (grid_range=0.6):")
        plot_comparison(comparison_logs, results_dir / 'comparison_18M_vs_100M.png',
                       'Model Comparison: 18M vs 100M (grid_range=0.6)')

    # All models comparison if grid1.2 exists
    if log_grid12_100m.exists():
        all_logs = comparison_logs.copy()
        all_logs.append((log_grid12_100m, '100M Model (grid=1.2)'))
        print("\nAll models comparison:")
        plot_comparison(all_logs, results_dir / 'comparison_all_models.png',
                       'Model Comparison: All Trained Models')

    print("\n" + "=" * 60)
    print("PLOTS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {results_dir}")


if __name__ == '__main__':
    main()
