#!/usr/bin/env python3
"""
Comprehensive evaluation of causal multi-horizon prediction.

Compares our model against baselines:
1. Dead reckoning (perfect - uses true velocity)
2. Last position (predicts 0 displacement)
3. Zero prediction (always predicts origin)
4. Our model

Tests:
- Multiple trajectories with different velocities
- Different source positions (varying input history length)
- Different prediction horizons
- Irregular time spacing
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import json

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

from run_experiment import (
    CausalMultiHorizonModel,
    VariableSpacedDataset,
    compute_soft_target_loss,
)

RESULTS_DIR = Path(__file__).parent / 'comprehensive_results'
RESULTS_DIR.mkdir(exist_ok=True)


def train_model(num_epochs=30):
    """Train the model."""
    print("=" * 70, flush=True)
    print("TRAINING MODEL", flush=True)
    print("=" * 70, flush=True)

    model = CausalMultiHorizonModel(
        d_model=64, nhead=4, num_layers=2, max_horizon=5,
        grid_size=64, num_freqs=8, grid_range=40.0
    ).to(DEVICE)

    train_data = VariableSpacedDataset(5000, seq_len=15, velocity_range=(0.5, 4.5))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for pos, dt in train_loader:
            pos, dt = pos.to(DEVICE), dt.to(DEVICE)
            optimizer.zero_grad()
            log_dens, tgt = model.forward_train(pos, dt, 5)
            loss = compute_soft_target_loss(log_dens, tgt, 40.0, 64, 1.5)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch}: loss={epoch_loss/len(train_loader):.4f}", flush=True)

    return model


def get_model_prediction(model, positions, dt_values, src_idx, horizon, grid):
    """Get model's predicted displacement."""
    seq_len = positions.shape[1]
    tgt_idx = src_idx + horizon

    with torch.no_grad():
        # Only use positions up to src_idx (causal)
        input_pos = positions[:, :src_idx+1, :]
        input_dt = dt_values[:, :src_idx+1]

        x = torch.cat([input_pos, input_dt.unsqueeze(-1)], dim=-1)
        x = model.input_proj(x)
        x = x + model.pe[:, :src_idx+1]
        mask = model._get_causal_mask(src_idx+1, DEVICE)
        embeddings = model.transformer(x, mask=mask)

        # Get cumulative dt from src to tgt
        cumsum_dt = torch.cumsum(dt_values, dim=1)
        cum_dt = cumsum_dt[0, tgt_idx] - cumsum_dt[0, src_idx]

        # Predict
        src_emb = embeddings[0, -1:, :]  # Last embedding (at src_idx)
        time_enc = model.time_encoder(cum_dt.unsqueeze(0))
        combined = torch.cat([src_emb, time_enc], dim=-1)
        conditioned = model.horizon_proj(combined)
        log_dens = model.fourier_head(conditioned)
        dens = torch.exp(log_dens[0])

        pred_x = (dens.sum(dim=1) * grid).sum().item()
        pred_y = (dens.sum(dim=0) * grid).sum().item()

    return pred_x, pred_y


def get_dead_reckoning_prediction(positions, dt_values, src_idx, horizon):
    """Perfect prediction using true velocity."""
    # Compute velocity from the trajectory up to src_idx
    if src_idx < 1:
        return 0.0, 0.0

    # Use velocity between last two observed points
    dx = positions[0, src_idx, 0] - positions[0, src_idx-1, 0]
    dy = positions[0, src_idx, 1] - positions[0, src_idx-1, 1]
    dt_last = dt_values[0, src_idx].item()

    if dt_last < 0.01:
        return 0.0, 0.0

    vx = dx / dt_last
    vy = dy / dt_last

    # Cumulative dt to target
    cumsum_dt = torch.cumsum(dt_values, dim=1)
    tgt_idx = src_idx + horizon
    cum_dt = (cumsum_dt[0, tgt_idx] - cumsum_dt[0, src_idx]).item()

    # Dead reckoning: displacement = velocity * time
    pred_x = vx * cum_dt
    pred_y = vy * cum_dt

    return pred_x.item(), pred_y.item()


def evaluate_all_methods(model, test_loader, grid):
    """Evaluate all methods and collect statistics."""
    print("\n" + "=" * 70, flush=True)
    print("EVALUATING ALL METHODS", flush=True)
    print("=" * 70, flush=True)

    results = {
        'model': {'errors': [], 'sq_errors': []},
        'dead_reckoning': {'errors': [], 'sq_errors': []},
        'last_position': {'errors': [], 'sq_errors': []},  # Predicts 0 displacement
        'zero_prediction': {'errors': [], 'sq_errors': []},  # Always predicts (0,0)
    }

    # Also track by horizon and source position
    by_horizon = {h: {m: [] for m in results.keys()} for h in range(1, 6)}
    by_src_pos = {s: {m: [] for m in results.keys()} for s in [3, 5, 7, 9, 11]}

    detailed_results = []

    for batch_idx, (positions, dt_values) in enumerate(test_loader):
        positions = positions.to(DEVICE)
        dt_values = dt_values.to(DEVICE)

        seq_len = positions.shape[1]

        for src_idx in [3, 5, 7, 9, 11]:
            for horizon in range(1, 6):
                tgt_idx = src_idx + horizon
                if tgt_idx >= seq_len:
                    continue

                # Ground truth displacement
                target_x = (positions[0, tgt_idx, 0] - positions[0, src_idx, 0]).item()
                target_y = (positions[0, tgt_idx, 1] - positions[0, src_idx, 1]).item()
                target_magnitude = np.sqrt(target_x**2 + target_y**2)

                # 1. Our model
                pred_x, pred_y = get_model_prediction(model, positions, dt_values, src_idx, horizon, grid)
                error_model = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)

                # 2. Dead reckoning (perfect baseline)
                dr_x, dr_y = get_dead_reckoning_prediction(positions, dt_values, src_idx, horizon)
                error_dr = np.sqrt((dr_x - target_x)**2 + (dr_y - target_y)**2)

                # 3. Last position (predicts 0 displacement)
                error_last = target_magnitude  # Error = distance to target

                # 4. Zero prediction
                error_zero = target_magnitude  # Same as last position for relative coords

                # Store results
                results['model']['errors'].append(error_model)
                results['model']['sq_errors'].append(error_model**2)
                results['dead_reckoning']['errors'].append(error_dr)
                results['dead_reckoning']['sq_errors'].append(error_dr**2)
                results['last_position']['errors'].append(error_last)
                results['last_position']['sq_errors'].append(error_last**2)
                results['zero_prediction']['errors'].append(error_zero)
                results['zero_prediction']['sq_errors'].append(error_zero**2)

                by_horizon[horizon]['model'].append(error_model)
                by_horizon[horizon]['dead_reckoning'].append(error_dr)
                by_horizon[horizon]['last_position'].append(error_last)
                by_horizon[horizon]['zero_prediction'].append(error_zero)

                by_src_pos[src_idx]['model'].append(error_model)
                by_src_pos[src_idx]['dead_reckoning'].append(error_dr)
                by_src_pos[src_idx]['last_position'].append(error_last)
                by_src_pos[src_idx]['zero_prediction'].append(error_zero)

                detailed_results.append({
                    'src_idx': src_idx,
                    'horizon': horizon,
                    'target': (target_x, target_y),
                    'target_mag': target_magnitude,
                    'model_pred': (pred_x, pred_y),
                    'model_error': error_model,
                    'dr_pred': (dr_x, dr_y),
                    'dr_error': error_dr,
                })

        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1} samples", flush=True)

    return results, by_horizon, by_src_pos, detailed_results


def print_summary_statistics(results, by_horizon, by_src_pos):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("=" * 70, flush=True)

    print("\n### Overall MSE and Mean Error ###\n")
    print(f"{'Method':<20} {'MSE':>10} {'RMSE':>10} {'Mean Error':>12}")
    print("-" * 55)

    for method in ['model', 'dead_reckoning', 'last_position', 'zero_prediction']:
        mse = np.mean(results[method]['sq_errors'])
        rmse = np.sqrt(mse)
        mean_err = np.mean(results[method]['errors'])
        print(f"{method:<20} {mse:>10.4f} {rmse:>10.4f} {mean_err:>12.4f}")

    print("\n### MSE by Horizon ###\n")
    print(f"{'Horizon':<10}", end="")
    for method in ['model', 'dead_reckoning', 'last_position']:
        print(f"{method:>15}", end="")
    print()
    print("-" * 55)

    for h in range(1, 6):
        print(f"{h:<10}", end="")
        for method in ['model', 'dead_reckoning', 'last_position']:
            mse = np.mean([e**2 for e in by_horizon[h][method]])
            print(f"{mse:>15.4f}", end="")
        print()

    print("\n### MSE by Source Position (input history length) ###\n")
    print(f"{'Src Pos':<10}", end="")
    for method in ['model', 'dead_reckoning', 'last_position']:
        print(f"{method:>15}", end="")
    print()
    print("-" * 55)

    for s in [3, 5, 7, 9, 11]:
        print(f"{s:<10}", end="")
        for method in ['model', 'dead_reckoning', 'last_position']:
            mse = np.mean([e**2 for e in by_src_pos[s][method]])
            print(f"{mse:>15.4f}", end="")
        print()

    # Improvement over baselines
    print("\n### Model Improvement Over Baselines ###\n")
    model_mse = np.mean(results['model']['sq_errors'])
    for baseline in ['last_position', 'zero_prediction']:
        baseline_mse = np.mean(results[baseline]['sq_errors'])
        improvement = (baseline_mse - model_mse) / baseline_mse * 100
        print(f"vs {baseline}: {improvement:.1f}% MSE reduction")


def create_visualizations(model, test_loader, grid, detailed_results):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 70, flush=True)
    print("CREATING VISUALIZATIONS", flush=True)
    print("=" * 70, flush=True)

    # 1. Example predictions on multiple trajectories
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    positions_list = []
    dt_list = []
    for i, (pos, dt) in enumerate(test_loader):
        if i >= 3:
            break
        positions_list.append(pos)
        dt_list.append(dt)

    for row in range(3):
        positions = positions_list[row].to(DEVICE)
        dt_values = dt_list[row].to(DEVICE)
        positions_np = positions[0].cpu().numpy()

        for col, (src_idx, horizon) in enumerate([(5, 2), (7, 3), (9, 4), (5, 5)]):
            tgt_idx = src_idx + horizon
            if tgt_idx >= 15:
                continue

            ax = axes[row, col]

            # Get predictions
            pred_x, pred_y = get_model_prediction(model, positions, dt_values, src_idx, horizon, grid)
            dr_x, dr_y = get_dead_reckoning_prediction(positions, dt_values, src_idx, horizon)

            target_rel = positions_np[tgt_idx] - positions_np[src_idx]

            # Plot in RELATIVE coordinates
            ax.scatter([0], [0], c='green', s=200, marker='o', zorder=10, label='source (0,0)')
            ax.scatter([target_rel[0]], [target_rel[1]], c='red', s=200, marker='X',
                      zorder=10, label=f'target ({target_rel[0]:.1f}, {target_rel[1]:.1f})')
            ax.scatter([pred_x], [pred_y], c='blue', s=200, marker='+', linewidths=3,
                      zorder=10, label=f'model ({pred_x:.1f}, {pred_y:.1f})')
            ax.scatter([dr_x], [dr_y], c='orange', s=150, marker='s',
                      zorder=9, label=f'dead_reck ({dr_x:.1f}, {dr_y:.1f})')

            # Draw arrows
            ax.annotate('', xy=(target_rel[0], target_rel[1]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=2))
            ax.annotate('', xy=(pred_x, pred_y), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5, lw=2))

            error = np.sqrt((pred_x - target_rel[0])**2 + (pred_y - target_rel[1])**2)
            disp = np.sqrt(target_rel[0]**2 + target_rel[1]**2)

            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'src={src_idx}, h={horizon}\nerror={error:.2f} ({error/disp*100:.0f}%)')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Model Predictions vs Dead Reckoning (Relative Coordinates)', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'predictions_grid.png', dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'predictions_grid.png'}", flush=True)
    plt.close()

    # 2. MSE comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overall MSE
    ax = axes[0]
    methods = ['Model', 'Dead Reck', 'Last Pos', 'Zero Pred']
    method_keys = ['model', 'dead_reckoning', 'last_position', 'zero_prediction']

    # Compute MSE by method
    from collections import defaultdict
    mse_by_method = defaultdict(list)
    for r in detailed_results:
        mse_by_method['model'].append(r['model_error']**2)
        mse_by_method['dead_reckoning'].append(r['dr_error']**2)
        mse_by_method['last_position'].append(r['target_mag']**2)
        mse_by_method['zero_prediction'].append(r['target_mag']**2)

    mses = [np.mean(mse_by_method[k]) for k in method_keys]
    colors = ['blue', 'orange', 'gray', 'red']
    bars = ax.bar(methods, mses, color=colors)
    ax.set_ylabel('MSE')
    ax.set_title('Overall MSE by Method')
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mse:.2f}', ha='center', va='bottom', fontsize=10)

    # MSE by horizon
    ax = axes[1]
    horizons = [1, 2, 3, 4, 5]

    mse_model_h = []
    mse_dr_h = []
    mse_last_h = []

    for h in horizons:
        h_results = [r for r in detailed_results if r['horizon'] == h]
        mse_model_h.append(np.mean([r['model_error']**2 for r in h_results]))
        mse_dr_h.append(np.mean([r['dr_error']**2 for r in h_results]))
        mse_last_h.append(np.mean([r['target_mag']**2 for r in h_results]))

    x = np.arange(len(horizons))
    width = 0.25
    ax.bar(x - width, mse_model_h, width, label='Model', color='blue')
    ax.bar(x, mse_dr_h, width, label='Dead Reckoning', color='orange')
    ax.bar(x + width, mse_last_h, width, label='Last Position', color='gray')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('MSE')
    ax.set_title('MSE by Prediction Horizon')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()

    # MSE by source position
    ax = axes[2]
    src_positions = [3, 5, 7, 9, 11]

    mse_model_s = []
    mse_dr_s = []
    mse_last_s = []

    for s in src_positions:
        s_results = [r for r in detailed_results if r['src_idx'] == s]
        mse_model_s.append(np.mean([r['model_error']**2 for r in s_results]))
        mse_dr_s.append(np.mean([r['dr_error']**2 for r in s_results]))
        mse_last_s.append(np.mean([r['target_mag']**2 for r in s_results]))

    x = np.arange(len(src_positions))
    ax.bar(x - width, mse_model_s, width, label='Model', color='blue')
    ax.bar(x, mse_dr_s, width, label='Dead Reckoning', color='orange')
    ax.bar(x + width, mse_last_s, width, label='Last Position', color='gray')
    ax.set_xlabel('Source Position (Input History Length)')
    ax.set_ylabel('MSE')
    ax.set_title('MSE by Input History Length')
    ax.set_xticks(x)
    ax.set_xticklabels(src_positions)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'mse_comparison.png', dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'mse_comparison.png'}", flush=True)
    plt.close()

    # 3. Scatter plot: Model prediction vs Target
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_preds_x = [r['model_pred'][0] for r in detailed_results]
    model_preds_y = [r['model_pred'][1] for r in detailed_results]
    targets_x = [r['target'][0] for r in detailed_results]
    targets_y = [r['target'][1] for r in detailed_results]

    ax = axes[0]
    ax.scatter(targets_x, model_preds_x, alpha=0.3, s=20)
    lims = [-40, 40]
    ax.plot(lims, lims, 'r--', label='Perfect prediction')
    ax.set_xlabel('Target X')
    ax.set_ylabel('Predicted X')
    ax.set_title('Model: Predicted vs Target (X)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.set_aspect('equal')

    ax = axes[1]
    ax.scatter(targets_y, model_preds_y, alpha=0.3, s=20)
    ax.plot(lims, lims, 'r--', label='Perfect prediction')
    ax.set_xlabel('Target Y')
    ax.set_ylabel('Predicted Y')
    ax.set_title('Model: Predicted vs Target (Y)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'scatter_pred_vs_target.png', dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'scatter_pred_vs_target.png'}", flush=True)
    plt.close()

    # 4. Error distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    model_errors = [r['model_error'] for r in detailed_results]
    dr_errors = [r['dr_error'] for r in detailed_results]

    ax.hist(model_errors, bins=50, alpha=0.5, label=f'Model (mean={np.mean(model_errors):.2f})', color='blue')
    ax.hist(dr_errors, bins=50, alpha=0.5, label=f'Dead Reckoning (mean={np.mean(dr_errors):.2f})', color='orange')
    ax.axvline(np.mean(model_errors), color='blue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(dr_errors), color='orange', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution: Model vs Dead Reckoning')
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'error_distribution.png', dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'error_distribution.png'}", flush=True)
    plt.close()

    # 5. Relative error (error / displacement)
    fig, ax = plt.subplots(figsize=(10, 6))

    rel_errors_model = [r['model_error'] / r['target_mag'] * 100 for r in detailed_results if r['target_mag'] > 0.1]
    rel_errors_dr = [r['dr_error'] / r['target_mag'] * 100 for r in detailed_results if r['target_mag'] > 0.1]

    ax.hist(rel_errors_model, bins=50, alpha=0.5, label=f'Model (mean={np.mean(rel_errors_model):.1f}%)', color='blue')
    ax.hist(rel_errors_dr, bins=50, alpha=0.5, label=f'Dead Reckoning (mean={np.mean(rel_errors_dr):.1f}%)', color='orange')
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Relative Error Distribution (Error / Displacement %)')
    ax.set_xlim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'relative_error_distribution.png', dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'relative_error_distribution.png'}", flush=True)
    plt.close()


def main():
    print("=" * 70, flush=True)
    print("COMPREHENSIVE EVALUATION: CAUSAL MULTI-HORIZON PREDICTION", flush=True)
    print("=" * 70, flush=True)

    # Train model
    model = train_model(num_epochs=30)

    # Create test set
    print("\nCreating test set...", flush=True)
    test_data = VariableSpacedDataset(500, seq_len=15, velocity_range=(0.5, 4.5))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    grid = torch.linspace(-40, 40, 64, device=DEVICE)

    # Evaluate all methods
    results, by_horizon, by_src_pos, detailed_results = evaluate_all_methods(
        model, test_loader, grid
    )

    # Print statistics
    print_summary_statistics(results, by_horizon, by_src_pos)

    # Create visualizations
    test_loader2 = DataLoader(test_data, batch_size=1, shuffle=True)
    create_visualizations(model, test_loader2, grid, detailed_results)

    # Save results to JSON
    summary = {
        'overall_mse': {
            'model': float(np.mean(results['model']['sq_errors'])),
            'dead_reckoning': float(np.mean(results['dead_reckoning']['sq_errors'])),
            'last_position': float(np.mean(results['last_position']['sq_errors'])),
            'zero_prediction': float(np.mean(results['zero_prediction']['sq_errors'])),
        },
        'overall_mean_error': {
            'model': float(np.mean(results['model']['errors'])),
            'dead_reckoning': float(np.mean(results['dead_reckoning']['errors'])),
            'last_position': float(np.mean(results['last_position']['errors'])),
        },
        'mse_by_horizon': {
            str(h): {
                'model': float(np.mean([e**2 for e in by_horizon[h]['model']])),
                'dead_reckoning': float(np.mean([e**2 for e in by_horizon[h]['dead_reckoning']])),
                'last_position': float(np.mean([e**2 for e in by_horizon[h]['last_position']])),
            }
            for h in range(1, 6)
        },
    }

    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {RESULTS_DIR / 'summary.json'}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print(f"Results saved to: {RESULTS_DIR}", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
