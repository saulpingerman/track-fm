#!/usr/bin/env python
"""
Correctly visualize 8-step ahead predictions
This time actually computing the PDF for the 8th future position
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import polars as pl
import boto3
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Circle
import random

# Import model components
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from four_head_2D_LR import FourierHead2DLR
from nano_gpt_trajectory import TrajectoryGPT, TrajectoryGPTConfig

# Import training components
from train_causal_multihorizon import (
    NanoGPTTrajectoryForecaster,
    StreamingMultiHorizonAISDataset,
    latlon_to_local_uv
)

def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same config as training
    model = NanoGPTTrajectoryForecaster(
        seq_len=20,
        d_model=128,
        nhead=8,
        num_layers=6,
        ff_hidden=512,
        fourier_m=128,
        fourier_rank=4
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from step {checkpoint['global_step']}")
    return model

def get_first_sample(dataset):
    """Get the first suitable sample from dataset"""
    data_iter = iter(dataset)
    
    while True:
        try:
            input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
            
            if causal_position == 19:
                return {
                    'input_uv': input_uv,
                    'centre_ll': centre_ll,
                    'target_uv': target_uv,
                    'logJ': logJ,
                    'causal_position': torch.tensor(causal_position)
                }
        except StopIteration:
            break
    
    return None

def evaluate_pdf_for_horizon(model, input_uv, centre_ll, causal_position, horizon_step, grid_size=200, extent=0.2):
    """
    Evaluate model's PDF for a specific horizon step (0-indexed)
    This correctly computes P(x_{t+horizon_step+1} | x_{1:t})
    """
    device = next(model.parameters()).device
    
    # Create grid for evaluation
    x = torch.linspace(-extent, extent, grid_size)
    y = torch.linspace(-extent, extent, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    
    # Prepare inputs
    input_uv = input_uv.unsqueeze(0).to(device)
    centre_ll = centre_ll.unsqueeze(0).to(device)
    causal_position = causal_position.unsqueeze(0).to(device)
    
    # Get model hidden representation
    B, S, _ = input_uv.shape
    context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
    input_sequence = torch.cat([input_uv, context_expanded], dim=-1)
    
    with torch.no_grad():
        causal_hidden = model.gpt(input_sequence, causal_position)
        output_repr = model.output_proj(causal_hidden)
    
    # The model predicts multiple horizons. We need to evaluate the PDF
    # for the specific horizon step we're interested in.
    # Since the model's forward method expects target_uv of shape (B, H, 2),
    # we'll create a dummy target and extract just the horizon we want.
    
    # Create dummy targets for all horizons
    H = 10  # model's horizon
    dummy_targets = torch.zeros(1, H, 2).to(device)
    
    # Evaluate PDF at grid points for our specific horizon
    pdf_values = []
    batch_size = 500
    
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_pdf_values = []
        
        for point in batch_points:
            # Set the target at our horizon to this grid point
            dummy_targets[0, horizon_step, :] = point
            
            # Get log probabilities for all horizons
            with torch.no_grad():
                log_probs = model(input_uv, centre_ll, dummy_targets, causal_position)
            
            # Extract just the horizon we care about
            log_prob = log_probs[0, horizon_step]
            pdf_value = torch.exp(log_prob)
            batch_pdf_values.append(pdf_value)
        
        pdf_values.extend(batch_pdf_values)
    
    pdf_values = torch.stack(pdf_values).reshape(grid_size, grid_size)
    
    # Calculate integral
    dx = (2 * extent) / grid_size
    dy = (2 * extent) / grid_size
    integral = pdf_values.sum().item() * dx * dy
    
    return xx.cpu().numpy(), yy.cpu().numpy(), pdf_values.cpu().numpy(), integral

def plot_correct_8step_forecast(model, sample, save_path):
    """Create correct visualization of 8-step ahead forecast"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot 1: Full trajectory
    ax = axes[0]
    ax.set_title('Full Trajectory Overview', fontsize=14)
    
    input_np = input_uv.numpy()
    target_np = target_uv.numpy()
    
    ax.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
    ax.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=100, marker='s', label='Current position', zorder=5)
    ax.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=8, 
             label='True future track', alpha=0.7)
    
    # Highlight different horizon predictions
    horizons_to_show = [0, 2, 4, 6, 7]  # 1, 3, 5, 7, 8 steps
    colors = ['orange', 'purple', 'brown', 'pink', 'lime']
    
    for h, color in zip(horizons_to_show, colors):
        if h < len(target_np):
            ax.scatter(target_np[h, 0], target_np[h, 1], c=color, s=150, 
                      marker='*', label=f'{h+1} steps', zorder=6, 
                      edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Local X', fontsize=12)
    ax.set_ylabel('Local Y', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot PDFs for different horizons
    for plot_idx, (horizon, color) in enumerate(zip(horizons_to_show, colors)):
        ax = axes[plot_idx + 1]
        ax.set_title(f'{horizon+1}-Step Ahead PDF', fontsize=12)
        
        print(f"Computing PDF for {horizon+1} steps ahead...")
        
        # Compute PDF for this specific horizon
        extent = 0.05 * (horizon + 1)  # Increase extent with horizon
        xx, yy, pdf_values, integral = evaluate_pdf_for_horizon(
            model, input_uv, centre_ll, causal_position, 
            horizon_step=horizon, grid_size=150, extent=extent
        )
        
        # Plot PDF
        if pdf_values.max() > 0:
            im = ax.contourf(xx, yy, pdf_values, levels=30, cmap='hot')
            ax.contour(xx, yy, pdf_values, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        # Show trajectory context (relative to current position)
        last_pos = input_np[-1]
        
        # Recent track
        n_recent = 8
        recent_track = input_np[-n_recent:] - last_pos
        ax.plot(recent_track[:, 0], recent_track[:, 1], 'c.-', 
                linewidth=2, markersize=6, alpha=0.7, label='Recent')
        
        # Current position (origin)
        ax.scatter(0, 0, c='cyan', s=150, marker='s', 
                  edgecolor='black', linewidth=2, zorder=5)
        
        # True position at this horizon
        if horizon < len(target_np):
            true_pos = target_np[horizon] - last_pos
            ax.scatter(true_pos[0], true_pos[1], c=color, s=150, 
                      marker='*', edgecolor='black', linewidth=2, zorder=5,
                      label=f'True {horizon+1}-step')
            
            # Show path to this position
            path_to_horizon = target_np[:horizon+1] - last_pos
            ax.plot(path_to_horizon[:, 0], path_to_horizon[:, 1], 
                   'w--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Relative X', fontsize=10)
        ax.set_ylabel('Relative Y', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        
        # Show stats
        max_pdf = pdf_values.max()
        ax.text(0.02, 0.98, f'∫PDF={integral:.3f}\nmax={max_pdf:.1f}', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        if plot_idx == 0:
            ax.legend(fontsize=8)
    
    # Add title
    lat0 = centre_ll[0].item() * 90
    lon0 = centre_ll[1].item() * 180
    fig.suptitle(f'Correct Multi-Horizon Predictions - Center: ({lat0:.2f}°, {lon0:.2f}°)', 
                 fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correct multi-horizon plot to {save_path}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_path = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep/correct_multihorizon_sample_1.png")
    
    # Load model
    model = load_checkpoint(checkpoint_path, device)
    
    # Create dataset
    print("Creating dataset...")
    import os
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    dataset = StreamingMultiHorizonAISDataset(
        bucket_name="ais-pipeline-data-10179bbf-us-east-1",
        seq_len=20,
        horizon=10
    )
    
    # Get first sample
    print("Getting sample...")
    sample = get_first_sample(dataset)
    
    if sample is None:
        print("Could not find suitable sample")
        return
    
    # Create correct visualization
    print("Generating correct multi-horizon visualization...")
    plot_correct_8step_forecast(model, sample, output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()