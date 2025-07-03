#!/usr/bin/env python
"""
Create a highly zoomed visualization for a specific sample
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

def plot_ultra_zoomed_forecast(model, sample, save_path, forecast_step=7, zoom_levels=[0.075, 0.04, 0.02, 0.01]):
    """Create multiple zoom levels for the same forecast"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    # Create figure with 2x3 subplots for different zoom levels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # First subplot: Full trajectory
    ax = axes[0]
    ax.set_title('Full Trajectory', fontsize=14)
    
    # Plot full tracks
    input_np = input_uv.numpy()
    target_np = target_uv.numpy()
    
    ax.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
    ax.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=100, marker='s', label='Current position', zorder=5)
    ax.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=8, 
             label='True future track', alpha=0.7)
    
    # Highlight the prediction target
    if forecast_step < len(target_np):
        ax.scatter(target_np[forecast_step, 0], target_np[forecast_step, 1], 
                   c='lime', s=200, marker='*', label=f'Target (8 steps)', 
                   zorder=6, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Local X (normalized)', fontsize=12)
    ax.set_ylabel('Local Y (normalized)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Get model predictions once
    device = next(model.parameters()).device
    input_uv_t = input_uv.unsqueeze(0).to(device)
    centre_ll_t = centre_ll.unsqueeze(0).to(device)
    causal_position_t = causal_position.unsqueeze(0).to(device)
    
    B, S, _ = input_uv_t.shape
    context_expanded = centre_ll_t.unsqueeze(1).expand(-1, S, -1)
    input_sequence = torch.cat([input_uv_t, context_expanded], dim=-1)
    
    with torch.no_grad():
        causal_hidden = model.gpt(input_sequence, causal_position_t)
        output_repr = model.output_proj(causal_hidden)
    
    # Plot PDFs at different zoom levels
    for plot_idx, extent in enumerate(zoom_levels):
        ax = axes[plot_idx + 1]
        ax.set_title(f'8-Step Forecast (zoom={extent:.3f})', fontsize=12)
        
        # Create grid
        grid_size = 250  # Higher resolution for zoomed views
        x = torch.linspace(-extent, extent, grid_size)
        y = torch.linspace(-extent, extent, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
        
        # Evaluate PDF
        pdf_values = []
        batch_size = 1000
        
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i+batch_size]
            batch_output_repr = output_repr.expand(len(batch_points), -1)
            
            with torch.no_grad():
                pdfs = model.fourier_head(batch_output_repr, batch_points)
                pdf_values.append(pdfs)
        
        pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size).cpu().numpy()
        
        # Plot PDF
        im = ax.contourf(xx.cpu().numpy(), yy.cpu().numpy(), pdf_values, levels=50, cmap='hot')
        
        # Overlay trajectory information (all relative to current position)
        # Show recent track
        n_recent = 10
        recent_track = input_np[-n_recent:] - input_np[-1]
        ax.plot(recent_track[:, 0], recent_track[:, 1], 'c.-', linewidth=2, 
                markersize=6, label='Recent track', alpha=0.8)
        
        # Current position at origin
        ax.scatter(0, 0, c='cyan', s=150, marker='s', label='Current', 
                  zorder=5, edgecolor='black', linewidth=2)
        
        # Show true future trajectory
        if forecast_step < len(target_np):
            future_traj = target_np[:forecast_step+1] - input_np[-1]
            ax.plot(future_traj[:, 0], future_traj[:, 1], 'w--', linewidth=2, 
                    label='True path', alpha=0.8)
            
            # Target position
            target_pos = target_np[forecast_step] - input_np[-1]
            ax.scatter(target_pos[0], target_pos[1], c='lime', s=150, marker='*', 
                       label='True pos', zorder=6, edgecolor='black', linewidth=2)
        
        # Add contour lines
        ax.contour(xx.cpu().numpy(), yy.cpu().numpy(), pdf_values, levels=10, 
                   colors='white', alpha=0.3, linewidths=0.5)
        
        # Add colorbar for first zoomed plot
        if plot_idx == 0:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('PDF', fontsize=10)
        
        ax.set_xlabel('Relative X', fontsize=10)
        ax.set_ylabel('Relative Y', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        
        # Calculate and show integral
        dx = (2 * extent) / grid_size
        dy = (2 * extent) / grid_size
        integral = pdf_values.sum() * dx * dy
        max_pdf = pdf_values.max()
        
        ax.text(0.02, 0.98, f'∫={integral:.3f}\nmax={max_pdf:.1f}', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add zoom box in previous plot to show what we're zooming into
        if plot_idx > 0 and plot_idx < len(zoom_levels):
            prev_ax = axes[plot_idx]
            prev_extent = zoom_levels[plot_idx-1]
            rect = Rectangle((-extent, -extent), 2*extent, 2*extent, 
                           fill=False, edgecolor='green', linewidth=2, linestyle='--')
            prev_ax.add_patch(rect)
    
    # Last subplot: Show PDF values along a line
    ax = axes[5]
    ax.set_title('PDF Cross-Section', fontsize=12)
    
    # Get PDF values along the line from current to target
    if forecast_step < len(target_np):
        target_pos = target_np[forecast_step] - input_np[-1]
        
        # Create points along the line
        n_points = 100
        t = np.linspace(0, 1, n_points)
        line_x = t * target_pos[0]
        line_y = t * target_pos[1]
        
        # Evaluate PDF along this line
        line_points = torch.tensor(np.stack([line_x, line_y], axis=1), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            line_output_repr = output_repr.expand(len(line_points), -1)
            line_pdfs = model.fourier_head(line_output_repr, line_points).cpu().numpy()
        
        # Calculate distance along line
        distances = np.sqrt(line_x**2 + line_y**2)
        
        ax.plot(distances, line_pdfs, 'r-', linewidth=2)
        ax.axvline(x=np.sqrt(target_pos[0]**2 + target_pos[1]**2), 
                   color='lime', linestyle='--', linewidth=2, label='True position')
        
        ax.set_xlabel('Distance from current position', fontsize=12)
        ax.set_ylabel('PDF value', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add sample info
    lat0 = centre_ll[0].item() * 90
    lon0 = centre_ll[1].item() * 180
    fig.suptitle(f'Ultra-Zoomed 8-Step Forecast - Sample 1 - Center: ({lat0:.2f}°, {lon0:.2f}°)', 
                 fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ultra-zoomed plot to {save_path}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_path = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep/ultra_zoomed_8step_sample_1.png")
    
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
    
    # Get first sample (same as sample 1 from previous runs)
    print("Getting sample...")
    sample = get_first_sample(dataset)
    
    if sample is None:
        print("Could not find suitable sample")
        return
    
    # Create ultra-zoomed visualization
    print("Generating ultra-zoomed visualization...")
    plot_ultra_zoomed_forecast(model, sample, output_path, 
                              forecast_step=7,  # 8 steps ahead (0-indexed)
                              zoom_levels=[0.075, 0.04, 0.02, 0.01])
    
    print("Done!")

if __name__ == "__main__":
    main()