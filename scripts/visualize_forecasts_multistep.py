#!/usr/bin/env python
"""
Multi-step forecast visualization with better zoom levels
Shows PDFs for multiple prediction horizons
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
from matplotlib.patches import Circle
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

def get_sample_tracks(dataset, num_samples=4):
    """Get sample tracks from dataset"""
    samples = []
    
    # Create a data iterator
    data_iter = iter(dataset)
    
    # Collect unique tracks
    seen_tracks = set()
    while len(samples) < num_samples:
        try:
            # Get next sample
            input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
            
            # Only use samples at the end of input sequence (causal_position = 19)
            if causal_position == 19:
                # Create a simple hash to avoid duplicate tracks
                track_hash = hash(tuple(input_uv.flatten().tolist()[:6]))  # First few points
                
                if track_hash not in seen_tracks:
                    seen_tracks.add(track_hash)
                    samples.append({
                        'input_uv': input_uv,
                        'centre_ll': centre_ll,
                        'target_uv': target_uv,
                        'logJ': logJ,
                        'causal_position': torch.tensor(causal_position)
                    })
                    print(f"Collected sample {len(samples)}/{num_samples}")
        except StopIteration:
            break
    
    return samples

def evaluate_pdf_on_grid_multistep(model, input_uv, centre_ll, causal_position, target_step=0, grid_size=150):
    """Evaluate model's PDF on a grid for a specific prediction step"""
    device = next(model.parameters()).device
    
    # Determine extent based on how far ahead we're predicting
    # Further predictions need larger extent
    base_extent = 0.05
    extent = base_extent * (target_step + 1)
    
    # Create grid centered at origin (predictions are relative to last input)
    x = torch.linspace(-extent, extent, grid_size)
    y = torch.linspace(-extent, extent, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten grid for batch evaluation
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    
    # Prepare inputs
    input_uv = input_uv.unsqueeze(0).to(device)  # Add batch dimension
    centre_ll = centre_ll.unsqueeze(0).to(device)
    causal_position = causal_position.unsqueeze(0).to(device)
    
    # Get model output representation
    B, S, _ = input_uv.shape
    context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
    input_sequence = torch.cat([input_uv, context_expanded], dim=-1)
    
    with torch.no_grad():
        causal_hidden = model.gpt(input_sequence, causal_position)
        output_repr = model.output_proj(causal_hidden)
    
    # Evaluate PDF at each grid point for the specified prediction step
    pdf_values = []
    batch_size = 1000  # Process grid points in batches
    
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_output_repr = output_repr.expand(len(batch_points), -1)
        
        with torch.no_grad():
            # Note: This evaluates PDF for the first step. For multi-step, 
            # we'd need to modify the model to return PDFs for all steps
            pdfs = model.fourier_head(batch_output_repr, batch_points)
            pdf_values.append(pdfs)
    
    pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size)
    
    # Calculate integral
    dx = (2 * extent) / grid_size
    dy = (2 * extent) / grid_size
    integral = pdf_values.sum().item() * dx * dy
    
    return xx.numpy(), yy.numpy(), pdf_values.cpu().numpy(), integral, extent

def plot_multistep_forecast(model, sample, sample_idx, save_path):
    """Create visualization showing PDFs for multiple forecast steps"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    # Create figure with 2x3 subplots for different prediction horizons
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot full trajectory in first subplot
    ax = axes[0]
    ax.set_title('Full Trajectory Overview', fontsize=14)
    
    # Plot input track
    input_np = input_uv.numpy()
    ax.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=6, label='Input track')
    ax.scatter(input_np[0, 0], input_np[0, 1], c='green', s=80, marker='o', label='Start', zorder=5)
    ax.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=80, marker='s', label='Current', zorder=5)
    
    # Plot target positions
    target_np = target_uv.numpy()
    ax.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=6, 
             label='True future', alpha=0.7)
    
    # Highlight specific future positions that we'll show PDFs for
    steps_to_show = [0, 2, 4, 6, 9]  # 1, 3, 5, 7, and 10 steps ahead
    colors = ['orange', 'purple', 'brown', 'pink', 'gray']
    for i, (step, color) in enumerate(zip(steps_to_show, colors)):
        if step < len(target_np):
            ax.scatter(target_np[step, 0], target_np[step, 1], c=color, s=150, 
                      marker='*', label=f'{step+1} steps ahead', zorder=6, 
                      edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Local X (normalized)', fontsize=12)
    ax.set_ylabel('Local Y (normalized)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot PDFs for different prediction horizons
    for plot_idx, (step, color) in enumerate(zip(steps_to_show, colors)):
        ax = axes[plot_idx + 1]
        
        # For now, we'll show the 1-step PDF with different zoom levels
        # In a full implementation, we'd need to modify the model to predict at different horizons
        xx, yy, pdf_values, integral, extent = evaluate_pdf_on_grid_multistep(
            model, input_uv, centre_ll, causal_position, 
            target_step=step, grid_size=150
        )
        
        # Adjust title
        ax.set_title(f'{step+1}-Step Ahead PDF (zoom={extent:.3f})', fontsize=12)
        
        # Plot PDF as heatmap
        im = ax.contourf(xx, yy, pdf_values, levels=30, cmap='hot')
        
        # Show the trajectory context
        # Transform to relative coordinates (relative to last input position)
        last_input = input_np[-1]
        
        # Show last few input positions
        n_show = min(5, len(input_np))
        recent_input = input_np[-n_show:] - last_input
        ax.plot(recent_input[:, 0], recent_input[:, 1], 'c.-', 
                linewidth=2, markersize=6, alpha=0.7)
        
        # Current position (origin)
        ax.scatter(0, 0, c='cyan', s=150, marker='s', 
                  edgecolor='black', linewidth=2, zorder=5)
        
        # True position at this step
        if step < len(target_np):
            true_pos = target_np[step] - last_input
            ax.scatter(true_pos[0], true_pos[1], c=color, s=150, 
                      marker='*', edgecolor='black', linewidth=2, zorder=5)
            
            # Show trajectory up to this point
            if step > 0:
                traj_to_step = target_np[:step+1] - last_input
                ax.plot(traj_to_step[:, 0], traj_to_step[:, 1], 'k--', 
                       linewidth=1, alpha=0.5)
        
        # Add contour lines
        ax.contour(xx, yy, pdf_values, levels=8, colors='white', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('Relative X', fontsize=10)
        ax.set_ylabel('Relative Y', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add integral value
        ax.text(0.02, 0.98, f'∫PDF = {integral:.3f}', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add sample info
    lat0 = centre_ll[0].item() * 90
    lon0 = centre_ll[1].item() * 180
    fig.suptitle(f'Multi-Step Forecast - Sample {sample_idx} - Center: ({lat0:.2f}°, {lon0:.2f}°)', 
                 fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-step plot to {save_path}")

def plot_single_forecast_better_zoom(model, sample, sample_idx, save_path, forecast_step=5):
    """Create a single well-zoomed forecast visualization"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subplot 1: Trajectory with zoom box
    ax1.set_title('Trajectory with Forecast Region', fontsize=14)
    
    # Plot full tracks
    input_np = input_uv.numpy()
    target_np = target_uv.numpy()
    
    ax1.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
    ax1.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=100, marker='s', label='Current position', zorder=5)
    ax1.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=8, 
             label='True future track', alpha=0.7)
    
    # Highlight the prediction target
    if forecast_step < len(target_np):
        ax1.scatter(target_np[forecast_step, 0], target_np[forecast_step, 1], 
                   c='lime', s=200, marker='*', label=f'Target ({forecast_step+1} steps)', 
                   zorder=6, edgecolor='black', linewidth=2)
    
    # Draw zoom box around the area of interest
    last_pos = input_np[-1]
    box_size = 0.15  # Size of the zoom box
    from matplotlib.patches import Rectangle
    rect = Rectangle((last_pos[0] - box_size/2, last_pos[1] - box_size/2), 
                    box_size, box_size, fill=False, edgecolor='orange', 
                    linewidth=3, linestyle='--', label='Zoom region')
    ax1.add_patch(rect)
    
    ax1.set_xlabel('Local X (normalized)', fontsize=12)
    ax1.set_ylabel('Local Y (normalized)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Subplot 2: Zoomed PDF view
    ax2.set_title(f'Forecast PDF ({forecast_step+1} steps ahead)', fontsize=14)
    
    # Evaluate PDF with appropriate zoom
    grid_size = 200
    extent = 0.075  # Half of the box size for good coverage
    
    x = torch.linspace(-extent, extent, grid_size)
    y = torch.linspace(-extent, extent, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    
    # Get model predictions
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
    
    # Evaluate PDF
    pdf_values = []
    batch_size = 1000
    grid_points_device = grid_points.to(device)
    
    for i in range(0, len(grid_points_device), batch_size):
        batch_points = grid_points_device[i:i+batch_size]
        batch_output_repr = output_repr.expand(len(batch_points), -1)
        
        with torch.no_grad():
            pdfs = model.fourier_head(batch_output_repr, batch_points)
            pdf_values.append(pdfs)
    
    pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size).cpu().numpy()
    
    # Plot PDF
    im = ax2.contourf(xx.numpy(), yy.numpy(), pdf_values, levels=50, cmap='hot')
    
    # Overlay trajectory information (all relative to current position)
    # Show recent track
    n_recent = 8
    recent_track = input_np[-n_recent:] - input_np[-1]
    ax2.plot(recent_track[:, 0], recent_track[:, 1], 'c.-', linewidth=3, 
             markersize=8, label='Recent track', alpha=0.8)
    
    # Current position at origin
    ax2.scatter(0, 0, c='cyan', s=200, marker='s', label='Current position', 
                zorder=5, edgecolor='black', linewidth=2)
    
    # Show true future trajectory up to target
    if forecast_step < len(target_np):
        future_traj = target_np[:forecast_step+1] - input_np[-1]
        ax2.plot(future_traj[:, 0], future_traj[:, 1], 'w--', linewidth=2, 
                label='True path', alpha=0.8)
        
        # Target position
        target_pos = target_np[forecast_step] - input_np[-1]
        ax2.scatter(target_pos[0], target_pos[1], c='lime', s=200, marker='*', 
                   label=f'True position', zorder=6, edgecolor='black', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Probability Density', fontsize=12)
    
    # Add contour lines
    ax2.contour(xx.numpy(), yy.numpy(), pdf_values, levels=10, colors='white', 
                alpha=0.3, linewidths=0.5)
    
    ax2.set_xlabel('Relative X from current', fontsize=12)
    ax2.set_ylabel('Relative Y from current', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-extent, extent)
    ax2.set_ylim(-extent, extent)
    
    # Calculate and show integral
    dx = (2 * extent) / grid_size
    dy = (2 * extent) / grid_size
    integral = pdf_values.sum() * dx * dy
    
    ax2.text(0.02, 0.02, f'PDF integral: {integral:.3f}', 
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved zoomed forecast to {save_path}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_dir = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep")
    output_dir.mkdir(exist_ok=True)
    
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
    
    # Get sample tracks
    print("Collecting sample tracks...")
    samples = get_sample_tracks(dataset, num_samples=4)
    
    # Generate plots
    print("\nGenerating forecast visualizations...")
    
    for i, sample in enumerate(samples):
        # Multi-step overview
        plot_path = output_dir / f"multistep_sample_{i+1}.png"
        plot_multistep_forecast(model, sample, i+1, plot_path)
        
        # Single well-zoomed forecast at 5 steps
        plot_path = output_dir / f"zoomed_5step_sample_{i+1}.png"
        plot_single_forecast_better_zoom(model, sample, i+1, plot_path, forecast_step=4)
        
        # Single well-zoomed forecast at 8 steps
        plot_path = output_dir / f"zoomed_8step_sample_{i+1}.png"
        plot_single_forecast_better_zoom(model, sample, i+1, plot_path, forecast_step=7)
    
    print(f"\nAll plots saved to {output_dir}")

if __name__ == "__main__":
    main()