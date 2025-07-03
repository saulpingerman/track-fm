#!/usr/bin/env python
"""
Visualize trajectory forecasts from trained model
Shows input track, forecast PDF, and true target positions
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

def evaluate_pdf_on_grid(model, input_uv, centre_ll, causal_position, grid_size=100, extent=1.0):
    """Evaluate model's PDF on a grid around the last input position"""
    device = next(model.parameters()).device
    
    # Create grid centered at origin (since positions are relative to last input)
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
    
    # Evaluate PDF at each grid point for first prediction step
    pdf_values = []
    batch_size = 1000  # Process grid points in batches
    
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_output_repr = output_repr.expand(len(batch_points), -1)
        
        with torch.no_grad():
            pdfs = model.fourier_head(batch_output_repr, batch_points)
            pdf_values.append(pdfs)
    
    pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size)
    
    return xx.numpy(), yy.numpy(), pdf_values.cpu().numpy()

def plot_forecast(model, sample, sample_idx, save_path):
    """Create visualization of forecast for a single sample"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    # Evaluate PDF on grid
    xx, yy, pdf_values = evaluate_pdf_on_grid(
        model, input_uv, centre_ll, causal_position,
        grid_size=150, extent=0.8
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subplot 1: Full trajectory view
    ax1.set_title('Full Trajectory', fontsize=14)
    
    # Plot input track
    input_np = input_uv.numpy()
    ax1.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
    ax1.scatter(input_np[0, 0], input_np[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=100, marker='s', label='Current position', zorder=5)
    
    # Plot target positions
    target_np = target_uv.numpy()
    ax1.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=8, 
             label='True future track', alpha=0.7)
    
    # Add arrows to show direction
    for i in range(0, len(input_np)-1, 3):
        dx = input_np[i+1, 0] - input_np[i, 0]
        dy = input_np[i+1, 1] - input_np[i, 1]
        ax1.arrow(input_np[i, 0], input_np[i, 1], dx*0.8, dy*0.8,
                 head_width=0.02, head_length=0.02, fc='blue', ec='blue', alpha=0.5)
    
    ax1.set_xlabel('Local X (normalized)', fontsize=12)
    ax1.set_ylabel('Local Y (normalized)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Subplot 2: Forecast PDF with zoomed view
    ax2.set_title('Forecast PDF (1-step ahead)', fontsize=14)
    
    # Plot PDF as heatmap
    im = ax2.contourf(xx, yy, pdf_values, levels=50, cmap='hot', norm=LogNorm())
    
    # Overlay input track (last few points)
    n_show = 5
    start_idx = max(0, len(input_np) - n_show)
    recent_track = input_np[start_idx:]
    ax2.plot(recent_track[:, 0], recent_track[:, 1], 'c.-', linewidth=3, 
             markersize=10, label='Recent track', alpha=0.8)
    ax2.scatter(input_np[-1, 0], input_np[-1, 1], c='cyan', s=200, 
                marker='s', label='Current position', zorder=5, edgecolor='black', linewidth=2)
    
    # Plot true next position
    ax2.scatter(target_np[0, 0], target_np[0, 1], c='lime', s=200, 
                marker='*', label='True next position', zorder=5, edgecolor='black', linewidth=2)
    
    # Add contour lines
    ax2.contour(xx, yy, pdf_values, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Probability Density', fontsize=12)
    
    ax2.set_xlabel('Local X (normalized)', fontsize=12)
    ax2.set_ylabel('Local Y (normalized)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add sample info
    lat0 = centre_ll[0].item() * 90
    lon0 = centre_ll[1].item() * 180
    fig.suptitle(f'Sample {sample_idx} - Center: ({lat0:.2f}°, {lon0:.2f}°)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {save_path}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_dir = Path("/home/ec2-user/repos/track-fm/forecast_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_checkpoint(checkpoint_path, device)
    
    # Create dataset
    print("Creating dataset...")
    import os
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'  # Set region to avoid warning
    dataset = StreamingMultiHorizonAISDataset(
        bucket_name="ais-pipeline-data-10179bbf-us-east-1",
        seq_len=20,
        horizon=10
    )
    
    # Get sample tracks
    print("Collecting sample tracks...")
    samples = get_sample_tracks(dataset, num_samples=6)
    
    # Generate plots
    print("Generating forecast visualizations...")
    for i, sample in enumerate(samples):
        plot_path = output_dir / f"forecast_sample_{i+1}.png"
        plot_forecast(model, sample, i+1, plot_path)
    
    print(f"\nAll plots saved to {output_dir}")

if __name__ == "__main__":
    main()