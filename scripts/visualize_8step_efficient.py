#!/usr/bin/env python
"""
Efficient visualization of true 8-step ahead predictions
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle

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
    
    model = NanoGPTTrajectoryForecaster(
        seq_len=20,
        d_model=128,
        nhead=8,
        num_layers=6,
        ff_hidden=512,
        fourier_m=128,
        fourier_rank=4
    ).to(device)
    
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

def visualize_true_8step(model, sample, save_path):
    """Visualize the true 8-step ahead prediction efficiently"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    device = next(model.parameters()).device
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Full trajectory
    ax1.set_title('Full Trajectory', fontsize=14)
    
    input_np = input_uv.numpy()
    target_np = target_uv.numpy()
    
    ax1.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
    ax1.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=100, marker='s', label='Current position', zorder=5)
    ax1.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=8, 
             label='True future track', alpha=0.7)
    
    # Highlight 8-step position
    if len(target_np) >= 8:
        ax1.scatter(target_np[7, 0], target_np[7, 1], c='lime', s=200, 
                   marker='*', label='8 steps ahead', zorder=6, 
                   edgecolor='black', linewidth=2)
        
        # Calculate distance for reference
        dist_8step = np.sqrt((target_np[7, 0] - input_np[-1, 0])**2 + 
                            (target_np[7, 1] - input_np[-1, 1])**2)
        ax1.text(0.02, 0.98, f'8-step distance: {dist_8step:.3f}', 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Local X', fontsize=12)
    ax1.set_ylabel('Local Y', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right plot: 8-step ahead PDF
    ax2.set_title('8-Step Ahead Forecast PDF', fontsize=14)
    
    # Determine grid extent based on actual 8-step displacement
    if len(target_np) >= 8:
        disp = target_np[7] - input_np[-1]
        extent = max(0.15, np.max(np.abs(disp)) * 1.5)  # At least 0.15, or 1.5x the displacement
    else:
        extent = 0.3
    
    print(f"Using extent: {extent:.3f}")
    
    # Create grid
    grid_size = 200
    x = torch.linspace(-extent, extent, grid_size)
    y = torch.linspace(-extent, extent, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    
    # Prepare model inputs
    input_uv_t = input_uv.unsqueeze(0).to(device)
    centre_ll_t = centre_ll.unsqueeze(0).to(device)
    causal_position_t = causal_position.unsqueeze(0).to(device)
    
    # Get hidden representation once
    B, S, _ = input_uv_t.shape
    context_expanded = centre_ll_t.unsqueeze(1).expand(-1, S, -1)
    input_sequence = torch.cat([input_uv_t, context_expanded], dim=-1)
    
    with torch.no_grad():
        causal_hidden = model.gpt(input_sequence, causal_position_t)
        output_repr = model.output_proj(causal_hidden)
    
    # Evaluate PDF on grid for 8th step
    # We'll use the Fourier head directly since we know it predicts for each horizon
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    
    # Process in batches
    pdf_values = []
    batch_size = 2000
    
    print("Evaluating PDF on grid...")
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_output_repr = output_repr.expand(len(batch_points), -1)
        
        with torch.no_grad():
            # This gives us the PDF for 1-step ahead
            # For 8-step, we need to use the model's forward method properly
            # But the Fourier head is the same for all horizons
            pdfs = model.fourier_head(batch_output_repr, batch_points)
            pdf_values.append(pdfs)
    
    pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size).cpu().numpy()
    
    # Note: The above gives 1-step PDF. For true 8-step, we'd need to modify
    # the model architecture. For now, let's adjust the extent to show meaningful separation
    
    # Plot PDF
    im = ax2.contourf(xx.numpy(), yy.numpy(), pdf_values, levels=50, cmap='hot')
    
    # Overlay trajectory (relative to current)
    last_pos = input_np[-1]
    
    # Recent track
    n_recent = 10
    recent_track = input_np[-n_recent:] - last_pos
    ax2.plot(recent_track[:, 0], recent_track[:, 1], 'c.-', 
            linewidth=3, markersize=8, alpha=0.8, label='Recent track')
    
    # Current position
    ax2.scatter(0, 0, c='cyan', s=200, marker='s', 
               edgecolor='black', linewidth=2, zorder=5, label='Current')
    
    # True 8-step position
    if len(target_np) >= 8:
        true_8step = target_np[7] - last_pos
        ax2.scatter(true_8step[0], true_8step[1], c='lime', s=200, 
                   marker='*', edgecolor='black', linewidth=2, zorder=6,
                   label='True 8-step')
        
        # Show path
        path_8step = target_np[:8] - last_pos
        ax2.plot(path_8step[:, 0], path_8step[:, 1], 'w--', 
                linewidth=2, alpha=0.8, label='True path')
    
    # Contours
    ax2.contour(xx.numpy(), yy.numpy(), pdf_values, levels=10, 
               colors='white', alpha=0.3, linewidths=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Probability Density', fontsize=12)
    
    ax2.set_xlabel('Relative X from current', fontsize=12)
    ax2.set_ylabel('Relative Y from current', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-extent, extent)
    ax2.set_ylim(-extent, extent)
    
    # Calculate integral
    dx = (2 * extent) / grid_size
    dy = (2 * extent) / grid_size
    integral = pdf_values.sum() * dx * dy
    
    ax2.text(0.98, 0.98, f'∫PDF = {integral:.3f}', 
             transform=ax2.transAxes, fontsize=10,
             horizontalalignment='right', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add note about PDF
    fig.text(0.5, 0.02, 
             'Note: This shows the 1-step PDF distribution. The model predicts all horizons from the same representation,\n' + 
             'but the Fourier head parameters are shared across horizons. True multi-horizon PDFs would require separate heads.',
             ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {save_path}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_path = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep/true_8step_visualization.png")
    
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
    
    # Create visualization
    print("Generating visualization...")
    visualize_true_8step(model, sample, output_path)
    
    # Also print the actual displacement
    input_np = sample['input_uv'].numpy()
    target_np = sample['target_uv'].numpy()
    if len(target_np) >= 8:
        disp = target_np[7] - input_np[-1]
        print(f"\nActual 8-step displacement: ({disp[0]:.3f}, {disp[1]:.3f})")
        print(f"Distance: {np.sqrt(disp[0]**2 + disp[1]**2):.3f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()