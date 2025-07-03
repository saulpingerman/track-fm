#!/usr/bin/env python
"""
Create a video showing true multi-step forecasts
Shows input track, then step by step predictions with their actual PDFs
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os

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

def get_sample_with_movement(dataset, min_total_displacement=0.02):
    """Get a sample with reasonable movement"""
    data_iter = iter(dataset)
    
    for _ in range(200):
        try:
            input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
            
            if causal_position == 19 and len(target_uv) >= 10:
                input_np = input_uv.numpy()
                target_np = target_uv.numpy()
                
                # Check total displacement over all horizons
                total_disp = 0
                for i in range(min(10, len(target_np))):
                    disp = target_np[i] - input_np[-1]
                    dist = np.sqrt(disp[0]**2 + disp[1]**2)
                    total_disp += dist
                
                if total_disp >= min_total_displacement:
                    print(f"Found sample with total displacement: {total_disp:.4f}")
                    return {
                        'input_uv': input_uv,
                        'centre_ll': centre_ll,
                        'target_uv': target_uv,
                        'logJ': logJ,
                        'causal_position': torch.tensor(causal_position)
                    }
                    
        except StopIteration:
            break
    
    print("No sample with significant movement found, using first available")
    # Fallback to first sample
    data_iter = iter(dataset)
    input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
    while causal_position != 19:
        input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
    
    return {
        'input_uv': input_uv,
        'centre_ll': centre_ll,
        'target_uv': target_uv,
        'logJ': logJ,
        'causal_position': torch.tensor(causal_position)
    }

def evaluate_pdf_for_step(model, input_uv, centre_ll, causal_position, step, grid_size=150, extent=0.1):
    """
    Evaluate the model's PDF for a specific prediction step
    This correctly uses the model's multi-horizon prediction capability
    """
    device = next(model.parameters()).device
    
    # Create grid
    x = torch.linspace(-extent, extent, grid_size)
    y = torch.linspace(-extent, extent, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
    
    # Prepare inputs
    input_uv_t = input_uv.unsqueeze(0).to(device)
    centre_ll_t = centre_ll.unsqueeze(0).to(device)
    causal_position_t = causal_position.unsqueeze(0).to(device)
    
    # Create target tensor with grid points at the specified step
    H = 10  # model horizon
    target_tensor = torch.zeros(1, H, 2).to(device)
    
    # Evaluate PDF for each grid point
    pdf_values = []
    batch_size = 1000
    
    for i in range(0, len(grid_points), batch_size):
        batch_points = grid_points[i:i+batch_size]
        batch_size_actual = len(batch_points)
        
        # Create batch of target tensors
        batch_targets = target_tensor.repeat(batch_size_actual, 1, 1)
        batch_targets[:, step, :] = batch_points
        
        # Expand other inputs to match batch size
        batch_input_uv = input_uv_t.repeat(batch_size_actual, 1, 1)
        batch_centre_ll = centre_ll_t.repeat(batch_size_actual, 1)
        batch_causal_pos = causal_position_t.repeat(batch_size_actual)
        
        with torch.no_grad():
            # Get log probabilities for all horizons
            log_probs = model(batch_input_uv, batch_centre_ll, batch_targets, batch_causal_pos)
            
            # Extract probabilities for our step
            step_log_probs = log_probs[:, step]  # (batch_size_actual,)
            step_probs = torch.exp(step_log_probs)
            
            pdf_values.append(step_probs)
    
    pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size).cpu().numpy()
    
    return xx.numpy(), yy.numpy(), pdf_values

def create_forecast_animation(model, sample, save_path):
    """Create animation showing step-by-step forecasts"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    input_np = input_uv.numpy()
    target_np = target_uv.numpy()
    
    # Determine extent based on maximum displacement
    max_disp = 0
    for i in range(min(10, len(target_np))):
        disp = target_np[i] - input_np[-1]
        dist = np.sqrt(disp[0]**2 + disp[1]**2)
        max_disp = max(max_disp, dist)
    
    extent = max(0.05, max_disp * 2)  # Use 2x the max displacement as extent
    print(f"Using extent: {extent:.4f}")
    
    # Pre-compute all PDFs to avoid computation during animation
    print("Pre-computing PDFs for all steps...")
    pdfs = {}
    grids = {}
    
    for step in range(10):
        print(f"Computing PDF for step {step+1}...")
        xx, yy, pdf_values = evaluate_pdf_for_step(
            model, input_uv, centre_ll, causal_position, 
            step, grid_size=100, extent=extent
        )
        pdfs[step] = pdf_values
        grids[step] = (xx, yy)
    
    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    def animate(frame):
        # Clear both axes
        ax1.clear()
        ax2.clear()
        
        if frame == 0:
            # Initial frame: just show input track
            ax1.set_title('Input Trajectory', fontsize=14)
            ax1.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
            ax1.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=150, marker='s', 
                       label='Current position', zorder=5, edgecolor='black', linewidth=2)
            
            ax1.set_xlabel('Local X', fontsize=12)
            ax1.set_ylabel('Local Y', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Empty right plot
            ax2.set_title('Forecast PDF', fontsize=14)
            ax2.set_xlabel('Relative X from current', fontsize=12)
            ax2.set_ylabel('Relative Y from current', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            ax2.set_xlim(-extent, extent)
            ax2.set_ylim(-extent, extent)
            
        else:
            step = frame - 1  # Convert frame to step (0-indexed)
            
            # Left plot: trajectory with current prediction step highlighted
            ax1.set_title(f'Trajectory - Predicting {step+1} Steps Ahead', fontsize=14)
            ax1.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=8, label='Input track')
            ax1.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=150, marker='s', 
                       label='Current position', zorder=5, edgecolor='black', linewidth=2)
            
            # Show true future track up to current step
            if step < len(target_np):
                future_track = target_np[:step+1]
                ax1.plot(future_track[:, 0], future_track[:, 1], 'k--', linewidth=2, 
                        alpha=0.5, label='True path')
                
                # Highlight the target position for this step
                ax1.scatter(target_np[step, 0], target_np[step, 1], c='lime', s=200, 
                           marker='*', label=f'True {step+1}-step position', zorder=6, 
                           edgecolor='black', linewidth=2)
            
            ax1.set_xlabel('Local X', fontsize=12)
            ax1.set_ylabel('Local Y', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Right plot: PDF for this step
            ax2.set_title(f'{step+1}-Step Ahead Forecast PDF', fontsize=14)
            
            # Plot the PDF for this step
            xx, yy = grids[step]
            pdf_values = pdfs[step]
            
            if pdf_values.max() > 0:
                im = ax2.contourf(xx, yy, pdf_values, levels=30, cmap='hot', alpha=0.8)
                ax2.contour(xx, yy, pdf_values, levels=8, colors='white', alpha=0.3, linewidths=0.5)
            
            # Show current position (at origin)
            ax2.scatter(0, 0, c='cyan', s=150, marker='s', 
                       edgecolor='black', linewidth=2, zorder=5, label='Current')
            
            # Show true position for this step
            if step < len(target_np):
                true_pos = target_np[step] - input_np[-1]
                ax2.scatter(true_pos[0], true_pos[1], c='lime', s=150, 
                           marker='*', edgecolor='black', linewidth=2, zorder=6,
                           label=f'True {step+1}-step')
            
            # Show recent input track for context
            n_recent = 5
            recent_track = input_np[-n_recent:] - input_np[-1]
            ax2.plot(recent_track[:, 0], recent_track[:, 1], 'c.-', 
                    linewidth=2, markersize=6, alpha=0.7, label='Recent track')
            
            ax2.set_xlabel('Relative X from current', fontsize=12)
            ax2.set_ylabel('Relative Y from current', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            ax2.set_xlim(-extent, extent)
            ax2.set_ylim(-extent, extent)
            
            # Add PDF statistics
            integral = pdf_values.sum() * ((2 * extent) / pdf_values.shape[0])**2
            max_pdf = pdf_values.max()
            ax2.text(0.02, 0.98, f'∫PDF = {integral:.3f}\nmax = {max_pdf:.1f}', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add overall title with step info
        if frame == 0:
            fig.suptitle('Multi-Step Trajectory Forecasting - Input Track', fontsize=16)
        else:
            fig.suptitle(f'Multi-Step Trajectory Forecasting - {frame} Steps Ahead', fontsize=16)
        
        plt.tight_layout()
    
    # Create animation (11 frames: 1 for input + 10 for predictions)
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=11, interval=1500, repeat=True)
    
    # Save as MP4
    print(f"Saving animation to {save_path}")
    anim.save(save_path, writer='ffmpeg', fps=0.7, bitrate=1800, dpi=150)
    plt.close()
    
    print("Animation complete!")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_path = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep/multistep_forecast_animation.mp4")
    
    # Load model
    model = load_checkpoint(checkpoint_path, device)
    
    # Create dataset
    print("Creating dataset...")
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    dataset = StreamingMultiHorizonAISDataset(
        bucket_name="ais-pipeline-data-10179bbf-us-east-1",
        seq_len=20,
        horizon=10
    )
    
    # Get sample with some movement
    print("Finding sample with movement...")
    sample = get_sample_with_movement(dataset)
    
    # Create animation
    print("Creating forecast animation...")
    create_forecast_animation(model, sample, output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()