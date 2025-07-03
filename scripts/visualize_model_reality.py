#!/usr/bin/env python
"""
Visualize the reality of what the model actually predicts
Shows both 1-step and the largest displacement available
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    
    return model

def find_largest_displacement_sample(dataset, max_samples=100):
    """Find sample with largest 8-step displacement"""
    data_iter = iter(dataset)
    best_sample = None
    max_displacement = 0
    
    print("Finding sample with largest displacement...")
    
    for i in range(max_samples):
        try:
            input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
            
            if causal_position == 19 and len(target_uv) >= 8:
                input_np = input_uv.numpy()
                target_np = target_uv.numpy()
                
                displacement = target_np[7] - input_np[-1]
                distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
                
                if distance > max_displacement:
                    max_displacement = distance
                    best_sample = {
                        'input_uv': input_uv,
                        'centre_ll': centre_ll,
                        'target_uv': target_uv,
                        'logJ': logJ,
                        'causal_position': torch.tensor(causal_position),
                        'displacement': distance
                    }
                    print(f"New best: {distance:.4f}")
                    
        except StopIteration:
            break
    
    return best_sample

def analyze_model_predictions(model, sample):
    """Analyze what the model actually predicts for different horizons"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    causal_position = sample['causal_position']
    
    device = next(model.parameters()).device
    
    # Prepare inputs
    input_uv_t = input_uv.unsqueeze(0).to(device)
    centre_ll_t = centre_ll.unsqueeze(0).to(device)
    target_uv_t = target_uv.unsqueeze(0).to(device)
    causal_position_t = causal_position.unsqueeze(0).to(device)
    
    # Get model predictions for all horizons
    with torch.no_grad():
        log_probs = model(input_uv_t, centre_ll_t, target_uv_t, causal_position_t)
    
    # Convert to probabilities
    probs = torch.exp(log_probs).cpu().numpy()[0]  # Remove batch dimension
    
    return probs

def create_reality_visualization(model, sample, save_path):
    """Create visualization showing the reality of model predictions"""
    input_uv = sample['input_uv']
    centre_ll = sample['centre_ll']
    target_uv = sample['target_uv']
    
    # Analyze model predictions
    probs = analyze_model_predictions(model, sample)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1])
    
    # Main trajectory plot
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_title('Complete Trajectory with Model Predictions', fontsize=16)
    
    input_np = input_uv.numpy()
    target_np = target_uv.numpy()
    
    # Plot trajectory
    ax_main.plot(input_np[:, 0], input_np[:, 1], 'b.-', linewidth=2, markersize=6, label='Input track')
    ax_main.scatter(input_np[-1, 0], input_np[-1, 1], c='red', s=100, marker='s', label='Current position', zorder=5)
    ax_main.plot(target_np[:, 0], target_np[:, 1], 'k.-', linewidth=2, markersize=6, 
                 label='True future track', alpha=0.7)
    
    # Highlight different horizon positions with their probabilities
    horizons = [0, 2, 4, 7, 9]  # 1, 3, 5, 8, 10 steps
    colors = ['orange', 'purple', 'brown', 'lime', 'gray']
    
    for h, color in zip(horizons, colors):
        if h < len(target_np) and h < len(probs):
            ax_main.scatter(target_np[h, 0], target_np[h, 1], c=color, s=150*probs[h], 
                           marker='*', label=f'{h+1}-step (p={probs[h]:.1e})', 
                           zorder=6, edgecolor='black', linewidth=1, alpha=0.8)
    
    ax_main.set_xlabel('Local X (normalized)', fontsize=12)
    ax_main.set_ylabel('Local Y (normalized)', fontsize=12)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    
    # PDF visualization for different horizons
    device = next(model.parameters()).device
    input_uv_t = input_uv.unsqueeze(0).to(device)
    centre_ll_t = centre_ll.unsqueeze(0).to(device)
    causal_position_t = sample['causal_position'].unsqueeze(0).to(device)
    
    # Get hidden representation
    B, S, _ = input_uv_t.shape
    context_expanded = centre_ll_t.unsqueeze(1).expand(-1, S, -1)
    input_sequence = torch.cat([input_uv_t, context_expanded], dim=-1)
    
    with torch.no_grad():
        causal_hidden = model.gpt(input_sequence, causal_position_t)
        output_repr = model.output_proj(causal_hidden)
    
    # Create PDF plots for first 4 horizons
    extent = max(0.02, sample['displacement'] * 1.5)
    
    for plot_idx, (h, color) in enumerate(zip(horizons[:4], colors[:4])):
        ax = fig.add_subplot(gs[1, plot_idx])
        ax.set_title(f'{h+1}-Step PDF', fontsize=12)
        
        # Create grid
        grid_size = 100
        x = torch.linspace(-extent, extent, grid_size)
        y = torch.linspace(-extent, extent, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(device)
        
        # Evaluate PDF (note: this is the same for all horizons due to shared Fourier head)
        pdf_values = []
        batch_size = 1000
        
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i+batch_size]
            batch_output_repr = output_repr.expand(len(batch_points), -1)
            
            with torch.no_grad():
                pdfs = model.fourier_head(batch_output_repr, batch_points)
                pdf_values.append(pdfs)
        
        pdf_values = torch.cat(pdf_values).reshape(grid_size, grid_size).cpu().numpy()
        
        # Plot
        im = ax.contourf(xx.numpy(), yy.numpy(), pdf_values, levels=20, cmap='hot')
        
        # Current position
        ax.scatter(0, 0, c='cyan', s=100, marker='s', zorder=5)
        
        # True position at this horizon
        if h < len(target_np):
            true_pos = target_np[h] - input_np[-1]
            ax.scatter(true_pos[0], true_pos[1], c=color, s=100, marker='*', zorder=6)
        
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Probability analysis plot
    ax_prob = fig.add_subplot(gs[2, :])
    ax_prob.set_title('Model Probability Predictions vs Horizon', fontsize=14)
    
    # Calculate distances for each horizon
    distances = []
    for h in range(len(target_np)):
        disp = target_np[h] - input_np[-1]
        dist = np.sqrt(disp[0]**2 + disp[1]**2)
        distances.append(dist)
    
    horizons_range = list(range(1, len(probs) + 1))
    
    # Plot probabilities
    ax_prob.bar(horizons_range, probs[:len(horizons_range)], alpha=0.7, color='blue', label='Model probability')
    ax_prob.set_xlabel('Prediction Horizon (steps ahead)', fontsize=12)
    ax_prob.set_ylabel('Probability', fontsize=12)
    ax_prob.set_yscale('log')
    ax_prob.grid(True, alpha=0.3)
    
    # Overlay distances on secondary y-axis
    ax_dist = ax_prob.twinx()
    ax_dist.plot(horizons_range[:len(distances)], distances, 'ro-', alpha=0.7, label='True distance')
    ax_dist.set_ylabel('Distance from current position', fontsize=12, color='red')
    ax_dist.tick_params(axis='y', labelcolor='red')
    
    # Add legends
    ax_prob.legend(loc='upper left')
    ax_dist.legend(loc='upper right')
    
    # Add explanatory text
    fig.text(0.5, 0.02, 
             'IMPORTANT: The model uses a single Fourier head for all horizons, so the PDF shape is identical across time steps.\n' +
             'Only the probability magnitudes differ. This explains why all PDF plots look the same.\n' +
             f'Max 8-step displacement in this sample: {sample["displacement"]:.4f} (very small movement)',
             ha='center', fontsize=11, style='italic', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive analysis to {save_path}")

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_path = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep/model_reality_analysis.png")
    
    # Load model
    print("Loading model...")
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
    
    # Find sample with largest displacement
    sample = find_largest_displacement_sample(dataset, max_samples=500)
    
    if sample is None:
        print("Could not find any suitable sample")
        return
    
    print(f"Using sample with displacement: {sample['displacement']:.4f}")
    
    # Create analysis
    create_reality_visualization(model, sample, output_path)
    
    print("\nModel Architecture Reality Check:")
    print("- The model uses a SINGLE Fourier head for ALL prediction horizons")
    print("- This means the PDF shape is IDENTICAL for 1-step, 8-step, etc.")
    print("- Only the probability magnitudes differ across horizons")
    print("- The AIS data contains mostly slow-moving vessels")
    print(f"- Largest 8-step displacement found: {sample['displacement']:.4f}")
    print("\nThis explains why the visualizations look similar across horizons!")

if __name__ == "__main__":
    main()