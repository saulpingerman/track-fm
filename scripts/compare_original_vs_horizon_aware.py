#!/usr/bin/env python
"""
Compare original model vs horizon-aware model to show the fundamental difference
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import models
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train_causal_multihorizon import NanoGPTTrajectoryForecaster
from create_horizon_aware_model import HorizonAwareTrajectoryForecaster, create_horizon_aware_model_from_checkpoint

def load_original_model(checkpoint_path, device='cuda'):
    """Load the original model"""
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

def create_comparison_visualization(original_model, horizon_model, save_path):
    """Create visualization comparing both models"""
    
    device = next(original_model.parameters()).device
    
    # Create test inputs
    B, S, H = 1, 20, 10
    torch.manual_seed(42)  # For reproducible results
    input_uv = torch.randn(B, S, 2).to(device) * 0.1  # Small movements
    centre_ll = torch.randn(B, 2).to(device) * 0.1
    causal_position = torch.tensor([19]).to(device)
    
    # Test 1: Same target position for all horizons
    same_target = torch.tensor([[0.05, 0.03]]).to(device)  # Fixed target
    target_uv_same = same_target.unsqueeze(0).repeat(B, H, 1)
    
    # Test 2: Different target positions (realistic scenario)
    target_positions = []
    for h in range(H):
        # Simulate positions getting further from origin
        pos = torch.tensor([[0.01 * (h+1), 0.005 * (h+1)]]).to(device)
        target_positions.append(pos)
    target_uv_diff = torch.stack(target_positions, dim=1).squeeze(2)  # (B, H, 2)
    
    # Get predictions from both models
    with torch.no_grad():
        # Original model predictions
        orig_probs_same = torch.exp(original_model(input_uv, centre_ll, target_uv_same, causal_position))
        orig_probs_diff = torch.exp(original_model(input_uv, centre_ll, target_uv_diff, causal_position))
        
        # Horizon-aware model predictions  
        horizon_probs_same = torch.exp(horizon_model(input_uv, centre_ll, target_uv_same, causal_position))
        horizon_probs_diff = torch.exp(horizon_model(input_uv, centre_ll, target_uv_diff, causal_position))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data for plotting
    horizons = list(range(1, H+1))
    orig_same = orig_probs_same[0].cpu().numpy()
    orig_diff = orig_probs_diff[0].cpu().numpy()
    horizon_same = horizon_probs_same[0].cpu().numpy()
    horizon_diff = horizon_probs_diff[0].cpu().numpy()
    
    # Row 1: Same target position
    axes[0, 0].bar(horizons, orig_same, alpha=0.7, color='red', label='Original Model')
    axes[0, 0].set_title('Original Model\n(Same Target Position)', fontsize=14)
    axes[0, 0].set_xlabel('Prediction Horizon')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_ylim(0, max(max(orig_same), max(horizon_same)) * 1.1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add text showing values are identical
    axes[0, 0].text(0.05, 0.95, f'All values: {orig_same[0]:.6f}', 
                   transform=axes[0, 0].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   verticalalignment='top')
    
    axes[0, 1].bar(horizons, horizon_same, alpha=0.7, color='blue', label='Horizon-Aware Model')
    axes[0, 1].set_title('Horizon-Aware Model\n(Same Target Position)', fontsize=14)
    axes[0, 1].set_xlabel('Prediction Horizon')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_ylim(0, max(max(orig_same), max(horizon_same)) * 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Comparison
    axes[0, 2].bar([h-0.2 for h in horizons], orig_same, width=0.4, alpha=0.7, color='red', label='Original')
    axes[0, 2].bar([h+0.2 for h in horizons], horizon_same, width=0.4, alpha=0.7, color='blue', label='Horizon-Aware')
    axes[0, 2].set_title('Comparison\n(Same Target Position)', fontsize=14)
    axes[0, 2].set_xlabel('Prediction Horizon')
    axes[0, 2].set_ylabel('Probability')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Different target positions
    axes[1, 0].bar(horizons, orig_diff, alpha=0.7, color='red')
    axes[1, 0].set_title('Original Model\n(Different Target Positions)', fontsize=14)
    axes[1, 0].set_xlabel('Prediction Horizon')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(horizons, horizon_diff, alpha=0.7, color='blue')
    axes[1, 1].set_title('Horizon-Aware Model\n(Different Target Positions)', fontsize=14)
    axes[1, 1].set_xlabel('Prediction Horizon')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Comparison
    x_pos = np.arange(len(horizons))
    axes[1, 2].bar(x_pos - 0.2, orig_diff, width=0.4, alpha=0.7, color='red', label='Original')
    axes[1, 2].bar(x_pos + 0.2, horizon_diff, width=0.4, alpha=0.7, color='blue', label='Horizon-Aware')
    axes[1, 2].set_title('Comparison\n(Different Target Positions)', fontsize=14)
    axes[1, 2].set_xlabel('Prediction Horizon')
    axes[1, 2].set_ylabel('Probability')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(horizons)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add overall explanation
    fig.text(0.5, 0.02, 
             'KEY INSIGHT: Original model produces IDENTICAL probabilities for all horizons at the same target position\n' +
             'because the Fourier head receives the same hidden representation for all time steps.\n' +
             'Horizon-aware model includes time-to-prediction information, allowing different PDFs per horizon.',
             ha='center', fontsize=12, style='italic', color='darkblue',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print numerical comparison
    print("\nNUMERICAL COMPARISON")
    print("="*50)
    print(f"Same target position:")
    print(f"  Original model variance: {np.var(orig_same):.10f}")
    print(f"  Horizon-aware variance: {np.var(horizon_same):.10f}")
    print(f"  Improvement ratio: {np.var(horizon_same) / np.var(orig_same) if np.var(orig_same) > 0 else 'inf'}")
    
    print(f"\nDifferent target positions:")
    print(f"  Original model variance: {np.var(orig_diff):.10f}")
    print(f"  Horizon-aware variance: {np.var(horizon_diff):.10f}")
    
    print(f"\nOriginal model probabilities (same target):")
    for i, p in enumerate(orig_same):
        print(f"  Horizon {i+1}: {p:.10f}")
    
    print(f"\nHorizon-aware model probabilities (same target):")
    for i, p in enumerate(horizon_same):
        print(f"  Horizon {i+1}: {p:.10f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    output_path = Path("/home/ec2-user/repos/track-fm/forecast_plots_multistep/original_vs_horizon_aware_comparison.png")
    
    print("Loading models...")
    
    # Load original model
    original_model = load_original_model(checkpoint_path, device)
    print("✅ Original model loaded")
    
    # Create horizon-aware model
    horizon_model = create_horizon_aware_model_from_checkpoint(checkpoint_path, device)
    print("✅ Horizon-aware model created")
    
    # Create comparison
    print("\nCreating comparison visualization...")
    create_comparison_visualization(original_model, horizon_model, output_path)
    
    print(f"\n✅ Comparison saved to: {output_path}")
    print("\nCONCLUSION:")
    print("The original model has a fundamental architectural flaw - it cannot produce")
    print("different PDFs for different prediction horizons because the Fourier head")
    print("receives identical input representations for all time steps.")
    print("\nThe horizon-aware model fixes this by including time-to-prediction information.")

if __name__ == "__main__":
    main()