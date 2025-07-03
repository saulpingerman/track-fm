#!/usr/bin/env python
"""
Test the original model to confirm it produces identical PDFs for all horizons
"""

import sys
import torch
from pathlib import Path

# Import original model
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train_causal_multihorizon import NanoGPTTrajectoryForecaster

def test_original_model_horizon_issue():
    """Test that the original model produces identical outputs for all horizons"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    
    # Load original model
    print(f"Loading original model from: {checkpoint_path}")
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
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy inputs
    B, S, H = 2, 20, 10
    input_uv = torch.randn(B, S, 2).to(device)
    centre_ll = torch.randn(B, 2).to(device)
    causal_position = torch.tensor([19, 19]).to(device)
    
    # Test with SAME target position for all horizons
    same_target_pos = torch.randn(B, 2).to(device)  # Single target position
    target_uv = same_target_pos.unsqueeze(1).repeat(1, H, 1)  # Repeat for all horizons
    
    print(f"\nTesting original model with SAME target position for all horizons...")
    print(f"Target position: {same_target_pos[0].cpu().numpy()}")
    
    with torch.no_grad():
        log_probs = model(input_uv, centre_ll, target_uv, causal_position)
        probs = torch.exp(log_probs)
        
        print(f"Output shape: {log_probs.shape}")
        print("Probabilities at SAME target position for different horizons (sample 1):")
        for i in range(H):
            print(f"  Horizon {i+1}: {probs[0, i].item():.8f}")
        
        # Check if they're identical
        horizon_differences = torch.abs(probs[0, 1:] - probs[0, 0]).sum()
        print(f"\nSum of absolute differences from horizon 1: {horizon_differences.item():.8f}")
        
        # Check max difference
        max_diff = torch.abs(probs[0, 1:] - probs[0, 0]).max()
        print(f"Maximum difference: {max_diff.item():.8f}")
        
        if horizon_differences < 1e-6:
            print("❌ CONFIRMED PROBLEM: Original model produces IDENTICAL outputs for all horizons!")
            print("    This explains why all the PDFs looked the same in the visualizations.")
        else:
            print("🤔 Unexpected: Original model shows some variation between horizons")
    
    # Let's also test the internal representations
    print(f"\n" + "="*60)
    print("TESTING INTERNAL REPRESENTATIONS")
    print("="*60)
    
    # Get the hidden representation that goes to the Fourier head
    B, S, _ = input_uv.shape
    context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
    input_sequence = torch.cat([input_uv, context_expanded], dim=-1)
    
    with torch.no_grad():
        causal_hidden = model.gpt(input_sequence, causal_position)
        output_repr = model.output_proj(causal_hidden)
        
        print(f"Hidden representation shape: {output_repr.shape}")
        print(f"Hidden representation (sample 1): {output_repr[0][:5].cpu().numpy()}")  # First 5 elements
        
        # Test Fourier head directly with same representation and target
        print(f"\nTesting Fourier head directly with SAME inputs:")
        
        # Same output_repr and same target for all calls
        target_pos = same_target_pos[0:1]  # Just first sample
        repr_input = output_repr[0:1]      # Just first sample
        
        fourier_outputs = []
        for i in range(5):  # Test first 5 horizons
            pdf_value = model.fourier_head(repr_input, target_pos)
            fourier_outputs.append(pdf_value.item())
            print(f"  Call {i+1}: {pdf_value.item():.8f}")
        
        fourier_diffs = [abs(x - fourier_outputs[0]) for x in fourier_outputs[1:]]
        max_fourier_diff = max(fourier_diffs) if fourier_diffs else 0
        
        print(f"\nMax difference in Fourier head outputs: {max_fourier_diff:.8f}")
        
        if max_fourier_diff < 1e-6:
            print("❌ CONFIRMED: Fourier head produces IDENTICAL outputs")
            print("    This is because it receives the SAME hidden representation every time!")
        else:
            print("🤔 Unexpected: Fourier head shows variation")

if __name__ == "__main__":
    test_original_model_horizon_issue()