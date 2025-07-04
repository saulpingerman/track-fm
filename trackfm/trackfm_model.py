#!/usr/bin/env python
"""
Create a horizon-aware model that includes time-to-prediction information
"""

import sys
import torch
from torch import nn
from pathlib import Path

# Import existing components
from .four_head_2D_LR import FourierHead2DLR
from .nano_gpt_trajectory import TrajectoryGPT, TrajectoryGPTConfig

class HorizonAwareFourierHead2DLR(nn.Module):
    """
    Modified Fourier head that takes horizon information as input
    """
    
    def __init__(
        self,
        dim_input: int,
        num_frequencies: int,
        max_horizon: int = 10,
        rank: int = 1,
        regularisation_gamma: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        super().__init__()
        self.dim_input = dim_input
        self.max_horizon = max_horizon
        
        # Add horizon embedding
        self.horizon_embedding = nn.Embedding(max_horizon, dim_input // 4)  # Small embedding
        
        # Modified input dimension to include horizon embedding
        modified_dim_input = dim_input + dim_input // 4
        
        # Original Fourier head with modified input size
        self.fourier_head = FourierHead2DLR(
            dim_input=modified_dim_input,
            num_frequencies=num_frequencies,
            rank=rank,
            regularisation_gamma=regularisation_gamma,
            dtype=dtype,
            device=device
        )
    
    def forward(self, output_repr, target_pos, horizon_step):
        """
        Args:
            output_repr: (B, dim_input) - hidden representation
            target_pos: (B, 2) - target position 
            horizon_step: (B,) or int - which horizon step (0-indexed)
        """
        batch_size = output_repr.shape[0]
        
        # Handle horizon_step as int or tensor
        if isinstance(horizon_step, int):
            horizon_step = torch.full((batch_size,), horizon_step, dtype=torch.long, device=output_repr.device)
        elif len(horizon_step.shape) == 0:  # scalar tensor
            horizon_step = horizon_step.unsqueeze(0).repeat(batch_size)
        
        # Get horizon embedding
        horizon_emb = self.horizon_embedding(horizon_step)  # (B, dim_input//4)
        
        # Concatenate output representation with horizon embedding
        enhanced_repr = torch.cat([output_repr, horizon_emb], dim=-1)  # (B, dim_input + dim_input//4)
        
        # Forward through Fourier head
        return self.fourier_head(enhanced_repr, target_pos)

class HorizonAwareTrajectoryForecaster(nn.Module):
    """
    Trajectory forecaster that is aware of prediction horizons
    """
    
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_hidden: int,
        fourier_m: int,
        fourier_rank: int,
        max_horizon: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Create nanoGPT configuration
        self.gpt_config = TrajectoryGPTConfig(
            block_size=seq_len,
            n_layer=num_layers,
            n_head=nhead,
            n_embd=d_model,
            dropout=dropout,
            bias=False,
            input_dim=4,  # (x, y, context_lat, context_lon)
            fourier_m=fourier_m,
            fourier_rank=fourier_rank
        )
        
        # Create the nanoGPT backbone
        self.gpt = TrajectoryGPT(self.gpt_config)
        
        # Horizon-aware Fourier head
        self.fourier_head = HorizonAwareFourierHead2DLR(
            dim_input=d_model,
            num_frequencies=fourier_m,
            max_horizon=max_horizon,
            rank=fourier_rank,
            device="cpu"  # Will be moved with .to(device)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def to(self, device):
        """Override to method to handle Fourier head device"""
        super().to(device)
        # Update Fourier head device
        self.fourier_head.fourier_head.dev = device
        return self
        
    def forward(self, input_uv, centre_ll, target_uv, causal_position):
        """
        Args:
            input_uv: (B, seq_len, 2) - input positions
            centre_ll: (B, 2) - normalized center coordinates
            target_uv: (B, horizon, 2) - target positions
            causal_position: (B,) - which position's output to use for prediction
        """
        B, S, _ = input_uv.shape
        H = target_uv.shape[1]  # horizon length
        
        # Prepare input: concatenate position with context
        context_expanded = centre_ll.unsqueeze(1).expand(-1, S, -1)
        input_sequence = torch.cat([input_uv, context_expanded], dim=-1)  # (B, S, 4)
        
        # Forward through nanoGPT backbone
        causal_hidden = self.gpt(input_sequence, causal_position)  # (B, d_model)
        
        # Project for output
        output_repr = self.output_proj(causal_hidden)  # (B, d_model)
        
        # Compute PDF for each target position WITH horizon information
        log_probs = []
        for t in range(H):
            target_t = target_uv[:, t, :]  # (B, 2)
            
            # NOW we pass the horizon step information!
            pdf = self.fourier_head(output_repr, target_t, horizon_step=t)  # (B,)
            log_probs.append(pdf.log())
        
        return torch.stack(log_probs, dim=1)  # (B, H)

def create_horizon_aware_model_from_checkpoint(checkpoint_path, device='cuda'):
    """
    Create a horizon-aware model and initialize it from an existing checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create horizon-aware model
    model = HorizonAwareTrajectoryForecaster(
        seq_len=20,
        d_model=128,
        nhead=8,
        num_layers=6,
        ff_hidden=512,
        fourier_m=128,
        fourier_rank=4,
        max_horizon=10
    ).to(device)
    
    # Load compatible weights from checkpoint
    # We can load the GPT and output_proj weights directly
    model_dict = model.state_dict()
    checkpoint_dict = checkpoint['model_state_dict']
    
    # Filter out incompatible keys (Fourier head will be different)
    compatible_dict = {}
    for k, v in checkpoint_dict.items():
        if k.startswith('gpt.') or k.startswith('output_proj.'):
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_dict[k] = v
                print(f"Loaded: {k}")
            else:
                print(f"Skipped (shape mismatch): {k}")
        else:
            print(f"Skipped (incompatible): {k}")
    
    # Load the compatible weights
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    print(f"Created horizon-aware model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Note: Fourier head weights are randomly initialized and would need retraining")
    
    return model

def test_horizon_awareness():
    """Test that the model produces different outputs for different horizons"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
    
    # Create horizon-aware model
    model = create_horizon_aware_model_from_checkpoint(checkpoint_path, device)
    
    # Create dummy inputs
    B, S, H = 2, 20, 10
    input_uv = torch.randn(B, S, 2).to(device)
    centre_ll = torch.randn(B, 2).to(device)
    target_uv = torch.randn(B, H, 2).to(device)
    causal_position = torch.tensor([19, 19]).to(device)
    
    print("\nTesting horizon awareness...")
    
    with torch.no_grad():
        # Get predictions
        log_probs = model(input_uv, centre_ll, target_uv, causal_position)
        probs = torch.exp(log_probs)
        
        print(f"Output shape: {log_probs.shape}")
        print("Probabilities for each horizon (sample 1):")
        for i in range(H):
            print(f"  Horizon {i+1}: {probs[0, i].item():.6f}")
        
        # Test if different horizons give different results
        # Use same target position for different horizons
        same_target = target_uv[:, 0:1, :].repeat(1, H, 1)  # Same position for all horizons
        log_probs_same = model(input_uv, centre_ll, same_target, causal_position)
        probs_same = torch.exp(log_probs_same)
        
        print("\nProbabilities at SAME target position for different horizons:")
        for i in range(H):
            print(f"  Horizon {i+1}: {probs_same[0, i].item():.6f}")
        
        # Check if they're different
        horizon_differences = torch.abs(probs_same[0, 1:] - probs_same[0, 0]).sum()
        print(f"\nSum of differences from horizon 1: {horizon_differences.item():.6f}")
        
        if horizon_differences > 1e-6:
            print("✅ SUCCESS: Model produces different outputs for different horizons!")
        else:
            print("❌ PROBLEM: Model produces identical outputs for all horizons")

if __name__ == "__main__":
    test_horizon_awareness()