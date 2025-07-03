# Architectural Flaw Analysis: Multi-Horizon Trajectory Forecasting

## Problem Identified

The original multi-horizon trajectory forecasting model has a **fundamental architectural flaw** that prevents it from producing different probability distributions for different prediction horizons.

## Root Cause

The model architecture follows this pattern:

```python
# 1. Compute hidden representation ONCE
causal_hidden = self.gpt(input_sequence, causal_position)  # (B, d_model)
output_repr = self.output_proj(causal_hidden)              # (B, d_model)

# 2. Use SAME representation for all horizons
for t in range(H):
    target_t = target_uv[:, t, :]  # Different targets
    pdf = self.fourier_head(output_repr, target_t)  # SAME output_repr!
```

**The problem**: The Fourier head receives **identical input representations** for all prediction horizons. It has no way to distinguish between "predicting 1 step ahead" vs "predicting 10 steps ahead".

## Experimental Verification

### Test: Same Target Position, Different Horizons

**Original Model Results:**
```
Horizon 1: 0.2178144604
Horizon 2: 0.2178144604  ← IDENTICAL
Horizon 3: 0.2178144604  ← IDENTICAL
...
Horizon 10: 0.2178144604 ← IDENTICAL
Variance: 0.0000000000
```

**Horizon-Aware Model Results:**
```
Horizon 1: 0.2401338518
Horizon 2: 0.1103231832  ← DIFFERENT
Horizon 3: 0.4341757596  ← DIFFERENT
...
Horizon 10: 0.1140524596 ← DIFFERENT
Variance: 0.0153548103
```

## Impact on Visualizations

This explains why all PDF visualizations looked identical across different prediction horizons. The model was literally producing the same spatial probability distribution for all time steps.

## Solution: Horizon-Aware Architecture

The fix involves adding horizon information to the Fourier head:

```python
class HorizonAwareFourierHead2DLR(nn.Module):
    def __init__(self, dim_input, max_horizon=10, ...):
        # Add horizon embedding
        self.horizon_embedding = nn.Embedding(max_horizon, dim_input // 4)
        
    def forward(self, output_repr, target_pos, horizon_step):
        # Get horizon embedding
        horizon_emb = self.horizon_embedding(horizon_step)
        
        # Concatenate output representation with horizon embedding
        enhanced_repr = torch.cat([output_repr, horizon_emb], dim=-1)
        
        # Forward through Fourier head with enhanced representation
        return self.fourier_head(enhanced_repr, target_pos)
```

## Key Changes

1. **Horizon Embedding**: Each prediction horizon (1-10 steps) gets a learnable embedding
2. **Enhanced Representation**: Concatenate horizon embedding with hidden representation
3. **Horizon-Aware Predictions**: Fourier head now knows which time step it's predicting

## Implementation Status

- ✅ **Problem identified and verified** through controlled experiments
- ✅ **Horizon-aware architecture implemented** with embedding approach
- ✅ **Experimental validation** shows models produce different outputs per horizon
- ⚠️ **Requires retraining** - current model weights are for the flawed architecture

## Training Implications

The current trained model at step 20,000 has learned to optimize the **average performance across all horizons** but cannot distinguish between them. To properly leverage multi-horizon capabilities:

1. **Retrain with horizon-aware architecture** from scratch or fine-tune
2. **Modified loss function** that accounts for horizon-specific performance
3. **Evaluation metrics** that measure performance per horizon

## Architectural Lesson

This highlights the importance of **information flow analysis** in neural architectures. The model can only learn what it has access to - without horizon information, it cannot produce horizon-specific behaviors.

The fix is conceptually simple but requires architectural modification and retraining to realize the full potential of multi-horizon trajectory forecasting.