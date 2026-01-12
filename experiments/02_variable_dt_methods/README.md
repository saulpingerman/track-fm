# Variable Time Horizon (Δt) Experiment Report

## Executive Summary

**Research Question:** How should we incorporate a variable prediction time horizon (Δt) into a transformer + 2D Fourier head model?

**Finding:** **Concat to Input** is the best method, achieving 0.247 error vs 0.366 for regression baseline (32% improvement).

**Critical Bug Fixed:** Initial experiments used grid_range=5.0, but max displacement = 4.5 velocity × 4.0 Δt = 18 units. Targets were clipped to grid edges, causing artifacts. Fixed by using grid_range=20.0.

---

## Experiment Setup

### Task
- **Predict displacement:** displacement = velocity × Δt
- **Input:** 10 positions (x, y) + time horizon Δt
- **Δt range:** [0.5, 4.0] timesteps
- **Velocity range:** [0.1, 4.5] units/timestep
- **Grid range:** 20.0 (covers max displacement of 18 units)
- **Training:** 10,000 samples, 30 epochs

---

## Results

### Final Rankings

| Rank | Method | Error | vs Baseline |
|------|--------|-------|-------------|
| 1 | **Concat to Input** | **0.247** | **32% better** |
| 2 | Δt as Token | 0.250 | 32% better |
| 3 | Cross-Attention | 0.309 | 16% better |
| 4 | *Regression Baseline* | *0.366* | *reference* |
| 5 | FiLM Conditioning | 0.386 | 5% worse |
| 6 | Concat to Output | 0.461 | 26% worse |

### Key Finding

**Three Fourier methods beat the regression baseline!** Concat to Input, Δt as Token, and Cross-Attention all outperform direct (x, y) regression.

---

## Δt Conditioning Methods

### Method 1: Concat to Output
```
Architecture:
  Trajectory → Transformer → [z]
  Δt → Embed → [dt_emb]
  [z, dt_emb] → MLP → Fourier Head → Density
```
- **Error:** 0.461 (worst)
- Late fusion of Δt after transformer may limit scaling ability

### Method 2: FiLM Conditioning
```
Architecture:
  Trajectory → Transformer → z
  Δt → Embed → (γ, β)
  z * (1 + γ) + β → Fourier Head → Density
```
- **Error:** 0.386
- Affine modulation insufficient for multiplicative displacement relationship

### Method 3: Concat to Input (BEST)
```
Architecture:
  (x, y, Δt) → Transformer → Fourier Head → Density
```
- **Error:** 0.247 (best)
- Early fusion allows transformer to learn velocity × Δt relationship
- Δt broadcast to all positions provides consistent context

### Method 4: Δt as Token
```
Architecture:
  [pos_tokens, dt_token] → Transformer → last token → Fourier Head
```
- **Error:** 0.250
- Second best method
- Transformer attention can directly integrate Δt with trajectory

### Method 5: Cross-Attention
```
Architecture:
  Trajectory → Transformer → z
  z attends to Δt embedding → Fourier Head
```
- **Error:** 0.309
- Third best method
- Explicit attention mechanism for Δt integration

---

## Why Concat to Input Works Best

1. **Early Integration:** Δt is available at every layer of the transformer, not just at the end
2. **Multiplicative Learning:** Transformer can learn displacement = velocity × Δt as a single operation
3. **Consistent Context:** Every position token sees the same Δt, providing stable conditioning

---

## Configuration Used

```python
config = {
    'num_train': 10000,
    'num_val': 1000,
    'seq_len': 10,
    'velocity_range': (0.1, 4.5),
    'dt_range': (0.5, 4.0),
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'grid_size': 64,
    'num_freqs': 8,
    'grid_range': 20.0,  # Critical: must cover max displacement (4.5 × 4.0 = 18)
    'batch_size': 64,
    'learning_rate': 3e-4,
    'num_epochs': 30,
}
```

---

## Files Generated

```
results/
├── Method_1_Concat_to_Output_predictions.png
├── Method_1_Concat_to_Output_dt_effect.png
├── Method_2_FiLM_Conditioning_predictions.png
├── Method_2_FiLM_Conditioning_dt_effect.png
├── Method_3_Concat_to_Input_predictions.png
├── Method_3_Concat_to_Input_dt_effect.png
├── Method_4_dt_as_Token_predictions.png
├── Method_4_dt_as_Token_dt_effect.png
├── Method_5_Cross-Attention_predictions.png
├── Method_5_Cross-Attention_dt_effect.png
└── experiment_results.json
```

---

## Lessons Learned

### Critical: Grid Range Must Cover Max Displacement

**Bug:** Initial grid_range=5.0 caused targets outside the grid to be clipped to edges, producing artifacts and misleading results.

**Fix:** grid_range = velocity_max × Δt_max = 4.5 × 4.0 = 18 → use 20.0

### Method Rankings Changed After Fix

| Method | Error (grid=5.0) | Error (grid=20.0) |
|--------|-----------------|-------------------|
| Concat to Output | 3.10 (best) | 0.461 (worst) |
| Concat to Input | 3.15 | **0.247 (best)** |
| Δt as Token | 3.16 | 0.250 |

The grid bug caused methods that happened to predict near the origin to appear better.

---

## Recommendations

1. **Use Concat to Input** for Δt conditioning - it's simple and performs best
2. **Always verify grid range** covers the maximum expected displacement
3. **Fourier heads can beat regression** when properly configured
