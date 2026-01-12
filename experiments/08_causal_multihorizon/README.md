# Experiment 8: Causal Multi-Horizon Prediction

## Question

Can we efficiently train a causal decoder transformer for trajectory prediction by:
1. Using variable time spacing in the data
2. Predicting multiple horizons (1, 2, ..., k positions ahead) at each position
3. Conditioning predictions on cumulative Δt to target

## Motivation

For real applications like AIS vessel tracking, we need a model that:
- Accepts variable-length input tracks
- Predicts at arbitrary future time horizons
- Outputs density (not point predictions) to capture uncertainty
- Uses causal architecture for efficient streaming inference

## Key Innovation

Instead of predicting at fixed time intervals, we leverage the natural variable spacing in the data:

```
Position:  p0 ----p1 --p2 ------p3 ---p4 --p5
Time gap:      2min  1min  5min   2min  1min

At p0, predict:
  - p1 (Δt = 2min)
  - p2 (Δt = 3min)
  - p3 (Δt = 8min)
```

This gives us:
- Variable Δt training "for free"
- ~L × k training examples per trajectory (massive data efficiency)
- Natural causality (each position only sees past)

## Architecture

```
Input: [(dx0,dy0,Δt0), (dx1,dy1,Δt1), ..., (dxN,dyN,ΔtN)]
       where dx, dy = displacement from previous position
                            ↓
         Causal Decoder Transformer
                            ↓
              embedding[i] for each position
                            ↓
         For each horizon h: concat(embedding[i], encode(cumulative_Δt))
                            ↓
                      Fourier Head
                            ↓
                    Density[i, h]
```

### Components

1. **Input tokens**: `(dx, dy, Δt_since_previous)` - relative displacements (NOT absolute coords)
2. **Causal transformer**: Each position only attends to previous positions
3. **Time encoding**: Sinusoidal encoding of cumulative Δt to target
4. **Horizon conditioning**: Concatenate embedding with time encoding
5. **Fourier head**: Output 2D density over prediction grid

## Training

For each trajectory of length L with max_horizon k:
- At each position i (0 to L-2)
- For each horizon h (1 to k, if i+h < L)
- Target = position[i+h] relative to position[i]
- Condition on cumulative_Δt = sum(dt[i+1:i+h+1])

**Data efficiency**: ~(L-1) × k / 2 examples per trajectory

## Inference

Two modes:

### 1. Multi-horizon output
```python
embeddings, log_densities, cumulative_dt = model.forward_multihorizon(positions, dt_values)
# log_densities: (batch, seq_len, max_horizon, grid_size, grid_size)
```

### 2. Query interface
```python
# Query specific time horizon
density = model(positions, dt_values, query_dt=torch.tensor([5.0]))
```

## How to Run

```bash
cd experiments/08_causal_multihorizon
python3 run_experiment.py
```

## Expected Output

- Training curves showing loss for each horizon
- Prediction error increasing with horizon (further = harder)
- Visualization of density predictions at different horizons
- Error vs Δt plot showing model generalization

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| seq_len | 20 | Positions per trajectory |
| max_horizon | 5 | Predict up to 5 positions ahead |
| dt_mean | 1.0 | Mean time gap between positions |
| dt_std | 0.3 | Std of time gaps |
| d_model | 64 | Transformer dimension |
| num_layers | 2 | Transformer layers |

## Results

(To be filled after running with fixed translation-invariant model)

| Horizon | Mean Error |
|---------|------------|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |

### Key Fix: Translation Invariance

The model now uses **relative displacements** `(dx, dy)` instead of absolute coordinates `(x, y)`.
This ensures the model is translation invariant - shifting a trajectory by any offset produces
identical predictions.

Without this fix, the model learned to exploit the fact that all training trajectories
started at origin (0, 0), predicting (0, 0) regardless of velocity.
