# Experiment 6: Relative Displacement Prediction

## Executive Summary

**Question:** Do results change when predicting relative displacement (normalized grid) instead of absolute position?

**Finding:** YES - complete reversal! True pointwise now works well (0.100), while soft targets fail (0.248).

---

## Results

| Method | Error | Rank |
|--------|-------|------|
| **Regression Baseline** | **0.057** | 1st |
| True Pointwise NLL | 0.100 | 2nd |
| Grid-Interpolated NLL | 0.109 | 3rd |
| Hard Target (one-hot) | 0.112 | 4th |
| Soft Target (σ=0.2) | 0.133 | 5th |
| Soft Target (σ=0.5) | 0.248 | 6th (worst) |

---

## Comparison Across Experiments

| Method | Exp 4 (absolute) | Exp 5 (variable Δt) | Exp 6 (relative) |
|--------|------------------|---------------------|------------------|
| Regression | 0.079 | 0.271 | **0.057** |
| True Pointwise | 0.191 | 0.586 (fails) | **0.100** |
| Grid-Interpolated | 2.96 (fails) | 6.73 (fails) | **0.109** |
| Hard Target | 0.163 | 0.401 | 0.112 |
| Soft σ=0.5 | **0.076** | **0.232** | 0.248 (fails) |
| Soft σ=0.2 | 0.110 | 0.298 | 0.133 |

---

## Key Finding: σ Must Scale with Grid Range

**Why soft targets fail here:**

The grid range changed from 5.0-20.0 (experiments 4-5) to 1.0 (experiment 6).

With grid_range=1.0 and σ=0.5:
- The Gaussian blob covers **half the entire grid**
- This provides too much smoothing - the target signal is smeared across 50% of cells
- The model can't learn precise predictions

**Rule of thumb:** σ should be proportional to grid_range / grid_size (one cell width).

| Experiment | Grid Range | Grid Size | Cell Width | σ=0.5 coverage |
|------------|------------|-----------|------------|----------------|
| Exp 4 | 5.0 | 64 | 0.156 | ~3 cells |
| Exp 5 | 20.0 | 64 | 0.625 | ~1 cell |
| Exp 6 | 1.0 | 64 | 0.031 | ~16 cells (too wide!) |

---

## Why Pointwise Works Here

1. **Simpler task:** Relative displacement is translation-invariant - easier to learn
2. **Normalized grid:** Targets always in [-1, 1], no extreme values
3. **Fixed Δt:** Only one thing to learn (velocity extraction)

---

## Implications

1. **σ is not universal** - it must scale with your grid resolution
2. **For normalized grids**, use smaller σ (e.g., σ=0.05 for 64×64 grid over [-1,1])
3. **True pointwise works** when the task is simple enough

---

## Configuration

```python
config = {
    'grid_size': 64,
    'grid_range': 1.0,  # Normalized [-1, 1]
    'endpoint_range': 10.0,  # Tracks end anywhere in [-10, 10]
    'velocity_range': (0.1, 4.5),
    'dt': 1.0,
    'scale_factor': 4.5,  # max_velocity * dt
}
```
