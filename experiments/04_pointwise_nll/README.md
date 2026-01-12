# Experiment 4: Loss Function Comparison for Fourier Density Head

## Executive Summary

**Question:** How should we compute the loss for training the Fourier density head?

**Finding:** Soft target with σ=0.5 is best (0.076 error), even beating regression baseline.

---

## Methods Compared

| # | Method | How it works |
|---|--------|--------------|
| 1 | **True Pointwise** | Evaluate Fourier at exact (x,y), no grid for target |
| 2 | **Grid-Interpolated** | Bilinear interpolation from density grid |
| 3 | **Hard Target** | One-hot on nearest grid cell, cross-entropy |
| 4 | **Soft Target (σ=0.5)** | Gaussian blob on grid, cross-entropy |
| 5 | **Soft Target (σ=0.2)** | Tighter Gaussian blob, cross-entropy |

---

## Results

| Method | Error | Status |
|--------|-------|--------|
| **Soft Target (σ=0.5)** | **0.076** | **Best** |
| Regression Baseline | 0.079 | Reference |
| Soft Target (σ=0.2) | 0.110 | Good |
| Hard Target (one-hot) | 0.163 | Works |
| True Pointwise NLL | 0.191 | Works |
| Grid-Interpolated NLL | 2.956 | Fails |

---

## Key Findings

### 1. Soft Target Beats Everything (even regression!)
σ=0.5 achieves 0.076 error, better than the 0.079 regression baseline. The Gaussian smoothing provides optimal gradient signal.

### 2. Wider Gaussian is Better
σ=0.5 beats σ=0.2. More smoothing = better optimization, even though it's "less precise."

### 3. Hard Target Works (but not great)
One-hot on nearest grid cell (0.163 error) is essentially the "grid version" of pointwise. It works but the sparse gradient signal limits performance.

### 4. Grid Interpolation Kills Gradients
Bilinear interpolation from the softmax'd grid (2.956 error) completely fails. The gradient signal is destroyed by the interpolation.

### 5. True Pointwise Works
Evaluating Fourier directly at target (0.191 error) works better than grid-interpolated, but worse than soft targets due to sparse gradients.

---

## Method Details

### True Pointwise NLL
```python
# Evaluate Fourier at exact target (no grid for target!)
logit_at_target = fourier_eval(coeffs, target_x, target_y)
log_Z = logsumexp(logits_on_grid)  # grid only for normalization
loss = -logit_at_target + log_Z
```

### Hard Target (One-hot)
```python
# Find nearest grid cell
x_idx = argmin(|grid_x - target_x|)
y_idx = argmin(|grid_y - target_y|)
# Cross-entropy against one-hot
loss = -log(density[x_idx, y_idx])
```

### Soft Target (Gaussian)
```python
# Gaussian blob over whole grid
soft_target = exp(-dist_to_target² / (2σ²))
soft_target = soft_target / sum(soft_target)
# Cross-entropy over all cells
loss = -sum(soft_target * log(density))
```

---

## Recommendation

**Use soft target loss with σ=0.5** for training Fourier density heads:
- Best accuracy (beats regression!)
- Smooth optimization landscape
- Robust to initialization

The wider Gaussian provides gradient signal from many grid cells, making optimization much easier than pointwise or hard target approaches.

---

## Configuration

```python
config = {
    'grid_size': 64,
    'grid_range': 5.0,
    'num_freqs': 8,
    'num_epochs': 50,
    'learning_rate': 3e-4,
}
```
