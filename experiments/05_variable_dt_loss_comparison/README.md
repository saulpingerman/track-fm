# Experiment 5: Loss Function Comparison with Variable Δt

## Executive Summary

**Question:** Do loss function results from Experiment 4 hold when we add variable Δt?

**Finding:** No! True pointwise NLL **fails** with variable Δt (0.586 error) while it worked with fixed Δt (0.191 error). Soft target σ=0.5 remains best.

---

## Results

| Method | Fixed Δt (Exp 4) | Variable Δt (Exp 5) | Change |
|--------|------------------|---------------------|--------|
| **Soft Target (σ=0.5)** | **0.076** | **0.232** | Best in both |
| Regression Baseline | 0.079 | 0.271 | Reference |
| Soft Target (σ=0.2) | 0.110 | 0.298 | Good |
| Hard Target (one-hot) | 0.163 | 0.401 | Mediocre |
| True Pointwise NLL | 0.191 | 0.586 | **FAILS** |
| Grid-Interpolated NLL | 2.956 | 6.725 | Fails |

---

## Key Finding: Why Pointwise Fails with Variable Δt

**Fixed Δt (works):** The model only needs to learn velocity extraction. The target distribution is relatively simple - all predictions fall within a narrow annulus.

**Variable Δt (fails):** The model must learn both velocity extraction AND Δt conditioning. The target distribution spans a much larger area (grid_range=20 vs 5). Pointwise NLL provides gradient signal from only ONE point per sample.

**The problem:** With variable Δt, the learning problem is harder but pointwise NLL provides the same (sparse) gradient signal. The model can't find the right coefficients.

**Soft targets work:** They provide gradient signal from MANY grid cells, giving the optimizer more information to work with. The wider Gaussian (σ=0.5) works better than tighter (σ=0.2) because it provides even more gradient signal.

---

## Implications for Production

If your production system uses:
- Variable prediction horizons (Δt varies)
- True pointwise NLL loss

**This is likely why it's failing.** The sparse gradient signal from pointwise NLL is insufficient for learning the more complex mapping.

**Recommendation:** Switch to soft target loss (σ=0.5 relative to your grid spacing).

---

## Configuration

```python
config = {
    'grid_size': 64,
    'grid_range': 20.0,  # Must cover max displacement: 4.5 * 4.0 = 18
    'num_freqs': 8,
    'dt_range': (0.5, 4.0),
    'velocity_range': (0.1, 4.5),
    'num_epochs': 30,
}
```
