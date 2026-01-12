# Fourier Head Trajectory Prediction Experiments

This repository contains experiments testing whether a transformer backbone with a 2D Fourier head can learn trajectory prediction (dead reckoning).

## Repository Structure

```
fourier_trajectory_test/
├── README.md                 # This file
├── src/                      # Shared code modules
│   ├── models.py            # Model architectures
│   ├── data.py              # Data generation
│   └── training.py          # Training utilities
├── experiments/              # Individual experiments
│   ├── 01_fixed_dt_baseline/ # Fixed time horizon experiment
│   │   ├── run_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   ├── 02_variable_dt_methods/ # Variable Δt conditioning methods
│   │   ├── run_variable_dt_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   ├── 03_multi_horizon/     # Multi-horizon prediction
│   │   ├── run_multi_horizon_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   ├── 04_pointwise_nll/     # Pointwise vs soft target NLL
│   │   ├── run_pointwise_nll_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   ├── 05_variable_dt_loss_comparison/ # Loss methods with variable Δt
│   │   ├── run_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   ├── 06_relative_displacement/ # Relative displacement with normalized grid
│   │   ├── run_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   ├── 07_decoder_transformer/  # Causal vs bidirectional attention
│   │   ├── run_experiment.py
│   │   ├── results/
│   │   └── README.md        # Experiment report
│   └── 08_causal_multihorizon/  # Causal decoder + multi-horizon prediction
│       ├── run_experiment.py
│       ├── comprehensive_evaluation.py
│       ├── results/
│       ├── comprehensive_results/
│       └── README.md        # Experiment report
```

## Experiments

### Experiment 1: Fixed Δt Baseline
**Question:** Can transformer + 2D Fourier head learn dead reckoning with NLL loss?

**Answer:** YES - with wide velocity range [0.1, 4.5], curriculum learning achieves 0.077 error (vs 0.074 regression baseline).

See: `experiments/01_fixed_dt_baseline/README.md`

### Experiment 2: Variable Δt Methods
**Question:** How should we incorporate variable prediction time horizon (Δt)?

**Finding:** Concat to Input (append Δt to each position as (x, y, Δt)) is best method, achieving 0.247 error vs 0.366 regression baseline (32% improvement).

See: `experiments/02_variable_dt_methods/README.md`

### Experiment 3: Multi-Horizon Prediction
**Question:** Can we efficiently predict multiple future horizons by encoding the trajectory once?

**Finding:** Yes! Encode once + duplicate embedding is **4.7x faster** than encoding separately for each horizon, with equal or better accuracy.

See: `experiments/03_multi_horizon/README.md`

### Experiment 4: Loss Function Comparison
**Question:** How should we compute the loss for the Fourier density head?

**Finding:** Soft target (σ=0.5) is best (0.076 error), beating even regression! Hard target (one-hot) works (0.163). True pointwise works (0.191). Grid-interpolated fails (2.96).

See: `experiments/04_pointwise_nll/README.md`

### Experiment 5: Loss Comparison with Variable Δt
**Question:** Do the loss function results hold when adding variable Δt?

**Finding:** NO! True pointwise NLL **fails** with variable Δt (0.586 error) while it worked with fixed Δt (0.191). Soft target σ=0.5 remains best (0.232 vs 0.271 regression).

See: `experiments/05_variable_dt_loss_comparison/README.md`

### Experiment 6: Relative Displacement Prediction
**Question:** Do results change when predicting relative displacement with a normalized grid?

**Finding:** Complete reversal! True pointwise works well (0.100), soft targets fail (0.248). σ must scale with grid range.

See: `experiments/06_relative_displacement/README.md`

### Experiment 7: Decoder Transformer Backbone
**Question:** Does causal (autoregressive) attention work better than bidirectional attention?

**Setup:** Compares encoder transformer (bidirectional) vs decoder transformer (causal masking) for trajectory prediction.

See: `experiments/07_decoder_transformer/README.md`

### Experiment 8: Causal Multi-Horizon Prediction
**Question:** Can we efficiently train a causal decoder transformer for trajectory prediction with variable time spacing and multiple prediction horizons?

**Finding:** YES! The causal multi-horizon model achieves **99.4% MSE reduction** vs naive baselines (last position / zero prediction). Model predicts 1-5 positions ahead with MSE ranging from 0.12 (horizon 1) to 1.13 (horizon 5), vs 8.2-192.8 for baselines.

Key features validated:
- Causal attention (autoregressive) works for trajectory prediction
- Multi-horizon prediction (1-5 steps ahead) from each position
- Variable time spacing with Δt conditioning
- Works with different input history lengths (3-11 positions)

See: `experiments/08_causal_multihorizon/README.md`

## Quick Start

```bash
# Run experiment 1
cd experiments/01_fixed_dt_baseline
python3 run_experiment.py

# Run experiment 2
cd experiments/02_variable_dt_methods
python3 run_variable_dt_experiment.py

# Run experiment 3
cd experiments/03_multi_horizon
python3 run_multi_horizon_experiment.py

# Run experiment 4
cd experiments/04_pointwise_nll
python3 run_pointwise_nll_experiment.py

# Run experiment 5
cd experiments/05_variable_dt_loss_comparison
python3 run_experiment.py

# Run experiment 6
cd experiments/06_relative_displacement
python3 run_experiment.py

# Run experiment 7
cd experiments/07_decoder_transformer
python3 run_experiment.py

# Run experiment 8
cd experiments/08_causal_multihorizon
python3 run_experiment.py
# For comprehensive evaluation with baselines:
python3 comprehensive_evaluation.py
```

## Key Findings

1. **Transformer extracts velocity** from position sequences (no explicit velocity input needed)
2. **2D Fourier head works** but requires careful training (soft targets, curriculum learning)
3. **Wide velocity range critical** to prevent marginal distribution exploitation
4. **Concat to Input** is best method for incorporating Δt (32% better than regression)
5. **Grid range critical** for variable Δt - must cover max displacement (velocity × Δt)
6. **Multi-horizon is efficient** - encode trajectory once, reuse embedding for all horizons (4.7x speedup)
7. **Soft targets best for training** - true pointwise NLL works for fixed Δt but soft targets (σ=0.5) optimize better
8. **Pointwise fails with variable Δt** - sparse gradient signal insufficient for harder learning problems
9. **σ must scale with grid range** - soft target σ=0.5 works for grid_range=5-20, but fails for normalized grids (range=1)
10. **Causal attention works** - decoder transformer (autoregressive) successfully predicts trajectories
11. **Multi-horizon + causal training validated** - single model predicts 1-5 steps ahead with 99.4% MSE reduction vs baselines
12. **Δt conditioning critical** - error increases 6x without time conditioning (2.63 vs 0.44)

## Hardware Used
- NVIDIA L4 GPU (23GB)
- CUDA 13.0
