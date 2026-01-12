# Fourier Head Trajectory Prediction Experiment Report

## Executive Summary

**Research Question:** Can a transformer backbone with a 2D Fourier head output learn dead reckoning (constant velocity prediction) using NLL loss?

**Answer: YES.** With a sufficiently wide velocity range, the model learns to predict in the correct direction. The best approach (curriculum learning) achieves an expected value error of **0.077 units** vs regression baseline of 0.074 units.

---

## 1. Experimental Setup

### 1.1 Hardware
- **GPU:** NVIDIA L4 (23GB VRAM)
- **CUDA Version:** 13.0
- **Driver:** 580.95.05

### 1.2 Software
- Python 3.9
- PyTorch 2.8.0
- NumPy 2.0.2

---

## 2. Dataset Specification

### 2.1 Data Generation
Synthetic straight-line trajectories with constant velocity:

| Parameter | Value |
|-----------|-------|
| Training samples | 10,000 (each with unique random velocity & angle) |
| Validation samples | 1,000 (each with unique random velocity & angle) |
| Sequence length | 10 timesteps |
| Prediction horizon | 1 timestep |
| Velocity range | **[0.1, 4.5] units/timestep** (wide range) |
| Angle range | [0, 2π] (uniform) |
| Noise std | 0.0 (perfect lines) |

### 2.2 Data Format
- **Input:** 10 (x, y) positions relative to last observed position → shape (10, 2)
- **Output:** Next position displacement relative to last position → shape (2,)
- **NO velocity is given to the model** - it must infer velocity from positions

### 2.3 How Data is Generated
```python
for _ in range(num_samples):  # 10,000 unique trajectories
    velocity = np.random.uniform(0.1, 4.5)  # Random continuous value
    angle = np.random.uniform(0, 2 * np.pi)  # Random continuous value
    # Generate straight line: position = velocity × time
```

### 2.4 Target Statistics (Training Set)
| Metric | Value |
|--------|-------|
| Target magnitude mean | 2.251 |
| Target magnitude std | 1.313 |
| Target X mean | -0.110 |
| Target X std | 1.848 |
| Target Y mean | -0.065 |
| Target Y std | 1.834 |

**Note:** The wide velocity range [0.1, 4.5] means targets can be anywhere from very close to the origin (0.1 units) to near the edge of the grid (4.5 units). This prevents the model from exploiting a narrow marginal distribution.

---

## 3. Model Architecture

### 3.1 Transformer Backbone

```
TrajectoryTransformer:
├── Input Embedding: Linear(2 → 64)
│   └── Maps (x, y) positions to 64-dim vectors
├── Positional Encoding: Sinusoidal (max_len=100)
│   └── Encodes sequence position (not spatial position)
└── Transformer Encoder:
    ├── Layers: 2
    ├── Attention heads: 4
    ├── Model dimension: 64
    ├── Feedforward dimension: 128
    └── Output: Last token embedding (64,)
```

### 3.2 2D Fourier Head

```
FourierHead2D:
├── Input: (batch, 64) from transformer
├── Coefficient predictor: Linear(64 → 578)
│   └── 289 cos coefficients + 289 sin coefficients
├── Frequency grid: 17×17 = (2×8+1)² frequencies
├── Spatial period: L = 10 units
├── Output grid: 64×64 over [-5, 5]²
└── Normalization: log_softmax over all 4096 grid cells
```

**Fourier Basis:**
$$p(x, y) = \text{softmax}\left( \sum_{j=-8}^{8} \sum_{k=-8}^{8} c_{jk} \cos\left(\frac{2\pi(jx + ky)}{10}\right) + s_{jk} \sin\left(\frac{2\pi(jx + ky)}{10}\right) \right)$$

### 3.3 Parameter Count
| Component | Parameters |
|-----------|------------|
| Input embedding | 192 |
| Transformer encoder | ~50,000 |
| Fourier head | 37,570 |
| **Total** | ~88,000 |

---

## 4. Training Configuration

### 4.1 Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Epochs | 100 |
| Warmup steps | 500 |
| Fourier coefficient L2 reg | 0.01 |
| Optimizer | AdamW |

### 4.2 Loss Functions Tested

1. **Standard NLL:** Hard assignment to nearest grid point
   - Loss = -log(p[target_grid_cell])

2. **Soft Target NLL:** Gaussian kernel (σ=0.5) centered on target
   - Creates soft target distribution, uses KL divergence loss

3. **Mixed Loss:** 0.5×NLL + 0.5×(MSE between density mode and target)
   - Combines likelihood with direct supervision on peak location

4. **Curriculum Learning:** Soft NLL with increasing velocity ranges
   - Stage 1: [0.1, 0.5] for 20 epochs
   - Stage 2: [0.1, 1.5] for 20 epochs
   - Stage 3: [0.1, 3.0] for 30 epochs
   - Stage 4: [0.1, 4.5] for 30 epochs

---

## 5. Results

### 5.1 Regression Baseline (Sanity Check)

**Purpose:** Verify transformer can extract velocity from position history.

| Metric | Value |
|--------|-------|
| Final Val MSE | 0.00362 |
| Final Val MAE | 0.0484 |
| Mean prediction error | 0.0741 |
| **Status** | **PASS** |

**Conclusion:** The transformer successfully extracts velocity from positions alone (no explicit velocity input).

### 5.2 Fourier Head Results

| Method | Expected Error | Mode Error |
|--------|---------------|------------|
| Standard NLL | 0.1129 | 0.1415 |
| Soft Targets | 0.0912 | 0.1370 |
| **Curriculum** | **0.0770** | **0.1199** |
| Mixed Loss | 0.0785 | 0.1082 |

### 5.3 Comparison to Baseline

| Model | Error | Relative to Regression |
|-------|-------|------------------------|
| Regression baseline | 0.0741 | 1.00× |
| Best Fourier (Curriculum) | 0.0770 | 1.04× |

**The Fourier head achieves nearly identical performance to direct regression!**

### 5.4 Visual Evidence: Model Uses Input

Looking at the curriculum learning predictions:
- **Different input trajectories produce different output densities**
- Density peaks align with trajectory direction
- Model correctly predicts displacement magnitude and direction

### 5.5 Training Dynamics

**Standard NLL:**
- Initial loss: 7.82 → Final: 1.71
- Entropy: 7.77 → 1.76 (distribution sharpens)
- Gradient norm stable at ~23

**Curriculum Learning:**
- Successfully progresses through velocity stages
- Final stage (v=0.1-4.5): Train loss = 0.069

---

## 6. Key Findings

### 6.1 Wide Velocity Range is Critical

With narrow velocity range [0.5, 2.0]:
- Targets cluster in a ring at radius ~1.25
- Model can achieve low error by predicting marginal distribution
- **Does not learn conditional prediction**

With wide velocity range [0.1, 4.5]:
- Targets spread across entire grid
- Model MUST use input to predict correctly
- **Learns true dead reckoning**

### 6.2 Curriculum Learning Works Best

Starting with small velocities and gradually increasing:
1. Allows model to first learn direction
2. Then learn to scale by velocity magnitude
3. Achieves lowest error (0.077)

### 6.3 Soft Targets Help Convergence

Gaussian soft targets (σ=0.5) provide smoother gradients than hard assignment, leading to faster and more stable training.

---

## 7. Conclusions

### 7.1 Main Results

1. **YES, the transformer + 2D Fourier head can learn dead reckoning**
2. The model achieves error (0.077) comparable to direct regression (0.074)
3. Critical requirement: velocity range must be wide enough to prevent marginal exploitation

### 7.2 Architecture Insights

- Transformer successfully extracts velocity from position sequences
- Fourier head can represent directional probability densities
- 2D Fourier basis with 17×17 frequencies is sufficient for this task

### 7.3 Training Recommendations

1. Use **curriculum learning** - start simple, increase complexity
2. Use **soft targets** (σ=0.3-0.5) for smoother gradients
3. Ensure **wide target distribution** in training data

---

## 8. Reproducibility

### 8.1 Random Seeds
```python
torch.manual_seed(42)
np.random.seed(42)
```

### 8.2 Code Location
```
/home/ec2-user/projects/trackfm-deadreckon/fourier_trajectory_test/
├── run_experiment.py    # Main experiment script
├── results/             # Output visualizations and JSON
└── EXPERIMENT_REPORT.md # This report
```

### 8.3 Running the Experiment
```bash
cd /home/ec2-user/projects/trackfm-deadreckon/fourier_trajectory_test
python3 run_experiment.py
```

---

## 9. Generated Files

```
results/
├── nll_predictions.png       # Standard NLL density visualizations
├── nll_diagnostics.png       # Training curves for NLL
├── soft_predictions.png      # Soft target density visualizations
├── soft_diagnostics.png      # Training curves for soft targets
├── curriculum_predictions.png # Curriculum learning results (BEST)
├── mixed_predictions.png     # Mixed loss density visualizations
├── mixed_diagnostics.png     # Training curves for mixed loss
└── experiment_results.json   # All numerical results
```

---

## 10. Full Configuration

```python
config = {
    # Data
    'num_train': 10000,
    'num_val': 1000,
    'seq_len': 10,
    'pred_horizon': 1,
    'velocity_range': (0.1, 4.5),  # WIDE RANGE - critical!
    'noise_std': 0.0,

    # Model
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 128,
    'grid_size': 64,
    'num_freqs': 8,
    'grid_range': 5.0,

    # Training
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'warmup_steps': 500,
    'fourier_coeff_reg': 0.01,
}
```
