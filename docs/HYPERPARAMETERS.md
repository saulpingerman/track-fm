# TrackFM Hyperparameter Reference

Complete hyperparameters for reproducing all paper results. If you're using your own training scripts, these are the values you need to match.

## Experiment 11: Pre-training (Foundation Model)

### Model Architecture (XLarge — used for paper results)

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| nhead | 16 |
| num_layers | 16 |
| dim_feedforward | 3072 |
| dropout | 0.1 |
| input_features | 6 (lat, lon, sog, cog_sin, cog_cos, dt) |
| max_seq_len | 128 |
| grid_size | 64 |
| grid_range | 0.3 degrees (~33km) |
| num_freqs (Fourier) | 12 |
| Parameters | ~116M |

### Training

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW |
| learning_rate | 3e-4 |
| weight_decay | 1e-5 |
| batch_size | 300 |
| max_horizon | 400 steps |
| num_horizon_samples | 4 (per position per batch) |
| warmup_steps | 20 |
| early_stop_patience | 4 validation checks |
| val_every_n_batches | 20 |
| use_amp | true (FP16) |
| gradient_accumulation | 1 |

### Loss

| Parameter | Value |
|-----------|-------|
| loss_type | Cross-entropy with soft targets |
| sigma (soft target) | 0.003 (in normalized grid coords) |
| dr_sigma (baseline) | 0.05 (dead reckoning baseline sigma) |

### Feature Normalization

| Feature | Transform | Notes |
|---------|-----------|-------|
| lat | `(lat - 56.25) / 1.0` | Centered on Danish waters |
| lon | `(lon - 11.5) / 2.0` | Centered on Danish waters |
| sog | `sog / 30.0` | Knots, clipped |
| cog | `sin(cog*π/180)`, `cos(cog*π/180)` | Two features from one |
| dt | `dt_seconds / 300.0` | 300s = 5 minutes |

### Data Filtering

| Parameter | Value |
|-----------|-------|
| min_track_length | 600 positions |
| min_sog | 3.0 knots (filter stationary) |
| gap_threshold | 1800 seconds (30 min, for segmentation) |
| region | 54–58.5°N, 7–16°E (Danish waters) |
| time_range | 69 days, Jan–Feb 2025 |
| train/val/test split | 80/10/10 (temporal) |

### Scaling Study Results (Table 1 in paper)

| Scale | d_model | heads | layers | FFN | params | vs DR | vs LP |
|-------|---------|-------|--------|-----|--------|-------|-------|
| Small | 128 | 8 | 4 | 512 | ~1M | 1.4x | 1.9x |
| Medium | 256 | 8 | 6 | 1024 | ~5M | 1.6x | 2.2x |
| Large | 384 | 16 | 8 | 2048 | ~18M | 2.1x | 3.1x |
| **XLarge** | **768** | **16** | **16** | **3072** | **~116M** | **2.9x** | **4.2x** |

---

## Experiment 12: Anomaly Detection

### Task Setup

| Parameter | Value |
|-----------|-------|
| Task | Binary classification (normal/anomalous) |
| Dataset | DTU Danish Waters AIS |
| Trajectories | 521 total, ~25 anomalies |
| Class ratio | ~10:1 (normal:anomaly) |
| max_seq_length | 512 |
| Evaluation | Nested 5-fold CV, 5 random seeds |
| Optimization | Bayesian (Ax library), 30 trials per condition |
| Primary metric | AUPRC |

### Model

| Parameter | Value |
|-----------|-------|
| Encoder | Same as Exp 11 XLarge (768d, 16 layers) |
| Classification head | MLP: 768 → 384 → 128 → 1 |
| Head dropout | 0.3 |
| Head activation | ReLU |
| Loss | BCE with positive_weight=10.0 |

### Best Hyperparameters (from Bayesian Optimization)

#### Pretrained condition (AUPRC = 0.872 ± 0.106)

| Parameter | Value |
|-----------|-------|
| learning_rate | 2.22e-4 |
| weight_decay | **0.0** |
| beta1 | 0.930 |
| beta2 | 0.9999 |
| pooling | **mean** |
| optimizer | AdamW |
| scheduler | cosine |
| warmup_epochs | 5 |
| max_epochs | 100 |
| batch_size | 64 |
| early_stopping_patience | 15 |
| use_amp | true |

#### Random init condition (AUPRC = 0.753 ± 0.168)

| Parameter | Value |
|-----------|-------|
| learning_rate | 1.94e-4 |
| weight_decay | **0.086** |
| beta1 | 0.897 |
| beta2 | 0.998 |
| pooling | **last** |
| optimizer | AdamW |
| scheduler | cosine |
| warmup_epochs | 5 |
| max_epochs | 100 |
| batch_size | 64 |
| early_stopping_patience | 15 |
| use_amp | true |

#### Frozen pretrained condition (AUPRC = 0.276 ± 0.138)

Encoder weights loaded from Exp 11 but frozen during training. Only the classification head is trained. This performed poorly, demonstrating that encoder adaptation is essential.

### Data Augmentation (applied during training)

| Augmentation | Parameter | Value |
|---|---|---|
| Temporal crop | min_ratio / max_ratio | 0.5 / 1.0 |
| Speed scaling | min_scale / max_scale | 0.85 / 1.15 |
| Coordinate jitter | std_degrees | 0.0001 |
| Point dropout | dropout_rate | 0.15 |

---

## Experiment 13: Vessel Classification

### Task Setup

| Parameter | Value |
|-----------|-------|
| Task | 4-class classification |
| Classes | Fishing (717), Cargo (624), Passenger (645), Tanker (271) |
| Total trajectories | 2,257 |
| max_seq_length | 512 |
| Train/Val/Test | 70/15/15 |
| Optimization | Bayesian (Ax library), 30 trials per condition |
| Primary metric | F1 macro |

### Model

| Parameter | Value |
|-----------|-------|
| Encoder | Same as Exp 11 XLarge (768d, 16 layers) |
| Classification head | MLP: 768 → 384 → 128 → 4 |
| Head dropout | 0.3 |
| Loss | Cross-entropy with label_smoothing=0.1 |

### Best Hyperparameters (from Bayesian Optimization)

#### Pretrained condition (Accuracy = 73.8%, F1 = 0.641)

| Parameter | Value |
|-----------|-------|
| learning_rate | 4.06e-4 |
| weight_decay | **0.001** |
| beta1 | 0.903 |
| beta2 | 0.979 |
| pooling | **mean** |
| optimizer | AdamW |
| scheduler | cosine |
| warmup_epochs | 3 |
| max_epochs | 50 |
| batch_size | 64 |
| early_stopping_patience | 10 |

#### Random init condition (Accuracy = 70.8%, F1 = 0.618)

| Parameter | Value |
|-----------|-------|
| learning_rate | 8.48e-5 |
| weight_decay | **0.017** |
| beta1 | 0.800 |
| beta2 | 0.991 |
| pooling | **last** |
| optimizer | AdamW |
| scheduler | cosine |
| warmup_epochs | 3 |
| max_epochs | 50 |
| batch_size | 64 |
| early_stopping_patience | 10 |

### Per-Class Performance (Pretrained)

| Class | F1 | Notes |
|-------|-----|-------|
| Fishing | ~0.77 | Distinctive motion patterns |
| Passenger | ~0.70 | Regular ferry schedules |
| Cargo | moderate | Hard to distinguish from Tanker |
| Tanker | ~0.07 | Nearly identical to Cargo motion |

---

## Experiment 14: Extended Horizon (Not in Paper)

### 18M Model (Large scale)

| Parameter | Value |
|-----------|-------|
| d_model | 384 |
| nhead | 16 |
| num_layers | 8 |
| dim_feedforward | 2048 |
| dropout | 0.1 |
| grid_size | **128** (vs 64 in Exp 11) |
| grid_range | **0.6** degrees (vs 0.3 in Exp 11) |
| max_horizon | **800** steps (~2 hours) |
| num_horizon_samples | 8 |
| batch_size | 88 |
| learning_rate | 3e-4 |
| weight_decay | 1e-5 |
| warmup_steps | 500 |
| Parameters | ~18M |

### 100M Model (run_100M_h800)

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| nhead | 16 |
| num_layers | **14** (vs 16 in Exp 11) |
| dim_feedforward | 3072 |
| grid_size | 128 |
| grid_range | 0.6 |
| max_horizon | 800 |
| All other params | Same as 18M run |

### Data

| Parameter | Value |
|-----------|-------|
| Source | 1 year of Danish AIS data |
| Training samples | 109M (pre-shuffled) |
| Validation samples | 313K |
| Window size | 928 positions (128 input + 800 horizon) |
| Format | MosaicML MDS (recommended) or parquet |
| Normalization | Same as Exp 11 |

---

## Key Pattern: Pretrained vs Random Init Hyperparameters

This pattern holds across both downstream tasks and is critical for reproducing results:

| Setting | Pretrained | Random Init |
|---------|-----------|-------------|
| Pooling | **mean** | **last** |
| Weight decay | **low/zero** | **moderate** |
| Learning rate | **higher** (2–4e-4) | **lower** (8e-5–2e-4) |

**Why this matters**: Using identical hyperparameters for both conditions produces misleading results. In Exp 13, default hyperparameters made random init appear better than pretrained. Only Bayesian optimization (30 trials each) revealed the true advantage of pre-training.

## Bayesian Optimization Search Ranges

Used for both Exp 12 and 13 (Ax library):

| Parameter | Range | Scale |
|-----------|-------|-------|
| learning_rate | [1e-6, 1e-2] | log |
| weight_decay | [0, 0.3] | linear |
| beta1 | [0.8, 0.99] | linear |
| beta2 | [0.95, 0.9999] | linear |
| pooling | {mean, last} | choice |
