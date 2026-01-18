# Experiment 13: Vessel Type Classification

## Overview

This experiment evaluates whether pre-training on trajectory forecasting (Experiment 11) improves vessel type classification - a behavior recognition task with thousands of labeled examples.

**Research Question**: Does the 116M parameter TrackFM encoder provide better representations for vessel type classification compared to training from scratch?

## Why This Task

Unlike anomaly detection (Exp 12) which had only 25 labeled anomalies, vessel classification uses the DMA data's `Ship type` field, giving us:
- **Thousands of vessels per class** (vs 25 anomalies)
- **Same domain as pre-training** (Danish waters)
- **Well-studied benchmark** with known baselines (82-87% accuracy)
- **Statistical power** to detect significant differences

## Dataset

**Source**: DMA Danish Maritime Authority AIS data (same as pre-training)

| Class | Vessels | Records |
|-------|---------|---------|
| Fishing | 766 | 70M |
| Cargo | 1,964 | 57M |
| Tanker | 837 | 24M |
| Passenger | 380 | 31M |

**Task**: Classify vessel trajectories by movement patterns, withholding the self-reported type field from the model.

## Experimental Conditions

### Base Conditions

| Condition | Description |
|-----------|-------------|
| **pretrained** | Load encoder from Exp 11, fine-tune all layers with differential LR |
| **random_init** | Randomly initialized encoder, train from scratch |
| **frozen_pretrained** | Load encoder from Exp 11, freeze encoder, only train classifier |
| **two_stage** | Stage 1: freeze encoder, train head (10 epochs); Stage 2: unfreeze all |

### Pooling Strategies

| Pooling | Description |
|---------|-------------|
| **mean** (default) | Mean pooling over sequence |
| **attention** | Learnable attention-weighted pooling |
| **mha** | Multi-head attention with learnable query |
| **hybrid** | Mean + Max concatenation |
| **hybrid_attention** | Mean + Max + Attention concatenation |

### Parameter-Efficient Fine-Tuning

| Technique | Description |
|-----------|-------------|
| **LoRA** | Low-Rank Adaptation (rank=4) - inject trainable low-rank matrices into attention |

## Model Architecture

```
Input (6 features: lat, lon, sog, cog_sin, cog_cos, dt)
    │
    ▼
┌─────────────────────────────────┐
│   TrackFM Encoder (116M params) │
│   - d_model: 768                │
│   - nhead: 16                   │
│   - num_layers: 16              │
│   - dim_feedforward: 3072       │
└─────────────────────────────────┘
    │
    ▼
    Mean Pooling
    │
    ▼
┌─────────────────────────────────┐
│   Classifier Head               │
│   - Linear(768, 384) + ReLU     │
│   - Dropout(0.3)                │
│   - Linear(384, 128) + ReLU     │
│   - Dropout(0.3)                │
│   - Linear(128, 4)              │
└─────────────────────────────────┘
    │
    ▼
    Softmax → Class Probabilities
```

## Quick Start

```bash
cd experiments/13_vessel_classification

# Run full experiment (all 3 conditions)
./run_all.sh my_experiment

# Run with learning curves (tests different data fractions)
./run_all.sh my_experiment --learning-curves

# Run single condition
python scripts/run_experiment.py --exp-name test --condition pretrained

# Extract data only
python scripts/extract_data.py
```

## Results

### Performance Summary

#### Base Conditions (Mean Pooling)

| Condition | Accuracy | F1 Macro | F1 Weighted |
|-----------|----------|----------|-------------|
| **random_init** | **68.6%** | **0.56** | **0.65** |
| pretrained | 66.3% | 0.55 | 0.64 |
| two_stage | 64.8% | 0.53 | 0.61 |
| frozen_pretrained | 64.0% | 0.52 | 0.61 |

#### Pooling Ablation (Pretrained Encoder)

| Pooling Strategy | Accuracy | F1 Macro | F1 Weighted |
|-----------------|----------|----------|-------------|
| **attention** | **65.7%** | **0.5694** | **0.6360** |
| hybrid_attention | 64.0% | 0.5487 | 0.6195 |
| mean (baseline) | 66.3% | 0.55 | 0.64 |
| mha | 56.4% | 0.4583 | 0.5315 |
| hybrid | 52.6% | 0.4081 | 0.4727 |

#### LoRA Fine-Tuning

| Condition | Accuracy | F1 Macro | F1 Weighted | Trainable Params |
|-----------|----------|----------|-------------|------------------|
| pretrained_lora | 63.4% | 0.5185 | 0.5999 | 104M (98K LoRA) |
| pretrained_attention_lora | 63.1% | 0.5352 | 0.6097 | 104M (98K LoRA) |

### Key Finding: No Transfer Benefit

**Pre-training on trajectory forecasting does not improve classification performance.**

Random initialization slightly outperforms all pre-trained conditions:
- **Random init beats pretrained by 2.3%** accuracy (68.6% vs 66.3%)
- **Two-stage fine-tuning** (warmup head, then unfreeze) doesn't help
- **Frozen pretrained performs worst** - pre-trained features alone are insufficient
- **Attention pooling** improves F1 macro slightly (0.55 → 0.5694) but still trails random init
- **LoRA does not help** - no benefit from parameter-efficient fine-tuning with sufficient data
- **MHA and hybrid pooling** actually hurt performance significantly

### Per-Class Analysis

| Class | Random Init F1 | Pretrained F1 | Δ |
|-------|----------------|---------------|---|
| Fishing | 0.77 | 0.78 | -0.01 |
| Cargo | 0.71 | 0.68 | +0.03 |
| Tanker | 0.04 | 0.07 | -0.03 |
| Passenger | 0.70 | 0.69 | +0.01 |

All models struggle with the Tanker class (only 42 test samples, similar movement patterns to Cargo).

### Critical Implementation Note: Normalization

**Input normalization must exactly match the pre-training setup.** Initial experiments showed pretrained performing 6% worse than random_init due to normalization mismatch:

| Setting | Exp 11 (pre-training) | Initial Exp 13 (wrong) | Fixed Exp 13 |
|---------|----------------------|------------------------|--------------|
| Latitude | (lat - 56.25) / 1.0 | per-trajectory center | (lat - 56.25) / 1.0 |
| Longitude | (lon - 11.5) / 2.0 | per-trajectory center | (lon - 11.5) / 2.0 |
| SOG | sog / 30.0 | sog / 50.0 | sog / 30.0 |
| Δt | dt / 300.0 | dt / 600.0 | dt / 300.0 |

After fixing normalization, pretrained improved from 49.3% → 66.3% accuracy.

### Why No Transfer Benefit?

1. **Sequence length mismatch**: Pre-training used 128-position sequences (~2 hours) while classification uses up to 512 positions (~4.6 hours average). The model's positional encodings were only trained on positions 0-127, creating distribution shift when processing longer sequences.

2. **Task mismatch**: Trajectory forecasting learns local velocity/acceleration patterns for predicting future positions. Classification needs global features: route structure, overall speed profiles, area-of-operation patterns.

3. **Class similarity**: Analysis revealed Cargo and Tanker vessels have nearly identical trajectory characteristics (straightness effect size = 0.034). Over 50% of both classes are straight-line shipping routes, making them fundamentally difficult to distinguish from movement patterns alone.

4. **Sufficient training data**: With ~1,500 training trajectories, random initialization has enough signal to learn task-specific features from scratch.

### Trajectory Analysis

To understand why transfer learning doesn't help, we analyzed the trajectory characteristics:

**Dataset Statistics:**
- Train: 1,577 trajectories | Val: 336 | Test: 344
- Trajectory length: 50-512 positions (avg 302)
- Duration: ~4.6 hours average (mean interval 61s, median 27s)

**Class Similarity Analysis:**

| Metric | Cargo | Tanker | Effect Size |
|--------|-------|--------|-------------|
| Straightness | 0.723 | 0.710 | 0.034 (tiny) |
| % Straight Lines | 57.9% | 51.3% | - |
| Speed Variability | 0.162 | 0.200 | 0.213 (small) |

Cargo and Tanker vessels exhibit nearly identical movement patterns - both are primarily straight-line shipping routes through the Danish maritime zone. This fundamental class overlap limits achievable accuracy regardless of model architecture.

**Sequence Length Distribution Shift:**

| Setting | Pre-training (Exp 11) | Classification (Exp 13) |
|---------|----------------------|-------------------------|
| Sequence length | 128 positions | Up to 512 positions |
| Duration | ~2 hours | ~4.6 hours average |

The model's sinusoidal positional encodings were trained on positions 0-127 but must generalize to positions 0-511 during classification.

### Conclusion

Pre-training on trajectory forecasting provides **no benefit** for vessel type classification when sufficient labeled data is available (~2K trajectories). Training from scratch achieves slightly better results (68.6% vs 65.7% best pretrained).

**We tested multiple techniques to improve transfer:**
- Attention pooling (best pretrained result)
- Multi-head attention pooling (hurt performance)
- Hybrid pooling (mean + max - hurt performance significantly)
- Two-stage fine-tuning (no benefit)
- LoRA parameter-efficient fine-tuning (no benefit)

None of these techniques made pretrained weights outperform random initialization. This suggests the forecasting task learns representations optimized for next-step prediction rather than vessel behavior discrimination.

## Learning Curves

The `--learning-curves` flag tests with limited training data:
- 1%, 5%, 10%, 25%, 50%, 100% of training data
- Pre-training should show largest gains at low data regimes

## File Structure

```
13_vessel_classification/
├── README.md
├── run_all.sh
├── EXPERIMENT_13_SPEC.md
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   │   ├── trajectories.npy
│   │   ├── labels.npy
│   │   ├── splits.json
│   │   └── stats.json
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── extract_trajectories.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── factory.py
│   │   └── lora.py
│   ├── training/
│   │   └── trainer.py
│   └── evaluation/
│       └── metrics.py
├── scripts/
│   ├── extract_data.py
│   ├── run_experiment.py
│   └── run_pooling_ablation.py
└── experiments/
    └── <exp_name>/
        ├── config.yaml
        ├── summary.json
        ├── pretrained_frac1.0/
        ├── random_init_frac1.0/
        └── frozen_pretrained_frac1.0/
```

## Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Macro**: Unweighted mean of per-class F1 scores
- **F1 Weighted**: Class-weighted F1 score
- **Per-class precision/recall/F1**: Detailed breakdown
- **Confusion Matrix**: Class-by-class error analysis

## Why Vessel Types Have Distinct Patterns

| Class | Movement Signature |
|-------|-------------------|
| **Fishing** | Irregular patterns, circling, speed changes, fishing grounds |
| **Cargo** | Direct routes, consistent speed, shipping lanes |
| **Tanker** | Similar to cargo, slower, wider turns |
| **Passenger** | Regular schedules, ferry routes, multiple stops |

A model pre-trained on trajectory forecasting has learned some of these patterns, but the hypothesis that these representations transfer to classification was **not supported** - random initialization performs equally well or better.

## Comparison to Experiment 12

| Aspect | Exp 12 (Anomaly) | Exp 13 (Classification) |
|--------|------------------|-------------------------|
| Positive examples | ~25 | 1000+ per class |
| Classes | 2 (binary) | 4 (multi-class) |
| Domain | DTU (different) | DMA (same as pre-train) |
| Statistical power | Low | High |
