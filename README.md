# Track-FM: Multi-Horizon Trajectory Forecasting

A high-performance vessel trajectory prediction system using horizon-aware transformers with Fourier probability density heads.

**Repository**: https://github.com/saulpingerman/track-fm

## Features

- **Horizon-Aware Architecture**: Each prediction horizon gets unique temporal context through learned embeddings
- **Fourier PDF Heads**: Continuous probability density modeling with 128 frequencies and rank-4 low-rank decomposition  
- **Multi-Horizon Prediction**: Simultaneous forecasting of 1-10 step-ahead vessel positions
- **Robust Training**: Built-in safeguards against numerical instabilities (inf/nan handling)
- **Streaming Pipeline**: Memory-efficient processing of large AIS trajectory datasets

## Key Innovation

**Problem Solved**: Original multi-horizon models produced identical PDFs for all prediction horizons due to architectural limitations.

**Solution**: Horizon embeddings provide each time step with unique temporal context:
```python
horizon_emb = self.horizon_embedding(horizon_step)  # Learned time embedding
enhanced_repr = torch.cat([gpt_output, horizon_emb], dim=-1)  # Temporal context
pdf = self.fourier_head(enhanced_repr, target_pos)  # Horizon-specific prediction
```

## Architecture

- **nanoGPT Backbone**: Decoder-only transformer (6 layers, 8 heads, 128D)
- **Horizon Embeddings**: 32D learned embeddings for temporal context (steps 0-9)
- **Fourier PDF Head**: 128×128 frequencies with rank-4 decomposition
- **Training Safeguards**: Automatic inf/nan detection and batch skipping

## Installation

```bash
# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Training

### Quick Start
```bash
# Default configuration
python train.py

# Background training with logging
./train.sh
```

### Flexible Configuration
```bash
# Use custom config file
python train.py --config config/experiments/high_res.yaml

# Override specific parameters
python train.py --override training.lr=2e-4 model.fourier_m=256 training.batch_size=512

# Create and run experiment
python train.py --experiment my_test --override training.epochs=10
```

### Configuration Files
- `config/default.yaml` - Default training configuration
- `config/experiments/high_res.yaml` - Higher resolution (256 frequencies, rank-8)
- `config/experiments/fast_training.yaml` - Quick training for debugging

## Model Performance

**Loss Analysis**: Training reaches fundamental limits of Fourier representation (~-8.2 NLL) due to:
- 128×128 frequencies provide ~0.008 normalized units resolution  
- Rank-4 decomposition limits maximum achievable PDF peak height
- Current architecture optimally balances expressiveness vs. computational efficiency

**Key Results**:
- Successfully learns horizon-specific predictions (variance: 0.0154 vs 0.0000 for original)
- Robust training with automatic error recovery
- Scales to 75M+ trajectory data points

## Configuration
- `seq_len=20`: Input sequence length
- `horizon=10`: Number of future steps to predict
- `fourier_m=128`: Number of Fourier frequencies
- `fourier_rank=4`: Low-rank decomposition rank
- `batch_size=1024`: Training batch size
- `ckpt_every=1000`: Checkpoint frequency (batches)

## Files

**Core Package** (`trackfm/`):
- `train_trackfm.py` - Main training script with safeguards
- `trackfm_model.py` - Horizon-aware model architecture  
- `trackfm_dataset.py` - AIS trajectory data loader
- `nano_gpt_trajectory.py`, `four_head_2D_LR.py` - Core model components

**Documentation** (`docs/`):
- `ARCHITECTURAL_FLAW_ANALYSIS.md` - Technical problem analysis
- `TRAINING_SAFEGUARDS.md` - Robustness improvements

**Results** (`results/`):
- Training loss curves and Fourier representation analysis plots

## Data Format

AIS trajectory data in Parquet format with columns:
- `LAT`, `LON`: Geographic coordinates
- `MMSI`: Vessel identifier  
- `BaseDateTime`: Timestamp