# Track-FM: Trajectory Forecasting with Multi-Horizon Transformers

A high-performance trajectory prediction system using decoder-only transformers with low-rank Fourier heads for multi-horizon forecasting.

## Features

- **Causal Multi-Horizon Prediction**: Predicts multiple future trajectory points using causal attention
- **Low-Rank Fourier Heads**: Efficient continuous probability density modeling with rank-4 decomposition
- **Streaming Data Pipeline**: Memory-efficient processing of large-scale AIS trajectory datasets
- **nanoGPT Architecture**: Optimized decoder-only transformer backbone
- **Mixed Precision Training**: Accelerated training with automatic mixed precision

## Architecture

The system combines:
- **nanoGPT backbone**: Decoder-only transformer with causal attention
- **Low-rank Fourier head**: Continuous PDF modeling with 64 frequencies, rank-4 decomposition
- **Multi-horizon output**: Simultaneous prediction of 10 future trajectory points
- **Coordinate transformation**: Local UV coordinates with proper Jacobian correction

## Performance

- **GPU Utilization**: 99% on Tesla T4 with batch size 2560
- **Training Speed**: 2,600+ samples/second
- **Memory Efficient**: Streaming dataset handles 75M+ trajectory points
- **Optimized Operations**: Custom einsum-based Fourier computations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd track-fm

# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

### Training

```bash
# Train on S3 data with optimized settings
python scripts/train_causal_multihorizon.py

# Monitor training progress
watch -n 2 python scripts/monitor_loss.py
```

### Configuration

Key parameters in `train_causal_multihorizon.py`:
- `seq_len=20`: Input sequence length
- `horizon=10`: Number of future steps to predict
- `fourier_m=64`: Number of Fourier frequencies
- `fourier_rank=4`: Low-rank decomposition rank
- `batch_size=2560`: Training batch size

## Dataset Format

The system expects trajectory data with columns:
- `lat`, `lon`: Geographic coordinates
- `mmsi`: Vessel identifier
- `timestamp`: Time information
- `track_id`: Unique track identifier

## Model Components

### Core Architecture
- `src/nano_gpt_trajectory.py`: nanoGPT-based transformer backbone
- `src/four_head_2D_LR.py`: Low-rank Fourier head implementation
- `scripts/train_causal_multihorizon.py`: Main training script

### Data Pipeline
- Streaming dataset with chunked S3 loading
- Random sampling across tracks to prevent overfitting
- Local coordinate transformation with Jacobian correction

### Monitoring
- `scripts/monitor_loss.py`: Real-time training metrics
- Loss curves, timing breakdowns, GPU utilization
- Training speed and trend analysis

## Research

This implementation focuses on:
- **Causal trajectory modeling**: No information leakage from future positions
- **Multi-horizon prediction**: Simultaneous forecasting of multiple time steps
- **Continuous PDFs**: Fourier-based density modeling vs discrete approaches
- **Scalable training**: Efficient processing of large maritime datasets

## Performance Optimizations

- **Einsum operations**: Optimized tensor contractions in Fourier head
- **Mixed precision**: FP16 for transformer, FP32 for Fourier computations
- **Streaming data**: Memory-efficient large dataset processing
- **Gradient scaling**: Stable training with large batch sizes

## File Structure

```
track-fm/
├── src/                    # Core model implementations
├── scripts/                # Training and monitoring scripts
├── notebooks/              # Research and analysis notebooks
├── output/                 # Visualization outputs
├── checkpoints/            # Model checkpoints
└── pyproject.toml         # Package configuration
```

## Citation

```bibtex
@misc{track-fm,
  title={Track-FM: Trajectory Forecasting with Multi-Horizon Transformers},
  year={2025},
  publisher={GitHub},
  url={<repository-url>}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.