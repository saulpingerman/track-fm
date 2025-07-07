# Track-FM: Trajectory Forecasting with Transformers

A high-performance vessel trajectory prediction system using GPT-style transformers with low-rank Fourier heads for probabilistic multi-step forecasting.

## Features

- **Multi-Step Trajectory Prediction**: Predicts 10 future positions (10-100 minutes ahead)
- **Time-Aware Architecture**: Incorporates temporal information for horizon-specific predictions
- **Probabilistic Outputs**: Full probability distributions using low-rank Fourier decomposition
- **Efficient Training**: Optimized for large-scale AIS vessel tracking data
- **Real-Time Inference**: Fast prediction suitable for operational use

## Architecture

Track-FM combines:
- **GPT-style Transformer**: Causal attention mechanism prevents information leakage
- **Low-Rank Fourier Head**: Efficient continuous probability density modeling (128 frequencies, rank 4)
- **Time Encoding**: Horizon-aware predictions based on time intervals
- **Local Coordinate System**: UV space transformation for translation invariance

## Performance

- **Training Speed**: ~1.5k samples/second on Tesla T4
- **Model Size**: 1.5M parameters
- **Loss**: Converges to -1.0 NLL on moving vessels
- **GPU Utilization**: 95%+ with optimized data loading

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/track-fm.git
cd track-fm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train Track-FM on vessel data
python scripts/train_trackfm.py

# Monitor training progress
python scripts/visualization/plot_loss_log_scale.py logs/latest.log
```

### Visualization

```bash
# Visualize predictions on real tracks
python scripts/visualization/viz_time_aware_real_data.py

# Analyze model behavior
python scripts/visualization/diagnose_predictions.py
```

## Data Format

Track-FM expects AIS vessel tracking data with:
- `lat`, `lon`: Geographic coordinates
- `track_id`: Unique trajectory identifier  
- `timestamp`: Time of observation
- Minimum average speed: 5 knots (for quality filtering)

## Model Configuration

Key parameters:
- `seq_len`: 20 (input sequence length)
- `horizon`: 10 (prediction steps)
- `d_model`: 128 (model dimension)
- `num_layers`: 6 (transformer layers)
- `fourier_m`: 128 (Fourier frequencies)
- `fourier_rank`: 4 (low-rank decomposition)

## Project Structure

```
track-fm/
├── src/
│   ├── nano_gpt_trajectory.py    # Core GPT architecture
│   └── four_head_2D_LR.py        # Low-rank Fourier head
├── scripts/
│   ├── train_*.py                # Training scripts
│   └── visualization/            # Analysis tools
├── logs/                         # Training logs
├── checkpoints/                  # Model checkpoints
├── output/                       # Visualizations
└── experiments.json              # Experiment tracking metadata
```

## Experiment Tracking

This project uses a simple experiment tracking system to manage hyperparameters and outputs:

- **Experiment IDs**: Short identifiers (e.g., `exp001`, `exp002`) replace verbose filenames
- **Metadata Storage**: All hyperparameters stored in `experiments.json`
- **File Naming**: Logs, checkpoints, and outputs use experiment IDs

### Example Experiment Entry

```json
{
  "exp001": {
    "timestamp": "20250706_020302",
    "model": "time_aware",
    "dataset": "training",
    "frequencies": 128,
    "batch_size": 512,
    "optimized": false,
    "description": "Initial time-aware training run",
    "log_file": "logs/exp001.log",
    "checkpoint_dir": "checkpoints/exp001/"
  }
}
```

### Viewing Experiment Details

```bash
# List all experiments
cat experiments.json | jq '.experiments | keys'

# View specific experiment
cat experiments.json | jq '.experiments.exp001'
```

## Key Insights

1. **Moving Vessels Only**: Training on vessels with >5 knot average speed prevents the model from learning trivial "no movement" predictions

2. **Time-Aware Predictions**: Each prediction horizon uses temporal encoding to produce appropriate uncertainty estimates

3. **Local Coordinates**: UV space transformation (±50 miles) provides translation invariance and numerical stability

## Training Tips

- Start with batch size 512 and increase based on GPU memory
- Use local data files instead of S3 for better GPU utilization
- Filter stationary vessels from training data
- Monitor NLL values - should be around -1.0 for good performance

## Citation

```bibtex
@misc{trackfm2025,
  title={Track-FM: Trajectory Forecasting with Transformers},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/track-fm}
}
```

## License

MIT License - see LICENSE file for details.