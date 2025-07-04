#!/usr/bin/env python
"""
TrackFM Training Entry Point

Examples:
    # Default training
    python train.py
    
    # Custom config
    python train.py --config config/experiments/high_res.yaml
    
    # Override specific parameters
    python train.py --override training.lr=2e-4 model.fourier_m=256
    
    # Create and run experiment
    python train.py --experiment my_experiment --override training.lr=1e-3
"""

if __name__ == "__main__":
    from trackfm.train_trackfm import main
    main()