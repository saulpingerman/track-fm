"""
TrackFM: Multi-Horizon Trajectory Forecasting

A high-performance vessel trajectory prediction system using horizon-aware 
transformers with Fourier probability density heads.
"""

__version__ = "0.1.0"

from .trackfm_model import HorizonAwareTrajectoryForecaster
from .trackfm_dataset import StreamingMultiHorizonAISDataset

__all__ = [
    "HorizonAwareTrajectoryForecaster",
    "StreamingMultiHorizonAISDataset",
]