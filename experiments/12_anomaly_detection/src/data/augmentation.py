"""
Trajectory augmentation for training data.

Augmentations are applied only to training data to improve generalization.
"""

import numpy as np
from typing import Optional


class TrajectoryAugmentor:
    """
    Apply various augmentations to trajectory data.

    Augmentations:
    - Temporal crop: Random subsequence
    - Speed scaling: Scale SOG by random factor
    - Coordinate jitter: Add small noise to positions
    - Point dropout: Randomly remove points
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Augmentation configuration from config.yaml
        """
        self.enabled = config.get('enabled', True)
        self.temporal_crop = config.get('temporal_crop', {})
        self.speed_scaling = config.get('speed_scaling', {})
        self.coordinate_jitter = config.get('coordinate_jitter', {})
        self.point_dropout = config.get('point_dropout', {})

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a trajectory.

        Args:
            trajectory: Array of shape (seq_len, 6) with features
                       [lat, lon, sog, cog_sin, cog_cos, dt]

        Returns:
            Augmented trajectory
        """
        if not self.enabled:
            return trajectory

        traj = trajectory.copy()

        # Apply augmentations in order
        if self.temporal_crop.get('enabled', False):
            traj = self._temporal_crop(traj)

        if self.point_dropout.get('enabled', False):
            traj = self._point_dropout(traj)

        if self.speed_scaling.get('enabled', False):
            traj = self._speed_scaling(traj)

        if self.coordinate_jitter.get('enabled', False):
            traj = self._coordinate_jitter(traj)

        return traj

    def _temporal_crop(self, traj: np.ndarray) -> np.ndarray:
        """Extract random temporal subsequence."""
        min_ratio = self.temporal_crop.get('min_ratio', 0.5)
        max_ratio = self.temporal_crop.get('max_ratio', 1.0)

        seq_len = len(traj)
        if seq_len < 10:
            return traj

        # Random crop ratio
        ratio = np.random.uniform(min_ratio, max_ratio)
        crop_len = max(10, int(seq_len * ratio))

        # Random start position
        max_start = seq_len - crop_len
        if max_start <= 0:
            return traj

        start = np.random.randint(0, max_start + 1)
        return traj[start:start + crop_len]

    def _point_dropout(self, traj: np.ndarray) -> np.ndarray:
        """Randomly remove points from trajectory."""
        dropout_rate = self.point_dropout.get('dropout_rate', 0.15)

        if len(traj) < 10:
            return traj

        # Keep at least 50% of points
        keep_prob = max(0.5, 1.0 - dropout_rate)

        # Random mask
        mask = np.random.random(len(traj)) < keep_prob

        # Always keep first and last
        mask[0] = True
        mask[-1] = True

        return traj[mask]

    def _speed_scaling(self, traj: np.ndarray) -> np.ndarray:
        """Scale speed (SOG) by random factor."""
        min_scale = self.speed_scaling.get('min_scale', 0.85)
        max_scale = self.speed_scaling.get('max_scale', 1.15)

        scale = np.random.uniform(min_scale, max_scale)
        traj[:, 2] = traj[:, 2] * scale  # SOG is column 2

        return traj

    def _coordinate_jitter(self, traj: np.ndarray) -> np.ndarray:
        """Add small Gaussian noise to coordinates."""
        std = self.coordinate_jitter.get('std_degrees', 0.0001)

        # Add noise to lat and lon
        noise = np.random.normal(0, std, (len(traj), 2))
        traj[:, 0:2] = traj[:, 0:2] + noise

        return traj


class NoAugmentation:
    """No-op augmentor for validation/test data."""

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        return trajectory
