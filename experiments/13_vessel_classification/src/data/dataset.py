"""
PyTorch Dataset for vessel type classification.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple, Optional


class VesselClassificationDataset(Dataset):
    """Dataset for vessel type classification."""

    def __init__(
        self,
        trajectories: List[np.ndarray],
        labels: List[int],
        max_length: int = 512,
        augment: bool = False,
    ):
        """
        Args:
            trajectories: List of trajectory arrays, each shape (seq_len, 6)
            labels: List of class indices
            max_length: Maximum sequence length (pad/truncate to this)
            augment: Whether to apply data augmentation
        """
        self.trajectories = trajectories
        self.labels = labels
        self.max_length = max_length
        self.augment = augment

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            features: Tensor of shape (max_length, 6)
            label: Class index
            length: Actual sequence length (before padding)
        """
        traj = self.trajectories[idx].copy()
        label = self.labels[idx]

        # Apply augmentation if enabled
        if self.augment:
            traj = self._augment(traj)

        # Get actual length
        length = len(traj)

        # Truncate if too long
        if length > self.max_length:
            # Random start for truncation during training
            if self.augment:
                start = np.random.randint(0, length - self.max_length + 1)
            else:
                start = 0
            traj = traj[start:start + self.max_length]
            length = self.max_length

        # Pad if too short
        if length < self.max_length:
            pad_length = self.max_length - length
            traj = np.pad(traj, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

        features = torch.from_numpy(traj).float()
        return features, torch.tensor(label, dtype=torch.long), length

    def _augment(self, traj: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Temporal crop (50-100% of original)
        if np.random.random() < 0.5 and len(traj) > 50:
            crop_ratio = np.random.uniform(0.5, 1.0)
            crop_len = int(len(traj) * crop_ratio)
            start = np.random.randint(0, len(traj) - crop_len + 1)
            traj = traj[start:start + crop_len]

        # Speed scaling (SOG is index 2)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.85, 1.15)
            traj[:, 2] = np.clip(traj[:, 2] * scale, 0, 1)

        # Small coordinate jitter (lat/lon are indices 0, 1)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.0001, (len(traj), 2))
            traj[:, :2] += noise

        return traj


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Dict:
    """Custom collate function for variable-length sequences."""
    features, labels, lengths = zip(*batch)

    features = torch.stack(features)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        'features': features,
        'labels': labels,
        'lengths': lengths,
    }
