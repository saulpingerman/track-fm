"""
Classification head for anomaly detection.

This module provides the MLP head that converts encoder representations
to binary anomaly predictions.
"""

import torch
import torch.nn as nn
from typing import List


class ClassifierHead(nn.Module):
    """
    MLP classification head for anomaly detection.

    Takes pooled encoder representations and outputs anomaly logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
        use_batch_norm: bool = False
    ):
        """
        Args:
            input_dim: Dimension of input features (d_model from encoder)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability between layers
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) pooled encoder representations

        Returns:
            (batch, 1) anomaly logits (before sigmoid)
        """
        return self.mlp(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence representations.

    Learns to weight different positions in the sequence.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            x: (batch, seq_len, hidden_dim) sequence representations
            lengths: (batch,) actual sequence lengths

        Returns:
            (batch, hidden_dim) pooled representations
        """
        batch_size, max_len, hidden = x.shape
        device = x.device

        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Create mask for padding
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over valid positions
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (batch, hidden)

        return pooled
