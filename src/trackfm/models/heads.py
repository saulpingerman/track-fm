"""Downstream heads on the TrackFM encoder.

Head architecture follows the exp 12/13 design (MLP d->d/2->128->out,
dropout 0.3, mean/last pooling). A key finding from those experiments:
pretrained encoders want `mean` pooling, random-init wants `last` —
keep pooling configurable and tune per condition.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from trackfm.models.encoder import CausalAISModel

Pooling = Literal["mean", "last"]


def _pool(emb: torch.Tensor, pooling: Pooling) -> torch.Tensor:
    if pooling == "mean":
        return emb.mean(dim=1)
    return emb[:, -1, :]


class MLPHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PortClassifier(nn.Module):
    """Origin or destination classification over discovered ports + edges."""

    def __init__(self, encoder: CausalAISModel, num_classes: int,
                 pooling: Pooling = "mean", dropout: float = 0.3,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.pooling: Pooling = pooling
        self.head = MLPHead(encoder.d_model, num_classes, dropout)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        emb = self.encoder.encode(windows)          # (B, L, d)
        return self.head(_pool(emb, self.pooling))  # (B, num_classes)


class EtaRegressor(nn.Module):
    """Time-to-arrival regression; predicts log1p(remaining seconds)."""

    def __init__(self, encoder: CausalAISModel, pooling: Pooling = "mean",
                 dropout: float = 0.3, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.pooling: Pooling = pooling
        self.head = MLPHead(encoder.d_model, 1, dropout)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        emb = self.encoder.encode(windows)
        return self.head(_pool(emb, self.pooling)).squeeze(-1)  # log1p(seconds)

    @staticmethod
    def target(remaining_s: torch.Tensor) -> torch.Tensor:
        return torch.log1p(remaining_s)

    @staticmethod
    def to_seconds(pred: torch.Tensor) -> torch.Tensor:
        return torch.expm1(pred.clamp(min=0.0, max=20.0))
