"""
Vessel type classifier model.

Architecture:
- TrackFM Encoder (from Experiment 11)
- Mean pooling over sequence
- MLP classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TrackFMEncoder(nn.Module):
    """
    Transformer encoder for trajectory sequences.
    Matches the architecture from Experiment 11.
    """

    def __init__(
        self,
        input_features: int = 6,
        d_model: int = 768,
        nhead: int = 16,
        num_layers: int = 16,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, seq_len, input_features)
            lengths: Actual sequence lengths (batch,)

        Returns:
            encoded: Encoded sequence (batch, seq_len, d_model)
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask for padding
        if lengths is not None:
            batch_size, seq_len = x.shape[:2]
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        else:
            mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Final layer norm
        x = self.norm(x)

        return x


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - True for valid positions
        Returns:
            pooled: (batch, d_model)
        """
        weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            weights = weights.masked_fill(~mask, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling with learnable query."""

    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - True for valid positions
        Returns:
            pooled: (batch, d_model)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class ClassificationHead(nn.Module):
    """MLP classification head with multiple pooling options."""

    def __init__(
        self,
        d_model: int,
        hidden_dims: list = [384, 128],
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()

        self.pooling_type = pooling

        # Initialize pooling-specific modules
        if pooling == "attention":
            self.attention_pool = AttentionPooling(d_model)
            in_dim = d_model
        elif pooling == "mha":  # Multi-head attention pooling
            self.mha_pool = MultiHeadAttentionPooling(d_model, nhead=4)
            in_dim = d_model
        elif pooling == "hybrid":  # Mean + Max concatenation
            in_dim = d_model * 2
        elif pooling == "hybrid_attention":  # Mean + Max + Attention
            self.attention_pool = AttentionPooling(d_model)
            in_dim = d_model * 3
        else:
            in_dim = d_model

        # Build MLP layers
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def _get_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Create boolean mask (True for valid positions)."""
        if lengths is None:
            return None
        return torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) < lengths.unsqueeze(1)

    def _mean_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            return (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)

    def _max_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        return x.max(dim=1)[0]

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded sequence (batch, seq_len, d_model)
            lengths: Actual sequence lengths (batch,)

        Returns:
            logits: Classification logits (batch, num_classes)
        """
        mask = self._get_mask(x, lengths)

        if self.pooling_type == "mean":
            pooled = self._mean_pool(x, mask)
        elif self.pooling_type == "max":
            pooled = self._max_pool(x, mask)
        elif self.pooling_type == "last":
            if lengths is not None:
                idx = (lengths - 1).clamp(min=0)
                pooled = x[torch.arange(x.size(0), device=x.device), idx]
            else:
                pooled = x[:, -1]
        elif self.pooling_type == "attention":
            pooled = self.attention_pool(x, mask)
        elif self.pooling_type == "mha":
            pooled = self.mha_pool(x, mask)
        elif self.pooling_type == "hybrid":
            mean_pool = self._mean_pool(x, mask)
            max_pool = self._max_pool(x, mask)
            pooled = torch.cat([mean_pool, max_pool], dim=-1)
        elif self.pooling_type == "hybrid_attention":
            mean_pool = self._mean_pool(x, mask)
            max_pool = self._max_pool(x, mask)
            attn_pool = self.attention_pool(x, mask)
            pooled = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_type}")

        return self.mlp(pooled)


class VesselClassifier(nn.Module):
    """
    Full vessel type classifier.

    Combines TrackFM encoder with classification head.
    """

    def __init__(
        self,
        input_features: int = 6,
        d_model: int = 768,
        nhead: int = 16,
        num_layers: int = 16,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        classifier_hidden_dims: list = [384, 128],
        num_classes: int = 4,
        classifier_dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()

        self.encoder = TrackFMEncoder(
            input_features=input_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        self.classifier = ClassificationHead(
            d_model=d_model,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=classifier_dropout,
            pooling=pooling,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, seq_len, input_features)
            lengths: Actual sequence lengths (batch,)

        Returns:
            logits: Classification logits (batch, num_classes)
        """
        encoded = self.encoder(x, lengths)
        logits = self.classifier(encoded, lengths)
        return logits

    def get_embeddings(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get pooled embeddings (for visualization)."""
        encoded = self.encoder(x, lengths)

        # Mean pooling
        if lengths is not None:
            mask = torch.arange(encoded.size(1), device=encoded.device).expand(encoded.size(0), -1) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)

        return pooled
