"""
Model factory for creating anomaly detection models.

Creates models for three experimental conditions:
- pretrained: Load encoder from experiment 11, fine-tune all
- random_init: Random initialization, train from scratch
- frozen_pretrained: Load encoder from experiment 11, freeze encoder
"""

import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding (matches experiment 11)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ClassifierHead(nn.Module):
    """MLP classification head for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Binary output
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AnomalyDetector(nn.Module):
    """
    Anomaly detection model combining encoder + classifier.

    The encoder architecture matches CausalAISModel from experiment 11,
    but replaces the Fourier head with a classification head.
    """

    def __init__(
        self,
        encoder_config: Dict,
        classifier_config: Dict,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.d_model = encoder_config['d_model']
        self.freeze_encoder = freeze_encoder

        # Input projection (matches experiment 11)
        self.input_proj = nn.Linear(
            encoder_config['input_features'],
            encoder_config['d_model']
        )

        # Positional encoding (matches experiment 11)
        self.pos_encoder = SinusoidalEncoding(
            encoder_config['d_model'],
            max_len=encoder_config['max_seq_length']
        )

        # Transformer encoder (matches experiment 11)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_config['d_model'],
            nhead=encoder_config['nhead'],
            dim_feedforward=encoder_config['dim_feedforward'],
            dropout=encoder_config['dropout'],
            batch_first=True,
            norm_first=True  # Pre-norm for stability (matches exp 11)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_config['num_layers']
        )

        # Classifier head (new for anomaly detection)
        self.classifier = ClassifierHead(
            input_dim=encoder_config['d_model'],
            hidden_dims=classifier_config['hidden_dims'],
            dropout=classifier_config['dropout']
        )

        self.pooling = classifier_config['pooling']

        # Causal mask cache
        self._causal_masks = {}

        if freeze_encoder:
            self._freeze_encoder()

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask."""
        if seq_len not in self._causal_masks:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self._causal_masks[seq_len] = mask
        return self._causal_masks[seq_len].to(device)

    def _freeze_encoder(self):
        """Freeze encoder weights for frozen_pretrained condition."""
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        # Note: pos_encoder has no learnable parameters (uses register_buffer)

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pool sequence representations to single vector."""
        batch_size, max_len, hidden = x.shape
        device = x.device

        if self.pooling == "mean":
            # Masked mean pooling
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return pooled

        elif self.pooling == "max":
            # Masked max pooling
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)
            x_masked = x.masked_fill(~mask, float('-inf'))
            return x_masked.max(dim=1)[0]

        elif self.pooling == "last":
            # Get representation at last valid position
            indices = (lengths - 1).clamp(min=0)
            return x[torch.arange(batch_size, device=device), indices]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for anomaly detection.

        Args:
            features: (batch, seq_len, 6) input features
            lengths: (batch,) actual sequence lengths

        Returns:
            (batch, 1) anomaly probabilities
        """
        batch_size, seq_len, _ = features.shape
        device = features.device

        # Input projection
        x = self.input_proj(features)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask
        causal_mask = self._get_causal_mask(seq_len, device)

        # Transformer encoding
        x = self.transformer(x, mask=causal_mask)

        # Pool to single vector
        pooled = self._pool(x, lengths)

        # Classify
        logits = self.classifier(pooled)

        return logits  # Return raw logits, apply sigmoid in loss/inference

    def get_embeddings(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get encoder embeddings for visualization (e.g., t-SNE)."""
        batch_size, seq_len, _ = features.shape
        device = features.device

        x = self.input_proj(features)
        x = self.pos_encoder(x)

        causal_mask = self._get_causal_mask(seq_len, device)
        x = self.transformer(x, mask=causal_mask)

        return self._pool(x, lengths)

    def get_encoder_params(self):
        """Get encoder parameters (for differential learning rates)."""
        return list(self.input_proj.parameters()) + list(self.transformer.parameters())

    def get_classifier_params(self):
        """Get classifier parameters (for differential learning rates)."""
        return list(self.classifier.parameters())


def load_pretrained_weights(
    model: AnomalyDetector,
    checkpoint_path: str,
    device: str = "cuda"
) -> AnomalyDetector:
    """
    Load pre-trained encoder weights from experiment 11.

    Args:
        model: AnomalyDetector model
        checkpoint_path: Path to experiment 11 checkpoint
        device: Device to load weights on

    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading pre-trained weights from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pretrained_state = checkpoint['model_state_dict']
    else:
        pretrained_state = checkpoint

    # Get current model state
    model_state = model.state_dict()

    # Keys to load from pretrained model
    encoder_keys = ['input_proj', 'transformer', 'pos_encoder']

    loaded_keys = []
    skipped_keys = []

    for key in model_state.keys():
        # Check if this is an encoder key
        if any(key.startswith(ek) for ek in encoder_keys):
            if key in pretrained_state:
                # Check shape compatibility
                if model_state[key].shape == pretrained_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    loaded_keys.append(key)
                elif key == 'pos_encoder.pe':
                    # Handle positional encoding truncation (checkpoint may have longer seq_len)
                    # Sinusoidal PE is deterministic, so we can truncate safely
                    target_len = model_state[key].shape[1]
                    model_state[key] = pretrained_state[key][:, :target_len, :]
                    loaded_keys.append(key)
                    print(f"  Truncated {key}: {pretrained_state[key].shape} -> {model_state[key].shape}")
                else:
                    print(f"Shape mismatch for {key}: "
                          f"model={model_state[key].shape}, "
                          f"checkpoint={pretrained_state[key].shape}")
                    skipped_keys.append(key)
            else:
                skipped_keys.append(key)

    model.load_state_dict(model_state)

    print(f"  Loaded {len(loaded_keys)} encoder parameters")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} parameters (not in checkpoint or shape mismatch)")

    return model


def create_model(
    condition: str,
    config: Dict,
    device: str = "cuda"
) -> AnomalyDetector:
    """
    Create model for specified experimental condition.

    Args:
        condition: One of "pretrained", "random_init", "frozen_pretrained"
        config: Full experiment configuration
        device: Device to load model on

    Returns:
        AnomalyDetector model
    """
    encoder_config = config['model']['encoder']
    classifier_config = config['model']['classifier']

    freeze_encoder = (condition == "frozen_pretrained")

    model = AnomalyDetector(
        encoder_config=encoder_config,
        classifier_config=classifier_config,
        freeze_encoder=freeze_encoder
    )

    if condition in ["pretrained", "frozen_pretrained"]:
        checkpoint_path = config['model']['pretrained_checkpoint']
        model = load_pretrained_weights(model, checkpoint_path, device)

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {condition}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model
