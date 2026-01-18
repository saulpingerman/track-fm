"""
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Injects trainable low-rank matrices into Linear layers while freezing original weights.
Based on: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Original weight is frozen, and low-rank matrices A and B are trained.
    Output = W @ x + (B @ A) @ x * scaling
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze original weights
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialize A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = F.linear(x, self.weight, self.bias)
        # Add LoRA contribution
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result + lora_out * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Apply LoRA to specific linear layers in a model.

    Args:
        model: The model to modify
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to target (e.g., ['q_proj', 'v_proj'])
                       If None, targets all linear layers in attention

    Returns:
        model: Modified model with LoRA layers
    """
    if target_modules is None:
        # Default: target query and value projections in attention
        target_modules = ['in_proj', 'out_proj']

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be targeted
            should_replace = any(target in name for target in target_modules)
            if should_replace:
                # Get parent module
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    child_name = name

                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                replaced += 1

    print(f"Applied LoRA to {replaced} layers (rank={rank}, alpha={alpha})")
    return model


def get_lora_params(model: nn.Module) -> list:
    """Get only the LoRA parameters (for optimizer)."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_lora_params(model: nn.Module) -> tuple:
    """Count trainable LoRA params vs frozen params."""
    lora_params = 0
    frozen_params = 0
    other_trainable = 0

    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params += param.numel()
        elif not param.requires_grad:
            frozen_params += param.numel()
        else:
            other_trainable += param.numel()

    return lora_params, other_trainable, frozen_params
