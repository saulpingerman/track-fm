"""
Model factory for creating and loading models.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

from .classifier import VesselClassifier
from .lora import apply_lora_to_model, get_lora_params, count_lora_params


def parse_condition(condition: str) -> dict:
    """
    Parse condition string to extract base condition and modifiers.

    Examples:
        'pretrained' -> {'base': 'pretrained', 'pooling': None, 'lora': False}
        'pretrained_attention' -> {'base': 'pretrained', 'pooling': 'attention', 'lora': False}
        'pretrained_lora' -> {'base': 'pretrained', 'pooling': None, 'lora': True}
        'pretrained_attention_lora' -> {'base': 'pretrained', 'pooling': 'attention', 'lora': True}
    """
    parts = condition.split('_')
    base = parts[0]

    pooling = None
    lora = False

    for part in parts[1:]:
        if part == 'lora':
            lora = True
        elif part in ['attention', 'mha', 'hybrid', 'hybrid_attention', 'max', 'last']:
            pooling = part
        elif part in ['init', 'pretrained', 'stage']:  # Part of base condition
            base = f"{base}_{part}"

    return {'base': base, 'pooling': pooling, 'lora': lora}


def create_model(config: Dict, condition: str) -> VesselClassifier:
    """
    Create a VesselClassifier model based on configuration and condition.

    Args:
        config: Configuration dictionary
        condition: Condition string (e.g., 'pretrained', 'pretrained_attention_lora')

    Returns:
        model: VesselClassifier instance
    """
    # Parse condition for modifiers
    parsed = parse_condition(condition)
    base_condition = parsed['base']
    pooling_override = parsed['pooling']
    use_lora = parsed['lora']

    encoder_cfg = config['model']['encoder']
    classifier_cfg = config['model']['classifier']

    # Use pooling override if specified, otherwise use config default
    pooling = pooling_override if pooling_override else classifier_cfg['pooling']

    model = VesselClassifier(
        input_features=encoder_cfg['input_features'],
        d_model=encoder_cfg['d_model'],
        nhead=encoder_cfg['nhead'],
        num_layers=encoder_cfg['num_layers'],
        dim_feedforward=encoder_cfg['dim_feedforward'],
        dropout=encoder_cfg['dropout'],
        max_seq_length=encoder_cfg['max_seq_length'],
        classifier_hidden_dims=classifier_cfg['hidden_dims'],
        num_classes=classifier_cfg['num_classes'],
        classifier_dropout=classifier_cfg['dropout'],
        pooling=pooling,
    )

    # Load pretrained weights if needed
    if base_condition in ['pretrained', 'frozen_pretrained', 'two_stage']:
        checkpoint_path = config['model']['pretrained_checkpoint']
        load_pretrained_weights(model, checkpoint_path)

        # Apply LoRA if requested (before freezing for frozen conditions)
        if use_lora:
            lora_cfg = config.get('lora', {})
            rank = lora_cfg.get('rank', 4)
            alpha = lora_cfg.get('alpha', 1.0)
            apply_lora_to_model(model.encoder, rank=rank, alpha=alpha)  # Modifies encoder in-place
            lora_p, other_p, frozen_p = count_lora_params(model)
            print(f"LoRA params: {lora_p:,}, Other trainable: {other_p:,}, Frozen: {frozen_p:,}")

        # Freeze encoder if frozen_pretrained or two_stage (stage 1)
        if base_condition in ['frozen_pretrained', 'two_stage']:
            for name, param in model.encoder.named_parameters():
                # For LoRA, keep lora params trainable
                if 'lora_' not in name:
                    param.requires_grad = False
            print("Encoder weights frozen (LoRA params remain trainable)" if use_lora else "Encoder weights frozen")

    return model


def load_pretrained_weights(model: VesselClassifier, checkpoint_path: str) -> None:
    """
    Load pre-trained encoder weights from Experiment 11 checkpoint.

    The checkpoint has different structure (full trajectory model),
    so we need to map weights carefully.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained weights from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        pretrained_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_state = checkpoint['state_dict']
    else:
        pretrained_state = checkpoint

    # Get model state dict
    model_state = model.state_dict()

    # Map weights from pretrained model to classifier model
    # The pretrained model has: input_proj, pos_encoder, encoder, norm
    # Our model has: encoder.input_proj, encoder.pos_encoder, encoder.transformer, encoder.norm

    loaded_keys = []
    skipped_keys = []

    for key in pretrained_state.keys():
        # Map pretrained keys to our encoder keys
        # Checkpoint structure: input_proj, pos_encoder, transformer.layers.X
        if key.startswith('input_proj'):
            new_key = f'encoder.{key}'
        elif key.startswith('pos_encoder'):
            new_key = f'encoder.{key}'
        elif key.startswith('transformer.'):
            # pretrained: transformer.layers.X -> ours: encoder.transformer.layers.X
            new_key = f'encoder.{key}'
        elif key.startswith('norm'):
            new_key = f'encoder.{key}'
        else:
            # Skip non-encoder keys (e.g., Fourier head, horizon_proj, time_proj)
            skipped_keys.append(key)
            continue

        if new_key in model_state:
            # Check shape compatibility
            if pretrained_state[key].shape == model_state[new_key].shape:
                model_state[new_key] = pretrained_state[key]
                loaded_keys.append(f"{key} -> {new_key}")
            elif 'pos_encoder.pe' in key:
                # Handle positional encoding truncation
                target_len = model_state[new_key].shape[1]
                model_state[new_key] = pretrained_state[key][:, :target_len, :]
                loaded_keys.append(f"{key} -> {new_key} (truncated)")
            else:
                skipped_keys.append(f"{key} (shape mismatch: {pretrained_state[key].shape} vs {model_state[new_key].shape})")
        else:
            skipped_keys.append(f"{key} (not found in model)")

    # Load the mapped state dict
    model.load_state_dict(model_state)

    print(f"Loaded {len(loaded_keys)} weight tensors")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys (non-encoder weights)")


def get_optimizer(model: VesselClassifier, config: Dict, condition: str) -> torch.optim.Optimizer:
    """
    Create optimizer with differential learning rates.

    Args:
        model: VesselClassifier model
        config: Configuration dictionary
        condition: Training condition

    Returns:
        optimizer: Configured optimizer
    """
    training_cfg = config['training']
    parsed = parse_condition(condition)
    base_condition = parsed['base']
    use_lora = parsed['lora']

    if base_condition == 'random_init':
        # Single learning rate for random init
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg['random_init_lr'],
            weight_decay=training_cfg['weight_decay'],
        )
    elif base_condition in ['frozen_pretrained', 'two_stage']:
        # Get trainable params (classifier + LoRA if enabled)
        trainable_params = []
        trainable_params.extend(model.classifier.parameters())
        if use_lora:
            lora_params = get_lora_params(model)
            trainable_params.extend(lora_params)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_cfg['classifier_lr'],
            weight_decay=training_cfg['weight_decay'],
        )
    else:  # pretrained (with or without pooling/lora modifiers)
        if use_lora:
            # LoRA: freeze encoder base weights, train only LoRA + classifier
            lora_params = get_lora_params(model)
            param_groups = [
                {
                    'params': lora_params,
                    'lr': training_cfg.get('lora_lr', training_cfg['encoder_lr']),
                },
                {
                    'params': model.classifier.parameters(),
                    'lr': training_cfg['classifier_lr'],
                },
            ]
        else:
            # Standard: differential LR for encoder and classifier
            param_groups = [
                {
                    'params': model.encoder.parameters(),
                    'lr': training_cfg['encoder_lr'],
                },
                {
                    'params': model.classifier.parameters(),
                    'lr': training_cfg['classifier_lr'],
                },
            ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=training_cfg['weight_decay'],
        )

    return optimizer


def get_optimizer_stage2(model: VesselClassifier, config: Dict) -> torch.optim.Optimizer:
    """
    Create optimizer for stage 2 of two-stage fine-tuning.
    Uses differential learning rates: lower for encoder, higher for classifier.
    """
    training_cfg = config['training']

    param_groups = [
        {
            'params': model.encoder.parameters(),
            'lr': training_cfg['encoder_lr'],
        },
        {
            'params': model.classifier.parameters(),
            'lr': training_cfg['classifier_lr'],
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=training_cfg['weight_decay'],
    )

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    training_cfg = config['training']

    warmup_steps = training_cfg['warmup_epochs'] * (num_training_steps // training_cfg['max_epochs'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=1e-7,
    )

    # Wrap with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return warmup_scheduler
