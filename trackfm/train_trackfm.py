#!/usr/bin/env python
"""
Train the horizon-aware multi-horizon trajectory forecasting model
This fixes the fundamental architectural flaw by including horizon information
"""

import argparse
import math
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, List, Tuple

import boto3
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# Import components
from trackfm.four_head_2D_LR import FourierHead2DLR
from trackfm.nano_gpt_trajectory import TrajectoryGPT, TrajectoryGPTConfig

# Import the horizon-aware model and dataset
from trackfm.trackfm_model import HorizonAwareTrajectoryForecaster
from trackfm.trackfm_dataset import StreamingMultiHorizonAISDataset
from trackfm.config import load_config, TrackFMConfig

def find_latest_valid_checkpoint(ckpt_dir, model, device):
    """Find the latest valid checkpoint that loads without errors"""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None, 0, 0
    
    # Get all checkpoint files
    checkpoint_files = sorted(ckpt_path.glob("step_*.pt"), reverse=True)
    
    for ckpt_file in checkpoint_files:
        try:
            print(f"🔍 Trying to load checkpoint: {ckpt_file}")
            checkpoint = torch.load(ckpt_file, map_location=device)
            
            # Verify checkpoint integrity
            if 'model_state_dict' not in checkpoint:
                print(f"   ❌ Missing model_state_dict")
                continue
                
            # Try loading the model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Check for any nan/inf in model parameters
            has_bad_params = False
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"   ❌ Non-finite parameters in {name}")
                    has_bad_params = True
                    break
            
            if has_bad_params:
                continue
            
            epoch = checkpoint.get('epoch', 1)
            global_step = checkpoint.get('global_step', 0)
            
            print(f"   ✅ Valid checkpoint found: epoch={epoch}, step={global_step}")
            return checkpoint, epoch, global_step
            
        except Exception as e:
            print(f"   ❌ Error loading {ckpt_file}: {e}")
            continue
    
    print("⚠️  No valid checkpoints found")
    return None, 1, 0

def train_trackfm(config: TrackFMConfig):
    """Train TrackFM model with configuration"""
    
    # Determine device
    if config.system.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.system.device
    
    print(f"🚀 Training TrackFM model on {device}")
    print(f"📊 Config: {config.paths.model_tag}")
    
    # Create streaming dataset
    dataset = StreamingMultiHorizonAISDataset(
        bucket_name=config.data.bucket_name, 
        seq_len=config.model.seq_len, 
        horizon=config.model.horizon,
        chunk_size=config.data.chunk_size,
        prefix=config.data.prefix
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # Dataset handles shuffling
        drop_last=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers
    )
    
    # Create horizon-aware model
    model = HorizonAwareTrajectoryForecaster(
        seq_len=config.model.seq_len,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        ff_hidden=config.model.ff_hidden,
        fourier_m=config.model.fourier_m,
        fourier_rank=config.model.fourier_rank,
        max_horizon=config.model.max_horizon,
        dropout=config.model.dropout
    ).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optionally warm-start from checkpoint
    start_epoch = 1
    global_step = 0
    
    if config.paths.warmup_checkpoint:
        print(f"🔥 Warm-starting from checkpoint: {config.paths.warmup_checkpoint}")
        checkpoint = torch.load(config.paths.warmup_checkpoint, map_location=device)
        
        # Load compatible weights (GPT backbone and output projection)
        model_dict = model.state_dict()
        checkpoint_dict = checkpoint['model_state_dict']
        
        compatible_dict = {}
        for k, v in checkpoint_dict.items():
            if k.startswith('gpt.') or k.startswith('output_proj.'):
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                    print(f"  Loaded: {k}")
        
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(compatible_dict)} compatible parameters")
        print(f"  Fourier head will be trained from scratch with horizon awareness")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.lr, 
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    
    # Create checkpoint directory
    ckpt_path = Path(config.paths.ckpt_dir) / config.paths.model_tag
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Checkpoints will be saved to: {ckpt_path}")
    
    # Check for existing TrackFM checkpoints to resume from
    print("\n🔍 Checking for existing TrackFM checkpoints...")
    recovery_checkpoint, start_epoch, global_step = find_latest_valid_checkpoint(
        ckpt_path, model, device
    )
    
    if recovery_checkpoint is not None:
        print(f"📤 Resuming from TrackFM checkpoint at step {global_step}")
        # Load optimizer and scheduler state if available
        if 'optimizer_state_dict' in recovery_checkpoint:
            optimizer.load_state_dict(recovery_checkpoint['optimizer_state_dict'])
            print("   Loaded optimizer state")
        if 'scheduler_state_dict' in recovery_checkpoint:
            scheduler.load_state_dict(recovery_checkpoint['scheduler_state_dict'])
            print("   Loaded scheduler state")
    else:
        print("🔄 No valid TrackFM checkpoints found, starting fresh")
    
    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start = time.time()
        
        print(f"🚀 Starting epoch {epoch}/{config.training.epochs}")
        
        for batch_idx, (input_uv, centre_ll, target_uv, logJ, causal_position) in enumerate(dataloader):
            batch_start_time = time.time()
            global_step += 1
            
            if batch_idx == 0:
                print(f"📦 Processing first batch of epoch {epoch}")
                print(f"  Input shape: {input_uv.shape}")
                print(f"  Target shape: {target_uv.shape}")
            
            # Move to device
            data_move_start = time.time()
            input_uv = input_uv.to(device)
            centre_ll = centre_ll.to(device)
            target_uv = target_uv.to(device)
            logJ = logJ.to(device)
            causal_position = causal_position.to(device)
            data_move_time = time.time() - data_move_start
            
            # Forward pass
            forward_start = time.time()
            log_probs = model(input_uv, centre_ll, target_uv, causal_position)
            forward_time = time.time() - forward_start
            
            # Loss calculation with safeguards
            loss_start = time.time()
            loss = -(log_probs + logJ.unsqueeze(1)).mean()
            
            # Safeguard against inf/nan losses
            if not torch.isfinite(loss):
                print(f"⚠️  WARNING: Non-finite loss detected at batch {batch_idx}")
                print(f"   Loss value: {loss.item()}")
                print(f"   log_probs stats: min={log_probs.min().item():.4f}, max={log_probs.max().item():.4f}")
                print(f"   logJ stats: min={logJ.min().item():.4f}, max={logJ.max().item():.4f}")
                print("   Skipping this batch and continuing...")
                continue
            
            loss_time = time.time() - loss_start
            
            # Backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            
            # Check for nan gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"⚠️  WARNING: Non-finite gradients in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"   Skipping gradient update at batch {batch_idx}")
                continue
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.training.grad_clip_norm
            )
            
            # Check gradient norm
            if not torch.isfinite(grad_norm):
                print(f"⚠️  WARNING: Non-finite gradient norm at batch {batch_idx}")
                print("   Skipping gradient update...")
                continue
            
            optimizer.step()
            backward_time = time.time() - backward_start
            
            # Update metrics (only for valid losses)
            total_loss += loss.item() * input_uv.size(0)
            total_samples += input_uv.size(0)
            
            batch_total_time = time.time() - batch_start_time
            
            # Print detailed timing every N batches
            if batch_idx % config.logging.log_every == 0:
                print(f"  Batch {batch_idx}: Total={batch_total_time:.3f}s, "
                      f"Forward={forward_time:.3f}s, Loss={loss_time:.3f}s, "
                      f"Backward={backward_time:.3f}s, LossVal={loss.item():.4f}, "
                      f"GradNorm={grad_norm:.4f}")
            
            # Checkpoint - save every N batches
            if global_step % config.training.ckpt_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'architecture': 'trackfm',  # Mark the architecture type
                    'horizon_aware': True,
                    'config': config.to_dict(),  # Save config with checkpoint
                }
                ckpt_file = ckpt_path / f"step_{global_step:07d}.pt"
                torch.save(checkpoint, ckpt_file)
                print(f"💾 TrackFM checkpoint saved: {ckpt_file}")
        
        # End of epoch
        scheduler.step()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_time = time.time() - epoch_start
        
        print(f"🟢 Epoch {epoch}/{config.training.epochs} | Loss: {avg_loss:.4f} | "
              f"Time: {epoch_time/60:.1f}min | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ TrackFM training complete!")
    return model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train TrackFM Multi-Horizon Trajectory Forecaster")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--override",
        nargs="*",
        help="Override config values using dot notation (e.g., training.lr=2e-4 model.fourier_m=256)"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (creates config/experiments/{name}.yaml)"
    )
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            if "=" in override:
                key, value = override.split("=", 1)
                # Try to convert to appropriate type
                try:
                    # Try float first (handles scientific notation like 1e-4)
                    value = float(value)
                    # If it's a whole number, convert to int
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    # Keep as string otherwise
                overrides[key] = value
    
    # Load configuration
    print("=" * 60)
    print("TRAINING TRACKFM MULTI-HORIZON TRAJECTORY FORECASTER")
    print("=" * 60)
    print("This model includes horizon information to fix the architectural flaw")
    print("where all prediction horizons produced identical PDFs.")
    print()
    
    print(f"📋 Loading config from: {args.config}")
    config = load_config(args.config, overrides)
    
    if overrides:
        print(f"🔧 Applied overrides: {overrides}")
    
    if args.experiment:
        # Save experiment config
        from trackfm.config import create_experiment_config
        experiment_config_path = create_experiment_config(
            args.config, args.experiment, overrides
        )
        print(f"💾 Saved experiment config: {experiment_config_path}")
    
    # Train model
    model = train_trackfm(config)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("The new model should now produce different PDFs for different horizons!")
    print("Use the visualization scripts to verify the fix worked.")

if __name__ == "__main__":
    main()