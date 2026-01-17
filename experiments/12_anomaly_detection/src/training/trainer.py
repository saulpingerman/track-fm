"""
Training loop for anomaly detection fine-tuning.

Supports:
- Differential learning rates for encoder vs classifier
- Weighted BCE loss for class imbalance
- Early stopping based on validation metrics
- Cosine learning rate schedule with warmup
- Automatic mixed precision (AMP) for faster training
- torch.compile() optimization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm

from evaluation.metrics import compute_all_metrics


class Trainer:
    """
    Trainer for anomaly detection models.

    Handles training loop, validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        condition: str,
        exp_dir: Optional[Path] = None,
        fold_info: Optional[Dict] = None
    ):
        """
        Args:
            model: AnomalyDetector model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Full experiment configuration
            condition: One of "pretrained", "random_init", "frozen_pretrained"
            exp_dir: Experiment directory for saving checkpoints
            fold_info: Dictionary with fold information for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.condition = condition
        self.exp_dir = exp_dir
        self.fold_info = fold_info or {}

        self.device = config['experiment']['device']
        self.train_config = config['training']

        # Performance settings
        self.use_amp = self.train_config.get('use_amp', False) and self.device == 'cuda'
        self.use_compile = self.train_config.get('use_compile', False)

        # Apply torch.compile() for faster execution
        if self.use_compile and hasattr(torch, 'compile'):
            print("Applying torch.compile() to model...")
            self.model = torch.compile(self.model)

        # Set up AMP scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Using automatic mixed precision (AMP)")

        # Set up optimizer with differential learning rates
        self.optimizer = self._create_optimizer()

        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Set up loss function
        self.criterion = self._create_loss()

        # Training state
        self.current_epoch = 0
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with differential learning rates."""
        if self.condition == "random_init":
            # Single learning rate for random init
            lr = self.train_config['random_init_lr']
            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.train_config['weight_decay']
            )
        else:
            # Differential learning rates for pre-trained
            encoder_lr = self.train_config['encoder_lr']
            classifier_lr = self.train_config['classifier_lr']

            param_groups = [
                {
                    'params': self.model.get_encoder_params(),
                    'lr': encoder_lr,
                    'name': 'encoder'
                },
                {
                    'params': self.model.get_classifier_params(),
                    'lr': classifier_lr,
                    'name': 'classifier'
                }
            ]

            optimizer = AdamW(
                param_groups,
                weight_decay=self.train_config['weight_decay']
            )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_epochs = self.train_config['warmup_epochs']
        max_epochs = self.train_config['max_epochs']

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Main scheduler (cosine decay)
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-7
        )

        # Combine warmup and main scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )

        return scheduler

    def _create_loss(self) -> nn.Module:
        """Create weighted BCE loss for class imbalance."""
        pos_weight = torch.tensor([self.train_config['positive_weight']]).to(self.device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_epoch(self) -> float:
        """Run one training epoch with optional AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", leave=False):
            features = batch['features'].to(self.device, non_blocking=True)
            lengths = batch['length'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(features, lengths)
                    loss = self.criterion(logits.squeeze(), labels)

                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                logits = self.model(features, lengths)
                loss = self.criterion(logits.squeeze(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Dict:
        """Run validation and compute metrics with optional AMP."""
        self.model.eval()
        loader = loader or self.val_loader

        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            features = batch['features'].to(self.device, non_blocking=True)
            lengths = batch['length'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            # Forward pass with AMP
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(features, lengths)
                    loss = self.criterion(logits.squeeze(), labels)
            else:
                logits = self.model(features, lengths)
                loss = self.criterion(logits.squeeze(), labels)

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        # Aggregate predictions
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid

        # Compute metrics
        metrics = compute_all_metrics(all_labels, all_probs.squeeze())
        metrics['loss'] = total_loss / num_batches

        return metrics

    def fit(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        max_epochs = self.train_config['max_epochs']
        patience = self.train_config['early_stopping_patience']
        metric_name = self.train_config['early_stopping_metric']

        print(f"\nTraining {self.condition} model for up to {max_epochs} epochs")
        print(f"Early stopping: patience={patience}, metric={metric_name}")

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            # Training
            train_loss = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(
                [pg['lr'] for pg in self.optimizer.param_groups]
            )

            # Early stopping check
            current_metric = val_metrics.get(metric_name, val_metrics.get('auprc', 0))

            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.patience_counter = 0

                # Save best model
                if self.exp_dir:
                    self._save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1

            # Logging
            lr_str = f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"{metric_name}={current_metric:.4f}, {lr_str}")

            # Check early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of test metrics
        """
        # Load best model if available
        if self.exp_dir:
            best_path = self.exp_dir / 'checkpoints' / 'best_model.pt'
            if best_path.exists():
                self._load_checkpoint('best_model.pt')

        return self.validate(test_loader)

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.exp_dir is None:
            return

        checkpoint_dir = self.exp_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_val_metric': self.best_val_metric,
            'condition': self.condition,
            'fold_info': self.fold_info
        }

        torch.save(checkpoint, checkpoint_dir / filename)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.exp_dir / 'checkpoints' / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length sequences.

    Pads sequences to max length in batch.
    """
    features = torch.stack([item['features'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'features': features,
        'length': lengths,
        'label': labels
    }
