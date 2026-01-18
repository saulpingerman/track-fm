"""
Training loop for vessel classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time

from ..models.classifier import VesselClassifier
from ..evaluation.metrics import compute_metrics


class Trainer:
    """Trainer for vessel classification."""

    def __init__(
        self,
        model: VesselClassifier,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        output_dir: str,
        condition: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.condition = condition

        # Training settings
        training_cfg = config['training']
        self.max_epochs = training_cfg['max_epochs']
        self.patience = training_cfg['early_stopping_patience']
        self.metric_name = training_cfg['early_stopping_metric']
        self.use_amp = training_cfg.get('use_amp', True)
        self.label_smoothing = training_cfg.get('label_smoothing', 0.1)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # AMP scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Scheduler
        self.scheduler = None

        # Tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {'train': [], 'val': []}

    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(features, lengths)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(features, lengths)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            num_classes=self.config['data']['num_classes'],
        )
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in val_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)

            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(features, lengths)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(features, lengths)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            num_classes=self.config['data']['num_classes'],
            probs=np.array(all_probs),
        )
        metrics['loss'] = avg_loss

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Full training loop with early stopping.

        Returns:
            results: Dictionary with training history and best metrics
        """
        print(f"\nTraining {self.condition} model...")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Early stopping: {self.patience} epochs on {self.metric_name}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train'].append(train_metrics)

            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history['val'].append(val_metrics)

            # Get current metric
            current_metric = val_metrics.get(self.metric_name.replace('val_', ''), 0.0)

            # Check for improvement
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': self.best_metric,
                    'val_metrics': val_metrics,
                }, checkpoint_dir / 'best_model.pt')
            else:
                self.epochs_without_improvement += 1

            # Print progress
            epoch_time = time.time() - epoch_start
            lr_str = f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            print(f"Epoch {epoch:3d}: train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_metrics['f1_macro']:.4f}, "
                  f"{lr_str}, "
                  f"time={epoch_time:.1f}s")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best {self.metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")

        # Load best model for final evaluation
        best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])

        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return {
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'total_time': total_time,
            'history': self.history,
        }
