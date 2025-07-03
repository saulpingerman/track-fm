#!/usr/bin/env python
"""Show summary of training progress"""

import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("/home/ec2-user/repos/track-fm/checkpoints/causal_multihorizon_128f/step_0020000.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=== Training Summary ===")
print(f"Checkpoint: {checkpoint_path.name}")
print(f"Global Step: {checkpoint['global_step']:,}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Last Loss: {checkpoint['loss']:.4f}")
print(f"Learning Rate: {checkpoint['scheduler_state_dict']['_last_lr'][0]:.2e}")

# Calculate approximate samples seen (batch_size * steps)
batch_size = 1024  # from training config
samples_seen = checkpoint['global_step'] * batch_size
print(f"Approximate samples processed: {samples_seen:,}")