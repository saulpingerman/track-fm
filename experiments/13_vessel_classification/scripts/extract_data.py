#!/usr/bin/env python3
"""
Extract labeled trajectories from DMA data for vessel classification.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import argparse
import numpy as np

from src.data.extract_trajectories import (
    extract_trajectories_with_labels,
    create_splits,
    save_processed_data,
)


def main():
    parser = argparse.ArgumentParser(description='Extract labeled trajectories')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']

    # Set random seed
    np.random.seed(config['experiment']['seed'])

    print("="*60)
    print("Extracting labeled trajectories for vessel classification")
    print("="*60)

    # Extract trajectories
    trajectories, labels, track_ids, stats = extract_trajectories_with_labels(
        data_path=data_cfg['raw_path'],
        year=data_cfg['year'],
        month=data_cfg['month'],
        classes=data_cfg['classes'],
        min_length=data_cfg['min_trajectory_length'],
        max_length=data_cfg['max_trajectory_length'],
        max_trajectories_per_class=data_cfg['max_trajectories_per_class'],
    )

    # Create splits
    splits = create_splits(
        trajectories=trajectories,
        labels=labels,
        track_ids=track_ids,
        train_ratio=data_cfg['train_ratio'],
        val_ratio=data_cfg['val_ratio'],
        seed=config['experiment']['seed'],
    )

    # Save processed data
    output_dir = Path(__file__).parent.parent / data_cfg['processed_path']
    save_processed_data(
        trajectories=trajectories,
        labels=labels,
        track_ids=track_ids,
        splits=splits,
        stats=stats,
        output_dir=output_dir,
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
