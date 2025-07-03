#!/usr/bin/env python
"""
Find a sample with significant 8-step displacement for better visualization
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Import training components
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train_causal_multihorizon import StreamingMultiHorizonAISDataset

def find_fast_sample(dataset, min_displacement=0.05, max_samples=50):
    """Find a sample with significant 8-step displacement"""
    data_iter = iter(dataset)
    samples_checked = 0
    
    print("Searching for samples with significant movement...")
    
    for i in range(max_samples):
        try:
            input_uv, centre_ll, target_uv, logJ, causal_position = next(data_iter)
            samples_checked += 1
            
            if causal_position == 19 and len(target_uv) >= 8:
                # Calculate 8-step displacement
                input_np = input_uv.numpy()
                target_np = target_uv.numpy()
                
                displacement = target_np[7] - input_np[-1]
                distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
                
                print(f"Sample {samples_checked}: 8-step displacement = {distance:.4f}")
                
                if distance >= min_displacement:
                    print(f"Found good sample! Displacement: {distance:.4f}")
                    return {
                        'input_uv': input_uv,
                        'centre_ll': centre_ll,
                        'target_uv': target_uv,
                        'logJ': logJ,
                        'causal_position': torch.tensor(causal_position),
                        'displacement': distance
                    }
                    
        except StopIteration:
            break
    
    print(f"Checked {samples_checked} samples, none with displacement >= {min_displacement}")
    return None

def main():
    # Create dataset
    print("Creating dataset...")
    import os
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    dataset = StreamingMultiHorizonAISDataset(
        bucket_name="ais-pipeline-data-10179bbf-us-east-1",
        seq_len=20,
        horizon=10
    )
    
    # Find fast-moving sample
    sample = find_fast_sample(dataset, min_displacement=0.05, max_samples=200)
    
    if sample:
        print(f"\nFound sample with 8-step displacement: {sample['displacement']:.4f}")
        
        # Print some stats
        input_np = sample['input_uv'].numpy()
        target_np = sample['target_uv'].numpy()
        
        displacements = []
        for i in range(len(target_np)):
            disp = target_np[i] - input_np[-1]
            dist = np.sqrt(disp[0]**2 + disp[1]**2)
            displacements.append(dist)
            print(f"  {i+1}-step displacement: {dist:.4f}")
        
        print(f"\nThis sample shows good separation for visualization!")
        print(f"Location: ({sample['centre_ll'][0].item() * 90:.2f}°, {sample['centre_ll'][1].item() * 180:.2f}°)")
        
    else:
        print("\nNo samples found with significant movement. AIS data might contain mostly stationary or slow-moving vessels.")

if __name__ == "__main__":
    main()