#!/usr/bin/env python
"""
Training Monitor - Shows model predictions vs reference trajectories
Run with: watch -n 5 python scripts/monitor_training.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from train_causal_multihorizon import (
        StreamingMultiHorizonAISDataset, 
        CausalTransformerForecaster,
        latlon_to_local_uv
    )
except ImportError:
    print("❌ Could not import training modules")
    sys.exit(1)

def load_latest_checkpoint():
    """Load the most recent checkpoint"""
    ckpt_dir = Path("checkpoints/causal_multihorizon_maxbatch")
    if not ckpt_dir.exists():
        return None
    
    ckpt_files = list(ckpt_dir.glob("step_*.pt"))
    if not ckpt_files:
        return None
    
    latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split('_')[1]))
    return latest_ckpt

def create_model():
    """Create model with same config as training"""
    model = CausalTransformerForecaster(
        seq_len=20,
        d_model=128,
        nhead=8,
        num_layers=6,
        ff_hidden=512,
        fourier_m=64,
        fourier_rank=4
    )
    return model

def get_sample_data():
    """Get a few samples from the dataset"""
    try:
        dataset = StreamingMultiHorizonAISDataset(
            bucket_name="ais-pipeline-data-10179bbf-us-east-1",
            seq_len=20,
            horizon=10
        )
        
        samples = []
        for i, sample in enumerate(dataset):
            if i >= 3:  # Get 3 samples
                break
            samples.append(sample)
        
        return samples
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def format_trajectory(traj, name="", max_points=5):
    """Format trajectory for display"""
    if len(traj) > max_points:
        # Show first few and last few points
        show_first = max_points // 2
        show_last = max_points - show_first
        points_str = []
        
        for i in range(show_first):
            x, y = traj[i]
            points_str.append(f"({x:+.3f},{y:+.3f})")
        
        points_str.append("...")
        
        for i in range(-show_last, 0):
            x, y = traj[i]
            points_str.append(f"({x:+.3f},{y:+.3f})")
            
        result = " → ".join(points_str)
    else:
        points_str = []
        for x, y in traj:
            points_str.append(f"({x:+.3f},{y:+.3f})")
        result = " → ".join(points_str)
    
    return f"{name:>12}: {result}"

def monitor_training():
    """Main monitoring function"""
    print("=" * 80)
    print(f"🔍 TRAJECTORY PREDICTION MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    # Check for checkpoint
    latest_ckpt = load_latest_checkpoint()
    if latest_ckpt is None:
        print("⏳ No checkpoints found yet - training may still be starting...")
        return
    
    print(f"📁 Latest checkpoint: {latest_ckpt.name}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model().to(device)
    
    try:
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        step = checkpoint.get('global_step', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        
        print(f"📊 Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}" if isinstance(loss, float) else f"📊 Epoch: {epoch}, Step: {step}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Get sample data
    print("\n🌊 Loading sample trajectories...")
    samples = get_sample_data()
    if samples is None:
        return
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Show predictions
    print("\n🎯 MODEL PREDICTIONS vs REFERENCE:")
    print("-" * 80)
    
    with torch.no_grad():
        for i, (input_uv, centre_ll, target_uv, logJ, causal_position) in enumerate(samples):
            print(f"\n📍 Sample {i+1} (causal_pos={causal_position}):")
            
            # Prepare inputs
            input_uv = input_uv.unsqueeze(0).to(device)       # (1, 20, 2)
            centre_ll = centre_ll.unsqueeze(0).to(device)     # (1, 2)  
            target_uv_batch = target_uv.unsqueeze(0).to(device)  # (1, 10, 2)
            causal_pos_batch = torch.tensor([causal_position]).to(device)  # (1,)
            
            try:
                # Get model prediction (log probabilities)
                log_probs = model(input_uv, centre_ll, target_uv_batch, causal_pos_batch)  # (1, 10)
                
                # Convert to probabilities
                probs = torch.exp(log_probs).cpu().numpy()[0]  # (10,)
                
                # Display input trajectory (last 5 points)
                input_traj = input_uv[0].cpu().numpy()[-5:]  # Last 5 points
                print(format_trajectory(input_traj, "Input (last5)", max_points=5))
                
                # Display reference trajectory
                ref_traj = target_uv.cpu().numpy()
                print(format_trajectory(ref_traj, "Reference", max_points=5))
                
                # Display prediction quality (probabilities for reference trajectory)
                prob_str = " ".join([f"{p:.3f}" for p in probs[:5]])
                if len(probs) > 5:
                    prob_str += f" ... {probs[-1]:.3f}"
                print(f"  Pred Probs: [{prob_str}] (higher=better fit)")
                print(f"  Avg Prob: {np.mean(probs):.4f}, Min: {np.min(probs):.4f}, Max: {np.max(probs):.4f}")
                
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
    
    print("\n" + "=" * 80)
    print("📈 GPU Status:")
    
    # Try to show GPU status
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_stats = result.stdout.strip().split(', ')
            util, mem_used, mem_total, temp = gpu_stats
            print(f"🔥 GPU: {util}% util, {mem_used}MB/{mem_total}MB ({int(mem_used)/int(mem_total)*100:.1f}%), {temp}°C")
        else:
            print("⚠️  Could not get GPU status")
    except:
        print("⚠️  nvidia-smi not available")

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")