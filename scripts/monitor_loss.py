#!/usr/bin/env python
"""
Loss Monitor - Shows training progress and loss curves
Run with: watch -n 2 python scripts/monitor_loss.py
"""

import re
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess

def get_gpu_stats():
    """Get current GPU utilization"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=3)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            util, mem_used, mem_total, temp, power = stats
            return {
                'util': int(util),
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temp': int(temp),
                'power': float(power) if power != '[N/A]' else 0
            }
    except:
        pass
    return None

def parse_training_log():
    """Parse the training log for loss values and timing"""
    log_file = Path("training_optimized.log")
    
    if not log_file.exists():
        return None, None, None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except:
        return None, None, None
    
    # Extract batch information
    # Pattern: "Batch X: Total=Y.YYYs, GPT=Z.ZZZs, Fourier=W.WWWs, Loss=V.VVVs, Backward=U.UUUs, LossVal=T.TTTT"
    batch_pattern = r'Batch (\d+): Total=([\d.]+)s, GPT=([\d.]+)s, Fourier=([\d.]+)s, Loss=([\d.]+)s, Backward=([\d.]+)s, LossVal=([-\d.]+)'
    
    matches = re.findall(batch_pattern, content)
    
    if not matches:
        return None, None, None
    
    batches = []
    losses = []
    timings = []
    
    for match in matches:
        batch_num, total_time, gpt_time, fourier_time, loss_time, backward_time, loss_val = match
        
        batches.append(int(batch_num))
        losses.append(float(loss_val))
        
        timings.append({
            'total': float(total_time),
            'gpt': float(gpt_time),
            'fourier': float(fourier_time),
            'loss': float(loss_time),
            'backward': float(backward_time)
        })
    
    return batches, losses, timings

def format_timing_stats(timings):
    """Format timing statistics"""
    if not timings:
        return "No timing data"
    
    recent_timings = timings[-10:]  # Last 10 batches
    
    avg_total = np.mean([t['total'] for t in recent_timings])
    avg_gpt = np.mean([t['gpt'] for t in recent_timings])
    avg_fourier = np.mean([t['fourier'] for t in recent_timings])
    avg_backward = np.mean([t['backward'] for t in recent_timings])
    
    gpt_pct = (avg_gpt / avg_total) * 100
    fourier_pct = (avg_fourier / avg_total) * 100
    backward_pct = (avg_backward / avg_total) * 100
    
    return f"""
  📊 Timing (avg last 10 batches):
     Total: {avg_total:.3f}s/batch
     GPT: {avg_gpt:.3f}s ({gpt_pct:.1f}%)
     Fourier: {avg_fourier:.3f}s ({fourier_pct:.1f}%)  
     Backward: {avg_backward:.3f}s ({backward_pct:.1f}%)"""

def create_loss_sparkline(losses, width=50):
    """Create a simple ASCII sparkline for losses"""
    if len(losses) < 2:
        return "Insufficient data"
    
    # Use recent losses
    recent_losses = losses[-width:] if len(losses) > width else losses
    
    min_loss = min(recent_losses)
    max_loss = max(recent_losses)
    
    if max_loss == min_loss:
        return "▁" * len(recent_losses) + f" (constant: {min_loss:.3f})"
    
    # Normalize to 0-7 range for sparkline characters
    chars = "▁▂▃▄▅▆▇█"
    normalized = [(l - min_loss) / (max_loss - min_loss) * 7 for l in recent_losses]
    sparkline = "".join(chars[int(n)] for n in normalized)
    
    return f"{sparkline} [{min_loss:.3f} to {max_loss:.3f}]"

def monitor_loss():
    """Main monitoring function"""
    print("=" * 80)
    print(f"📈 TRAINING LOSS MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    # Parse training log
    batches, losses, timings = parse_training_log()
    
    if batches is None:
        print("⏳ No training data found yet...")
        print("   Make sure training_optimized.log exists and contains batch data")
        return
    
    # Current status
    current_batch = batches[-1]
    current_loss = losses[-1]
    total_batches = len(batches)
    
    print(f"🚀 Current Batch: {current_batch}")
    print(f"📉 Current Loss: {current_loss:.4f}")
    print(f"📊 Total Batches Processed: {total_batches}")
    
    # Loss statistics
    if len(losses) >= 10:
        recent_losses = losses[-10:]
        avg_loss = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        
        print(f"📈 Recent Loss (last 10): {avg_loss:.4f} ± {loss_std:.4f}")
        
        # Loss trend
        if len(losses) >= 20:
            old_avg = np.mean(losses[-20:-10])
            new_avg = np.mean(losses[-10:])
            trend = "📉 improving" if new_avg < old_avg else "📈 increasing" if new_avg > old_avg else "➡️  stable"
            change = abs(new_avg - old_avg)
            print(f"📊 Trend: {trend} (change: {change:.4f})")
    
    # Loss curve (sparkline)
    print(f"\n📈 Loss Curve (last {min(50, len(losses))} batches):")
    print(f"   {create_loss_sparkline(losses)}")
    
    # Training speed
    if timings:
        recent_timings = timings[-5:]
        avg_time = np.mean([t['total'] for t in recent_timings])
        batches_per_min = 60 / avg_time
        samples_per_sec = 2560 / avg_time  # batch_size = 2560
        
        print(f"\n⚡ Training Speed:")
        print(f"   {avg_time:.3f}s/batch")
        print(f"   {batches_per_min:.1f} batches/min")
        print(f"   {samples_per_sec:.0f} samples/sec")
    
    # Timing breakdown
    if timings:
        print(format_timing_stats(timings))
    
    # GPU Status
    gpu_stats = get_gpu_stats()
    if gpu_stats:
        print(f"\n🔥 GPU Status:")
        print(f"   Utilization: {gpu_stats['util']}%")
        print(f"   Memory: {gpu_stats['mem_used']}MB / {gpu_stats['mem_total']}MB ({gpu_stats['mem_used']/gpu_stats['mem_total']*100:.1f}%)")
        print(f"   Temperature: {gpu_stats['temp']}°C")
        if gpu_stats['power'] > 0:
            print(f"   Power: {gpu_stats['power']:.1f}W")
    
    # Check if training is still running
    try:
        result = subprocess.run(['pgrep', '-f', 'train_causal_multihorizon.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"\n✅ Training is running (PID: {', '.join(pids)})")
        else:
            print(f"\n⚠️  Training process not detected")
    except:
        print(f"\n❓ Could not check training process status")
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        monitor_loss()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
    except Exception as e:
        print(f"❌ Error: {e}")