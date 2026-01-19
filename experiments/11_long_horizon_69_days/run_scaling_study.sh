#!/bin/bash
# Run scaling study: tiny (200k), small (1M), medium (5M) parameter models
# All run sequentially to avoid GPU memory conflicts

set -e

cd /home/ec2-user/trackfm/experiments/11_long_horizon_69_days
source /opt/pytorch/bin/activate

echo "=========================================="
echo "SCALING STUDY: 200K, 1M, 5M PARAMETERS"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Tiny model (~194k params): d_model=64, nhead=4, layers=2, dim_ff=256
echo "=========================================="
echo "1/3: Training TINY model (~194K params)"
echo "=========================================="
python run_experiment.py \
    --exp-name 69days_causal_tiny_194k \
    --d-model 64 \
    --nhead 4 \
    --num-layers 2 \
    --dim-feedforward 256 \
    --batch-size 0 \
    --target-gpu-pct 0.50 \
    --num-epochs 100 \
    --num-horizons 4 \
    --early-stop-patience 4

echo ""
echo "Tiny model complete: $(date)"
echo ""

# Small model (~1M params): d_model=128, nhead=8, layers=4, dim_ff=512
echo "=========================================="
echo "2/3: Training SMALL model (~1M params)"
echo "=========================================="
python run_experiment.py \
    --exp-name 69days_causal_small_1M \
    --d-model 128 \
    --nhead 8 \
    --num-layers 4 \
    --dim-feedforward 512 \
    --batch-size 0 \
    --target-gpu-pct 0.50 \
    --num-epochs 100 \
    --num-horizons 4 \
    --early-stop-patience 4

echo ""
echo "Small model complete: $(date)"
echo ""

# Medium model (~5.3M params): d_model=256, nhead=8, layers=6, dim_ff=1024
echo "=========================================="
echo "3/3: Training MEDIUM model (~5.3M params)"
echo "=========================================="
python run_experiment.py \
    --exp-name 69days_causal_medium_5M \
    --d-model 256 \
    --nhead 8 \
    --num-layers 6 \
    --dim-feedforward 1024 \
    --batch-size 0 \
    --target-gpu-pct 0.50 \
    --num-epochs 100 \
    --num-horizons 4 \
    --early-stop-patience 4

echo ""
echo "=========================================="
echo "SCALING STUDY COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - experiments/69days_causal_tiny_194k/   (~194K params)"
echo "  - experiments/69days_causal_small_1M/    (~1M params)"
echo "  - experiments/69days_causal_medium_5M/   (~5.3M params)"
