#!/bin/bash
# Experiment 13: Vessel Type Classification
# Usage: ./run_all.sh [exp_name] [--learning-curves]

set -e

cd "$(dirname "$0")"

EXP_NAME=${1:-"vessel_class_$(date +%Y%m%d_%H%M%S)"}
LEARNING_CURVES=${2:-""}

echo "=============================================="
echo "Experiment 13: Vessel Type Classification"
echo "=============================================="
echo "Experiment name: $EXP_NAME"

# Activate environment
source /opt/pytorch/bin/activate 2>/dev/null || true

# Step 1: Extract data (if not already done)
if [ ! -f "data/processed/trajectories.npy" ]; then
    echo ""
    echo "Step 1: Extracting labeled trajectories..."
    python scripts/extract_data.py --config configs/config.yaml
else
    echo ""
    echo "Step 1: Data already extracted, skipping..."
fi

# Step 2: Run experiment
echo ""
echo "Step 2: Running experiment..."

if [ "$LEARNING_CURVES" == "--learning-curves" ]; then
    echo "Running with learning curves ablation..."
    python scripts/run_experiment.py --exp-name "$EXP_NAME" --learning-curves
else
    python scripts/run_experiment.py --exp-name "$EXP_NAME"
fi

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Results saved to: experiments/$EXP_NAME"
echo "=============================================="
