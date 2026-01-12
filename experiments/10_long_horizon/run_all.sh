#!/bin/bash
# Run training and visualization pipeline for Experiment 10
# Usage: ./run_all.sh <exp-name> [num-epochs] [max-horizon-video]
#
# Examples:
#   ./run_all.sh my_experiment          # 5 epochs, 400 horizon video
#   ./run_all.sh my_experiment 10       # 10 epochs, 400 horizon video
#   ./run_all.sh my_experiment 5 100    # 5 epochs, 100 horizon video

set -e  # Exit on error

# Parse arguments
EXP_NAME=${1:?"Usage: $0 <exp-name> [num-epochs] [max-horizon-video]"}
NUM_EPOCHS=${2:-5}
MAX_HORIZON_VIDEO=${3:-400}
NUM_TRACKS=${4:-6}

# Activate environment
source /opt/pytorch/bin/activate

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "EXPERIMENT 10: LONG HORIZON PREDICTION PIPELINE"
echo "========================================================================"
echo "Experiment name: $EXP_NAME"
echo "Epochs: $NUM_EPOCHS"
echo "Max horizon for video: $MAX_HORIZON_VIDEO"
echo "Number of tracks: $NUM_TRACKS"
echo "========================================================================"

# Step 1: Training
echo ""
echo "[1/3] Running training..."
echo "========================================================================"
python3 run_experiment.py --exp-name "$EXP_NAME" --num-epochs "$NUM_EPOCHS"

# Step 2: Training history plot
echo ""
echo "[2/3] Generating training history plot..."
echo "========================================================================"
python3 visualize_predictions.py --exp-name "$EXP_NAME"

# Step 3: Horizon videos (GIFs)
echo ""
echo "[3/3] Generating horizon videos..."
echo "========================================================================"
python3 make_horizon_videos.py --exp-name "$EXP_NAME" --max-horizon "$MAX_HORIZON_VIDEO" --num-tracks "$NUM_TRACKS"

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo "Results saved to: experiments/$EXP_NAME/"
echo "  - checkpoints/"
echo "  - results/"
echo "    - training.log"
echo "    - training_history.png"
echo "    - horizon_video_track*.gif"
echo "========================================================================"
