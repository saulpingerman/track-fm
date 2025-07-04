#!/bin/bash
# Simple script to start TrackFM training

echo "Starting TrackFM Multi-Horizon Trajectory Forecasting Training..."
echo "================================================================="

# Activate virtual environment
source .venv/bin/activate

# Run training with logging
python train.py > training.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Logs are being written to: training.log"
echo ""
echo "To monitor progress:"
echo "  tail -f training.log"
echo ""
echo "To stop training:"
echo "  kill $PID"