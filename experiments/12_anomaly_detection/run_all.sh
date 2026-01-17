#!/bin/bash
#
# Experiment 12: Anomaly Detection Fine-tuning
#
# This script runs the complete anomaly detection experiment using the
# pre-trained 116M parameter encoder from experiment 11.
#
# Usage:
#   ./run_all.sh                    # Full experiment
#   ./run_all.sh --quick            # Quick test with reduced iterations
#   ./run_all.sh --condition pretrained  # Run single condition
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "Experiment 12: Anomaly Detection Fine-tuning"
echo "============================================================"
echo ""

# Parse arguments
QUICK_MODE=false
CONDITION=""
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --condition)
            CONDITION="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run quick test with reduced iterations"
            echo "  --condition NAME  Run single condition (pretrained, random_init, frozen_pretrained)"
            echo "  --skip-download   Skip data download step"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check required packages
python3 -c "import torch, pandas, numpy, sklearn, scipy, tqdm, yaml" 2>/dev/null || {
    echo -e "${RED}Error: Missing required Python packages${NC}"
    echo "Please install: torch pandas numpy scikit-learn scipy tqdm pyyaml"
    exit 1
}

# Check for pre-trained weights
PRETRAINED_WEIGHTS="../11_causal_subwindow_training/outputs/best_116m_20250115.pt"
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo -e "${YELLOW}Warning: Pre-trained weights not found at $PRETRAINED_WEIGHTS${NC}"
    echo "The experiment will still run but 'pretrained' and 'frozen_pretrained' conditions"
    echo "will train from scratch."
    echo ""
fi

# Step 1: Download data (if needed)
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo -e "${GREEN}Step 1: Checking/Downloading DTU Dataset${NC}"
    echo "------------------------------------------------------------"

    if [ -d "data/raw" ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then
        echo "Data directory exists and is not empty."
        echo "Skipping download. Use scripts/download_data.py to re-download."
    else
        echo "Running data download script..."
        python3 scripts/download_data.py --output data/raw/
    fi
else
    echo ""
    echo -e "${YELLOW}Skipping data download (--skip-download)${NC}"
fi

# Step 2: Run experiment
echo ""
echo -e "${GREEN}Step 2: Running Experiment${NC}"
echo "------------------------------------------------------------"

# Generate experiment name with timestamp
EXP_NAME="exp_$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS="--exp-name $EXP_NAME"

if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}Running in QUICK MODE (reduced iterations for testing)${NC}"
    EXTRA_ARGS="$EXTRA_ARGS --quick"
fi

if [ -n "$CONDITION" ]; then
    echo "Running single condition: $CONDITION"
    EXTRA_ARGS="$EXTRA_ARGS --condition $CONDITION"
fi

echo "Experiment name: $EXP_NAME"

# Run the experiment
python3 scripts/run_experiment.py $EXTRA_ARGS

# Step 3: Summary
echo ""
echo "============================================================"
echo -e "${GREEN}Experiment Complete!${NC}"
echo "============================================================"
echo ""
echo "Results saved to: outputs/"
echo ""
echo "Key files:"
echo "  - outputs/results_summary.csv    : Aggregated metrics"
echo "  - outputs/statistical_tests.csv  : Significance tests"
echo "  - outputs/full_results.pkl       : Complete results"
echo ""

# Print summary if available
if [ -f "outputs/results_summary.csv" ]; then
    echo "Results Summary:"
    echo "------------------------------------------------------------"
    python3 -c "
import pandas as pd
df = pd.read_csv('outputs/results_summary.csv')
print(df[['condition', 'auroc', 'auprc', 'f1_optimal']].to_string(index=False))
"
fi

echo ""
echo "Done!"
