#!/usr/bin/env bash
# Scaling study: Small -> XLarge on the full materialized dataset.
# Each run logs loss/DR-ratio/MFU to MLflow experiment trackfm/scaling.
# Usage: scripts/run_scaling_study.sh [scales...]   (default: all four)
set -euo pipefail
cd "$(dirname "$0")/.."

SCALES=("${@:-small medium large xlarge}")
for scale in ${SCALES[@]}; do
    echo "=== scaling study: $scale ==="
    uv run trackfm pretrain --config "configs/pretrain/${scale}.yaml"
done
