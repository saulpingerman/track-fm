#!/bin/bash
# =============================================================================
# Create download bundles for collaborators
# Run from the trackfm/ root directory
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_DIR="/home/ec2-user/trackfm_bundles"
mkdir -p "$BUNDLE_DIR"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Creating TrackFM collaborator bundles"
echo "============================================"

# -----------------------------------------------
# Bundle 1: Essential (checkpoint + finetuning data + configs)
# Expected size: ~500MB
# -----------------------------------------------
echo ""
echo "[1/3] Creating essential bundle (model + data + configs)..."

ESSENTIAL_DIR="$BUNDLE_DIR/trackfm_essential"
rm -rf "$ESSENTIAL_DIR"
mkdir -p "$ESSENTIAL_DIR/checkpoints"
mkdir -p "$ESSENTIAL_DIR/data/anomaly_detection/raw"
mkdir -p "$ESSENTIAL_DIR/data/anomaly_detection/processed"
mkdir -p "$ESSENTIAL_DIR/data/anomaly_detection/splits"
mkdir -p "$ESSENTIAL_DIR/data/vessel_classification"
mkdir -p "$ESSENTIAL_DIR/configs"

# Pre-trained backbone (the critical file)
echo "  Copying Exp 11 XLarge checkpoint (478MB)..."
cp experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/checkpoints/best_model.pt \
   "$ESSENTIAL_DIR/checkpoints/exp11_xlarge_116M_best_model.pt"

# Finetuning datasets
echo "  Copying finetuning datasets..."
cp experiments/12_anomaly_detection/data/raw/*.pkl "$ESSENTIAL_DIR/data/anomaly_detection/raw/"
cp experiments/12_anomaly_detection/data/processed/* "$ESSENTIAL_DIR/data/anomaly_detection/processed/"
cp experiments/12_anomaly_detection/data/splits/* "$ESSENTIAL_DIR/data/anomaly_detection/splits/"
cp experiments/13_vessel_classification/data/processed/* "$ESSENTIAL_DIR/data/vessel_classification/"

# All config files
echo "  Copying configs..."
cp experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/config.json \
   "$ESSENTIAL_DIR/configs/exp11_pretraining_config.json"
cp experiments/12_anomaly_detection/configs/config.yaml \
   "$ESSENTIAL_DIR/configs/exp12_anomaly_detection_config.yaml"
cp experiments/13_vessel_classification/configs/config.yaml \
   "$ESSENTIAL_DIR/configs/exp13_vessel_classification_config.yaml"

# Bayesian optimization results
echo "  Copying Bayesian optimization results..."
mkdir -p "$ESSENTIAL_DIR/results/exp12_bayesian_opt"
mkdir -p "$ESSENTIAL_DIR/results/exp13_bayesian_opt"
cp experiments/12_anomaly_detection/experiments/bayesian_opt_v1/comparison.json \
   "$ESSENTIAL_DIR/results/exp12_bayesian_opt/"
cp experiments/12_anomaly_detection/experiments/bayesian_opt_v1/pretrained_optimization.json \
   "$ESSENTIAL_DIR/results/exp12_bayesian_opt/"
cp experiments/12_anomaly_detection/experiments/bayesian_opt_v1/random_init_optimization.json \
   "$ESSENTIAL_DIR/results/exp12_bayesian_opt/"
cp experiments/13_vessel_classification/experiments/bayesian_opt_v1/comparison.json \
   "$ESSENTIAL_DIR/results/exp13_bayesian_opt/"
cp experiments/13_vessel_classification/experiments/bayesian_opt_v1/pretrained_optimization.json \
   "$ESSENTIAL_DIR/results/exp13_bayesian_opt/"
cp experiments/13_vessel_classification/experiments/bayesian_opt_v1/random_init_optimization.json \
   "$ESSENTIAL_DIR/results/exp13_bayesian_opt/"

# Key docs
cp HYPERPARAMETERS.md "$ESSENTIAL_DIR/"
cp MATERIALIZED_DATA.md "$ESSENTIAL_DIR/"
cp TRAINING_HANDOFF.md "$ESSENTIAL_DIR/"

# Create the zip
echo "  Creating zip..."
cd "$BUNDLE_DIR"
zip -r trackfm_essential.zip trackfm_essential/
cd "$SCRIPT_DIR"
echo "  Done: $BUNDLE_DIR/trackfm_essential.zip ($(du -sh trackfm_essential.zip | cut -f1))"

# -----------------------------------------------
# Bundle 2: Experiment 14 checkpoints
# Expected size: ~640MB
# -----------------------------------------------
echo ""
echo "[2/3] Creating Exp 14 checkpoints bundle..."

EXP14_DIR="$BUNDLE_DIR/trackfm_exp14_checkpoints"
rm -rf "$EXP14_DIR"
mkdir -p "$EXP14_DIR"

if [ -f experiments/14_800_horizon_1_year/experiments/run_18M_h800/checkpoints/best_model.pt ]; then
    cp experiments/14_800_horizon_1_year/experiments/run_18M_h800/checkpoints/best_model.pt \
       "$EXP14_DIR/exp14_18M_h800_best_model.pt"
    cp experiments/14_800_horizon_1_year/experiments/run_18M_h800/config.json \
       "$EXP14_DIR/exp14_18M_h800_config.json"
fi

if [ -f experiments/14_800_horizon_1_year/experiments/run_100M_h800/checkpoints/best_model.pt ]; then
    cp experiments/14_800_horizon_1_year/experiments/run_100M_h800/checkpoints/best_model.pt \
       "$EXP14_DIR/exp14_100M_h800_best_model.pt"
    cp experiments/14_800_horizon_1_year/experiments/run_100M_h800/config.json \
       "$EXP14_DIR/exp14_100M_h800_config.json"
fi

cd "$BUNDLE_DIR"
zip -r trackfm_exp14_checkpoints.zip trackfm_exp14_checkpoints/
cd "$SCRIPT_DIR"
echo "  Done: $BUNDLE_DIR/trackfm_exp14_checkpoints.zip ($(du -sh trackfm_exp14_checkpoints.zip | cut -f1))"

# -----------------------------------------------
# Bundle 3: Videos and visualizations
# Expected size: ~350MB
# -----------------------------------------------
echo ""
echo "[3/3] Creating videos bundle..."

VIDEOS_DIR="$BUNDLE_DIR/trackfm_videos"
rm -rf "$VIDEOS_DIR"
mkdir -p "$VIDEOS_DIR/exp11_predictions"
mkdir -p "$VIDEOS_DIR/exp11_interesting"
mkdir -p "$VIDEOS_DIR/exp12_results"
mkdir -p "$VIDEOS_DIR/exp14_results"

# Exp 11 prediction videos
for f in experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/results/horizon_video_track*.gif; do
    [ -f "$f" ] && cp "$f" "$VIDEOS_DIR/exp11_predictions/"
done

# Exp 11 interesting tracks
for f in experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/results/interesting_track*.gif; do
    [ -f "$f" ] && cp "$f" "$VIDEOS_DIR/exp11_interesting/"
done

# Exp 11 leak test
[ -f experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/results/leak_test_input_only.gif ] && \
    cp experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/results/leak_test_input_only.gif "$VIDEOS_DIR/exp11_predictions/"

# Key images
mkdir -p "$VIDEOS_DIR/figures"
cp experiments/11_long_horizon_69_days/experiments/69days_causal_v4_100M/results/training_history.png "$VIDEOS_DIR/figures/" 2>/dev/null || true
cp experiments/12_anomaly_detection/experiments/bayesian_opt_v1/*.png "$VIDEOS_DIR/figures/" 2>/dev/null || true
cp experiments/14_800_horizon_1_year/results/*.png "$VIDEOS_DIR/figures/" 2>/dev/null || true

# Exp 14 videos
for f in experiments/14_800_horizon_1_year/results/*.gif experiments/14_800_horizon_1_year/results/**/*.gif; do
    [ -f "$f" ] && cp "$f" "$VIDEOS_DIR/exp14_results/" 2>/dev/null || true
done

cd "$BUNDLE_DIR"
zip -r trackfm_videos.zip trackfm_videos/
cd "$SCRIPT_DIR"
echo "  Done: $BUNDLE_DIR/trackfm_videos.zip ($(du -sh trackfm_videos.zip | cut -f1))"

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo ""
echo "============================================"
echo "All bundles created in $BUNDLE_DIR/"
echo "============================================"
echo ""
ls -lh "$BUNDLE_DIR"/*.zip
echo ""
echo "To share with collaborators:"
echo "  1. Upload these zips to a shared location (Google Drive, S3, etc.)"
echo "  2. Share the download links"
echo ""
echo "Bundle contents:"
echo "  trackfm_essential.zip     - Pre-trained model, finetuning data, configs, hyperparameters"
echo "  trackfm_exp14_checkpoints.zip - Experiment 14 model checkpoints (not in paper)"
echo "  trackfm_videos.zip        - Prediction visualizations and result figures"
