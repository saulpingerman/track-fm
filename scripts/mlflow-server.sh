#!/usr/bin/env bash
# Local MLflow tracking server for TrackFM.
# Backend + artifacts live under ~/data/trackfm/mlflow (never in git).
set -euo pipefail

MLFLOW_DIR="${MLFLOW_DIR:-$HOME/data/trackfm/mlflow}"
mkdir -p "$MLFLOW_DIR/artifacts"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

exec "$UV_BIN" --directory "$(dirname "$0")/.." run mlflow server \
    --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" \
    --artifacts-destination "$MLFLOW_DIR/artifacts" \
    --host 127.0.0.1 \
    --allowed-hosts "snorlax.tail0517b6.ts.net,localhost,127.0.0.1,localhost:5000,127.0.0.1:5000" \
    --cors-allowed-origins "https://snorlax.tail0517b6.ts.net" \
    --port 5000
