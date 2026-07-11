# TrackFM

A foundation model for vessel trajectory understanding: a decoder-only causal
transformer with a 2D Fourier density head, pretrained via next-position
prediction on AIS data from Danish waters. Pretrained representations transfer
to downstream maritime tasks (anomaly detection, vessel-type classification).

This is the consolidated monorepo: data pipeline, model, training,
experiments, and paper in one place. It supersedes the original
[`track-fm`](https://github.com/saulpingerman/track-fm) experiment history
(archived at branch `archive-main` / tag `archive/2026-pre-overhaul`) and the
standalone [`ais-analysis`](https://github.com/saulpingerman/ais-analysis)
pipeline (archived; absorbed here as `trackfm.data`).

## Layout

```
src/trackfm/
├── data/        # AIS ingest + cleaning pipeline (DMA zips -> partitioned parquet)
├── datasets/    # windowing, pre-shuffled shard materialization, torch loaders
├── models/      # canonical encoder + Fourier density head + downstream heads
├── training/    # trainer, checkpointing, MLflow integration
├── eval/        # forecasting metrics vs dead-reckoning/last-position baselines
└── viz/         # track/density/video visualization
configs/         # YAML configs (pydantic-validated) for every stage
experiments/     # thin, config-driven entry points
paper/           # LaTeX paper
tests/           # pytest suite
```

## Quick start

```bash
uv sync --extra dev
uv run trackfm --help

# Stage pipeline
uv run trackfm download --start-date 2023-01-01 --end-date 2025-02-26
uv run trackfm clean
uv run trackfm materialize
uv run trackfm pretrain --config configs/pretrain/xlarge.yaml
```

Experiment tracking: local MLflow at `http://127.0.0.1:5000`
(`scripts/mlflow-server.sh`). Data and checkpoints live under `~/data/` —
never in git.

## Hardware

Developed on a single RTX Pro 6000 (Blackwell, 96GB), 48-core CPU, 125GB RAM.
