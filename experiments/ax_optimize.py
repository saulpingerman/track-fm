#!/usr/bin/env python
"""Ax Bayesian hyperparameter optimization over the shared fine-tune trainer.

Reproduces the paper's methodology (30 trials/condition, search over lr,
weight_decay, betas, pooling) for any downstream task config. Each trial is
a full fine-tune logged to MLflow; the Ax state is checkpointed so an
interrupted sweep resumes.

Usage:
  uv run python experiments/ax_optimize.py --config configs/downstream/ports_dest.yaml \
      --trials 30 [--backbone ~/data/trackfm/checkpoints/xlarge-26mo/best.pt]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ax.service.ax_client import AxClient, ObjectiveProperties

from trackfm.config import load_config
from trackfm.training.finetune import FinetuneConfig, finetune_port_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("ax_optimize")

# Search space from HYPERPARAMETERS.md (legacy exp 12/13 Ax setup)
SEARCH_SPACE = [
    {"name": "learning_rate", "type": "range", "bounds": [1e-6, 1e-2], "log_scale": True},
    {"name": "weight_decay", "type": "range", "bounds": [0.0, 0.3]},
    {"name": "pooling", "type": "choice", "values": ["mean", "last"], "is_ordered": False,
     "sort_values": False},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--backbone", default=None,
                    help="override backbone (None in config = random init)")
    ap.add_argument("--state", default=None, help="Ax state JSON (default: alongside config)")
    args = ap.parse_args()

    base: FinetuneConfig = load_config(args.config, FinetuneConfig)
    if args.backbone:
        base.backbone = Path(args.backbone)
    condition = "pretrained" if base.backbone else "random-init"

    state_path = Path(args.state or f"{args.config}.ax_{condition}.json")
    maximize = base.task == "classification"
    metric = "f1_macro" if maximize else "mae_minutes"

    if state_path.exists():
        ax = AxClient.load_from_json_file(str(state_path))
        logger.info(f"resumed Ax state: {len(ax.experiment.trials)} trials done")
    else:
        ax = AxClient()
        ax.create_experiment(
            name=f"{base.label_column}-{condition}",
            parameters=SEARCH_SPACE,
            objectives={metric: ObjectiveProperties(minimize=not maximize)},
        )

    while len(ax.experiment.trials) < args.trials:
        params, idx = ax.get_next_trial()
        cfg = base.model_copy(deep=True)
        cfg.train.learning_rate = params["learning_rate"]
        cfg.train.weight_decay = params["weight_decay"]
        cfg.train.pooling = params["pooling"]
        cfg.mlflow.run_name = f"{base.label_column}-{condition}-t{idx:02d}"
        logger.info(f"trial {idx}: {params}")
        try:
            result = finetune_port_task(cfg)
            ax.complete_trial(idx, raw_data={metric: result[metric]})
        except Exception as e:
            logger.error(f"trial {idx} failed: {e}")
            ax.log_trial_failure(idx)
        ax.save_to_json_file(str(state_path))

    best_params, values = ax.get_best_parameters()
    logger.info(f"BEST [{condition}]: {best_params} -> {values[0]}")
    print(json.dumps({"condition": condition, "best": best_params,
                      "value": values[0]}, indent=2))


if __name__ == "__main__":
    main()
