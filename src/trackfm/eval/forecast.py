"""Per-horizon forecasting evaluation vs analytic baselines.

Reproduces the paper's headline metrics: model loss at fixed horizons,
compared against dead reckoning (constant velocity) and last position
(zero displacement), each scored with the same soft-target loss on the
same grid. Reports per-horizon losses and the DR/LP ratios (paper Table 1).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trackfm.config import PretrainConfig
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.models.factory import build_model
from trackfm.training.losses import (
    compute_soft_target_loss,
    dead_reckoning_displacement,
    gaussian_log_density_loss,
)

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 50, 100, 200, 400, 800]


@torch.no_grad()
def evaluate_forecasting(
    checkpoint: Path,
    cfg: PretrainConfig,
    split: str = "test",
    horizons: list[int] | None = None,
    max_batches: int = 200,
) -> dict:
    """Returns {horizon: {model, dr, lp, dr_ratio, lp_ratio}} + aggregates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = horizons or [h for h in DEFAULT_HORIZONS if h <= cfg.train.max_horizon]
    m, t = cfg.model, cfg.train

    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu",
                       weights_only=False)
    model = build_model(m, cfg.normalization, t.max_horizon,
                        len(horizons)).to(device).eval()
    model.load_state_dict(state["model"])
    logger.info(f"loaded {checkpoint} (step {state.get('step', '?')})")

    ds = ShardedWindowDataset(cfg.data_dir / split, batch_size=t.batch_size,
                              shuffle_shards=False)
    loader = DataLoader(ds, batch_size=None, num_workers=4,
                        pin_memory=device.type == "cuda")

    h_idx = torch.tensor(horizons, device=device)
    sums = {h: {"model": 0.0, "dr": 0.0, "lp": 0.0} for h in horizons}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            ld, tgt, _, _ = model.forward_train(batch, horizon_indices=h_idx,
                                                causal=False)
        ld, tgt = ld.float(), tgt.float()
        dr = dead_reckoning_displacement(batch, h_idx, m.max_seq_len,
                                         cfg.normalization)
        for k, h in enumerate(horizons):
            sums[h]["model"] += compute_soft_target_loss(
                ld[:, k:k+1], tgt[:, k:k+1], m.grid_range, m.grid_size, t.sigma).item()
            sums[h]["dr"] += gaussian_log_density_loss(
                dr[:, k:k+1], tgt[:, k:k+1], m.grid_range, m.grid_size,
                t.sigma, t.dr_sigma).item()
            sums[h]["lp"] += gaussian_log_density_loss(
                torch.zeros_like(dr[:, k:k+1]), tgt[:, k:k+1], m.grid_range,
                m.grid_size, t.sigma, t.dr_sigma).item()
        n += 1

    out: dict = {"horizons": {}, "checkpoint": str(checkpoint), "split": split,
                 "batches": n}
    agg = {"model": 0.0, "dr": 0.0, "lp": 0.0}
    for h in horizons:
        r = {k: v / n for k, v in sums[h].items()}
        r["dr_ratio"] = r["dr"] / r["model"]
        r["lp_ratio"] = r["lp"] / r["model"]
        out["horizons"][h] = r
        for k in agg:
            agg[k] += r[k]
    out["mean_dr_ratio"] = (agg["dr"] / len(horizons)) / (agg["model"] / len(horizons))
    out["mean_lp_ratio"] = (agg["lp"] / len(horizons)) / (agg["model"] / len(horizons))

    logger.info(f"{'h':>5} {'model':>8} {'DR':>8} {'LP':>8} {'DRx':>6} {'LPx':>6}")
    for h, r in out["horizons"].items():
        logger.info(f"{h:>5} {r['model']:8.3f} {r['dr']:8.3f} {r['lp']:8.3f} "
                    f"{r['dr_ratio']:6.2f} {r['lp_ratio']:6.2f}")
    logger.info(f"MEAN ratios: {out['mean_dr_ratio']:.2f}x vs DR, "
                f"{out['mean_lp_ratio']:.2f}x vs LP")
    return out
