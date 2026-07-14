"""Cross-geometry containment scoring: compare checkpoints on ONE physical
basis, regardless of grid geometry (fixed vs cone) or head (fourier vs
direct).

Raw CE and raw cell-counts are NOT comparable across geometries — a cone
cell at 2h is physically ~17x a fixed cell, and the fixed grid clamps
targets its window can't reach. This scorer reports what is comparable:

  * containment in PHYSICAL km^2 to capture 90% of vessels, per time
    bucket (15m/30m/1h/2h) — Paul's search-effort metric in real units;
  * the ceiling (on-canvas fraction) per bucket, reported separately —
    a geometry that clips 25% of 2h targets simply CANNOT reach 90%, and
    that shows up here as "unreachable", not as a silently-good number.

Long horizons are the point; h1-style short-horizon CE is deliberately
not the headline.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trackfm.config import PretrainConfig
from trackfm.datasets.loaders import ShardedWindowDataset
from trackfm.eval.horizons import TIME_BUCKETS, time_bucket_indices
from trackfm.eval.search import capture_curve, search_ranks, summarize_ranks
from trackfm.models.factory import build_model
from trackfm.training.losses import cone_elapsed_seconds, cone_ranges

LAT0 = 56.25
KM_PER_DEG = 111.32


def _cell_area_km2(cell_deg: float) -> float:
    """Physical area of one square grid cell of side cell_deg, at LAT0."""
    return (cell_deg * KM_PER_DEG) * (cell_deg * KM_PER_DEG * math.cos(math.radians(LAT0)))


@torch.no_grad()
def score_geometry(checkpoint: Path, cfg: PretrainConfig, split: str = "test",
                   max_batches: int = 120) -> dict:
    """Per-bucket containment in km^2, ceiling, ranks — geometry-agnostic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, t = cfg.model, cfg.train
    cone = m.grid_mode == "cone"
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu",
                       weights_only=False)
    model = build_model(m, cfg.normalization, t.max_horizon,
                        len(TIME_BUCKETS)).to(device).eval()
    model.load_state_dict(state["model"])

    ds = ShardedWindowDataset(cfg.data_dir / split, batch_size=t.batch_size,
                              shuffle_shards=False)
    loader = DataLoader(ds, batch_size=None, num_workers=4,
                        pin_memory=device.type == "cuda")

    buckets = list(TIME_BUCKETS)                       # ["15m","30m","1h","2h"]
    rank_chunks = {b: [] for b in buckets}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        step_idx, valid = time_bucket_indices(batch, cfg.normalization, m.max_seq_len)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            ld, tgt = model.forward_at_indices(batch, step_idx)
        ld, tgt = ld.float(), tgt.float()
        g_rng = 1.0 if cone else m.grid_range
        if cone:
            el = cone_elapsed_seconds(batch, step_idx, m.max_seq_len,
                                      cfg.normalization.dt_scale, causal=False)
            tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
        ranks = search_ranks(ld, tgt, g_rng)           # (B, H) canvas/physical cells
        # samples not available at a bucket are dropped from THAT bucket
        ranks = torch.where(valid, ranks, torch.full_like(ranks, -2))
        for k, b in enumerate(buckets):
            r = ranks[:, k]
            rank_chunks[b].append(r[r != -2].cpu())    # keep censored(-1), drop unavail(-2)

    n_cells = m.grid_size * m.grid_size
    out = {"checkpoint": str(checkpoint), "geometry": m.grid_mode,
           "head": m.head_type, "split": split, "buckets": {}}
    for k, b in enumerate(buckets):
        ranks = torch.cat(rank_chunks[b]) if rank_chunks[b] else torch.zeros(0)
        curve = capture_curve(ranks, n_cells)
        summ = summarize_ranks(ranks)
        tau = TIME_BUCKETS[b]
        if cone:
            R = m.cone_r0 + m.cone_v * tau ** m.cone_p
            cell_deg = (2.0 * 1.0 / m.grid_size) * R
        else:
            cell_deg = 2.0 * m.grid_range / m.grid_size
        cell_km2 = _cell_area_km2(cell_deg)
        k90 = curve["k@90"]
        out["buckets"][b] = {
            "n": int(ranks.numel()),
            "ceiling": round(curve["ceiling"], 4),          # on-canvas fraction
            "cell_km2": round(cell_km2, 4),
            "k90_cells": k90,
            "km2_to_capture90": round(k90 * cell_km2, 1) if k90 else None,
            "unreachable_reason": None if k90 else (
                "ceiling<0.9 (targets escape grid)" if curve["ceiling"] < 0.9
                else "ceiling>=0.9 but curve never hits 90% within grid"),
            "median_rank": summ.get("median_rank"),
            "p90_rank": summ.get("p90_rank"),
            "capture@10_km2": round(10 * cell_km2, 1),      # area of a 10-cell search
            "capture@10": summ.get("capture@10"),
        }
    return out
