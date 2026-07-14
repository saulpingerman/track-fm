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
    if m.grid_mode != "fixed":
        raise NotImplementedError(
            "evaluate_forecasting scores physical fixed-grid densities; "
            "cone-grid checkpoints go through the cross-geometry study "
            "harness (eval/xgeometry) instead.")

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
    BASELINES = ["dr", "lp", "dr_tuned", "lp_tuned", "kalman"]
    sums = {h: {k: 0.0 for k in ["model", *BASELINES]} for h in horizons}
    # ---- tune strong baselines on VAL (never on the scored split) ----
    from trackfm.eval.baselines import (fit_sigma_per_horizon,
                                        gaussian_density_loss_per_sample,
                                        kalman_cv_forecast, tune_kalman)
    val_ds = ShardedWindowDataset(cfg.data_dir / "val", batch_size=t.batch_size,
                                  shuffle_shards=False)
    val_batches = []
    for i, b in enumerate(DataLoader(val_ds, batch_size=None, num_workers=2)):
        val_batches.append(b.to(device))
        if i >= 7:
            break
    vf = torch.cat(val_batches)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        _, vtgt, _, _ = model.forward_train(vf, horizon_indices=h_idx, causal=False)
    vtgt = vtgt.float()
    vdr = dead_reckoning_displacement(vf, h_idx, m.max_seq_len, cfg.normalization)
    dr_sigmas = fit_sigma_per_horizon(vdr, vtgt, m.grid_range, m.grid_size, t.sigma)
    lp_sigmas = fit_sigma_per_horizon(torch.zeros_like(vdr), vtgt, m.grid_range,
                                      m.grid_size, t.sigma)
    kf_q, kf_r = tune_kalman(vf, vtgt, h_idx, m.max_seq_len, cfg.normalization,
                             m.grid_range, m.grid_size, t.sigma)
    logger.info(f"tuned per-horizon DR sigmas: {[round(float(s), 3) for s in dr_sigmas]}")
    del vf, val_batches

    n = 0
    rank_accum: dict[str, list] = {"model": [], "dr_tuned": [], "kalman": []}
    strat_accum: dict[str, list] = {"difficulty": [], "ce_model": [], "ce_dr_tuned": []}
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
        lp = torch.zeros_like(dr)
        kf_pred, kf_sig = kalman_cv_forecast(batch, h_idx, m.max_seq_len,
                                             cfg.normalization, kf_q, kf_r)
        from trackfm.eval.search import gaussian_grid_log_density, search_ranks
        from trackfm.eval.stratified import difficulty_score, per_sample_soft_ce
        strat_accum["difficulty"].append(difficulty_score(dr, tgt).cpu())
        strat_accum["ce_model"].append(per_sample_soft_ce(
            ld, tgt, m.grid_range, m.grid_size, t.sigma).cpu())
        strat_accum["ce_dr_tuned"].append(per_sample_soft_ce(
            gaussian_grid_log_density(dr, dr_sigmas.to(device).view(1, -1, 1).expand_as(dr),
                                      m.grid_range, m.grid_size),
            tgt, m.grid_range, m.grid_size, t.sigma).cpu())
        rank_accum["model"].append(search_ranks(ld, tgt, m.grid_range).cpu())
        rank_accum["dr_tuned"].append(search_ranks(
            gaussian_grid_log_density(dr, dr_sigmas.to(device).view(1, -1, 1).expand_as(dr),
                                      m.grid_range, m.grid_size),
            tgt, m.grid_range).cpu())
        rank_accum["kalman"].append(search_ranks(
            gaussian_grid_log_density(kf_pred, kf_sig, m.grid_range, m.grid_size),
            tgt, m.grid_range).cpu())
        for k, h in enumerate(horizons):
            s, sl = sums[h], slice(k, k + 1)
            s["model"] += compute_soft_target_loss(
                ld[:, sl], tgt[:, sl], m.grid_range, m.grid_size, t.sigma).item()
            # paper-comparable fixed-sigma baselines
            s["dr"] += gaussian_log_density_loss(
                dr[:, sl], tgt[:, sl], m.grid_range, m.grid_size,
                t.sigma, t.dr_sigma).item()
            s["lp"] += gaussian_log_density_loss(
                lp[:, sl], tgt[:, sl], m.grid_range, m.grid_size,
                t.sigma, t.dr_sigma).item()
            # strong baselines: per-horizon tuned sigma; Kalman covariance
            s["dr_tuned"] += gaussian_density_loss_per_sample(
                dr[:, sl], torch.full_like(dr[:, sl], float(dr_sigmas[k])),
                tgt[:, sl], m.grid_range, m.grid_size, t.sigma).item()
            s["lp_tuned"] += gaussian_density_loss_per_sample(
                lp[:, sl], torch.full_like(lp[:, sl], float(lp_sigmas[k])),
                tgt[:, sl], m.grid_range, m.grid_size, t.sigma).item()
            s["kalman"] += gaussian_density_loss_per_sample(
                kf_pred[:, sl], kf_sig[:, sl], tgt[:, sl],
                m.grid_range, m.grid_size, t.sigma).item()
        n += 1

    out: dict = {"horizons": {}, "checkpoint": str(checkpoint), "split": split,
                 "batches": n,
                 "baseline_params": {
                     "dr_sigmas": [float(s) for s in dr_sigmas],
                     "lp_sigmas": [float(s) for s in lp_sigmas],
                     "kalman_q": kf_q, "kalman_r": kf_r,
                 }}
    agg = {k: 0.0 for k in ["model", *BASELINES]}
    for h in horizons:
        r = {k: v / n for k, v in sums[h].items()}
        for b in BASELINES:
            r[f"{b}_ratio"] = r[b] / r["model"]
        out["horizons"][h] = r
        for k in agg:
            agg[k] += r[k]
    for b in BASELINES:
        out[f"mean_{b}_ratio"] = agg[b] / agg["model"]

    # search-effort / capture curves per horizon, model vs strong baselines
    from trackfm.eval.search import capture_curve, summarize_ranks
    n_cells = m.grid_size * m.grid_size
    out["search"] = {}
    for name, chunks in rank_accum.items():
        ranks = torch.cat(chunks)                       # (N, H)
        per_h = {}
        for k, h in enumerate(horizons):
            s = summarize_ranks(ranks[:, k])
            s["curve"] = capture_curve(ranks[:, k], n_cells)
            per_h[h] = s
        out["search"][name] = per_h
    for k, h in enumerate(horizons):
        row = {name: out["search"][name][h] for name in rank_accum}
        logger.info(f"search h={h}: " + " | ".join(
            f"{nm} k@90={v['curve']['k@90']} med={v.get('median_rank', float('nan')):.0f}"
            for nm, v in row.items()))

    # difficulty-stratified report (DR-residual deciles) per horizon
    from trackfm.eval.stratified import stratified_table
    diff = torch.cat(strat_accum["difficulty"])
    ce_m = torch.cat(strat_accum["ce_model"])
    ce_d = torch.cat(strat_accum["ce_dr_tuned"])
    model_ranks = torch.cat(rank_accum["model"])
    out["stratified"] = {}
    for k, h in enumerate(horizons):
        out["stratified"][h] = stratified_table(
            diff[:, k], {"ce_model": ce_m[:, k], "ce_dr_tuned": ce_d[:, k],
                         "rank_model": model_ranks[:, k].float()})
        top = out["stratified"][h].get(9, {})
        bot = out["stratified"][h].get(0, {})
        if top and bot:
            logger.info(
                f"stratified h={h}: easy-decile CE {bot['ce_model_mean']:.3f} "
                f"(DRtuned {bot['ce_dr_tuned_mean']:.3f}) | hard-decile CE "
                f"{top['ce_model_mean']:.3f} (DRtuned {top['ce_dr_tuned_mean']:.3f}), "
                f"hard capture@10 {top.get('capture10', float('nan')):.0%}")

    hdr = f"{'h':>5} {'model':>7} " + " ".join(f"{b:>9}" for b in BASELINES)
    logger.info(hdr)
    for h, r in out["horizons"].items():
        logger.info(f"{h:>5} {r['model']:7.3f} " + " ".join(
            f"{r[b]:5.2f}/{r[f'{b}_ratio']:.1f}x" for b in BASELINES))
    logger.info("MEAN ratios: " + ", ".join(
        f"{out[f'mean_{b}_ratio']:.2f}x vs {b}" for b in BASELINES))
    return out
