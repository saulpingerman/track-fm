"""Pretraining loop: bf16, cosine schedule, time-based validation, MLflow.

Replaces the exp-11/exp-14 monolith training loops. Loss semantics are
identical (soft-target KL over the Fourier grid, random horizon sampling,
causal subwindow training); the infrastructure is new: bf16 autocast
instead of fp16 GradScaler, MLflow instead of print-logs, time-based
validation for multi-day runs.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from trackfm.config import PretrainConfig
from trackfm.datasets.loaders import ShardedWindowDataset, num_samples
from trackfm.models.factory import build_model, count_parameters
from trackfm.training.losses import compute_soft_target_loss, cone_elapsed_seconds, \
    cone_ranges, dead_reckoning_displacement, gaussian_log_density_loss
from trackfm.training.mlflow_utils import start_run

logger = logging.getLogger(__name__)


def _mlflow_log(metrics: dict, step: int) -> None:
    """Log metrics, surviving tracking-server outages.

    A multi-day run must never die because MLflow hiccuped (learned the
    hard way: a server restart mid-run killed the first golden chain).
    Checkpoints and training continue; lost metric points are acceptable.
    """
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        logger.warning(f"mlflow logging failed at step {step} (continuing): {e}")



def should_stop_saturation(history: list[tuple[int, float]],
                            min_history: int = 48,
                            min_gain_frac: float = 0.004) -> tuple[bool, float]:
    """No-progress test, robust to per-validation noise (median blocks).

    Splits the full validation history into two halves and compares their
    MEDIANS: stop only when the second half improved on the first by less
    than `min_gain_frac`. Median noise shrinks as 1/sqrt(n) while the
    half-to-half signal of any healthy power law stays ~1%+, so noise
    cannot fake saturation (a window-local slope fit could not make this
    guarantee: its SNR falls below 1 mid-run — measured 26 consecutive
    false positives under +-1% val noise). Never fires before
    `min_history` validations (~24h at the 30-min cadence).

    Returns (stop, half_over_half_relative_gain).
    """
    import numpy as np

    n = len(history)
    if n < min_history:
        return False, float("inf")
    losses = np.array([h[1] for h in history], dtype=np.float64)
    half = n // 2
    first = float(np.median(losses[:half]))
    second = float(np.median(losses[half:]))
    gain = (first - second) / max(second, 1e-9)
    return gain < min_gain_frac, gain


def _lr_lambda(step: int, warmup: int, max_steps: int | None, schedule: str):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if schedule == "constant" or not max_steps:
        return 1.0
    progress = min(1.0, (step - warmup) / max(max_steps - warmup, 1))
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def fast_val(model, cached_batches, cfg: PretrainConfig, device, autocast_dtype):
    """Loss-only validation on a small FIXED subset with FIXED horizons —
    seconds per call, so it can run every few hundred steps and give
    GPT-3-density val curves. Full validate() (search metrics, selection)
    stays on its wall-clock schedule; this is a monitoring curve only.
    """
    model.eval()
    m, t = cfg.model, cfg.train
    horizons = torch.linspace(1, t.max_horizon, t.num_horizon_samples,
                              dtype=torch.long, device=device)
    total = 0.0
    for batch in cached_batches:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            ld, tgt, hz, _ = model.forward_train(batch, horizon_indices=horizons,
                                                 causal=False)
        if m.grid_mode == "cone":
            el = cone_elapsed_seconds(batch, hz, m.max_seq_len,
                                      cfg.normalization.dt_scale, causal=False)
            tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
            total += compute_soft_target_loss(ld.float(), tgt.float(), 1.0,
                                              m.grid_size,
                                              t.sigma / m.grid_range).item()
        else:
            total += compute_soft_target_loss(ld.float(), tgt.float(), m.grid_range,
                                              m.grid_size, t.sigma).item()
    model.train()
    return total / max(len(cached_batches), 1)


def validate(model, val_loader, cfg: PretrainConfig, device, autocast_dtype,
             max_batches: int = 100):
    """Val loss (causal=False) + DR ratio + search metrics.

    Returns (val_loss, dr_loss, search) where search maps metric names to
    values keyed by TIME bucket, e.g. val_p90rank_30m / val_capture10_2h. Search metrics are MONITORING
    only — early stopping stays on val_loss (a proper scoring rule); never
    select models on a pure ranking metric.
    """
    from trackfm.eval.horizons import TIME_BUCKETS, time_bucket_indices
    from trackfm.eval.search import search_ranks

    model.eval()
    m = cfg.model
    t = cfg.train
    total, dr_total, n = 0.0, 0.0, 0
    horizons = torch.linspace(1, t.max_horizon, t.num_horizon_samples,
                              dtype=torch.long, device=device)
    rank_chunks, valid_chunks = [], []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            # loss on the training-consistent step horizons (early stopping)
            ld, tgt, h, _ = model.forward_train(batch, horizon_indices=horizons, causal=False)
            # containment monitoring at OPERATIONAL time buckets (15m..2h):
            # step-indexed horizons conflate physically different questions
            # because dt varies per vessel
            step_idx, bucket_valid = time_bucket_indices(batch, cfg.normalization,
                                                         m.max_seq_len)
            ld_t, tgt_t = model.forward_at_indices(batch, step_idx)
        cone = m.grid_mode == "cone"
        if cone:
            el = cone_elapsed_seconds(batch, horizons, m.max_seq_len,
                                      cfg.normalization.dt_scale, causal=False)
            tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
            el_t = cone_elapsed_seconds(batch, step_idx, m.max_seq_len,
                                        cfg.normalization.dt_scale, causal=False)
            tgt_t = tgt_t / cone_ranges(el_t, m.cone_r0, m.cone_v, m.cone_p)
        g_rng = 1.0 if cone else m.grid_range
        g_sig = t.sigma / m.grid_range if cone else t.sigma
        loss = compute_soft_target_loss(ld.float(), tgt.float(), g_rng,
                                        m.grid_size, g_sig)
        dr_pred = dead_reckoning_displacement(batch, horizons, m.max_seq_len, cfg.normalization)
        if cone:
            dr_pred = dr_pred / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
        dr_loss = gaussian_log_density_loss(dr_pred, tgt.float(), g_rng,
                                            m.grid_size, g_sig,
                                            t.dr_sigma / m.grid_range if cone else t.dr_sigma)
        rank_chunks.append(search_ranks(ld_t.float(), tgt_t.float(), g_rng).cpu())
        # ALSO: physical fine-grid ranks per bucket. Two grids, so cone runs
        # log metrics directly comparable to (a) fixed's native cell size
        # and (b) a 1×1 km operational unit — the native cell size drifts
        # with the cone geometry and can't be compared across runs.
        from trackfm.eval.xgeometry import (ranks_on_fine_grid,
                                             FIXED_NATIVE_CELL_DEG)
        if not hasattr(model, "_fine_bucket_chunks"):
            model._fine_bucket_chunks = {b: [] for b in TIME_BUCKETS}
            model._fixcell_bucket_chunks = {b: [] for b in TIME_BUCKETS}
        for k_b, b_name in enumerate(TIME_BUCKETS):
            tau = TIME_BUCKETS[b_name]
            R_deg = (m.cone_r0 + m.cone_v * tau ** m.cone_p) if cone else m.grid_range
            tc = tgt_t[:, k_b].float() if cone else tgt_t[:, k_b].float() / m.grid_range
            fr1, _, _ = ranks_on_fine_grid(ld_t[:, k_b].float(), tc, R_deg,
                                            target_cell_km=1.0)
            model._fine_bucket_chunks[b_name].append(fr1.cpu())
            frF, _, _ = ranks_on_fine_grid(ld_t[:, k_b].float(), tc, R_deg,
                                            cell_deg=FIXED_NATIVE_CELL_DEG)
            model._fixcell_bucket_chunks[b_name].append(frF.cpu())
        valid_chunks.append(bucket_valid.cpu())
        total += loss.item()
        dr_total += dr_loss.item()
        n += 1
    model.train()
    if n == 0:
        return float("nan"), float("nan"), {}

    ranks = torch.cat(rank_chunks)
    bucket_ok = torch.cat(valid_chunks)
    search = {}
    # Slimmed metric set (Paul, 2026-07-14): keep only the metrics that
    # actually drive the flagship decision. Removed: val_p90rank/medrank
    # (native cell size drifts by geometry — incomparable), val_capture10
    # (same), val_avail (nearly-flat sanity), val_fine_medrank_km2 &
    # val_fixcell_medrank (median hides tail), dr_loss & dr_ratio (also
    # geometry-locked in raw form). Kept: val_ceiling (companion to km²@90),
    # val_km2_at_capture90 (operational, 1×1 km, cross-geometry-comparable),
    # val_fixcell_p90rank (matches fixed's native cell for direct compare).
    for k, name in enumerate(TIME_BUCKETS):
        col = torch.where(bucket_ok[:, k], ranks[:, k], torch.full_like(ranks[:, k], -1))
        n_avail = bucket_ok[:, k].sum().clamp(min=1).float()
        search[f"val_ceiling_{name}"] = float((col > 0).sum() / n_avail)
        # fine 1×1 km grid — operational km²@90
        fine = torch.cat(model._fine_bucket_chunks[name])[:len(bucket_ok)]
        fine_ok = torch.where(bucket_ok[:, k], fine, torch.full_like(fine, -1))
        fine_valid = fine_ok[fine_ok > 0].float()
        if len(fine_valid):
            search[f"val_km2_at_capture90_{name}"] = float(fine_valid.quantile(0.9))
        # fixed-native cell size (0.009375°) — directly comparable to fixed's own p90rank
        fx = torch.cat(model._fixcell_bucket_chunks[name])[:len(bucket_ok)]
        fx_ok = torch.where(bucket_ok[:, k], fx, torch.full_like(fx, -1))
        fx_valid = fx_ok[fx_ok > 0].float()
        if len(fx_valid):
            search[f"val_fixcell_p90rank_{name}"] = float(fx_valid.quantile(0.9))
    if hasattr(model, "_fine_bucket_chunks"):
        del model._fine_bucket_chunks
        del model._fixcell_bucket_chunks
    return total / n, dr_total / n, search


def run_pretraining(cfg: PretrainConfig) -> Path:
    t = cfg.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if (t.precision == "bf16" and device.type == "cuda") else None
    torch.manual_seed(t.seed)

    model = build_model(cfg.model, cfg.normalization, t.max_horizon,
                        t.num_horizon_samples).to(device)

    # Compile the loss: Inductor fuses its elementwise chain (dist^2 -> exp
    # -> normalize -> KL over B x pairs x G x G tensors). Measured on XLarge
    # @ batch 256: +16% throughput, -16% peak memory, bit-identical values.
    loss_fn = compute_soft_target_loss
    if device.type == "cuda":
        try:
            loss_fn = torch.compile(compute_soft_target_loss)
        except Exception as e:
            logger.warning(f"loss compile unavailable ({e}); using eager")

    # NOTE: t.compile wraps the whole model — only safe with FIXED horizon
    # indices (random horizons trigger value-specialized recompiles). Adds
    # no measurable speed over the compiled loss (GEMMs already optimal);
    # keep off for method-faithful random-horizon training.
    if t.compile:
        model = torch.compile(model)
    n_params = count_parameters(model)
    from trackfm.training.flops import train_flops_per_sample
    flops_per_sample = train_flops_per_sample(cfg.model, t.num_horizon_samples)
    logger.info(f"Model: {n_params/1e6:.1f}M params, "
                f"{flops_per_sample/1e9:.1f} GFLOPs/sample on {device}")

    train_ds = ShardedWindowDataset(cfg.data_dir / "train", batch_size=t.batch_size,
                                    seed=t.seed)
    val_ds = ShardedWindowDataset(cfg.data_dir / "val", batch_size=t.batch_size,
                                  shuffle_shards=False)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=t.num_workers,
                              pin_memory=device.type == "cuda", prefetch_factor=4
                              if t.num_workers else None)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=2,
                            pin_memory=device.type == "cuda")
    # fixed CPU-cached subset for the dense loss-only fast-val curve
    fast_batches = []
    if t.fast_val_interval_steps > 0:
        fast_batches = [b for _, b in
                        zip(range(t.fast_val_batches), val_loader)]

    n_train = num_samples(cfg.data_dir / "train")
    steps_per_epoch = max(1, n_train // t.batch_size)
    max_steps = t.max_steps or (steps_per_epoch * (t.max_epochs or 1))
    logger.info(f"{n_train:,} train samples, {steps_per_epoch:,} steps/epoch, "
                f"max_steps={max_steps:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=t.learning_rate,
                            weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, t.warmup_steps, max_steps, t.lr_schedule))

    run = start_run(cfg.mlflow, cfg, data_dir=cfg.data_dir,
                    extra_params={"n_params": n_params,
                                  "n_params_h": f"{n_params/1e6:.1f}M",
                                  "gflops_per_sample": f"{flops_per_sample/1e9:.1f}"})
    ckpt_dir = cfg.checkpoint_dir / (cfg.mlflow.run_name or run.info.run_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    best_val = float("inf")
    crashed = False
    val_history: list[tuple[int, float]] = []
    stop_streak = 0
    step = 0
    samples_seen = 0
    last_val_time = time.time()
    t_start = time.time()
    window_samples, window_t = 0, t_start
    m = cfg.model

    try:
        for epoch in range(1000):  # bounded by max_steps
            train_ds.set_epoch(epoch)
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                    enabled=autocast_dtype is not None):
                    ld, tgt, hz, _ = model.forward_train(batch, causal=True)
                    if m.grid_mode == "cone":
                        el = cone_elapsed_seconds(batch, hz, m.max_seq_len,
                                                  cfg.normalization.dt_scale,
                                                  causal=True)
                        tgt = tgt / cone_ranges(el, m.cone_r0, m.cone_v, m.cone_p)
                        loss = loss_fn(ld.float(), tgt.float(), 1.0,
                                       m.grid_size, t.sigma / m.grid_range)
                    else:
                        loss = loss_fn(ld.float(), tgt.float(),
                                       m.grid_range, m.grid_size, t.sigma)
                loss = loss / t.grad_accum_steps
                loss.backward()

                if (step + 1) % t.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                sched.step()

                samples_seen += batch.shape[0]
                step += 1

                if step % 50 == 0:
                    now = time.time()
                    # windowed rate (since last log), not cumulative — else
                    # one-time compile cost pollutes the whole run's curve
                    sps = (samples_seen - window_samples) / max(now - window_t, 1e-9)
                    window_samples, window_t = samples_seen, now
                    achieved_tflops = flops_per_sample * sps / 1e12
                    _mlflow_log({
                        "train_loss": loss.item() * t.grad_accum_steps,
                        "lr": sched.get_last_lr()[0],
                        "samples_per_s": sps,
                        "achieved_tflops": achieved_tflops,
                        "mfu": achieved_tflops / t.peak_tflops,
                    }, step=step)

                if fast_batches and step % t.fast_val_interval_steps == 0:
                    _mlflow_log({"val_loss_fast": fast_val(
                        model, fast_batches, cfg, device, autocast_dtype)},
                        step=step)

                if time.time() - last_val_time > t.val_interval_minutes * 60 \
                        or step >= max_steps:
                    val_loss, dr_loss, search = validate(model, val_loader, cfg,
                                                         device, autocast_dtype)
                    ratio = dr_loss / val_loss if val_loss and not math.isnan(val_loss) else float("nan")
                    # dr_loss/dr_ratio computed but not logged — geometry-locked
                    # (Paul, 2026-07-14). Kept in console for the log line.
                    _mlflow_log({"val_loss": val_loss, **search}, step=step)
                    logger.info(f"step {step}: val {val_loss:.4f}, DR-ratio {ratio:.2f}x, "
                                + ", ".join(f"{k}={v:.3g}" for k, v in search.items()
                                            if "p90rank" in k))
                    last_val_time = time.time()

                    torch.save({"model": model.state_dict(), "step": step,
                                "config": cfg.model_dump(mode="json")},
                               ckpt_dir / "last.pt")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({"model": model.state_dict(), "step": step,
                                    "val_loss": val_loss,
                                    "config": cfg.model_dump(mode="json")},
                                   ckpt_dir / "best.pt")
                    # saturation: opportunity-cost projection over remaining
                    # budget (power-law-safe; see should_stop_saturation)
                    val_history.append((step, val_loss))
                    stop, gain = should_stop_saturation(
                        val_history, t.early_stop_min_history,
                        t.early_stop_min_gain_frac)
                    if gain != float("inf"):
                        _mlflow_log({"half_over_half_gain": gain}, step)
                    stop_streak = stop_streak + 1 if stop else 0
                    if stop:
                        logger.info(f"saturation signal {stop_streak}/"
                                    f"{t.early_stop_confirmations}: half-over-half "
                                    f"gain {gain:.4%}")
                    if stop_streak >= t.early_stop_confirmations:
                        logger.info("Early stop: no meaningful progress "
                                    "(confirmed over consecutive validations)")
                        raise StopIteration
                if step >= max_steps:
                    raise StopIteration
    except StopIteration:
        pass
    except BaseException:
        crashed = True
        raise
    finally:
        try:
            mlflow.log_metric("best_val_loss", best_val)
            if (ckpt_dir / "best.pt").exists():
                mlflow.log_artifact(str(ckpt_dir / "best.pt"))
            # a crash (incl. OOM/KeyboardInterrupt) must not read FINISHED
            mlflow.end_run("FAILED" if crashed else "FINISHED")
        except Exception as e:
            logger.warning(f"mlflow finalization failed (checkpoints are safe "
                           f"on disk in {ckpt_dir}): {e}")

    logger.info(f"Done: best val {best_val:.4f}; checkpoints in {ckpt_dir}")
    return ckpt_dir / "best.pt"
