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
from trackfm.training.losses import compute_soft_target_loss, dead_reckoning_displacement, \
    gaussian_log_density_loss
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


def _lr_lambda(step: int, warmup: int, max_steps: int | None, schedule: str):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if schedule == "constant" or not max_steps:
        return 1.0
    progress = min(1.0, (step - warmup) / max(max_steps - warmup, 1))
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, val_loader, cfg: PretrainConfig, device, autocast_dtype,
             max_batches: int = 100):
    """Val loss (causal=False) + DR ratio + search metrics.

    Returns (val_loss, dr_loss, search) where search maps metric names to
    values, e.g. val_p90rank_h267 / val_capture10_h800. Search metrics are MONITORING
    only — early stopping stays on val_loss (a proper scoring rule); never
    select models on a pure ranking metric.
    """
    from trackfm.eval.search import search_ranks

    model.eval()
    m = cfg.model
    t = cfg.train
    total, dr_total, n = 0.0, 0.0, 0
    horizons = torch.linspace(1, t.max_horizon, t.num_horizon_samples,
                              dtype=torch.long, device=device)
    rank_chunks = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            ld, tgt, h, _ = model.forward_train(batch, horizon_indices=horizons, causal=False)
        loss = compute_soft_target_loss(ld.float(), tgt.float(), m.grid_range,
                                        m.grid_size, t.sigma)
        dr_pred = dead_reckoning_displacement(batch, horizons, m.max_seq_len, cfg.normalization)
        dr_loss = gaussian_log_density_loss(dr_pred, tgt.float(), m.grid_range,
                                            m.grid_size, t.sigma, t.dr_sigma)
        rank_chunks.append(search_ranks(ld.float(), tgt.float(), m.grid_range).cpu())
        total += loss.item()
        dr_total += dr_loss.item()
        n += 1
    model.train()
    if n == 0:
        return float("nan"), float("nan"), {}

    ranks = torch.cat(rank_chunks)
    search = {}
    # Monitor a mid horizon and the max horizon; h=1 is degenerate (the
    # vessel has barely moved). k@90-of-total is undefined whenever the
    # grid ceiling is below 90%, so report the 90th-percentile rank among
    # CAPTURABLE targets instead — always defined, actually tracks skill.
    mid = max(1, len(horizons) // 2)
    for k, h in ((mid, int(horizons[mid])), (len(horizons) - 1, int(horizons[-1]))):
        col = ranks[:, k]
        valid = col[col > 0].float()
        if len(valid):
            search[f"val_p90rank_h{h}"] = float(valid.quantile(0.9))
            search[f"val_medrank_h{h}"] = float(valid.median())
        search[f"val_capture10_h{h}"] = float(((col > 0) & (col <= 10)).float().mean())
        search[f"val_ceiling_h{h}"] = float((col > 0).float().mean())
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
    bad_vals = 0
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
                    ld, tgt, _, _ = model.forward_train(batch, causal=True)
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

                if time.time() - last_val_time > t.val_interval_minutes * 60 \
                        or step >= max_steps:
                    val_loss, dr_loss, search = validate(model, val_loader, cfg,
                                                         device, autocast_dtype)
                    ratio = dr_loss / val_loss if val_loss and not math.isnan(val_loss) else float("nan")
                    _mlflow_log({"val_loss": val_loss, "dr_loss": dr_loss,
                                        "dr_ratio": ratio, **search}, step=step)
                    logger.info(f"step {step}: val {val_loss:.4f}, DR-ratio {ratio:.2f}x, "
                                + ", ".join(f"{k}={v:.3g}" for k, v in search.items()
                                            if "rank" in k))
                    last_val_time = time.time()

                    torch.save({"model": model.state_dict(), "step": step,
                                "config": cfg.model_dump(mode="json")},
                               ckpt_dir / "last.pt")
                    if val_loss < best_val:
                        best_val = val_loss
                        bad_vals = 0
                        torch.save({"model": model.state_dict(), "step": step,
                                    "val_loss": val_loss,
                                    "config": cfg.model_dump(mode="json")},
                                   ckpt_dir / "best.pt")
                    else:
                        bad_vals += 1
                        if bad_vals >= t.early_stop_patience:
                            logger.info(f"Early stop after {bad_vals} bad validations")
                            raise StopIteration
                if step >= max_steps:
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        try:
            mlflow.log_metric("best_val_loss", best_val)
            if (ckpt_dir / "best.pt").exists():
                mlflow.log_artifact(str(ckpt_dir / "best.pt"))
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"mlflow finalization failed (checkpoints are safe "
                           f"on disk in {ckpt_dir}): {e}")

    logger.info(f"Done: best val {best_val:.4f}; checkpoints in {ckpt_dir}")
    return ckpt_dir / "best.pt"
