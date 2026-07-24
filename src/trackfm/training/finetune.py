"""Shared downstream fine-tuning loop.

One trainer for every fine-tune task: port origin/destination (classification),
ETA (regression), vessel classification, anomaly detection. Tasks differ only
in dataset adapter + head + metric; the loop (bf16, AdamW+cosine, early
stopping on the task metric, MLflow) is common — this replaces the
copy-pasted trainers of legacy experiments 12/13.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, model_validator
from torch.utils.data import DataLoader, Dataset

from trackfm.config import MLflowConfig, ModelConfig, NormalizationConfig
from trackfm.datasets.windowing import NUM_FEATURES, normalize_features
from trackfm.models.factory import build_model
from trackfm.models.heads import EtaRegressor, PortClassifier
from trackfm.training.mlflow_utils import start_run
from trackfm.training.mup import build_finetune_optimizer

logger = logging.getLogger(__name__)


class FinetuneTrainConfig(BaseModel):
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    # see TrainConfig.decay_bias_norm; moot at wd=0.0 but kept symmetric
    decay_bias_norm: bool = True
    batch_size: int = 256
    max_epochs: int = 50
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    # pretrained backbones want mean pooling, random-init wants last
    # (exp 12/13; pinned by docs/research/2026-07-finetuning-review.md)
    pooling: Literal["mean", "last"] = "mean"
    head_dropout: float = 0.3
    freeze_encoder: bool = False               # frozen for the ENTIRE run
    # strategy="lp-ft" (Kumar et al. 2022): freeze -> linear-probe the head
    # to early-stop -> reload BEST head -> unfreeze -> full FT. The
    # protection is the CONVERGED head suppressing feature drift, so the
    # probe phase runs to convergence, not a fixed warm-up epoch count.
    # The lp_*/ft_* fields govern ONLY the two lp-ft phases: a plain
    # strategy="lp" run uses learning_rate/max_epochs/
    # early_stopping_patience, identical to freeze_encoder=True.
    strategy: Literal["full", "lp", "lp-ft"] = "full"
    # linear_head=True replaces the 2-hidden-layer MLPHead with a single
    # nn.Linear — the TRUE linear probe. (Historical note: every strategy
    # ="lp" run before 2026-07-24 used the MLP head; those results are the
    # frozen-MLP rung of the ladder, not linear probes.)
    linear_head: bool = False
    lp_max_epochs: int = 30
    lp_patience: int = 5
    lp_learning_rate: Optional[float] = None   # None -> learning_rate
    ft_learning_rate: Optional[float] = None   # None -> learning_rate / 10
    precision: Literal["bf16", "fp32"] = "bf16"
    num_workers: int = 4
    seed: int = 17
    max_train_windows: Optional[int] = None   # subsample cap for quick runs

    @model_validator(mode="after")
    def _validate_strategy(self):
        if self.strategy == "lp-ft" and self.freeze_encoder:
            raise ValueError(
                "freeze_encoder pins the backbone for the entire run, but "
                "lp-ft must unfreeze it after the probe phase — use "
                "strategy='lp' for a fully frozen run.")
        return self


class FinetuneConfig(BaseModel):
    task: Literal["classification", "regression"]
    label_column: str                          # origin|destination|remaining_s|...
    data_dir: Path
    checkpoint_dir: Path = Path("~/data/trackfm/checkpoints/finetune")
    backbone: Optional[Path] = None            # None = random init
    input_len: int = 128
    model: ModelConfig = ModelConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    train: FinetuneTrainConfig = FinetuneTrainConfig()
    mlflow: MLflowConfig = MLflowConfig(experiment="trackfm/finetune")


# ------------------------------------------------------------------ datasets
class WindowTaskDataset(Dataset):
    """Port-task parquet windows -> (normalized window, target)."""

    def __init__(self, split_dir: Path, label_column: str, input_len: int,
                 norm: NormalizationConfig, vocab: dict[str, int] | None,
                 max_windows: int | None = None, seed: int = 0):
        import polars as pl

        split_dir = Path(split_dir)
        cols = ["features", label_column]
        legacy = split_dir / "windows.parquet"
        if legacy.exists():
            df = pl.read_parquet(legacy, columns=cols)
        else:  # v2 sharded layout: <data_dir>/port_windows/shard_XXXX.parquet
            shards = sorted((split_dir.parent / "port_windows").glob("shard_*.parquet"))
            df = (pl.scan_parquet(shards)
                  .filter(pl.col("split") == split_dir.name)
                  .select(cols).collect())
        if max_windows and df.height > max_windows:
            df = df.sample(max_windows, seed=seed)

        raw = df["features"].to_numpy().reshape(-1, input_len, NUM_FEATURES)
        self.x = normalize_features(
            raw, lat_center=norm.lat_center, lat_scale=norm.lat_scale,
            lon_center=norm.lon_center, lon_scale=norm.lon_scale,
            sog_scale=norm.sog_scale, dt_scale=norm.dt_scale)

        if vocab is not None:  # classification
            other = vocab.get("OTHER", len(vocab) - 1)
            self.y = np.array([vocab.get(v, other) for v in df[label_column].to_list()],
                              dtype=np.int64)
        else:                  # regression target (seconds)
            self.y = df[label_column].to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


class ArrayTaskDataset(Dataset):
    """Pre-processed (N, L, 6) arrays + integer labels (exp-13 format)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


# ------------------------------------------------------------------- metrics
def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    pred = logits.argmax(1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro"))}


def regression_metrics(pred_log: np.ndarray, y_seconds: np.ndarray) -> dict[str, float]:
    pred_s = np.expm1(np.clip(pred_log, 0, 20))
    mae_min = float(np.abs(pred_s - y_seconds).mean() / 60)
    medae_min = float(np.median(np.abs(pred_s - y_seconds)) / 60)
    return {"mae_minutes": mae_min, "medae_minutes": medae_min}


# --------------------------------------------------------------------- loop
def run_finetune(cfg: FinetuneConfig, datasets: dict[str, Dataset],
                 num_classes: int | None = None) -> dict[str, float]:
    """Train head(+encoder) on `datasets` {'train','val','test'}; returns test metrics.

    strategy="lp-ft" runs two phases through the same epoch loop: probe
    (frozen backbone, head to early-stop), then full FT from the BEST
    probe head at ft_learning_rate. Early stopping is phase-LOCAL (FT
    keeps its patience budget while improving even from below the probe's
    score), but checkpointing is GLOBAL: best.pt — and the final test
    evaluation — is the overall val winner, so if full FT never beats the
    probe, the probe model ships.
    """
    t = cfg.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if (t.precision == "bf16" and device.type == "cuda") else None
    torch.manual_seed(t.seed)

    encoder = build_model(cfg.model, cfg.normalization)
    if cfg.backbone:
        state = torch.load(Path(cfg.backbone).expanduser(), map_location="cpu",
                           weights_only=False)
        missing, unexpected = encoder.load_state_dict(state["model"], strict=False)
        logger.info(f"backbone loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")

    frozen = t.freeze_encoder or t.strategy in ("lp", "lp-ft")
    if cfg.task == "classification":
        assert num_classes, "num_classes required for classification"
        model = PortClassifier(encoder, num_classes, t.pooling, t.head_dropout,
                               frozen, t.linear_head).to(device)
        loss_fn = F.cross_entropy
        primary, mode = "f1_macro", "max"
    else:
        model = EtaRegressor(encoder, t.pooling, t.head_dropout,
                             frozen, t.linear_head).to(device)
        loss_fn = lambda out, y: F.huber_loss(out, EtaRegressor.target(y))
        primary, mode = "mae_minutes", "min"

    loaders = {
        k: DataLoader(ds, batch_size=t.batch_size, shuffle=(k == "train"),
                      num_workers=t.num_workers, pin_memory=device.type == "cuda")
        for k, ds in datasets.items()
    }
    steps_per_epoch = max(1, len(loaders["train"]))

    def make_opt(lr: float, max_epochs: int):
        opt = build_finetune_optimizer(model, t, cfg.model, lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
            (s + 1) / max(t.warmup_epochs * steps_per_epoch, 1),
            0.5 * (1 + np.cos(np.pi * min(1.0, s / max(max_epochs * steps_per_epoch, 1))))))
        return opt, sched

    if t.strategy == "lp-ft":
        lp_lr = (t.lp_learning_rate if t.lp_learning_rate is not None
                 else t.learning_rate)
        ft_lr = (t.ft_learning_rate if t.ft_learning_rate is not None
                 else t.learning_rate / 10)
        phases = [("lp", lp_lr, t.lp_max_epochs, t.lp_patience),
                  ("ft", ft_lr, t.max_epochs, t.early_stopping_patience)]
    else:
        phases = [(t.strategy, t.learning_rate, t.max_epochs,
                   t.early_stopping_patience)]

    run = start_run(cfg.mlflow, cfg, data_dir=cfg.data_dir)
    ckpt_dir = Path(cfg.checkpoint_dir).expanduser() / (cfg.mlflow.run_name or run.info.run_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(split: str) -> dict[str, float]:
        model.eval()
        outs, ys = [], []
        with torch.no_grad():
            for x, y in loaders[split]:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                    enabled=autocast_dtype is not None):
                    outs.append(model(x).float().cpu().numpy())
                ys.append(y.numpy())
        model.train()
        outs, ys = np.concatenate(outs), np.concatenate(ys)
        return (classification_metrics(outs, ys) if cfg.task == "classification"
                else regression_metrics(outs, ys))

    best = -np.inf if mode == "max" else np.inf
    epoch = 0                                   # global step across phases
    try:
        for pi, (tag, lr, max_epochs, patience) in enumerate(phases):
            if tag == "ft" and pi > 0:
                # LP-FT boundary: the converged head is the protection —
                # take the BEST probe head, not the last epoch's, then
                # unfreeze the backbone.
                if (ckpt_dir / "best.pt").exists():
                    model.load_state_dict(torch.load(
                        ckpt_dir / "best.pt", weights_only=False)["model"])
                for p in model.encoder.parameters():
                    p.requires_grad = True
            opt, sched = make_opt(lr, max_epochs)
            # early stopping is PHASE-local: full FT starting below the
            # converged probe's val score must get its patience budget as
            # long as it keeps improving, not race the probe's best.
            # best.pt (and the final test model) still track the GLOBAL
            # best. In a single-phase run the two are provably identical
            # to the historical single-`best` loop.
            phase_best = -np.inf if mode == "max" else np.inf
            bad = 0
            for _ in range(max_epochs):
                t0 = time.time()
                for x, y in loaders["train"]:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                        enabled=autocast_dtype is not None):
                        loss = loss_fn(model(x), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()

                vm = evaluate("val")
                mlflow.log_metrics({f"val_{k}": v for k, v in vm.items()}
                                   | {"train_loss": loss.item(),
                                      "epoch_seconds": time.time() - t0,
                                      "phase": float(pi)}, step=epoch)
                score = vm[primary]
                improved = score > best if mode == "max" else score < best
                phase_improved = (score > phase_best if mode == "max"
                                  else score < phase_best)
                logger.info(f"[{tag}] epoch {epoch}: val {primary}={score:.4f}"
                            f"{' *' if improved else ''}")
                epoch += 1
                if improved:
                    best = score
                    torch.save({"model": model.state_dict(),
                                "config": cfg.model_dump(mode="json")}, ckpt_dir / "best.pt")
                if phase_improved:
                    phase_best, bad = score, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
    finally:
        if (ckpt_dir / "best.pt").exists():
            model.load_state_dict(torch.load(ckpt_dir / "best.pt",
                                             weights_only=False)["model"])
        test = evaluate("test") if "test" in loaders else {}
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.end_run()

    logger.info(f"test: {test}")
    return test


# ---------------------------------------------------------- task assembly
def finetune_port_task(cfg: FinetuneConfig) -> dict[str, float]:
    """Wire a port-task fine-tune (origin/destination/eta) from its data dir."""
    data_dir = Path(cfg.data_dir).expanduser()
    vocab = None
    num_classes = None
    if cfg.task == "classification":
        vocab = json.loads((data_dir / "labels.json").read_text())[cfg.label_column]
        num_classes = len(vocab)

    t = cfg.train
    datasets = {
        split: WindowTaskDataset(
            data_dir / split, cfg.label_column, cfg.input_len, cfg.normalization,
            vocab, max_windows=t.max_train_windows if split == "train"
            else (t.max_train_windows // 4 if t.max_train_windows else None),
            seed=t.seed)
        for split in ("train", "val", "test")
    }
    sizes = {k: len(v) for k, v in datasets.items()}
    logger.info(f"datasets: {sizes}, classes: {num_classes}")
    return run_finetune(cfg, datasets, num_classes)


def finetune_vessel_task(cfg: FinetuneConfig) -> dict[str, float]:
    """Vessel-type classification through the SAME trainer as port tasks.

    Reuses ArrayTaskDataset + run_finetune. Class filter AND normalization
    are identical to scripts/ft_vessel_probe_v2.py (drop junk '1"', require
    >=500 train / >=100 val / >=100 test), so the MLP-frozen and full-FT
    rungs land on the exact 15-class task the linear probe used. The npz is
    ALREADY 6-col model format [lat,lon,sog,cog_SIN,cog_COS,dt] (builder
    did the sin/cos), so normalize ONLY lat/lon/sog/dt in place and leave
    cols 3/4 untouched. (Do NOT use normalize_features here — it expects
    5-col raw with cog_DEG and would re-sin the sin column to ~0.)
    """
    data_dir = Path(cfg.data_dir).expanduser()
    DROP = {'1"'}
    MIN = {"train": 500, "val": 100, "test": 100}
    name_of = {v: k for k, v in json.loads((data_dir / "labels.json").read_text()).items()}
    raw = {s: np.load(data_dir / f"{s}.npz") for s in ("train", "val", "test")}
    counts = {s: np.bincount(raw[s]["y"], minlength=len(name_of)) for s in raw}
    keep = [i for i in range(len(name_of))
            if name_of[i] not in DROP and all(counts[s][i] >= MIN[s] for s in raw)]
    remap = {old: new for new, old in enumerate(keep)}
    nm = cfg.normalization
    t = cfg.train
    datasets = {}
    for s, z in raw.items():
        sel = np.isin(z["y"], keep)
        x = z["x"][sel].astype(np.float32).copy()
        x[..., 0] = (x[..., 0] - nm.lat_center) / nm.lat_scale
        x[..., 1] = (x[..., 1] - nm.lon_center) / nm.lon_scale
        x[..., 2] = x[..., 2] / nm.sog_scale
        x[..., 5] = x[..., 5] / nm.dt_scale       # cols 3,4 = cog sin/cos, already done
        y = np.array([remap[v] for v in z["y"][sel]], dtype=np.int64)
        if s == "train" and t.max_train_windows and len(y) > t.max_train_windows:
            rng = np.random.default_rng(t.seed)
            idx = rng.choice(len(y), t.max_train_windows, replace=False)
            x, y = x[idx], y[idx]
        datasets[s] = ArrayTaskDataset(x, y)
    logger.info(f"vessel classes kept {len(keep)}/{len(name_of)}: "
                f"{[name_of[i] for i in keep]}")
    logger.info(f"datasets: { {k: len(v) for k, v in datasets.items()} }")
    return run_finetune(cfg, datasets, len(keep))
