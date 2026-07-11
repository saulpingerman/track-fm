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
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from trackfm.config import MLflowConfig, ModelConfig, NormalizationConfig
from trackfm.datasets.windowing import NUM_FEATURES, normalize_features
from trackfm.models.factory import build_model
from trackfm.models.heads import EtaRegressor, PortClassifier
from trackfm.training.mlflow_utils import start_run

logger = logging.getLogger(__name__)


class FinetuneTrainConfig(BaseModel):
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 50
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    pooling: Literal["mean", "last"] = "mean"
    head_dropout: float = 0.3
    freeze_encoder: bool = False
    precision: Literal["bf16", "fp32"] = "bf16"
    num_workers: int = 4
    seed: int = 17
    max_train_windows: Optional[int] = None   # subsample cap for quick runs


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

        cols = ["features", label_column]
        df = pl.read_parquet(Path(split_dir) / "windows.parquet", columns=cols)
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
    """Train head(+encoder) on `datasets` {'train','val','test'}; returns test metrics."""
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

    if cfg.task == "classification":
        assert num_classes, "num_classes required for classification"
        model = PortClassifier(encoder, num_classes, t.pooling, t.head_dropout,
                               t.freeze_encoder).to(device)
        loss_fn = F.cross_entropy
        primary, mode = "f1_macro", "max"
    else:
        model = EtaRegressor(encoder, t.pooling, t.head_dropout,
                             t.freeze_encoder).to(device)
        loss_fn = lambda out, y: F.huber_loss(out, EtaRegressor.target(y))
        primary, mode = "mae_minutes", "min"

    loaders = {
        k: DataLoader(ds, batch_size=t.batch_size, shuffle=(k == "train"),
                      num_workers=t.num_workers, pin_memory=device.type == "cuda")
        for k, ds in datasets.items()
    }

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=t.learning_rate, weight_decay=t.weight_decay)
    steps_per_epoch = max(1, len(loaders["train"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / max(t.warmup_epochs * steps_per_epoch, 1),
        0.5 * (1 + np.cos(np.pi * min(1.0, s / max(t.max_epochs * steps_per_epoch, 1))))))

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
    bad = 0
    try:
        for epoch in range(t.max_epochs):
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
                                  "epoch_seconds": time.time() - t0}, step=epoch)
            score = vm[primary]
            improved = score > best if mode == "max" else score < best
            logger.info(f"epoch {epoch}: val {primary}={score:.4f}"
                        f"{' *' if improved else ''}")
            if improved:
                best, bad = score, 0
                torch.save({"model": model.state_dict(),
                            "config": cfg.model_dump(mode="json")}, ckpt_dir / "best.pt")
            else:
                bad += 1
                if bad >= t.early_stopping_patience:
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
