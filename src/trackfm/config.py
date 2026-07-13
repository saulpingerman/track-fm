"""Unified pydantic configuration for all TrackFM stages.

Every CLI entry point loads one YAML file into one of these models.
Paths default to the snorlax data layout under ~/data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser()


# NOTE: the cleaning-pipeline config schema lives in trackfm.data.config
# (moved verbatim from ais-analysis); this module covers the ML stages.


# ------------------------------------------------------------- materialize
class MaterializeConfig(BaseModel):
    """Config for `trackfm materialize` (windowing + 2-pass shuffle)."""
    clean_dir: Path = Path("~/data/ais/clean")
    out_dir: Path = Path("~/data/trackfm/materialized/v1")
    window_size: int = 928           # 128 input + 800 horizon
    input_len: int = 128
    stride: int = 32                 # legacy materialize_samples.py value
    # Legacy materialization applied no extra filters beyond window length;
    # Exp-11-style filtering (600 / 3.0 kn) is available via these knobs.
    min_track_length: int = 928
    min_sog_knots: float = 0.0
    num_output_shards: int = 256
    shard_target_mb: int = 512
    split: Literal["temporal"] = "temporal"
    train_frac: float = 0.8
    val_frac: float = 0.1
    start_date: Optional[str] = None   # "YYYY-MM-DD" — golden-slice selection
    end_date: Optional[str] = None
    # 1 = legacy 5-feature rows; 3 = +heading feature and per-window
    # [t0, mmsi] meta slots (weather-join / conditioning prerequisite).
    format_version: Literal[1, 3] = 3
    seed: int = 17
    num_workers: int = 40

    @field_validator("clean_dir", "out_dir")
    @classmethod
    def _expanduser(cls, v: Path) -> Path:
        return _expand(v)


# ------------------------------------------------------------------- model
class ModelConfig(BaseModel):
    d_model: int = 768
    nhead: int = 16
    num_layers: int = 16
    dim_feedforward: int = 3072
    dropout: float = 0.1
    input_features: int = 6          # lat, lon, sog, cog_sin, cog_cos, dt
    max_seq_len: int = 128
    grid_size: int = 64
    grid_range: float = 0.3          # degrees
    num_freqs: int = 12
    head_type: Literal["fourier", "direct"] = "fourier"  # 'direct' = ablation


class NormalizationConfig(BaseModel):
    lat_center: float = 56.25
    lat_scale: float = 1.0
    lon_center: float = 11.5
    lon_scale: float = 2.0
    sog_scale: float = 30.0
    dt_scale: float = 300.0


# ---------------------------------------------------------------- training
class TrainConfig(BaseModel):
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 300
    max_horizon: int = 800
    num_horizon_samples: int = 4
    warmup_steps: int = 1000
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    sigma: float = 0.003             # soft-target Gaussian width (grid coords)
    dr_sigma: float = 0.05           # dead-reckoning baseline sigma
    precision: Literal["bf16", "fp32"] = "bf16"
    compile: bool = False
    grad_accum_steps: int = 1
    val_interval_minutes: float = 30.0
    # dense loss-only val curve (GPT-3-style); 0 disables. Full validate()
    # keeps its wall-clock schedule and remains the selection metric.
    fast_val_interval_steps: int = 200
    fast_val_batches: int = 3
    early_stop_patience: int = 10          # legacy knob, unused by pretrain
    early_stop_min_delta_frac: float = 0.001  # legacy knob, unused by pretrain
    # Saturation = opportunity cost: stop when the trend (fit over the last
    # `window` validations vs log-step) projects < min_remaining_frac
    # further improvement over the ENTIRE remaining budget. Power-law-safe.
    # Saturation = no-progress, noise-robust: compare MEDIANS of the two
    # halves of the validation history; stop when half-over-half gain stays
    # below min_gain_frac for `confirmations` consecutive validations.
    # Median blocks grow with the run (noise ~1/sqrt(n)), so noise cannot
    # fake saturation on a healthy power-law curve.
    early_stop_min_history: int = 48
    early_stop_min_gain_frac: float = 0.004
    early_stop_confirmations: int = 4
    num_workers: int = 8
    seed: int = 17
    # Practical bf16 peak of this GPU (scripts/gpu_peak_bench.py) — the
    # denominator for MFU. RTX Pro 6000 Blackwell Max-Q measured: 304.
    peak_tflops: float = 304.0


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment: str = "trackfm/pretrain"
    run_name: Optional[str] = None


class PretrainConfig(BaseModel):
    """Config for `trackfm pretrain`."""
    data_dir: Path = Path("~/data/trackfm/materialized/v1")
    checkpoint_dir: Path = Path("~/data/trackfm/checkpoints")
    model: ModelConfig = ModelConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    train: TrainConfig = TrainConfig()
    mlflow: MLflowConfig = MLflowConfig()

    @field_validator("data_dir", "checkpoint_dir")
    @classmethod
    def _expanduser(cls, v: Path) -> Path:
        return _expand(v)


# ------------------------------------------------------------------ loader
def load_config(path: str | Path, model_cls: type[BaseModel]) -> BaseModel:
    """Load a YAML file into the given pydantic config model."""
    with open(_expand(path)) as f:
        raw = yaml.safe_load(f) or {}
    return model_cls.model_validate(raw)
