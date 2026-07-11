"""Model scale presets (HYPERPARAMETERS.md scaling table) and construction."""
from __future__ import annotations

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel

# Scale -> (d_model, nhead, num_layers, dim_feedforward); params ~{1M, 5M, 18M, 116M}
SCALES: dict[str, dict] = {
    "small":  {"d_model": 128, "nhead": 8,  "num_layers": 4,  "dim_feedforward": 512},
    "medium": {"d_model": 256, "nhead": 8,  "num_layers": 6,  "dim_feedforward": 1024},
    "large":  {"d_model": 384, "nhead": 16, "num_layers": 8,  "dim_feedforward": 2048},
    "xlarge": {"d_model": 768, "nhead": 16, "num_layers": 16, "dim_feedforward": 3072},
}


def model_config_for_scale(scale: str, **overrides) -> ModelConfig:
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; choose from {sorted(SCALES)}")
    return ModelConfig(**{**SCALES[scale], **overrides})


def build_model(
    model_cfg: ModelConfig,
    norm_cfg: NormalizationConfig | None = None,
    max_horizon: int = 800,
    num_horizon_samples: int = 4,
) -> CausalAISModel:
    return CausalAISModel(
        model=model_cfg,
        norm=norm_cfg or NormalizationConfig(),
        max_horizon=max_horizon,
        num_horizon_samples=num_horizon_samples,
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
