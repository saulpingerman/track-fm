#!/usr/bin/env python
"""
Configuration management for TrackFM
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    seq_len: int = 20
    horizon: int = 10
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    ff_hidden: int = 512
    fourier_m: int = 128
    fourier_rank: int = 4
    max_horizon: int = 10
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training parameters configuration"""
    batch_size: int = 1024
    lr: float = 1e-4
    epochs: int = 50
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    ckpt_every: int = 1000


@dataclass
class DataConfig:
    """Data loading configuration"""
    bucket_name: str = "ais-pipeline-data-10179bbf-us-east-1"
    chunk_size: int = 200000
    prefix: str = "cleaned/"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class PathsConfig:
    """File paths configuration"""
    ckpt_dir: str = "checkpoints"
    model_tag: str = "trackfm_multihorizon_128f"
    warmup_checkpoint: Optional[str] = None


@dataclass
class SystemConfig:
    """System configuration"""
    device: str = "auto"
    mixed_precision: bool = False
    compile_model: bool = False


@dataclass
class WandBConfig:
    """Weights & Biases configuration"""
    enabled: bool = False
    project: str = "trackfm"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["multihorizon", "fourier", "transformer"])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_every: int = 10
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class TrackFMConfig:
    """Complete TrackFM configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrackFMConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        return cls.from_dict(yaml_data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackFMConfig":
        """Create configuration from dictionary"""
        
        # Handle nested configs
        model_config = ModelConfig(**data.get('model', {}))
        training_config = TrainingConfig(**data.get('training', {}))
        data_config = DataConfig(**data.get('data', {}))
        paths_config = PathsConfig(**data.get('paths', {}))
        system_config = SystemConfig(**data.get('system', {}))
        
        # Handle nested logging config
        logging_data = data.get('logging', {})
        wandb_config = WandBConfig(**logging_data.get('wandb', {}))
        logging_config = LoggingConfig(
            log_every=logging_data.get('log_every', 10),
            wandb=wandb_config
        )
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            paths=paths_config,
            system=system_config,
            logging=logging_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
        
        return dataclass_to_dict(self)
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def override(self, overrides: Dict[str, Any]) -> "TrackFMConfig":
        """Override configuration with new values using dot notation"""
        data = self.to_dict()
        
        for key, value in overrides.items():
            # Support dot notation: "model.lr" -> data["model"]["lr"]
            keys = key.split('.')
            current = data
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return self.from_dict(data)


def load_config(config_path: str = None, overrides: Dict[str, Any] = None) -> TrackFMConfig:
    """
    Load configuration with optional overrides
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
        overrides: Dictionary of configuration overrides using dot notation
        
    Returns:
        TrackFMConfig object
        
    Example:
        config = load_config(
            "config/experiment.yaml",
            overrides={
                "training.lr": 2e-4,
                "model.fourier_m": 256,
                "training.batch_size": 512
            }
        )
    """
    
    if config_path is None:
        # Use default configuration
        config = TrackFMConfig()
    else:
        config = TrackFMConfig.from_yaml(config_path)
    
    if overrides:
        config = config.override(overrides)
    
    return config


def create_experiment_config(base_config: str, experiment_name: str, 
                           overrides: Dict[str, Any]) -> str:
    """
    Create a new experiment configuration file
    
    Args:
        base_config: Path to base configuration
        experiment_name: Name for the new experiment
        overrides: Configuration overrides
        
    Returns:
        Path to the new experiment config file
    """
    
    # Load base config
    config = load_config(base_config, overrides)
    
    # Update model tag for experiment
    config.paths.model_tag = f"trackfm_{experiment_name}"
    
    # Save experiment config
    experiment_path = f"config/experiments/{experiment_name}.yaml"
    config.save_yaml(experiment_path)
    
    return experiment_path