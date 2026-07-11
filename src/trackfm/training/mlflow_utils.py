"""MLflow run setup and provenance logging."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

import mlflow
from pydantic import BaseModel

from trackfm.config import MLflowConfig

logger = logging.getLogger(__name__)


def _git_info() -> dict[str, str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"git_sha": sha, "git_dirty": str(dirty).lower()}
    except Exception:
        return {"git_sha": "unknown", "git_dirty": "unknown"}


def data_manifest_sha256(data_dir: Path) -> str:
    """Hash of the materialized dataset's MANIFEST.json (or 'missing')."""
    manifest = Path(data_dir).expanduser() / "MANIFEST.json"
    if not manifest.exists():
        return "missing"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


# Config fields that add noise, not information, in the params table:
# mlflow's own settings, and filesystem paths (kept in the config artifact).
_SKIP_PREFIXES = ("mlflow.",)
_SKIP_SUFFIXES = ("_dir", "_path", "backbone", "checkpoint")


def _flatten(cfg: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if key.startswith(_SKIP_PREFIXES) or key.endswith(_SKIP_SUFFIXES):
            continue
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = str(v)
    return out


def start_run(mlf: MLflowConfig, config: BaseModel, data_dir: Path | None = None,
              extra_params: dict | None = None):
    """Start an MLflow run with full provenance: curated flattened params
    (paths/mlflow noise excluded — the full config is attached as an
    artifact), git SHA/dirty and data-manifest tags. Returns the run."""
    mlflow.set_tracking_uri(mlf.tracking_uri)
    mlflow.set_experiment(mlf.experiment)
    run = mlflow.start_run(run_name=mlf.run_name)

    cfg_dict = config.model_dump(mode="json")
    params = _flatten(cfg_dict)
    if extra_params:
        params.update({k: str(v) for k, v in extra_params.items()})
    # MLflow caps params per batch; log in chunks
    items = list(params.items())
    for i in range(0, len(items), 90):
        mlflow.log_params(dict(items[i:i + 90]))

    mlflow.log_text(json.dumps(cfg_dict, indent=2), "config.json")

    tags = _git_info()
    if data_dir is not None:
        tags["data_manifest_sha256"] = data_manifest_sha256(data_dir)
    mlflow.set_tags(tags)
    return run
