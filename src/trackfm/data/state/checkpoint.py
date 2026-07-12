"""Processing checkpoint management for resumable processing (local FS)."""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "processing_checkpoint.json"


@dataclass
class ProcessingCheckpoint:
    """Checkpoint for resumable processing."""
    last_processed_file: str = ""
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    last_updated: str = ""
    processing_started: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_processed_file": self.last_processed_file,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "last_updated": self.last_updated,
            "processing_started": self.processing_started,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingCheckpoint":
        return cls(
            last_processed_file=data.get("last_processed_file", ""),
            processed_files=data.get("processed_files", []),
            failed_files=data.get("failed_files", []),
            last_updated=data.get("last_updated", ""),
            processing_started=data.get("processing_started", ""),
            stats=data.get("stats", {}),
        )

    def mark_processed(self, filename: str) -> None:
        if filename not in self.processed_files:
            self.processed_files.append(filename)
        self.last_processed_file = filename
        self.last_updated = datetime.utcnow().isoformat()

    def mark_failed(self, filename: str) -> None:
        if filename not in self.failed_files:
            self.failed_files.append(filename)
        self.last_updated = datetime.utcnow().isoformat()

    def get_pending_files(self, all_files: List[str]) -> List[str]:
        processed_set = set(self.processed_files)
        return [f for f in all_files if f not in processed_set]

    def update_stats(self, key: str, value: Any) -> None:
        self.stats[key] = value


def load_checkpoint(state_dir: Path) -> ProcessingCheckpoint:
    """Load checkpoint from <state_dir>/processing_checkpoint.json, or return a fresh one."""
    state_dir = Path(state_dir)
    checkpoint_path = state_dir / CHECKPOINT_FILENAME

    if not checkpoint_path.exists():
        logger.info("No existing checkpoint found, starting fresh")
        checkpoint = ProcessingCheckpoint()
        checkpoint.processing_started = datetime.utcnow().isoformat()
        return checkpoint

    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return ProcessingCheckpoint.from_dict(data)


def save_checkpoint(checkpoint: ProcessingCheckpoint, state_dir: Path) -> None:
    """Persist checkpoint atomically (write to .tmp then rename)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_dir / CHECKPOINT_FILENAME
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    checkpoint.last_updated = datetime.utcnow().isoformat()
    with open(tmp_path, "w") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)
    tmp_path.replace(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
