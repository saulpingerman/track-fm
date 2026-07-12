"""Cross-file track continuity state management (local FS).

Handles maintaining track state across multiple files to ensure:
- Track IDs are consistent for vessels spanning multiple days
- MMSI collision assignments are preserved
- Segment numbers continue correctly
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_FILENAME = "track_continuity.json"


@dataclass
class MMSIState:
    """State for a single MMSI."""
    last_position: Tuple[float, float]
    last_timestamp: str
    current_segment: int
    cluster_assignment: Optional[str] = None


@dataclass
class CollisionRegistryEntry:
    """Registry entry for a detected MMSI collision."""
    detected_date: str
    cluster_a_centroid: Tuple[float, float]
    cluster_b_centroid: Tuple[float, float]


@dataclass
class TrackContinuityState:
    """Complete state for track continuity across files."""
    last_updated: str = ""
    last_file_processed: str = ""
    mmsi_state: Dict[int, MMSIState] = field(default_factory=dict)
    collision_registry: Dict[int, CollisionRegistryEntry] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_updated": self.last_updated,
            "last_file_processed": self.last_file_processed,
            "mmsi_state": {
                str(k): {
                    "last_position": list(v.last_position),
                    "last_timestamp": v.last_timestamp,
                    "current_segment": v.current_segment,
                    "cluster_assignment": v.cluster_assignment,
                }
                for k, v in self.mmsi_state.items()
            },
            "collision_registry": {
                str(k): {
                    "detected_date": v.detected_date,
                    "cluster_a_centroid": list(v.cluster_a_centroid),
                    "cluster_b_centroid": list(v.cluster_b_centroid),
                }
                for k, v in self.collision_registry.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackContinuityState":
        state = cls(
            last_updated=data.get("last_updated", ""),
            last_file_processed=data.get("last_file_processed", ""),
        )

        for mmsi_str, mmsi_data in data.get("mmsi_state", {}).items():
            mmsi = int(mmsi_str)
            state.mmsi_state[mmsi] = MMSIState(
                last_position=tuple(mmsi_data["last_position"]),
                last_timestamp=mmsi_data["last_timestamp"],
                current_segment=mmsi_data["current_segment"],
                cluster_assignment=mmsi_data.get("cluster_assignment"),
            )

        for mmsi_str, collision_data in data.get("collision_registry", {}).items():
            mmsi = int(mmsi_str)
            state.collision_registry[mmsi] = CollisionRegistryEntry(
                detected_date=collision_data["detected_date"],
                cluster_a_centroid=tuple(collision_data["cluster_a_centroid"]),
                cluster_b_centroid=tuple(collision_data["cluster_b_centroid"]),
            )

        return state

    def get_starting_segment(self, mmsi: int, first_timestamp: datetime, gap_hours: float) -> int:
        if mmsi not in self.mmsi_state:
            return 0

        mmsi_state = self.mmsi_state[mmsi]
        last_ts = datetime.fromisoformat(mmsi_state.last_timestamp)
        time_gap_hours = (first_timestamp - last_ts).total_seconds() / 3600

        if time_gap_hours <= gap_hours:
            return mmsi_state.current_segment
        return mmsi_state.current_segment + 1

    def get_cluster_assignment(self, mmsi: int) -> Optional[str]:
        if mmsi in self.mmsi_state and self.mmsi_state[mmsi].cluster_assignment:
            return self.mmsi_state[mmsi].cluster_assignment
        return None

    def is_known_collision(self, mmsi: int) -> bool:
        return mmsi in self.collision_registry

    def get_collision_centroids(
        self, mmsi: int
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if mmsi in self.collision_registry:
            entry = self.collision_registry[mmsi]
            return (entry.cluster_a_centroid, entry.cluster_b_centroid)
        return None

    def update_mmsi_state(
        self,
        mmsi: int,
        last_position: Tuple[float, float],
        last_timestamp: datetime,
        current_segment: int,
        cluster_assignment: Optional[str] = None,
    ) -> None:
        self.mmsi_state[mmsi] = MMSIState(
            last_position=last_position,
            last_timestamp=last_timestamp.isoformat(),
            current_segment=current_segment,
            cluster_assignment=cluster_assignment,
        )

    def register_collision(
        self,
        mmsi: int,
        centroid_a: Tuple[float, float],
        centroid_b: Tuple[float, float],
        detected_date: str,
    ) -> None:
        self.collision_registry[mmsi] = CollisionRegistryEntry(
            detected_date=detected_date,
            cluster_a_centroid=centroid_a,
            cluster_b_centroid=centroid_b,
        )


def load_state(state_dir: Path) -> TrackContinuityState:
    """Load continuity state from <state_dir>/track_continuity.json, or return a fresh one."""
    state_dir = Path(state_dir)
    state_path = state_dir / STATE_FILENAME

    if not state_path.exists():
        logger.info("No existing state found, starting fresh")
        return TrackContinuityState()

    with open(state_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded state from {state_path}")
    return TrackContinuityState.from_dict(data)


def save_state(state: TrackContinuityState, state_dir: Path) -> None:
    """Persist state atomically (write to .tmp then rename)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / STATE_FILENAME
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")

    state.last_updated = datetime.utcnow().isoformat()
    with open(tmp_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    tmp_path.replace(state_path)
    logger.info(f"Saved state to {state_path}")
