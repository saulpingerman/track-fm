"""State management for cross-file track continuity."""
from .continuity import TrackContinuityState, load_state, save_state
from .checkpoint import ProcessingCheckpoint

__all__ = [
    "TrackContinuityState",
    "load_state",
    "save_state",
    "ProcessingCheckpoint",
]
