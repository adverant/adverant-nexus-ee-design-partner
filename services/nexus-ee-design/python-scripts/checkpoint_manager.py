"""
Checkpoint Manager for MAPO Schematic Pipeline.

Saves and loads pipeline state to disk so that interrupted runs can be
resumed from the last completed phase rather than restarting from scratch.

Checkpoint location (in order of preference):
  1. CHECKPOINT_DIR env var (explicit override)
  2. /data/artifacts/_checkpoints (PVC — shared with Terminal Computer, survives restarts)
  3. /tmp/nexus-schematic-checkpoints (ephemeral fallback)
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _resolve_checkpoint_base() -> Path:
    """Resolve checkpoint directory, preferring NFS over /tmp."""
    # 1. Explicit env override
    env_dir = os.environ.get("CHECKPOINT_DIR")
    if env_dir:
        return Path(env_dir)

    # 2. PVC mount (shared with Terminal Computer via nexus-user-workspaces PVC)
    # EE Design pod: /data/artifacts → Terminal Computer: /workspaces/ee-design/artifacts
    pvc_path = Path("/data/artifacts/_checkpoints")
    if pvc_path.parent.exists():
        pvc_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using PVC checkpoint storage: {pvc_path}")
        return pvc_path

    # 3. Ephemeral fallback
    fallback = Path("/tmp/nexus-schematic-checkpoints")
    logger.warning(f"NFS not available, using ephemeral checkpoint storage: {fallback}")
    return fallback


CHECKPOINT_BASE = _resolve_checkpoint_base()


class CheckpointManager:
    """Manages checkpoint persistence for schematic pipeline phases."""

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        self.checkpoint_dir = CHECKPOINT_BASE / operation_id
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"

    def save_checkpoint(
        self,
        phase: str,
        data: Dict[str, Any],
        completed_phases: Optional[List[str]] = None,
    ) -> Path:
        """
        Save a checkpoint after a pipeline phase completes.

        Args:
            phase: Name of the phase that just completed (e.g. "connections", "assembly").
            data: Serialisable dict with all outputs from the phase.
            completed_phases: Cumulative list of phases finished so far.

        Returns:
            Path to the written checkpoint file.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "operation_id": self.operation_id,
            "phase": phase,
            "completed_phases": completed_phases or [phase],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        # Atomic write: write to tmp then rename
        tmp_path = self.checkpoint_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(checkpoint, indent=2, default=str))
        tmp_path.rename(self.checkpoint_file)

        logger.info(
            f"Checkpoint saved: phase={phase}, "
            f"completed={completed_phases}, "
            f"path={self.checkpoint_file}"
        )
        return self.checkpoint_file

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the last saved checkpoint for this operation.

        Returns:
            Checkpoint dict or None if no checkpoint exists.
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            checkpoint = json.loads(self.checkpoint_file.read_text())
            logger.info(
                f"Checkpoint loaded: phase={checkpoint.get('phase')}, "
                f"completed={checkpoint.get('completed_phases')}"
            )
            return checkpoint
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load checkpoint: {exc}")
            return None

    def has_phase(self, phase: str) -> bool:
        """Check if a specific phase was completed in the checkpoint."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        return phase in checkpoint.get("completed_phases", [])

    def get_phase_data(self, phase: str) -> Optional[Dict[str, Any]]:
        """
        Get the data saved for a specific completed phase.

        The checkpoint stores the data from the most recent save_checkpoint() call.
        Phase-specific data is stored under the "data" key.
        """
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return None
        if phase not in checkpoint.get("completed_phases", []):
            return None
        return checkpoint.get("data")

    def increment_resume_count(self) -> int:
        """Increment and return the number of resume attempts for this operation."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return 0
        count = checkpoint.get("resume_count", 0) + 1
        checkpoint["resume_count"] = count
        # Atomic write
        tmp_path = self.checkpoint_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(checkpoint, indent=2, default=str))
        tmp_path.rename(self.checkpoint_file)
        return count

    def cleanup(self) -> None:
        """Remove checkpoint directory for this operation."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
            logger.info(f"Checkpoint cleaned up: {self.checkpoint_dir}")
