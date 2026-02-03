"""
MAPO v2.1 Schematic Core Data Structures

Provides immutable state management and configuration for the
LLM-orchestrated Gaming AI schematic generation pipeline.
"""

from .schematic_state import (
    SchematicState,
    SchematicSolution,
    ComponentInstance,
    Connection,
    Wire,
    WireSegment,
    Junction,
    ValidationResults,
    FitnessScores,
)
from .config import SchematicMAPOConfig

__all__ = [
    "SchematicState",
    "SchematicSolution",
    "ComponentInstance",
    "Connection",
    "Wire",
    "WireSegment",
    "Junction",
    "ValidationResults",
    "FitnessScores",
    "SchematicMAPOConfig",
]
