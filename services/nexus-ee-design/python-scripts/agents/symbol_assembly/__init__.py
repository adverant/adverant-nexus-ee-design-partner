"""
Symbol Assembly Agent module.

Pre-generation symbol, datasheet, and characterization gathering
with GraphRAG-first search and WebSocket progress streaming.
"""

from .symbol_assembler import (
    AssemblyPhase,
    AssemblyEventType,
    AssemblyProgressEmitter,
    AssemblyReport,
    ComponentGatherResult,
    ComponentRequirement,
    SymbolAssembler,
    emit_progress,
)

__all__ = [
    "AssemblyPhase",
    "AssemblyEventType",
    "AssemblyProgressEmitter",
    "AssemblyReport",
    "ComponentGatherResult",
    "ComponentRequirement",
    "SymbolAssembler",
    "emit_progress",
]
