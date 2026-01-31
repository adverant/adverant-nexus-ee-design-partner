"""
Schematic Assembler Agent module.

Assembles KiCad schematics using real symbols with intelligent placement.
"""

from .assembler_agent import (
    SchematicAssemblerAgent,
    SchematicSheet,
    SymbolInstance,
    BOMItem,
    Connection,
    BlockDiagram,
    Wire,
    Label,
    Junction,
    Pin,
    PinType,
)

__all__ = [
    'SchematicAssemblerAgent',
    'SchematicSheet',
    'SymbolInstance',
    'BOMItem',
    'Connection',
    'BlockDiagram',
    'Wire',
    'Label',
    'Junction',
    'Pin',
    'PinType',
]
