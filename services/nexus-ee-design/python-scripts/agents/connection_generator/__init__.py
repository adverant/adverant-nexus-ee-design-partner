"""
Connection Generator Agent module.

Infers circuit connections from BOM, design intent, and component knowledge.
"""

from .connection_generator_agent import (
    ConnectionGeneratorAgent,
    GeneratedConnection,
    ConnectionType,
)

__all__ = [
    'ConnectionGeneratorAgent',
    'GeneratedConnection',
    'ConnectionType',
]
