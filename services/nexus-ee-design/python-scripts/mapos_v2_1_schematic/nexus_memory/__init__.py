"""
MAPO v2.1 Schematic - Nexus-Memory Integration

Provides GraphRAG-powered memory for:
- Symbol resolution learning (remember successful symbol sources)
- Wiring pattern learning (remember successful connection patterns)
- Design pattern recall (leverage past designs for new schematics)

Author: Nexus EE Design Team
"""

from .symbol_memory import SymbolMemoryClient, SymbolResolution
from .wiring_memory import WiringMemoryClient, ConnectionPattern

__all__ = [
    "SymbolMemoryClient",
    "SymbolResolution",
    "WiringMemoryClient",
    "ConnectionPattern",
]
