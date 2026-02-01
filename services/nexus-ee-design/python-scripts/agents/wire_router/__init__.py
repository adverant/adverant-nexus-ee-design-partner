"""
Enhanced Wire Router Agent - Professional-grade Manhattan routing for KiCad schematics.

Implements industry best practices:
- Avoids 4-way junctions (IPC recommendation)
- Bus routing for parallel signals
- Power rail optimization
- Crossing minimization
- Wire length optimization for critical paths
"""

from .enhanced_wire_router import (
    EnhancedWireRouter,
    WireSegment,
    RoutingConstraint,
    RoutingResult,
    BusRoute,
)

__all__ = [
    "EnhancedWireRouter",
    "WireSegment",
    "RoutingConstraint",
    "RoutingResult",
    "BusRoute",
]
