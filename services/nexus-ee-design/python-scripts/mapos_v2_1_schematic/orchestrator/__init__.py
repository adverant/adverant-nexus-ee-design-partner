"""
MAPO v2.1 Schematic - Orchestrator Module

Main optimizer class that coordinates all components:
- Symbol Resolution (memory-enhanced)
- Connection Generation (memory-enhanced)
- Wire Routing
- Smoke Test Validation
- Gaming AI Optimization

Author: Nexus EE Design Team
"""

from .schematic_mapo_optimizer import SchematicMAPOOptimizer, OptimizationResult

__all__ = [
    "SchematicMAPOOptimizer",
    "OptimizationResult",
]
