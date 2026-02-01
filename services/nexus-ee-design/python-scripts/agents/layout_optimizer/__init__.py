"""
Layout Optimizer Agent module.

Optimizes schematic component placement for signal flow,
readability, and DFM compliance following IPC/IEEE standards.
"""

from .layout_optimizer_agent import (
    LayoutOptimizerAgent,
    LayoutConstraint,
    PlacementZone,
    OptimizationResult,
)

__all__ = [
    'LayoutOptimizerAgent',
    'LayoutConstraint',
    'PlacementZone',
    'OptimizationResult',
]
