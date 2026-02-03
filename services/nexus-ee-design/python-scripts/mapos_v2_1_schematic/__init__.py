"""
MAPO v2.1 Schematic - LLM-Orchestrated Gaming AI for Schematic Generation

A unified schematic generation system that combines:
1. LLM Orchestration (Opus 4.5 via OpenRouter) for intelligent decision-making
2. Nexus-Memory/GraphRAG for self-improving symbol resolution and design learning
3. Gaming AI (MAP-Elites, Red Queen, Ralph Wiggum) for quality-diversity optimization
4. SPICE Smoke Test to validate schematics work before output
5. Proper Wiring via ConnectionGeneratorAgent + EnhancedWireRouter

Philosophy: "Opus 4.5 Thinks, Gaming AI Explores, Algorithms Execute, Memory Learns"

Usage:
    from mapos_v2_1_schematic import SchematicMAPOOptimizer, SchematicMAPOConfig
    
    config = SchematicMAPOConfig.from_env()
    optimizer = SchematicMAPOOptimizer(config)
    
    result = await optimizer.optimize(
        bom=bom_items,
        design_intent="FOC ESC with STM32G4 and gate drivers",
        design_name="foc_esc_schematic",
    )
    
    if result.success and result.smoke_test_passed:
        print(f"Schematic: {result.schematic_path}")

Author: Nexus EE Design Team
Version: 2.1.0
"""

__version__ = "2.1.0"
__author__ = "Nexus EE Design Team"

# Core data structures
from .core import (
    SchematicState,
    SchematicSolution,
    ComponentInstance,
    Connection,
    Wire,
    WireSegment,
    Junction,
    ValidationResults,
    FitnessScores,
    SchematicMAPOConfig,
)

# Nexus-Memory integration
from .nexus_memory import (
    SymbolMemoryClient,
    SymbolResolution,
    WiringMemoryClient,
    ConnectionPattern,
)

# Validation
from .validation import (
    SmokeTestValidator,
    SmokeTestValidationResult,
)

# Gaming AI
from .gaming_ai import (
    LLMGuidedSchematicMAPElites,
    MAPElitesArchive,
    LLMGuidedRedQueen,
)

# Pipeline components
from .pipeline import (
    MemoryEnhancedSymbolResolver,
    MemoryEnhancedConnectionGenerator,
)

# Main optimizer
from .orchestrator import (
    SchematicMAPOOptimizer,
    OptimizationResult,
)

__all__ = [
    # Version
    "__version__",
    
    # Core
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
    
    # Nexus-Memory
    "SymbolMemoryClient",
    "SymbolResolution",
    "WiringMemoryClient",
    "ConnectionPattern",
    
    # Validation
    "SmokeTestValidator",
    "SmokeTestValidationResult",
    
    # Gaming AI
    "LLMGuidedSchematicMAPElites",
    "MAPElitesArchive",
    "LLMGuidedRedQueen",
    
    # Pipeline
    "MemoryEnhancedSymbolResolver",
    "MemoryEnhancedConnectionGenerator",
    
    # Main optimizer
    "SchematicMAPOOptimizer",
    "OptimizationResult",
]


async def generate_schematic(
    bom: list,
    design_intent: str,
    design_name: str = "schematic",
    design_type: str = "foc_esc",
    config: SchematicMAPOConfig = None,
) -> OptimizationResult:
    """
    Convenience function to generate a schematic.
    
    Args:
        bom: Bill of materials
        design_intent: Natural language design description
        design_name: Name for output file
        design_type: Type of design for pattern matching
        config: Optional configuration
    
    Returns:
        OptimizationResult with final schematic
    
    Example:
        from mapos_v2_1_schematic import generate_schematic
        
        result = await generate_schematic(
            bom=[{"part_number": "STM32G431CBT6", "category": "MCU", ...}],
            design_intent="FOC ESC controller",
        )
    """
    from .core.config import get_config
    
    cfg = config or get_config()
    optimizer = SchematicMAPOOptimizer(cfg)
    
    try:
        return await optimizer.optimize(
            bom=bom,
            design_intent=design_intent,
            design_name=design_name,
            design_type=design_type,
        )
    finally:
        await optimizer.close()
