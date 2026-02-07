"""
Validation module for Nexus EE Design.

Contains validators for schematic and PCB quality assurance:
- Vision-based validators (LLM-powered)
- S-expression validators (deterministic AST parsing)
- Legacy regex-based validators
"""

from .schematic_vision_validator import (
    SchematicVisionValidator,
    MAPOSchematicLoop,
    ExpertPersona,
    ValidationIssue,
    FixRecommendation,
    ExpertValidationResult,
    SchematicValidationReport,
)

# Try to import enhanced S-expression validator (requires sexpdata)
try:
    from .sexp_validator import (
        SExpValidator,
        validate_schematic_file,
        validate_schematic_content,
        KiCadSchematic,
        KiCadCoordinate,
        KiCadUUID,
        KiCadSymbolInstance,
        KiCadWire,
        KiCadJunction,
        KiCadLabel,
        ValidationSeverity,
        SExpValidationResult,
        SEXPDATA_AVAILABLE,
    )
    SEXP_VALIDATOR_AVAILABLE = True
except ImportError:
    SEXP_VALIDATOR_AVAILABLE = False
    # Provide stub for type checking
    SExpValidator = None
    validate_schematic_file = None
    validate_schematic_content = None

# Legacy regex-based validator (always available)
from .sexpression_validator import (
    SExpressionValidator,
    SExpressionValidationReport,
)

__all__ = [
    # Vision-based validators
    'SchematicVisionValidator',
    'MAPOSchematicLoop',
    'ExpertPersona',
    'ValidationIssue',
    'FixRecommendation',
    'ExpertValidationResult',
    'SchematicValidationReport',
    # Enhanced S-expression validator (AST-based)
    'SExpValidator',
    'validate_schematic_file',
    'validate_schematic_content',
    'KiCadSchematic',
    'KiCadCoordinate',
    'KiCadUUID',
    'KiCadSymbolInstance',
    'KiCadWire',
    'KiCadJunction',
    'KiCadLabel',
    'ValidationSeverity',
    'SExpValidationResult',
    'SEXP_VALIDATOR_AVAILABLE',
    # Legacy regex-based validator
    'SExpressionValidator',
    'SExpressionValidationReport',
]
