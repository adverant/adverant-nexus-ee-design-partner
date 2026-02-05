"""
Enhanced Visual Validator - Image-based schematic validation with visual feedback loop.

Features:
- Extracts PNG images via kicad-worker (NO fallbacks)
- Opus 4.5 ultrathinking for visual analysis with progress context
- Progress tracking with stagnation detection
- Automatic issue-to-fix transformation
- S-expression modification for fix application

NO FALLBACKS - Strict error handling with verbose reporting.

Author: Nexus EE Design Team
"""

from .dual_llm_validator import (
    DualLLMVisualValidator,
    VisualAnalysis,
    VisualIssue,
    VisualIssueCategory,
    IssueSeverity,
    ComparisonResult,
    ValidationLoop,
    ValidationLoopResult,
    export_schematic_to_image,
)

from .image_extractor import (
    SchematicImageExtractor,
    ImageExtractionResult,
    ImageExtractionError,
)

from .progress_tracker import (
    ProgressTracker,
    ProgressSnapshot,
    ProgressAnalysis,
    StagnationRisk,
    StagnationReason,
    StagnationError,
)

from .issue_to_fix import (
    IssueToFixTransformer,
    SchematicFix,
    FixCategory,
    FixOperation,
    TransformResult,
    FixTransformError,
)

from .fix_applicator import (
    SchematicFixApplicator,
    ApplyResult,
    FixApplicationError,
)

__all__ = [
    # Core validator
    "DualLLMVisualValidator",
    "VisualAnalysis",
    "VisualIssue",
    "VisualIssueCategory",
    "IssueSeverity",
    "ComparisonResult",
    "ValidationLoop",
    "ValidationLoopResult",
    "export_schematic_to_image",
    # Image extraction
    "SchematicImageExtractor",
    "ImageExtractionResult",
    "ImageExtractionError",
    # Progress tracking
    "ProgressTracker",
    "ProgressSnapshot",
    "ProgressAnalysis",
    "StagnationRisk",
    "StagnationReason",
    "StagnationError",
    # Issue-to-fix transformation
    "IssueToFixTransformer",
    "SchematicFix",
    "FixCategory",
    "FixOperation",
    "TransformResult",
    "FixTransformError",
    # Fix application
    "SchematicFixApplicator",
    "ApplyResult",
    "FixApplicationError",
]
