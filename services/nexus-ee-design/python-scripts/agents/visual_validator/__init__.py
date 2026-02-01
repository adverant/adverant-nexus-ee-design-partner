"""
Dual-LLM Visual Validator - Image-based schematic validation using Opus 4.5 and Kimi K2.5.

CRITICAL: This validator exports the actual schematic to PNG/PDF and analyzes the
real visual output - NOT mathematical analysis of the S-expression.

Features:
- Exports schematic to high-resolution PNG using KiCad CLI
- Sends rendered image to Claude Opus 4.5 (ultrathinking mode)
- Sends same image to Kimi K2.5 for independent analysis
- Compares results and calculates agreement score
- Loops until compliance target is met
"""

from .dual_llm_validator import (
    DualLLMVisualValidator,
    VisualAnalysis,
    ComparisonResult,
    ValidationLoop,
    export_schematic_to_image,
)

__all__ = [
    "DualLLMVisualValidator",
    "VisualAnalysis",
    "ComparisonResult",
    "ValidationLoop",
    "export_schematic_to_image",
]
