"""
Visual Verification Agent for MAPO v3.1

Provides visual quality assessment of electronic schematics using Claude Opus 4.5.
"""

from .visual_verifier import (
    VisualVerifier,
    VisualQualityReport,
    QualityIssue,
    VerificationError,
    verify_schematic
)

__all__ = [
    'VisualVerifier',
    'VisualQualityReport',
    'QualityIssue',
    'VerificationError',
    'verify_schematic'
]

__version__ = '1.0.0'
