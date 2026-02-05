"""
Ideation Artifact Generator Agent module.

LLM-powered generation of pre-schematic design documentation including:
- System overviews and executive summaries
- Schematic specifications per subsystem
- Architecture diagrams (Mermaid format)
- Bill of Materials with component selection rationale
- Design calculations and trade-off analysis
"""

from .artifact_generator import (
    IdeationArtifactGenerator,
    ArtifactType,
    ArtifactCategory,
    GeneratedArtifact,
    GenerationConfig,
    ARTIFACT_PROMPTS,
)

__all__ = [
    'IdeationArtifactGenerator',
    'ArtifactType',
    'ArtifactCategory',
    'GeneratedArtifact',
    'GenerationConfig',
    'ARTIFACT_PROMPTS',
]
