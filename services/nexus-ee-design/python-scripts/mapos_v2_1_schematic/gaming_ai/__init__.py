"""
MAPO v2.1 Schematic - Gaming AI Module

Provides LLM-guided quality-diversity optimization:
- MAP-Elites: Archive-based exploration with behavioral descriptors
- Red Queen: Adversarial co-evolution for robustness
- Ralph Wiggum: Iterative self-improvement loop

Philosophy: "Opus 4.6 Thinks, Gaming AI Explores, Algorithms Execute, Memory Learns"

Author: Nexus EE Design Team
"""

from .llm_guided_map_elites import LLMGuidedSchematicMAPElites, MAPElitesArchive
from .llm_guided_red_queen import LLMGuidedRedQueen

__all__ = [
    "LLMGuidedSchematicMAPElites",
    "MAPElitesArchive",
    "LLMGuidedRedQueen",
]
