"""
Gaming AI for MAPOS - Multi-Agent PCB Optimization System

This package implements cutting-edge gaming AI techniques for PCB optimization:

1. Digital Red Queen (DRQ): Adversarial co-evolution with MAP-Elites
2. Ralph Wiggum Loop: Persistent iteration until success
3. AlphaZero-style: Self-play with policy/value networks
4. MuZero-style: Learned world models for latent planning
5. AlphaFold-inspired: Iterative refinement with recycling

Architecture Overview:
    PCBGraphEncoder -> ValueNetwork -> PolicyNetwork -> DynamicsNetwork
                          |                |
                          v                v
                    MAPElitesArchive <- RedQueenEvolver <- RalphWiggumOptimizer

References:
- Digital Red Queen: https://sakana.ai/drq/
- AlphaZero: https://arxiv.org/abs/1712.01815
- MuZero: https://arxiv.org/abs/1911.08265
- AlphaFold: https://www.nature.com/articles/s41586-021-03819-2

Author: Adverant Inc.
License: MIT
"""

__version__ = "1.0.0"

# Core imports that don't require PyTorch
from .pcb_graph_encoder import PCBGraph, NodeType, EdgeType, GraphNode, GraphEdge
from .map_elites import MAPElitesArchive, BehavioralDescriptor, ArchiveCell, ArchiveStatistics
from .red_queen_evolver import RedQueenEvolver, EvolutionRound, Champion, GeneralityScore
from .ralph_wiggum_optimizer import (
    RalphWiggumOptimizer, CompletionCriteria, OptimizationState,
    OptimizationResult, OptimizationStatus, EscalationStrategy
)
from .training import ExperienceBuffer, Experience
from .mapos_bridge import GamingAIMultiAgentOptimizer, GamingAIConfig, GamingAIResult

# Conditional imports for PyTorch-dependent modules
try:
    import torch
    TORCH_AVAILABLE = True

    from .pcb_graph_encoder import PCBGraphEncoder
    from .value_network import ValueNetwork, ValuePrediction, ValueTrainingSample
    from .policy_network import (
        PolicyNetwork, ModificationHead, PolicyOutput,
        ModificationCategory, DRCContextEncoder
    )
    from .dynamics_network import DynamicsNetwork, WorldModel, DynamicsPrediction
    from .training import TrainingPipeline
    from .integration import MAPOSRQOptimizer, MAPOSRQConfig, MAPOSRQResult, optimize_pcb

except ImportError:
    TORCH_AVAILABLE = False

    # Provide fallback classes
    from .pcb_graph_encoder import PCBGraphEncoder  # Has fallback
    from .value_network import ValueNetwork  # Has fallback
    from .policy_network import PolicyNetwork, ModificationCategory, DRCContextEncoder  # Has fallback
    from .dynamics_network import WorldModel  # Has fallback
    from .integration import MAPOSRQOptimizer, MAPOSRQConfig, MAPOSRQResult, optimize_pcb

    # Placeholders
    class TrainingPipeline:
        """PyTorch required for training."""
        pass

__all__ = [
    # Version
    "__version__",
    "TORCH_AVAILABLE",

    # Graph representation
    "PCBGraph",
    "PCBGraphEncoder",
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",

    # Neural Networks
    "ValueNetwork",
    "PolicyNetwork",
    "ModificationHead",
    "WorldModel",
    "DynamicsNetwork",
    "ModificationCategory",
    "DRCContextEncoder",

    # Quality-Diversity
    "MAPElitesArchive",
    "BehavioralDescriptor",
    "ArchiveCell",
    "ArchiveStatistics",

    # Evolution
    "RedQueenEvolver",
    "EvolutionRound",
    "Champion",
    "GeneralityScore",

    # Ralph Wiggum
    "RalphWiggumOptimizer",
    "CompletionCriteria",
    "OptimizationState",
    "OptimizationResult",
    "OptimizationStatus",
    "EscalationStrategy",

    # Training
    "TrainingPipeline",
    "ExperienceBuffer",
    "Experience",

    # Integration
    "MAPOSRQOptimizer",
    "MAPOSRQConfig",
    "MAPOSRQResult",
    "optimize_pcb",

    # MAPOS Bridge
    "GamingAIMultiAgentOptimizer",
    "GamingAIConfig",
    "GamingAIResult",
]
