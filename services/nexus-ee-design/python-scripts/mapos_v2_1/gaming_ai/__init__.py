"""
Gaming AI for MAPOS - Multi-Agent PCB Optimization System

This package implements cutting-edge gaming AI techniques for PCB optimization:

1. Digital Red Queen (DRQ): Adversarial co-evolution with MAP-Elites
2. Ralph Wiggum Loop: Persistent iteration until success
3. AlphaZero-style: Self-play with policy/value networks (optional)
4. MuZero-style: Learned world models for latent planning (optional)
5. AlphaFold-inspired: Iterative refinement with recycling
6. LLM-First Mode: OpenRouter LLM backends (default, no PyTorch needed)

Architecture Overview (LLM-First Mode):
    LLMStateEncoder -> LLMValueEstimator -> LLMPolicyGenerator
                          |                      |
                          v                      v
                    MAPElitesArchive <- RedQueenEvolver <- RalphWiggumOptimizer

Architecture Overview (Neural Network Mode - Optional):
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

__version__ = "2.0.0"  # LLM-first version

# Configuration (always available)
from .config import (
    GamingAIConfig as FullGamingAIConfig,
    OptimizationMode,
    InferenceProvider,
    LLMConfig,
    DRCConfig,
    EvolutionConfig,
    RalphWiggumConfig,
    BoardConfig,
    NeuralNetworkConfig,
    get_config,
)

# LLM Backends (always available, no PyTorch needed)
from .llm_backends import (
    LLMClient,
    LLMStateEncoder,
    LLMValueEstimator,
    LLMPolicyGenerator,
    LLMDynamicsSimulator,
    StateEmbedding,
    ValueEstimate,
    ModificationSuggestion,
    DynamicsPrediction as LLMDynamicsPrediction,
    get_llm_backends,
)

# Optional GPU backends
from .optional_gpu_backend import (
    OptionalGPUInference,
    GPUBackend,
    RunPodBackend,
    ModalBackend,
    ReplicateBackend,
    get_inference_backend,
)

# Core imports that don't require PyTorch
from .pcb_graph_encoder import PCBGraph, NodeType, EdgeType, GraphNode, GraphEdge
from .map_elites import MAPElitesArchive, BehavioralDescriptor, ArchiveCell, ArchiveStatistics
from .red_queen_evolver import RedQueenEvolver, EvolutionRound, Champion, GeneralityScore
from .ralph_wiggum_optimizer import (
    RalphWiggumOptimizer, CompletionCriteria, OptimizationState,
    OptimizationResult, OptimizationStatus, EscalationStrategy,
    file_lock, atomic_write_json,
)
from .training import ExperienceBuffer, Experience
from .mapos_bridge import GamingAIMultiAgentOptimizer, GamingAIConfig, GamingAIResult

# Integration (works with or without PyTorch)
from .integration import MAPOSRQOptimizer, MAPOSRQConfig, MAPOSRQResult, optimize_pcb

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

except ImportError:
    TORCH_AVAILABLE = False

    # Provide fallback classes
    from .pcb_graph_encoder import PCBGraphEncoder  # Has fallback
    from .value_network import ValueNetwork  # Has fallback
    from .policy_network import PolicyNetwork, ModificationCategory, DRCContextEncoder  # Has fallback
    from .dynamics_network import WorldModel  # Has fallback

    # Placeholders
    class TrainingPipeline:
        """PyTorch required for training."""
        pass

    class DynamicsPrediction:
        """PyTorch required for dynamics prediction."""
        pass

__all__ = [
    # Version
    "__version__",
    "TORCH_AVAILABLE",

    # Configuration
    "FullGamingAIConfig",
    "OptimizationMode",
    "InferenceProvider",
    "LLMConfig",
    "DRCConfig",
    "EvolutionConfig",
    "RalphWiggumConfig",
    "BoardConfig",
    "NeuralNetworkConfig",
    "get_config",

    # LLM Backends (primary - no PyTorch needed)
    "LLMClient",
    "LLMStateEncoder",
    "LLMValueEstimator",
    "LLMPolicyGenerator",
    "LLMDynamicsSimulator",
    "StateEmbedding",
    "ValueEstimate",
    "ModificationSuggestion",
    "LLMDynamicsPrediction",
    "get_llm_backends",

    # Optional GPU Backends
    "OptionalGPUInference",
    "GPUBackend",
    "RunPodBackend",
    "ModalBackend",
    "ReplicateBackend",
    "get_inference_backend",

    # Graph representation
    "PCBGraph",
    "PCBGraphEncoder",
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",

    # Neural Networks (optional)
    "ValueNetwork",
    "PolicyNetwork",
    "ModificationHead",
    "WorldModel",
    "DynamicsNetwork",
    "DynamicsPrediction",
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
    "file_lock",
    "atomic_write_json",

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
