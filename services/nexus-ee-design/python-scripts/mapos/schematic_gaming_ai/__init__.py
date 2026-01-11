"""
Schematic Gaming AI - Quality-Diversity Optimization for Electronic Schematics

This package extends the MAPOS Gaming AI approach to schematic generation and optimization:

1. Schematic Behavioral Descriptors - 10D feature space characterizing design strategies
2. Multi-objective Fitness Functions - ERC, cost, reliability, manufacturability
3. MAP-Elites Archive - Maintain diverse high-quality schematic solutions
4. Red Queen Evolution - Adversarial co-evolution for schematic improvement
5. Ralph Wiggum Loop - Persistent iteration until target fitness achieved

Architecture Overview:
    SchematicEncoder -> SchematicFitness -> SchematicMAPElites
                            |                      |
                            v                      v
                    MutationOperators <- SchematicRedQueen <- SchematicRalphWiggum

The system integrates with:
- SchematicGenerator (TypeScript) for initial schematic generation
- SchematicReviewer (TypeScript) for ERC validation
- LLM backends (OpenRouter) for intelligent mutations
- Git for schematic history tracking

References:
- Gaming AI PCB: ../gaming_ai/
- MAP-Elites: https://arxiv.org/abs/1504.04909
- Digital Red Queen: https://arxiv.org/abs/2601.03335

Author: Adverant Inc.
License: MIT
"""

__version__ = "1.0.0"

# Re-use core gaming AI infrastructure
from ..gaming_ai.config import (
    GamingAIConfig as BaseGamingAIConfig,
    OptimizationMode,
    InferenceProvider,
    LLMConfig,
    get_config as get_base_config,
)

from ..gaming_ai.llm_backends import (
    LLMClient,
    get_llm_backends,
)

from ..gaming_ai.ralph_wiggum_optimizer import (
    file_lock,
    atomic_write_json,
    OptimizationStatus,
    EscalationStrategy,
)

# Schematic-specific components
from .behavior_descriptor import (
    SchematicBehaviorDescriptor,
    compute_schematic_descriptor,
)

from .fitness_function import (
    SchematicFitness,
    compute_schematic_fitness,
    FitnessWeights,
)

from .mutation_operators import (
    MutationStrategy,
    SchematicMutator,
    apply_mutation,
)

from .schematic_map_elites import (
    SchematicMAPElitesArchive,
    SchematicArchiveCell,
    SchematicArchiveStatistics,
)

from .schematic_red_queen import (
    SchematicRedQueenEvolver,
    SchematicEvolutionRound,
    SchematicChampion,
)

from .schematic_ralph_wiggum import (
    SchematicRalphWiggumOptimizer,
    SchematicCompletionCriteria,
    SchematicOptimizationState,
    SchematicOptimizationResult,
)

from .config import (
    SchematicGamingAIConfig,
    SchematicOptimizationMode,
    get_schematic_config,
)

from .integration import (
    optimize_schematic,
    SchematicOptimizer,
    SchematicOptimizationRequest,
    SchematicOptimizationResponse,
)

__all__ = [
    # Version
    "__version__",

    # Base config (re-exported)
    "BaseGamingAIConfig",
    "OptimizationMode",
    "InferenceProvider",
    "LLMConfig",
    "get_base_config",

    # LLM (re-exported)
    "LLMClient",
    "get_llm_backends",

    # File operations (re-exported)
    "file_lock",
    "atomic_write_json",
    "OptimizationStatus",
    "EscalationStrategy",

    # Schematic-specific
    "SchematicBehaviorDescriptor",
    "compute_schematic_descriptor",
    "SchematicFitness",
    "compute_schematic_fitness",
    "FitnessWeights",
    "MutationStrategy",
    "SchematicMutator",
    "apply_mutation",
    "SchematicMAPElitesArchive",
    "SchematicArchiveCell",
    "SchematicArchiveStatistics",
    "SchematicRedQueenEvolver",
    "SchematicEvolutionRound",
    "SchematicChampion",
    "SchematicRalphWiggumOptimizer",
    "SchematicCompletionCriteria",
    "SchematicOptimizationState",
    "SchematicOptimizationResult",

    # Configuration
    "SchematicGamingAIConfig",
    "SchematicOptimizationMode",
    "get_schematic_config",

    # Integration
    "optimize_schematic",
    "SchematicOptimizer",
    "SchematicOptimizationRequest",
    "SchematicOptimizationResponse",
]
