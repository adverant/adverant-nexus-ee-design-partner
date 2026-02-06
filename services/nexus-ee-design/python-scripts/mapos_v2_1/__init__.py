"""
MAPO v2.1 - LLM-Orchestrated Gaming AI for PCB Optimization

Merges LLM orchestration with Gaming AI algorithms:
- Core Philosophy: "Opus 4.6 Thinks, Gaming AI Explores, Algorithms Execute"

Key Features:
1. LLM Strategic Planning (Routing Strategist + Congestion Predictor)
2. MAP-Elites Quality-Diversity with LLM-guided exploration
3. Red Queen Adversarial Evolution with LLM mutation guidance
4. Elo Tournament Selection with Debate Validation
5. PathFinder + CBS Routing with LLM conflict resolution
6. Layer Assignment Optimization with DP + LLM hints
7. Signal Integrity with IPC-2141 formulas

Gaming AI Components (from v1.0):
- MAP-Elites: 10D behavioral descriptors, quality-diversity archive
- Red Queen: Adversarial co-evolution against historical champions
- Elo Tournaments: Fair multi-criteria ranking
- Ralph Wiggum: Self-referential optimization loops

LLM Agents (from v2.0):
- RoutingStrategistAgent: Pre-routing strategy analysis
- CongestionPredictorAgent: Congestion zone prediction
- ConflictResolverAgent: Routing conflict resolution
- SignalIntegrityAdvisorAgent: SI guidance
- LayerAssignmentStrategistAgent: Layer hints

Research Sources:
- CircuitLM Multi-Agent Framework (arxiv 2601.04505)
- Digital Red Queen (Sakana AI)
- MAP-Elites Quality-Diversity (arxiv 1504.04909)
- OrthoRoute PathFinder Algorithm
- AlphaFold2 Iterative Refinement
"""

__version__ = "2.1.0"
__author__ = "Adverant"

# v2.0 Agents
from .agents import (
    # Routing Strategist
    RoutingStrategistAgent,
    RoutingStrategy,
    NetRoutingStrategy,
    CongestionZone,
    RoutingMethod,
    NetPriority,
    create_routing_strategist,

    # Congestion Predictor
    CongestionPredictorAgent,
    CongestionPrediction,
    CongestionRegion,
    CongestionSeverity,
    create_congestion_predictor,

    # Conflict Resolver
    ConflictResolverAgent,
    ConflictResolution,
    RoutingConflict,
    ConflictType,
    ResolutionStrategy,
    create_conflict_resolver,

    # Signal Integrity Advisor
    SignalIntegrityAdvisorAgent,
    ImpedanceResult,
    SIGuidance,
    SignalType,
    ProtocolType,
    create_si_advisor,

    # Layer Strategist
    LayerAssignmentStrategistAgent,
    LayerHintsResult,
    NetLayerHint,
    LayerGrouping,
    LayerPreference,
    create_layer_strategist,
)

# v2.0 Core Components
from .debate_coordinator import (
    DebateAndCritiqueCoordinator,
    ConsensusResult,
    DebateOutcome,
    Proposal,
    Critique,
    Vote
)

from .pathfinder_router import (
    PathFinderRouter,
    RoutingSolution,
    Route,
    GridPoint,
    RouteSegment,
    CostMap,
    create_pathfinder_router
)

from .cbs_router import (
    CBSRouter,
    CBSSolution,
    CBSNode,
    Conflict,
    Constraint,
    HybridRouter,
    Net,
    create_cbs_router
)

from .layer_optimizer import (
    LayerAssignmentOptimizer,
    LayerOptimizationResult,
    LayerStackup,
    Layer,
    LayerType,
    Route2D,
    Route3D,
    NetType,
    create_layer_optimizer,
    create_2_layer_stackup,
    create_4_layer_stackup,
    create_6_layer_stackup,
    create_8_layer_stackup
)

from .multi_agent_optimizer import (
    MultiAgentOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationPhase,
    PhaseResult,
    DesignState,
    create_optimizer
)

# Gaming AI Components
from .gaming_ai.map_elites import (
    MAPElitesArchive,
    BehavioralDescriptor,
    ArchiveCell,
    ArchiveStatistics
)

from .gaming_ai.red_queen_evolver import (
    RedQueenEvolver,
    Champion,
    MutationStrategy,
    GeneralityScore,
    EvolutionRound
)

# LLM-Gaming Integration (v2.1 specific)
from .llm_gaming_integration import (
    IntegratedGamingLLMOptimizer,
    LLMGuidedMAPElites,
    LLMGuidedRedQueen,
    LLMGuidance,
    LLMGuidanceType,
    EnhancedBehavioralDescriptor,
    LLMGuidedMutation,
    create_integrated_optimizer,
    create_llm_map_elites,
    create_llm_red_queen
)

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Agents - Routing Strategist
    "RoutingStrategistAgent",
    "RoutingStrategy",
    "NetRoutingStrategy",
    "CongestionZone",
    "RoutingMethod",
    "NetPriority",
    "create_routing_strategist",

    # Agents - Congestion Predictor
    "CongestionPredictorAgent",
    "CongestionPrediction",
    "CongestionRegion",
    "CongestionSeverity",
    "create_congestion_predictor",

    # Agents - Conflict Resolver
    "ConflictResolverAgent",
    "ConflictResolution",
    "RoutingConflict",
    "ConflictType",
    "ResolutionStrategy",
    "create_conflict_resolver",

    # Agents - Signal Integrity Advisor
    "SignalIntegrityAdvisorAgent",
    "ImpedanceResult",
    "SIGuidance",
    "SignalType",
    "ProtocolType",
    "create_si_advisor",

    # Agents - Layer Strategist
    "LayerAssignmentStrategistAgent",
    "LayerHintsResult",
    "NetLayerHint",
    "LayerGrouping",
    "LayerPreference",
    "create_layer_strategist",

    # Debate Coordinator
    "DebateAndCritiqueCoordinator",
    "ConsensusResult",
    "DebateOutcome",
    "Proposal",
    "Critique",
    "Vote",

    # PathFinder Router
    "PathFinderRouter",
    "RoutingSolution",
    "Route",
    "GridPoint",
    "RouteSegment",
    "CostMap",
    "create_pathfinder_router",

    # CBS Router
    "CBSRouter",
    "CBSSolution",
    "CBSNode",
    "Conflict",
    "Constraint",
    "HybridRouter",
    "Net",
    "create_cbs_router",

    # Layer Optimizer
    "LayerAssignmentOptimizer",
    "LayerOptimizationResult",
    "LayerStackup",
    "Layer",
    "LayerType",
    "Route2D",
    "Route3D",
    "NetType",
    "create_layer_optimizer",
    "create_2_layer_stackup",
    "create_4_layer_stackup",
    "create_6_layer_stackup",
    "create_8_layer_stackup",

    # Multi-Agent Optimizer
    "MultiAgentOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationPhase",
    "PhaseResult",
    "DesignState",
    "create_optimizer",

    # Gaming AI - MAP-Elites
    "MAPElitesArchive",
    "BehavioralDescriptor",
    "ArchiveCell",
    "ArchiveStatistics",

    # Gaming AI - Red Queen
    "RedQueenEvolver",
    "Champion",
    "MutationStrategy",
    "GeneralityScore",
    "EvolutionRound",

    # LLM-Gaming Integration
    "IntegratedGamingLLMOptimizer",
    "LLMGuidedMAPElites",
    "LLMGuidedRedQueen",
    "LLMGuidance",
    "LLMGuidanceType",
    "EnhancedBehavioralDescriptor",
    "LLMGuidedMutation",
    "create_integrated_optimizer",
    "create_llm_map_elites",
    "create_llm_red_queen",
]
