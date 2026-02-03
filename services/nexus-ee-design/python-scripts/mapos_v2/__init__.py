"""
MAPO v2.0 - Enhanced Multi-Agent Prompt Optimization System

LLM-first architecture using Claude Opus 4.5 via OpenRouter.
Core Philosophy: "Opus 4.5 Thinks, Algorithms Execute"

Key Enhancements over v1.0:
1. LLM Strategic Planning Phase (Routing Strategist + Congestion Predictor)
2. Debate-and-Critique Coordination (CircuitLM inspired)
3. PathFinder Negotiation-Based Routing
4. CBS (Conflict-Based Search) for Complex Cases
5. Layer Assignment Optimization with DP
6. Signal Integrity Awareness Throughout

Research Sources:
- CircuitLM Multi-Agent Framework (arxiv 2601.04505)
- Multi-Agent Based Minimal-Layer Via Routing (ScienceDirect 2025)
- OrthoRoute PathFinder Algorithm
- AlphaFold2 Iterative Refinement
"""

__version__ = "2.0.0"
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
]
