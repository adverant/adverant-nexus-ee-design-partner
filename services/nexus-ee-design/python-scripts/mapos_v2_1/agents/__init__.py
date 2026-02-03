"""
MAPO v2.0 Enhanced Agents

New Opus 4.5 agents for strategic routing orchestration.
All agents use OpenRouter API for Claude Opus 4.5 access.

Agent Roles:
- RoutingStrategistAgent: Pre-routing strategy analysis
- CongestionPredictorAgent: Congestion zone prediction
- ConflictResolverAgent: Routing conflict resolution
- SignalIntegrityAdvisorAgent: SI-aware routing guidance
- LayerAssignmentStrategistAgent: Layer assignment strategy
"""

from .routing_strategist import (
    RoutingStrategistAgent,
    RoutingStrategy,
    NetRoutingStrategy,
    CongestionZone,
    RoutingMethod,
    NetPriority,
    create_routing_strategist
)

from .congestion_predictor import (
    CongestionPredictorAgent,
    CongestionPrediction,
    CongestionRegion,
    CongestionSeverity,
    create_congestion_predictor
)

from .conflict_resolver import (
    ConflictResolverAgent,
    ConflictResolution,
    RoutingConflict,
    ConflictType,
    ResolutionStrategy,
    create_conflict_resolver
)

from .si_advisor import (
    SignalIntegrityAdvisorAgent,
    ImpedanceResult,
    SIGuidance,
    SignalType,
    ProtocolType,
    create_si_advisor
)

from .layer_strategist import (
    LayerAssignmentStrategistAgent,
    LayerHintsResult,
    NetLayerHint,
    LayerGrouping,
    LayerPreference,
    create_layer_strategist
)

__all__ = [
    # Routing Strategist
    "RoutingStrategistAgent",
    "RoutingStrategy",
    "NetRoutingStrategy",
    "CongestionZone",
    "RoutingMethod",
    "NetPriority",
    "create_routing_strategist",

    # Congestion Predictor
    "CongestionPredictorAgent",
    "CongestionPrediction",
    "CongestionRegion",
    "CongestionSeverity",
    "create_congestion_predictor",

    # Conflict Resolver
    "ConflictResolverAgent",
    "ConflictResolution",
    "RoutingConflict",
    "ConflictType",
    "ResolutionStrategy",
    "create_conflict_resolver",

    # Signal Integrity Advisor
    "SignalIntegrityAdvisorAgent",
    "ImpedanceResult",
    "SIGuidance",
    "SignalType",
    "ProtocolType",
    "create_si_advisor",

    # Layer Strategist
    "LayerAssignmentStrategistAgent",
    "LayerHintsResult",
    "NetLayerHint",
    "LayerGrouping",
    "LayerPreference",
    "create_layer_strategist",
]
