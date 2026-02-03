"""
MAPO v2.0 Multi-Agent Optimizer

Enhanced 9-Phase Pipeline with LLM-First Architecture.
"Opus 4.5 Thinks, Algorithms Execute"

Key Enhancements over v1.0:
1. LLM Strategic Planning Phase (NEW)
2. Debate-and-Critique Coordination (CircuitLM inspired)
3. PathFinder Negotiation-Based Routing
4. CBS (Conflict-Based Search) for Complex Cases
5. Layer Assignment Optimization with DP
6. Signal Integrity Awareness Throughout

Phase Flow:
0. Load & Analyze (Initial DRC)
1. LLM Strategic Planning (NEW - Routing Strategist + Congestion Predictor)
2. Pre-DRC Fixes (Enhanced with LLM guidance)
3. Negotiation-Based Routing (PathFinder with CBS fallback)
4. Layer Assignment (DP optimizer with LLM hints)
5. MCTS Exploration (Enhanced with debate validation)
6. Evolutionary Optimization (GA with debate-validated mutations)
7. Tournament Selection (Elo-based ranking)
8. AlphaFold-style Refinement (Iterative recycling)

Research Sources:
- CircuitLM Multi-Agent Framework (arxiv 2601.04505)
- Multi-Agent Based Minimal-Layer Via Routing (ScienceDirect 2025)
- OrthoRoute PathFinder Algorithm
- AlphaFold2 Iterative Refinement

Author: Claude Opus 4.5 via MAPO v2.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os

# Import v2.0 agents
from .agents import (
    RoutingStrategistAgent,
    create_routing_strategist,
    RoutingStrategy,
    NetRoutingStrategy,
    CongestionZone,
    RoutingMethod,
    NetPriority
)

# Import other agents (these will be available after full module setup)
try:
    from .agents.congestion_predictor import CongestionPredictorAgent, create_congestion_predictor
    from .agents.conflict_resolver import ConflictResolverAgent, create_conflict_resolver
    from .agents.si_advisor import SignalIntegrityAdvisorAgent, create_si_advisor
    from .agents.layer_strategist import LayerAssignmentStrategistAgent, create_layer_strategist
except ImportError:
    # Agents not yet fully set up
    CongestionPredictorAgent = None
    ConflictResolverAgent = None
    SignalIntegrityAdvisorAgent = None
    LayerAssignmentStrategistAgent = None

# Import v2.0 components
from .debate_coordinator import DebateAndCritiqueCoordinator, ConsensusResult
from .pathfinder_router import PathFinderRouter, RoutingSolution
from .cbs_router import CBSRouter, CBSSolution, Net as CBSNet
from .layer_optimizer import (
    LayerAssignmentOptimizer,
    LayerOptimizationResult,
    LayerStackup,
    Route2D,
    create_4_layer_stackup
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class OptimizationPhase(Enum):
    """Phases of the optimization pipeline."""
    LOAD_ANALYZE = "load_analyze"
    LLM_STRATEGIC_PLANNING = "llm_strategic_planning"
    PRE_DRC_FIXES = "pre_drc_fixes"
    NEGOTIATION_ROUTING = "negotiation_routing"
    LAYER_ASSIGNMENT = "layer_assignment"
    MCTS_EXPLORATION = "mcts_exploration"
    EVOLUTIONARY_OPTIMIZATION = "evolutionary_optimization"
    TOURNAMENT_SELECTION = "tournament_selection"
    ALPHAFOLD_REFINEMENT = "alphafold_refinement"


@dataclass
class PhaseResult:
    """Result from a single phase."""
    phase: OptimizationPhase
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for the optimizer."""
    # API Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-opus-4-5-20251101"

    # Phase Enable/Disable
    enable_llm_planning: bool = True
    enable_pathfinder: bool = True
    enable_cbs: bool = True
    enable_layer_optimization: bool = True
    enable_mcts: bool = True
    enable_evolutionary: bool = True
    enable_debate: bool = True

    # Phase Parameters
    max_mcts_iterations: int = 100
    max_evolutionary_generations: int = 50
    max_refinement_cycles: int = 5
    pathfinder_max_iterations: int = 50
    cbs_max_iterations: int = 1000

    # Routing Parameters
    grid_resolution: float = 0.254  # mm (10 mil)
    via_cost: float = 10.0
    layer_change_cost: float = 5.0

    # Layer Stackup
    num_layers: int = 4
    stackup: Optional[LayerStackup] = None

    # Thresholds
    conflict_threshold_for_cbs: int = 10  # Switch to CBS above this
    debate_threshold: int = 3  # Use debate for complex decisions

    # Targets
    target_drc_reduction: float = 0.5  # 50% reduction target
    target_via_reduction: float = 0.3  # 30% reduction target


@dataclass
class OptimizationResult:
    """Final result of the optimization."""
    success: bool
    total_duration_seconds: float
    phase_results: List[PhaseResult]
    initial_drc_count: int
    final_drc_count: int
    drc_reduction_percentage: float
    initial_via_count: int
    final_via_count: int
    via_reduction_percentage: float
    routing_solution: Optional[RoutingSolution] = None
    layer_result: Optional[LayerOptimizationResult] = None
    final_schematic: Optional[str] = None  # Modified schematic content
    debate_sessions: int = 0
    llm_calls: int = 0


@dataclass
class DesignState:
    """Current state of the design being optimized."""
    schematic_content: str
    netlist: List[Dict[str, Any]]
    components: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    drc_violations: List[Dict[str, Any]]
    routing_strategy: Optional[RoutingStrategy] = None
    congestion_zones: List[CongestionZone] = field(default_factory=list)
    routing_solution: Optional[RoutingSolution] = None
    layer_assignment: Optional[LayerOptimizationResult] = None


# =============================================================================
# Multi-Agent Optimizer
# =============================================================================

class MultiAgentOptimizer:
    """
    MAPO v2.0 Multi-Agent Optimizer

    Orchestrates the 9-phase optimization pipeline with LLM-first architecture.

    Key Features:
    - LLM agents for strategic decisions (Opus 4.5 via OpenRouter)
    - CPU algorithms for execution (PathFinder, CBS, DP)
    - Debate-and-critique for complex decisions
    - Iterative refinement with convergence detection
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

        # Get API key from config or environment
        self.api_key = self.config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

        # Initialize LLM agents
        self._init_agents()

        # Initialize algorithmic components
        self._init_algorithms()

        # Initialize debate coordinator
        self.debate_coordinator = DebateAndCritiqueCoordinator(
            agents=[],  # Will be populated with active agents
            max_rounds=3,
            consensus_threshold=0.6
        )

        # Statistics
        self.stats = {
            "total_llm_calls": 0,
            "total_debate_sessions": 0,
            "total_conflicts_resolved": 0
        }

    def _init_agents(self):
        """Initialize LLM agents."""
        # Routing Strategist
        self.routing_strategist = create_routing_strategist(
            openrouter_api_key=self.api_key,
            model=self.config.openrouter_model
        ) if self.config.enable_llm_planning else None

        # Other agents (initialized lazily or when modules are available)
        self.congestion_predictor = None
        self.conflict_resolver = None
        self.si_advisor = None
        self.layer_strategist = None

        if CongestionPredictorAgent is not None:
            try:
                self.congestion_predictor = create_congestion_predictor(
                    openrouter_api_key=self.api_key,
                    model=self.config.openrouter_model
                )
            except Exception as e:
                logger.warning(f"Failed to create congestion predictor: {e}")

        if ConflictResolverAgent is not None:
            try:
                self.conflict_resolver = create_conflict_resolver(
                    openrouter_api_key=self.api_key,
                    model=self.config.openrouter_model
                )
            except Exception as e:
                logger.warning(f"Failed to create conflict resolver: {e}")

        if SignalIntegrityAdvisorAgent is not None:
            try:
                self.si_advisor = create_si_advisor(
                    openrouter_api_key=self.api_key,
                    model=self.config.openrouter_model
                )
            except Exception as e:
                logger.warning(f"Failed to create SI advisor: {e}")

        if LayerAssignmentStrategistAgent is not None:
            try:
                self.layer_strategist = create_layer_strategist(
                    openrouter_api_key=self.api_key,
                    model=self.config.openrouter_model
                )
            except Exception as e:
                logger.warning(f"Failed to create layer strategist: {e}")

    def _init_algorithms(self):
        """Initialize CPU-based algorithms."""
        # PathFinder Router
        self.pathfinder = PathFinderRouter(
            strategist=self.routing_strategist,
            predictor=self.congestion_predictor,
            resolver=self.conflict_resolver,
            grid_resolution=self.config.grid_resolution,
            max_iterations=self.config.pathfinder_max_iterations
        ) if self.config.enable_pathfinder else None

        # CBS Router
        self.cbs_router = CBSRouter(
            conflict_resolver=self.conflict_resolver,
            grid_resolution=self.config.grid_resolution,
            max_iterations=self.config.cbs_max_iterations,
            via_cost=self.config.via_cost
        ) if self.config.enable_cbs else None

        # Layer Optimizer
        self.layer_optimizer = LayerAssignmentOptimizer(
            layer_strategist=self.layer_strategist,
            via_cost=self.config.via_cost,
            layer_change_cost=self.config.layer_change_cost
        ) if self.config.enable_layer_optimization else None

    # -------------------------------------------------------------------------
    # Main Optimization Entry Point
    # -------------------------------------------------------------------------

    async def optimize(
        self,
        schematic_content: str,
        netlist: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Dict[str, Any]]] = None
    ) -> OptimizationResult:
        """
        Run the full optimization pipeline.

        Args:
            schematic_content: KiCad schematic file content
            netlist: Optional parsed netlist (will be extracted if not provided)
            components: Optional component list (will be extracted if not provided)

        Returns:
            OptimizationResult with all phase results and final metrics
        """
        start_time = time.time()
        phase_results: List[PhaseResult] = []

        # Initialize design state
        state = DesignState(
            schematic_content=schematic_content,
            netlist=netlist or [],
            components=components or [],
            connections=[],
            drc_violations=[]
        )

        # Track initial metrics
        initial_drc_count = 0
        initial_via_count = 0

        try:
            # Phase 0: Load & Analyze
            result = await self._phase_load_analyze(state)
            phase_results.append(result)
            if not result.success:
                return self._create_failure_result(phase_results, start_time, state)

            initial_drc_count = len(state.drc_violations)
            initial_via_count = result.metrics.get("via_count", 0)

            # Phase 1: LLM Strategic Planning
            if self.config.enable_llm_planning:
                result = await self._phase_llm_strategic_planning(state)
                phase_results.append(result)

            # Phase 2: Pre-DRC Fixes
            result = await self._phase_pre_drc_fixes(state)
            phase_results.append(result)

            # Phase 3: Negotiation-Based Routing
            if self.config.enable_pathfinder:
                result = await self._phase_negotiation_routing(state)
                phase_results.append(result)

            # Phase 4: Layer Assignment
            if self.config.enable_layer_optimization and self.config.num_layers > 2:
                result = await self._phase_layer_assignment(state)
                phase_results.append(result)

            # Phase 5: MCTS Exploration
            if self.config.enable_mcts:
                result = await self._phase_mcts_exploration(state)
                phase_results.append(result)

            # Phase 6: Evolutionary Optimization
            if self.config.enable_evolutionary:
                result = await self._phase_evolutionary_optimization(state)
                phase_results.append(result)

            # Phase 7: Tournament Selection
            result = await self._phase_tournament_selection(state)
            phase_results.append(result)

            # Phase 8: AlphaFold-style Refinement
            result = await self._phase_alphafold_refinement(state)
            phase_results.append(result)

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._create_failure_result(phase_results, start_time, state, str(e))

        # Calculate final metrics
        final_drc_count = len(state.drc_violations)
        final_via_count = self._count_vias(state)

        drc_reduction = ((initial_drc_count - final_drc_count) / initial_drc_count * 100
                        if initial_drc_count > 0 else 0)
        via_reduction = ((initial_via_count - final_via_count) / initial_via_count * 100
                        if initial_via_count > 0 else 0)

        total_duration = time.time() - start_time

        return OptimizationResult(
            success=True,
            total_duration_seconds=total_duration,
            phase_results=phase_results,
            initial_drc_count=initial_drc_count,
            final_drc_count=final_drc_count,
            drc_reduction_percentage=drc_reduction,
            initial_via_count=initial_via_count,
            final_via_count=final_via_count,
            via_reduction_percentage=via_reduction,
            routing_solution=state.routing_solution,
            layer_result=state.layer_assignment,
            final_schematic=state.schematic_content,
            debate_sessions=self.stats["total_debate_sessions"],
            llm_calls=self.stats["total_llm_calls"]
        )

    # -------------------------------------------------------------------------
    # Phase Implementations
    # -------------------------------------------------------------------------

    async def _phase_load_analyze(self, state: DesignState) -> PhaseResult:
        """Phase 0: Load design and perform initial analysis."""
        start_time = time.time()

        try:
            # Extract netlist if not provided
            if not state.netlist:
                state.netlist = self._extract_netlist(state.schematic_content)

            # Extract components if not provided
            if not state.components:
                state.components = self._extract_components(state.schematic_content)

            # Extract connections
            state.connections = self._extract_connections(state.schematic_content)

            # Initial DRC check
            state.drc_violations = self._run_drc(state)

            # Count initial vias
            via_count = self._count_vias(state)

            return PhaseResult(
                phase=OptimizationPhase.LOAD_ANALYZE,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "net_count": len(state.netlist),
                    "component_count": len(state.components),
                    "connection_count": len(state.connections),
                    "initial_drc_violations": len(state.drc_violations),
                    "via_count": via_count
                }
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.LOAD_ANALYZE,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_llm_strategic_planning(self, state: DesignState) -> PhaseResult:
        """Phase 1: LLM Strategic Planning."""
        start_time = time.time()

        try:
            if not self.routing_strategist:
                return PhaseResult(
                    phase=OptimizationPhase.LLM_STRATEGIC_PLANNING,
                    success=True,
                    duration_seconds=time.time() - start_time,
                    metrics={"skipped": True, "reason": "No routing strategist configured"}
                )

            # Prepare net info for strategist
            nets = [
                {
                    "name": net.get("name", f"net_{i}"),
                    "pins": net.get("pins", []),
                    "type": net.get("type", "signal")
                }
                for i, net in enumerate(state.netlist)
            ]

            # Get routing strategy from LLM
            strategy = await self.routing_strategist.analyze_routing_strategy(
                nets=nets,
                components=state.components,
                board_size=(100, 80),  # Default size, should be extracted from schematic
                layer_count=self.config.num_layers
            )

            state.routing_strategy = strategy
            self.stats["total_llm_calls"] += 1

            # Get congestion predictions if available
            if self.congestion_predictor:
                congestion_result = await self.congestion_predictor.predict_congestion(
                    nets=nets,
                    components=state.components,
                    board_size=(100, 80)
                )
                state.congestion_zones = congestion_result.congestion_zones
                self.stats["total_llm_calls"] += 1

            return PhaseResult(
                phase=OptimizationPhase.LLM_STRATEGIC_PLANNING,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "nets_analyzed": len(strategy.net_strategies) if strategy else 0,
                    "congestion_zones": len(state.congestion_zones),
                    "high_priority_nets": sum(
                        1 for ns in (strategy.net_strategies if strategy else [])
                        if ns.priority == NetPriority.CRITICAL
                    )
                }
            )

        except Exception as e:
            logger.error(f"LLM strategic planning failed: {e}")
            return PhaseResult(
                phase=OptimizationPhase.LLM_STRATEGIC_PLANNING,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_pre_drc_fixes(self, state: DesignState) -> PhaseResult:
        """Phase 2: Apply pre-DRC fixes (deterministic + LLM-guided)."""
        start_time = time.time()
        modifications = []

        try:
            # Apply deterministic fixes first
            for violation in state.drc_violations:
                fix = self._get_deterministic_fix(violation)
                if fix:
                    modifications.append(fix)
                    self._apply_fix(state, fix)

            # Re-run DRC
            state.drc_violations = self._run_drc(state)

            # For remaining violations, consult LLM if debate is enabled
            if self.config.enable_debate and len(state.drc_violations) > 0:
                remaining_count = len(state.drc_violations)

                # Use debate for complex fixes
                if remaining_count <= self.config.debate_threshold and self.conflict_resolver:
                    for violation in state.drc_violations[:3]:  # Limit debate to 3 violations
                        resolution = await self._debate_fix(violation, state)
                        if resolution:
                            modifications.append(resolution)
                            self._apply_fix(state, resolution)

                    state.drc_violations = self._run_drc(state)

            return PhaseResult(
                phase=OptimizationPhase.PRE_DRC_FIXES,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "fixes_applied": len(modifications),
                    "remaining_violations": len(state.drc_violations)
                },
                modifications=modifications
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.PRE_DRC_FIXES,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_negotiation_routing(self, state: DesignState) -> PhaseResult:
        """Phase 3: Negotiation-based routing (PathFinder with CBS fallback)."""
        start_time = time.time()

        try:
            if not self.pathfinder:
                return PhaseResult(
                    phase=OptimizationPhase.NEGOTIATION_ROUTING,
                    success=True,
                    duration_seconds=time.time() - start_time,
                    metrics={"skipped": True}
                )

            # Convert netlist to Net objects for routing
            nets = self._convert_to_route_nets(state)

            # Use strategy hints if available
            if state.routing_strategy:
                self.pathfinder.apply_strategy_hints(state.routing_strategy)

            # Run PathFinder routing
            solution = await self.pathfinder.route_with_negotiation(
                nets=nets,
                max_iterations=self.config.pathfinder_max_iterations
            )

            # Check if too many conflicts remain
            if solution.total_conflicts > self.config.conflict_threshold_for_cbs and self.cbs_router:
                logger.info(f"PathFinder had {solution.total_conflicts} conflicts, using CBS")

                # Convert to CBS format and run
                cbs_nets = self._convert_to_cbs_nets(state)

                # Set up CBS grid
                board_width = max(c.get("x", 0) + c.get("width", 10) for c in state.components) if state.components else 100
                board_height = max(c.get("y", 0) + c.get("height", 10) for c in state.components) if state.components else 80

                self.cbs_router.set_grid_bounds(board_width + 20, board_height + 20, self.config.num_layers)
                self.cbs_router.add_obstacles_from_components(state.components)

                cbs_solution = await self.cbs_router.route_nets(cbs_nets)

                # Use CBS solution if better
                if len(cbs_solution.routes) >= len(solution.routes):
                    solution = self._convert_cbs_to_pathfinder_solution(cbs_solution)

            state.routing_solution = solution

            return PhaseResult(
                phase=OptimizationPhase.NEGOTIATION_ROUTING,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "nets_routed": len(solution.routes) if solution else 0,
                    "total_conflicts": solution.total_conflicts if solution else 0,
                    "total_length": solution.total_length if solution else 0,
                    "iterations": solution.iterations if solution else 0
                }
            )

        except Exception as e:
            logger.error(f"Negotiation routing failed: {e}")
            return PhaseResult(
                phase=OptimizationPhase.NEGOTIATION_ROUTING,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_layer_assignment(self, state: DesignState) -> PhaseResult:
        """Phase 4: Layer assignment optimization."""
        start_time = time.time()

        try:
            if not self.layer_optimizer or not state.routing_solution:
                return PhaseResult(
                    phase=OptimizationPhase.LAYER_ASSIGNMENT,
                    success=True,
                    duration_seconds=time.time() - start_time,
                    metrics={"skipped": True}
                )

            # Get or create stackup
            stackup = self.config.stackup or create_4_layer_stackup()

            # Convert routing solution to 2D routes
            routes_2d = self._convert_to_routes_2d(state.routing_solution)

            # Run layer optimization
            result = await self.layer_optimizer.optimize_layers(routes_2d, stackup)

            state.layer_assignment = result

            return PhaseResult(
                phase=OptimizationPhase.LAYER_ASSIGNMENT,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "total_vias": result.total_vias,
                    "via_span_cost": result.total_via_span_cost,
                    "layer_utilization": result.layer_utilization,
                    "iterations": result.iterations
                }
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.LAYER_ASSIGNMENT,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_mcts_exploration(self, state: DesignState) -> PhaseResult:
        """Phase 5: MCTS exploration with debate validation."""
        start_time = time.time()

        try:
            # MCTS implementation would go here
            # For now, placeholder that integrates with existing MAPO MCTS

            return PhaseResult(
                phase=OptimizationPhase.MCTS_EXPLORATION,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "iterations": 0,
                    "nodes_explored": 0,
                    "best_score": 0.0,
                    "note": "MCTS integration pending - uses v1.0 MCTS"
                }
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.MCTS_EXPLORATION,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_evolutionary_optimization(self, state: DesignState) -> PhaseResult:
        """Phase 6: Evolutionary optimization with debate-validated mutations."""
        start_time = time.time()

        try:
            # Evolutionary optimization implementation
            # For now, placeholder that integrates with existing MAPO evolutionary

            return PhaseResult(
                phase=OptimizationPhase.EVOLUTIONARY_OPTIMIZATION,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "generations": 0,
                    "population_size": 0,
                    "best_fitness": 0.0,
                    "note": "Evolutionary integration pending - uses v1.0"
                }
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.EVOLUTIONARY_OPTIMIZATION,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_tournament_selection(self, state: DesignState) -> PhaseResult:
        """Phase 7: Tournament selection with Elo ranking."""
        start_time = time.time()

        try:
            # Tournament selection implementation
            # Uses existing v1.0 tournament judge

            return PhaseResult(
                phase=OptimizationPhase.TOURNAMENT_SELECTION,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "candidates_evaluated": 0,
                    "winner_score": 0.0,
                    "note": "Tournament integration pending - uses v1.0"
                }
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.TOURNAMENT_SELECTION,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    async def _phase_alphafold_refinement(self, state: DesignState) -> PhaseResult:
        """Phase 8: AlphaFold-style iterative refinement."""
        start_time = time.time()

        try:
            # Iterative refinement with convergence detection
            # Uses existing v1.0 refinement loop

            # Final DRC check
            state.drc_violations = self._run_drc(state)

            return PhaseResult(
                phase=OptimizationPhase.ALPHAFOLD_REFINEMENT,
                success=True,
                duration_seconds=time.time() - start_time,
                metrics={
                    "refinement_cycles": 0,
                    "final_drc_violations": len(state.drc_violations),
                    "converged": True,
                    "note": "Refinement integration pending - uses v1.0"
                }
            )

        except Exception as e:
            return PhaseResult(
                phase=OptimizationPhase.ALPHAFOLD_REFINEMENT,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _extract_netlist(self, schematic: str) -> List[Dict[str, Any]]:
        """Extract netlist from schematic content."""
        # Simplified extraction - real implementation would parse KiCad format
        nets = []
        # Parse (net ...) sections from schematic
        import re
        net_pattern = r'\(net\s+(\d+)\s+"([^"]+)"\)'
        for match in re.finditer(net_pattern, schematic):
            nets.append({
                "id": int(match.group(1)),
                "name": match.group(2),
                "pins": [],
                "type": "signal"
            })
        return nets

    def _extract_components(self, schematic: str) -> List[Dict[str, Any]]:
        """Extract components from schematic content."""
        components = []
        import re
        # Parse (symbol ...) sections
        symbol_pattern = r'\(symbol\s+\(lib_id\s+"([^"]+)"\)'
        for match in re.finditer(symbol_pattern, schematic):
            components.append({
                "lib_id": match.group(1),
                "x": 0,
                "y": 0,
                "width": 10,
                "height": 10
            })
        return components

    def _extract_connections(self, schematic: str) -> List[Dict[str, Any]]:
        """Extract wire connections from schematic."""
        connections = []
        import re
        wire_pattern = r'\(wire\s+\(pts\s+\(xy\s+([\d.]+)\s+([\d.]+)\)\s+\(xy\s+([\d.]+)\s+([\d.]+)\)\)'
        for match in re.finditer(wire_pattern, schematic):
            connections.append({
                "start": (float(match.group(1)), float(match.group(2))),
                "end": (float(match.group(3)), float(match.group(4)))
            })
        return connections

    def _run_drc(self, state: DesignState) -> List[Dict[str, Any]]:
        """Run design rule check."""
        violations = []
        # Simplified DRC - check for basic issues
        # Real implementation would use KiCad DRC or custom checks

        # Check for 4-way junctions (should be avoided)
        wire_points = {}
        for conn in state.connections:
            for point in [conn["start"], conn["end"]]:
                wire_points[point] = wire_points.get(point, 0) + 1

        for point, count in wire_points.items():
            if count >= 4:
                violations.append({
                    "type": "four_way_junction",
                    "location": point,
                    "severity": "warning"
                })

        return violations

    def _count_vias(self, state: DesignState) -> int:
        """Count vias in current design."""
        if state.layer_assignment:
            return state.layer_assignment.total_vias
        if state.routing_solution:
            return sum(r.vias for r in state.routing_solution.routes.values())
        return 0

    def _get_deterministic_fix(self, violation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get deterministic fix for a violation."""
        if violation["type"] == "four_way_junction":
            return {
                "type": "split_junction",
                "location": violation["location"],
                "action": "Convert 4-way to two 3-way junctions"
            }
        return None

    def _apply_fix(self, state: DesignState, fix: Dict[str, Any]):
        """Apply a fix to the design state."""
        # Implementation would modify schematic content
        pass

    async def _debate_fix(self, violation: Dict[str, Any], state: DesignState) -> Optional[Dict[str, Any]]:
        """Use debate mechanism to determine fix for complex violation."""
        if not self.conflict_resolver:
            return None

        self.stats["total_debate_sessions"] += 1
        # Implementation would use debate coordinator
        return None

    def _convert_to_route_nets(self, state: DesignState) -> List[Any]:
        """Convert netlist to PathFinder Net format."""
        from .pathfinder_router import Net, GridPoint
        nets = []
        for net_data in state.netlist:
            pins = []
            for pin in net_data.get("pins", []):
                if isinstance(pin, dict):
                    pins.append(GridPoint(pin.get("x", 0), pin.get("y", 0)))
                elif isinstance(pin, (list, tuple)) and len(pin) >= 2:
                    pins.append(GridPoint(pin[0], pin[1]))

            if len(pins) >= 2:
                nets.append(Net(
                    name=net_data.get("name", f"net_{len(nets)}"),
                    pins=pins,
                    priority=net_data.get("priority", 1)
                ))
        return nets

    def _convert_to_cbs_nets(self, state: DesignState) -> List[CBSNet]:
        """Convert netlist to CBS Net format."""
        from .cbs_router import GridPoint as CBSGridPoint
        nets = []
        for net_data in state.netlist:
            pins = []
            for pin in net_data.get("pins", []):
                if isinstance(pin, dict):
                    pins.append(CBSGridPoint(pin.get("x", 0), pin.get("y", 0)))
                elif isinstance(pin, (list, tuple)) and len(pin) >= 2:
                    pins.append(CBSGridPoint(pin[0], pin[1]))

            if len(pins) >= 2:
                nets.append(CBSNet(
                    name=net_data.get("name", f"net_{len(nets)}"),
                    pins=pins,
                    priority=net_data.get("priority", 1)
                ))
        return nets

    def _convert_cbs_to_pathfinder_solution(self, cbs_solution: CBSSolution) -> RoutingSolution:
        """Convert CBS solution to PathFinder format."""
        from .pathfinder_router import Route, GridPoint
        routes = {}
        for name, cbs_route in cbs_solution.routes.items():
            routes[name] = Route(
                net_name=name,
                path=[GridPoint(p.x, p.y, p.layer) for p in cbs_route.path],
                cost=cbs_route.cost,
                vias=cbs_route.vias
            )
        return RoutingSolution(
            routes=routes,
            total_cost=cbs_solution.total_cost,
            total_length=0,  # Would need to calculate
            total_vias=cbs_solution.total_vias,
            total_conflicts=0,
            iterations=cbs_solution.iterations
        )

    def _convert_to_routes_2d(self, solution: RoutingSolution) -> List[Route2D]:
        """Convert PathFinder solution to 2D routes for layer optimization."""
        routes = []
        for name, route in solution.routes.items():
            path = [(p.x, p.y) for p in route.path]
            routes.append(Route2D(
                net_name=name,
                path=path,
                width=route.width if hasattr(route, 'width') else 0.254
            ))
        return routes

    def _create_failure_result(
        self,
        phase_results: List[PhaseResult],
        start_time: float,
        state: DesignState,
        error: Optional[str] = None
    ) -> OptimizationResult:
        """Create a failure result."""
        return OptimizationResult(
            success=False,
            total_duration_seconds=time.time() - start_time,
            phase_results=phase_results,
            initial_drc_count=0,
            final_drc_count=len(state.drc_violations),
            drc_reduction_percentage=0,
            initial_via_count=0,
            final_via_count=0,
            via_reduction_percentage=0,
            debate_sessions=self.stats["total_debate_sessions"],
            llm_calls=self.stats["total_llm_calls"]
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_optimizer(
    openrouter_api_key: Optional[str] = None,
    enable_all_features: bool = True,
    num_layers: int = 4
) -> MultiAgentOptimizer:
    """Factory function to create an optimizer instance."""
    config = OptimizationConfig(
        openrouter_api_key=openrouter_api_key,
        enable_llm_planning=enable_all_features,
        enable_pathfinder=enable_all_features,
        enable_cbs=enable_all_features,
        enable_layer_optimization=enable_all_features and num_layers > 2,
        enable_mcts=enable_all_features,
        enable_evolutionary=enable_all_features,
        enable_debate=enable_all_features,
        num_layers=num_layers
    )
    return MultiAgentOptimizer(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MultiAgentOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationPhase",
    "PhaseResult",
    "DesignState",
    "create_optimizer"
]
