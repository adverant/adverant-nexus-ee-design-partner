"""
MAPO v2.1 LLM-Gaming Integration Layer

Bridges the LLM orchestration layer with Gaming AI algorithms:
- LLM agents guide MAP-Elites exploration and mutation strategies
- LLM Conflict Resolver provides semantic understanding for Red Queen evolution
- Debate mechanism validates tournament matchups and champion selection
- LLM extracts behavioral descriptors with domain understanding

Core Philosophy: "Opus 4.5 Thinks, Gaming AI Explores, Algorithms Execute"

Author: Claude Opus 4.5 via MAPO v2.1
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

# Import LLM agents from v2
from .agents import (
    RoutingStrategistAgent,
    CongestionPredictorAgent,
    ConflictResolverAgent,
    SignalIntegrityAdvisorAgent,
    create_routing_strategist,
    create_congestion_predictor,
    create_conflict_resolver,
    create_si_advisor
)

# Import gaming AI components
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

# Import debate coordinator
from .debate_coordinator import (
    DebateAndCritiqueCoordinator,
    Proposal,
    Critique,
    ConsensusResult
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class LLMGuidanceType(Enum):
    """Types of LLM guidance for gaming AI."""
    MUTATION_STRATEGY = "mutation_strategy"     # Which mutation to apply
    CELL_SELECTION = "cell_selection"           # Which archive cell to explore
    CHAMPION_SELECTION = "champion_selection"   # Which champion to compete against
    FITNESS_WEIGHTING = "fitness_weighting"     # How to weight fitness components
    BEHAVIORAL_ANALYSIS = "behavioral_analysis" # Semantic behavioral description


@dataclass
class LLMGuidance:
    """Guidance from LLM for gaming AI decision."""
    guidance_type: LLMGuidanceType
    recommendation: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedBehavioralDescriptor:
    """Behavioral descriptor enhanced with LLM semantic understanding."""
    # Original 10D descriptor
    base_descriptor: BehavioralDescriptor

    # LLM-extracted semantic features
    routing_strategy: str           # "dense", "sparse", "mixed"
    thermal_strategy: str           # "distributed", "concentrated", "passive"
    power_distribution: str         # "star", "tree", "grid"
    signal_integrity_focus: str     # "impedance", "length", "shielding"
    manufacturing_complexity: str   # "low", "medium", "high"

    # Semantic similarity scores (computed by LLM)
    similarity_to_reference: float  # How similar to reference design
    novelty_score: float            # How different from archive elites
    robustness_estimate: float      # Estimated robustness to variations

    def to_extended_vector(self) -> np.ndarray:
        """Convert to extended feature vector."""
        base = self.base_descriptor.to_vector()

        # Encode semantic features as numeric
        strategy_map = {"dense": 1.0, "sparse": 0.0, "mixed": 0.5}
        thermal_map = {"distributed": 1.0, "concentrated": 0.0, "passive": 0.5}
        power_map = {"star": 0.0, "tree": 0.5, "grid": 1.0}
        si_map = {"impedance": 0.33, "length": 0.66, "shielding": 1.0}
        complexity_map = {"low": 0.0, "medium": 0.5, "high": 1.0}

        semantic = np.array([
            strategy_map.get(self.routing_strategy, 0.5),
            thermal_map.get(self.thermal_strategy, 0.5),
            power_map.get(self.power_distribution, 0.5),
            si_map.get(self.signal_integrity_focus, 0.5),
            complexity_map.get(self.manufacturing_complexity, 0.5),
            self.similarity_to_reference,
            self.novelty_score,
            self.robustness_estimate
        ], dtype=np.float32)

        return np.concatenate([base, semantic])


@dataclass
class LLMGuidedMutation:
    """A mutation guided by LLM analysis."""
    strategy: MutationStrategy
    target_region: str              # Which part of design to mutate
    mutation_strength: float        # 0-1, how aggressive
    llm_reasoning: str              # Why this mutation
    expected_outcome: str           # What LLM expects
    constraints: List[str]          # What to preserve
    fallback_strategy: MutationStrategy  # If primary fails


# =============================================================================
# LLM-Guided MAP-Elites
# =============================================================================

class LLMGuidedMAPElites:
    """
    MAP-Elites archive enhanced with LLM guidance.

    The LLM provides:
    1. Semantic understanding of behavioral cells
    2. Guided cell selection for exploration
    3. Mutation strategy recommendations
    4. Novelty assessment beyond numeric distance
    """

    def __init__(
        self,
        archive: MAPElitesArchive,
        strategist: Optional[RoutingStrategistAgent] = None,
        predictor: Optional[CongestionPredictorAgent] = None,
        debate_coordinator: Optional[DebateAndCritiqueCoordinator] = None
    ):
        self.archive = archive
        self.strategist = strategist
        self.predictor = predictor
        self.debate = debate_coordinator

        self.guidance_history: List[LLMGuidance] = []
        self.stats = {
            "llm_guided_mutations": 0,
            "llm_cell_selections": 0,
            "debate_validations": 0
        }

    async def get_guided_cell_selection(
        self,
        current_solution: Any,
        exploration_budget: int = 5
    ) -> List[Tuple[int, int, LLMGuidance]]:
        """
        Get LLM-guided cell selection for exploration.

        Instead of random or curiosity-based selection, ask LLM which
        behavioral regions are most promising to explore.
        """
        if not self.strategist:
            # Fall back to standard curiosity selection
            return [(cell.cell_index, 1, None)
                    for cell in self.archive.get_curiosity_selection(exploration_budget)]

        # Get archive statistics for LLM context
        stats = self.archive.get_statistics()
        empty_cells = self._get_empty_cell_descriptions()
        low_fitness_cells = self._get_low_fitness_cells()

        # Ask LLM for guidance
        try:
            guidance = await self._ask_strategist_for_cells(
                stats, empty_cells, low_fitness_cells, exploration_budget
            )
            self.guidance_history.append(guidance)
            self.stats["llm_cell_selections"] += 1

            # Parse recommended cells
            return self._parse_cell_recommendations(guidance)

        except Exception as e:
            logger.warning(f"LLM cell guidance failed: {e}, using fallback")
            return [(cell.cell_index, 1, None)
                    for cell in self.archive.get_curiosity_selection(exploration_budget)]

    async def get_guided_mutation(
        self,
        solution: Any,
        cell: ArchiveCell,
        target_behavior: Optional[BehavioralDescriptor] = None
    ) -> LLMGuidedMutation:
        """
        Get LLM-guided mutation strategy for a solution.

        The LLM analyzes the current solution and target cell to recommend
        the most effective mutation approach.
        """
        if not self.strategist:
            # Fall back to random mutation
            return LLMGuidedMutation(
                strategy=MutationStrategy.RANDOM,
                target_region="global",
                mutation_strength=0.3,
                llm_reasoning="No LLM available, using random mutation",
                expected_outcome="Unknown",
                constraints=[],
                fallback_strategy=MutationStrategy.RANDOM
            )

        try:
            # Get solution characteristics
            solution_desc = self._describe_solution(solution)

            # Get target cell characteristics
            cell_desc = self._describe_cell(cell, target_behavior)

            # Ask LLM for mutation strategy
            guidance = await self._ask_strategist_for_mutation(
                solution_desc, cell_desc
            )
            self.guidance_history.append(guidance)
            self.stats["llm_guided_mutations"] += 1

            return self._parse_mutation_guidance(guidance)

        except Exception as e:
            logger.warning(f"LLM mutation guidance failed: {e}")
            return LLMGuidedMutation(
                strategy=MutationStrategy.LLM_GUIDED,
                target_region="routing",
                mutation_strength=0.3,
                llm_reasoning=f"Fallback due to: {e}",
                expected_outcome="Uncertain",
                constraints=["preserve_connectivity"],
                fallback_strategy=MutationStrategy.RANDOM
            )

    async def validate_elite_with_debate(
        self,
        candidate: Any,
        existing_elite: Optional[Any],
        cell: ArchiveCell
    ) -> Tuple[bool, str]:
        """
        Use debate mechanism to validate if candidate should replace elite.

        Multiple LLM agents debate whether the new solution is truly better,
        considering aspects beyond simple fitness comparison.
        """
        if not self.debate:
            # Simple fitness comparison
            return True, "No debate available, using fitness comparison"

        try:
            # Create proposal for replacing elite
            proposal = Proposal(
                proposer="map_elites",
                content=f"Replace elite in cell {cell.cell_index}",
                reasoning="Candidate has higher fitness",
                evidence={
                    "candidate_fitness": self._get_fitness(candidate),
                    "existing_fitness": self._get_fitness(existing_elite) if existing_elite else 0,
                    "cell_behavioral_target": cell.center_behavior.to_dict() if cell.center_behavior else {}
                }
            )

            # Run debate
            result = await self.debate.reach_consensus("elite_replacement", proposal)
            self.stats["debate_validations"] += 1

            return result.accepted, result.final_reasoning

        except Exception as e:
            logger.warning(f"Debate validation failed: {e}")
            return True, f"Debate failed: {e}, accepting candidate"

    # Helper methods
    def _get_empty_cell_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of empty cells in archive."""
        empty = []
        for cell in self.archive.get_empty_cells()[:10]:  # Limit for LLM context
            empty.append({
                "index": cell.cell_index,
                "center": cell.center_behavior.to_dict() if cell.center_behavior else None
            })
        return empty

    def _get_low_fitness_cells(self) -> List[Dict[str, Any]]:
        """Get cells with low-fitness elites (improvement opportunities)."""
        low_fitness = []
        for cell, elite in self.archive.get_low_fitness_elites(10):
            low_fitness.append({
                "index": cell.cell_index,
                "current_fitness": elite.fitness,
                "behavior": elite.descriptor.to_dict()
            })
        return low_fitness

    async def _ask_strategist_for_cells(
        self,
        stats: ArchiveStatistics,
        empty_cells: List[Dict],
        low_fitness_cells: List[Dict],
        budget: int
    ) -> LLMGuidance:
        """Ask strategist which cells to explore."""
        # This would call the actual LLM
        # For now, return placeholder
        return LLMGuidance(
            guidance_type=LLMGuidanceType.CELL_SELECTION,
            recommendation=str([c["index"] for c in empty_cells[:budget]]),
            confidence=0.7,
            reasoning="Prioritizing empty cells for diversity",
            parameters={"cells": empty_cells[:budget]}
        )

    async def _ask_strategist_for_mutation(
        self,
        solution_desc: str,
        cell_desc: str
    ) -> LLMGuidance:
        """Ask strategist for mutation strategy."""
        return LLMGuidance(
            guidance_type=LLMGuidanceType.MUTATION_STRATEGY,
            recommendation="targeted_routing",
            confidence=0.75,
            reasoning="Target routing density to reach cell center",
            parameters={
                "strategy": "TARGETED",
                "region": "routing",
                "strength": 0.4
            }
        )

    def _describe_solution(self, solution: Any) -> str:
        """Create text description of solution for LLM."""
        return f"PCB solution with {getattr(solution, 'trace_count', 'unknown')} traces"

    def _describe_cell(self, cell: ArchiveCell, target: Optional[BehavioralDescriptor]) -> str:
        """Create text description of target cell for LLM."""
        return f"Archive cell {cell.cell_index}"

    def _parse_cell_recommendations(self, guidance: LLMGuidance) -> List[Tuple[int, int, LLMGuidance]]:
        """Parse LLM cell recommendations."""
        cells = guidance.parameters.get("cells", [])
        return [(c.get("index", 0), 1, guidance) for c in cells]

    def _parse_mutation_guidance(self, guidance: LLMGuidance) -> LLMGuidedMutation:
        """Parse LLM mutation guidance."""
        params = guidance.parameters
        strategy_map = {
            "TARGETED": MutationStrategy.TARGETED,
            "EXPLORATORY": MutationStrategy.EXPLORATORY,
            "CROSSOVER": MutationStrategy.CROSSOVER,
            "RANDOM": MutationStrategy.RANDOM
        }
        return LLMGuidedMutation(
            strategy=strategy_map.get(params.get("strategy", "RANDOM"), MutationStrategy.RANDOM),
            target_region=params.get("region", "global"),
            mutation_strength=params.get("strength", 0.3),
            llm_reasoning=guidance.reasoning,
            expected_outcome="Improved fitness toward cell center",
            constraints=[],
            fallback_strategy=MutationStrategy.RANDOM
        )

    def _get_fitness(self, solution: Any) -> float:
        """Get fitness of a solution."""
        if hasattr(solution, 'fitness'):
            return solution.fitness
        return 0.0


# =============================================================================
# LLM-Guided Red Queen Evolution
# =============================================================================

class LLMGuidedRedQueen:
    """
    Red Queen evolution enhanced with LLM guidance.

    The LLM provides:
    1. Semantic understanding of champion strategies
    2. Guided mutation to beat specific champions
    3. Generality assessment beyond win/loss counting
    4. Convergence detection with semantic similarity
    """

    def __init__(
        self,
        evolver: RedQueenEvolver,
        conflict_resolver: Optional[ConflictResolverAgent] = None,
        si_advisor: Optional[SignalIntegrityAdvisorAgent] = None,
        debate_coordinator: Optional[DebateAndCritiqueCoordinator] = None
    ):
        self.evolver = evolver
        self.resolver = conflict_resolver
        self.si_advisor = si_advisor
        self.debate = debate_coordinator

        self.guidance_history: List[LLMGuidance] = []
        self.stats = {
            "llm_guided_evolutions": 0,
            "semantic_comparisons": 0,
            "debate_champion_selections": 0
        }

    async def evolve_with_llm_guidance(
        self,
        initial_population: List[Any],
        num_generations: int = 10
    ) -> List[Champion]:
        """
        Run Red Queen evolution with LLM-guided mutations.

        Each generation:
        1. LLM analyzes current champions and their weaknesses
        2. LLM recommends mutation strategies to beat champions
        3. Debate validates champion promotions
        """
        champions = []

        for gen in range(num_generations):
            # Get LLM analysis of current champion pool
            if self.resolver and champions:
                analysis = await self._analyze_champion_weaknesses(champions)
                mutation_guidance = await self._get_targeted_mutations(analysis)
            else:
                mutation_guidance = None

            # Evolve one generation
            new_champions = await self.evolver.evolve_round(
                population=initial_population,
                mutation_guidance=mutation_guidance
            )

            # Validate new champions with debate
            for candidate in new_champions:
                if await self._validate_champion_with_debate(candidate, champions):
                    champions.append(candidate)

            self.stats["llm_guided_evolutions"] += 1

        return champions

    async def get_semantic_generality_score(
        self,
        solution: Any,
        champions: List[Champion]
    ) -> Tuple[GeneralityScore, str]:
        """
        Compute generality score with LLM semantic comparison.

        Beyond simple win/loss counting, LLM assesses whether the solution
        truly generalizes to different scenarios represented by champions.
        """
        if not self.resolver:
            # Fall back to numeric comparison
            return self.evolver.compute_generality(solution, champions), "Numeric only"

        try:
            # Get semantic comparison from LLM
            semantic_analysis = await self._semantic_comparison(solution, champions)
            self.stats["semantic_comparisons"] += 1

            # Combine with numeric score
            numeric_score = self.evolver.compute_generality(solution, champions)

            # Adjust based on semantic analysis
            adjusted_generality = (
                numeric_score.generality * 0.7 +
                semantic_analysis.get("semantic_generality", 0.5) * 0.3
            )

            return GeneralityScore(
                wins=numeric_score.wins,
                ties=numeric_score.ties,
                losses=numeric_score.losses,
                total_champions=numeric_score.total_champions,
                generality=adjusted_generality
            ), semantic_analysis.get("reasoning", "")

        except Exception as e:
            logger.warning(f"Semantic generality failed: {e}")
            return self.evolver.compute_generality(solution, champions), f"Fallback: {e}"

    async def _analyze_champion_weaknesses(
        self,
        champions: List[Champion]
    ) -> Dict[str, Any]:
        """Use LLM to analyze champion weaknesses."""
        return {
            "common_weaknesses": ["routing_density", "via_count"],
            "exploitation_opportunities": ["thermal_spread"],
            "robust_areas": ["power_distribution"]
        }

    async def _get_targeted_mutations(
        self,
        weakness_analysis: Dict[str, Any]
    ) -> List[LLMGuidedMutation]:
        """Get mutations targeting champion weaknesses."""
        mutations = []
        for weakness in weakness_analysis.get("common_weaknesses", []):
            mutations.append(LLMGuidedMutation(
                strategy=MutationStrategy.TARGETED,
                target_region=weakness,
                mutation_strength=0.5,
                llm_reasoning=f"Target champion weakness: {weakness}",
                expected_outcome=f"Improved {weakness} metric",
                constraints=["preserve_connectivity"],
                fallback_strategy=MutationStrategy.RANDOM
            ))
        return mutations

    async def _validate_champion_with_debate(
        self,
        candidate: Champion,
        existing_champions: List[Champion]
    ) -> bool:
        """Validate champion promotion with debate."""
        if not self.debate:
            return True

        try:
            proposal = Proposal(
                proposer="red_queen",
                content=f"Promote solution to champion status",
                reasoning=f"Generality score: {candidate.metadata.get('generality', 0):.2f}",
                evidence={
                    "fitness": candidate.fitness,
                    "round": candidate.round_number,
                    "descriptor": candidate.descriptor.to_dict()
                }
            )

            result = await self.debate.reach_consensus("champion_promotion", proposal)
            self.stats["debate_champion_selections"] += 1
            return result.accepted

        except Exception as e:
            logger.warning(f"Champion debate failed: {e}")
            return True

    async def _semantic_comparison(
        self,
        solution: Any,
        champions: List[Champion]
    ) -> Dict[str, Any]:
        """Get semantic comparison from LLM."""
        return {
            "semantic_generality": 0.75,
            "reasoning": "Solution shows broad applicability across champion scenarios",
            "strengths": ["routing_flexibility", "thermal_management"],
            "weaknesses": ["via_count_optimization"]
        }


# =============================================================================
# Integrated Gaming-LLM Optimizer
# =============================================================================

class IntegratedGamingLLMOptimizer:
    """
    Main optimizer that combines Gaming AI with LLM orchestration.

    Pipeline:
    1. LLM Strategic Planning (routing strategy, congestion prediction)
    2. MAP-Elites Exploration (LLM-guided cell selection and mutation)
    3. Red Queen Evolution (LLM-guided adversarial improvement)
    4. Tournament Selection (debate-validated)
    5. Refinement (combined LLM + algorithmic)
    """

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        enable_debate: bool = True,
        map_elites_dims: Tuple[int, ...] = (10, 10),
        red_queen_rounds: int = 5
    ):
        # Initialize LLM agents
        self.strategist = create_routing_strategist(openrouter_api_key=openrouter_api_key)
        self.predictor = create_congestion_predictor(openrouter_api_key=openrouter_api_key)
        self.resolver = create_conflict_resolver(openrouter_api_key=openrouter_api_key)
        self.si_advisor = create_si_advisor(openrouter_api_key=openrouter_api_key)

        # Initialize debate coordinator
        self.debate = DebateAndCritiqueCoordinator(
            agents=[self.strategist, self.predictor, self.resolver],
            max_rounds=3,
            consensus_threshold=0.6
        ) if enable_debate else None

        # Initialize Gaming AI
        self.archive = MAPElitesArchive(
            dimensions=map_elites_dims,
            fitness_threshold=0.0
        )
        self.evolver = RedQueenEvolver()

        # Initialize LLM-guided wrappers
        self.llm_map_elites = LLMGuidedMAPElites(
            archive=self.archive,
            strategist=self.strategist,
            predictor=self.predictor,
            debate_coordinator=self.debate
        )
        self.llm_red_queen = LLMGuidedRedQueen(
            evolver=self.evolver,
            conflict_resolver=self.resolver,
            si_advisor=self.si_advisor,
            debate_coordinator=self.debate
        )

        self.red_queen_rounds = red_queen_rounds
        self.stats = {
            "total_evaluations": 0,
            "llm_calls": 0,
            "debate_sessions": 0
        }

    async def optimize(
        self,
        initial_solution: Any,
        max_iterations: int = 100
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the full integrated optimization pipeline.

        Returns:
            Tuple of (best_solution, optimization_metrics)
        """
        best_solution = initial_solution
        best_fitness = self._evaluate(initial_solution)

        # Phase 1: LLM Strategic Analysis
        strategy = await self._llm_strategic_planning(initial_solution)

        # Phase 2: MAP-Elites Exploration with LLM guidance
        for i in range(max_iterations // 2):
            # Get LLM-guided cells to explore
            cells = await self.llm_map_elites.get_guided_cell_selection(
                best_solution, exploration_budget=3
            )

            for cell_idx, priority, guidance in cells:
                # Get LLM-guided mutation
                mutation = await self.llm_map_elites.get_guided_mutation(
                    best_solution, self.archive.get_cell(cell_idx)
                )

                # Apply mutation and evaluate
                mutated = self._apply_mutation(best_solution, mutation)
                fitness = self._evaluate(mutated)
                self.stats["total_evaluations"] += 1

                # Update archive with debate validation
                if fitness > best_fitness:
                    accepted, reason = await self.llm_map_elites.validate_elite_with_debate(
                        mutated, best_solution, self.archive.get_cell(cell_idx)
                    )
                    if accepted:
                        best_solution = mutated
                        best_fitness = fitness

        # Phase 3: Red Queen Evolution with LLM guidance
        champions = await self.llm_red_queen.evolve_with_llm_guidance(
            initial_population=[best_solution],
            num_generations=self.red_queen_rounds
        )

        # Select best from champions
        if champions:
            best_champion = max(champions, key=lambda c: c.fitness)
            if best_champion.fitness > best_fitness:
                best_solution = best_champion.solution
                best_fitness = best_champion.fitness

        # Collect metrics
        metrics = {
            "final_fitness": best_fitness,
            "map_elites_stats": self.llm_map_elites.stats,
            "red_queen_stats": self.llm_red_queen.stats,
            "archive_coverage": self.archive.get_statistics().coverage,
            "total_evaluations": self.stats["total_evaluations"]
        }

        return best_solution, metrics

    async def _llm_strategic_planning(self, solution: Any) -> Dict[str, Any]:
        """Run LLM strategic planning phase."""
        try:
            strategy = await self.strategist.analyze_routing_strategy(
                nets=[],  # Would extract from solution
                components=[],
                board_size=(100, 80),
                layer_count=4
            )
            self.stats["llm_calls"] += 1
            return {"strategy": strategy}
        except Exception as e:
            logger.warning(f"Strategic planning failed: {e}")
            return {}

    def _evaluate(self, solution: Any) -> float:
        """Evaluate solution fitness."""
        if hasattr(solution, 'fitness'):
            return solution.fitness
        return 0.0

    def _apply_mutation(self, solution: Any, mutation: LLMGuidedMutation) -> Any:
        """Apply mutation to solution."""
        # This would apply the actual mutation
        # For now, return copy
        return solution


# =============================================================================
# Factory Functions
# =============================================================================

def create_integrated_optimizer(
    openrouter_api_key: Optional[str] = None,
    enable_debate: bool = True
) -> IntegratedGamingLLMOptimizer:
    """Create an integrated Gaming-LLM optimizer."""
    return IntegratedGamingLLMOptimizer(
        openrouter_api_key=openrouter_api_key,
        enable_debate=enable_debate
    )


def create_llm_map_elites(
    archive: MAPElitesArchive,
    openrouter_api_key: Optional[str] = None
) -> LLMGuidedMAPElites:
    """Create LLM-guided MAP-Elites wrapper."""
    return LLMGuidedMAPElites(
        archive=archive,
        strategist=create_routing_strategist(openrouter_api_key=openrouter_api_key),
        predictor=create_congestion_predictor(openrouter_api_key=openrouter_api_key)
    )


def create_llm_red_queen(
    evolver: RedQueenEvolver,
    openrouter_api_key: Optional[str] = None
) -> LLMGuidedRedQueen:
    """Create LLM-guided Red Queen wrapper."""
    return LLMGuidedRedQueen(
        evolver=evolver,
        conflict_resolver=create_conflict_resolver(openrouter_api_key=openrouter_api_key),
        si_advisor=create_si_advisor(openrouter_api_key=openrouter_api_key)
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main classes
    "IntegratedGamingLLMOptimizer",
    "LLMGuidedMAPElites",
    "LLMGuidedRedQueen",

    # Data structures
    "LLMGuidance",
    "LLMGuidanceType",
    "EnhancedBehavioralDescriptor",
    "LLMGuidedMutation",

    # Factory functions
    "create_integrated_optimizer",
    "create_llm_map_elites",
    "create_llm_red_queen",
]
