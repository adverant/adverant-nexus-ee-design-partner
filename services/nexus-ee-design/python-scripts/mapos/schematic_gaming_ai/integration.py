"""
Schematic Gaming AI Integration

High-level integration functions for schematic optimization
that combine MAP-Elites, Red Queen, and Ralph Wiggum approaches.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union

from ..gaming_ai.llm_backends import LLMClient, get_llm_backends
from .config import (
    SchematicGamingAIConfig,
    SchematicOptimizationMode,
    get_schematic_config,
)
from .behavior_descriptor import SchematicBehaviorDescriptor, compute_schematic_descriptor
from .fitness_function import SchematicFitness, compute_schematic_fitness, FitnessWeights
from .mutation_operators import SchematicMutator
from .schematic_map_elites import SchematicMAPElitesArchive, SchematicArchiveStatistics
from .schematic_red_queen import SchematicRedQueenEvolver, SchematicChampion
from .schematic_ralph_wiggum import (
    SchematicRalphWiggumOptimizer,
    SchematicOptimizationResult,
    SchematicOptimizationStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class SchematicOptimizationRequest:
    """Request for schematic optimization."""
    project_id: str
    schematic: Dict[str, Any]
    target_fitness: float = 0.95
    max_iterations: int = 500
    mode: SchematicOptimizationMode = SchematicOptimizationMode.HYBRID
    algorithms: List[str] = field(default_factory=lambda: ["map_elites", "red_queen", "ralph_wiggum"])
    timeout_seconds: float = 3600.0
    custom_weights: Optional[FitnessWeights] = None


@dataclass
class SchematicOptimizationResponse:
    """Response from schematic optimization."""
    success: bool
    optimized_schematic: Dict[str, Any]
    fitness: SchematicFitness
    descriptor: SchematicBehaviorDescriptor
    iterations_run: int
    completion_reason: str
    duration_seconds: float
    fitness_history: List[float]
    archive_statistics: Optional[SchematicArchiveStatistics] = None
    diverse_solutions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "fitness": self.fitness.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "iterations_run": self.iterations_run,
            "completion_reason": self.completion_reason,
            "duration_seconds": self.duration_seconds,
            "fitness_history": self.fitness_history,
            "archive_statistics": self.archive_statistics.to_dict() if self.archive_statistics else None,
            "diverse_solutions_count": len(self.diverse_solutions),
        }


class SchematicOptimizer:
    """
    Main orchestrator for schematic optimization.

    Combines MAP-Elites, Red Queen, and Ralph Wiggum approaches
    for comprehensive schematic quality-diversity optimization.
    """

    def __init__(
        self,
        config: Optional[SchematicGamingAIConfig] = None,
        llm_client: Optional[LLMClient] = None,
        validation_callback: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
    ):
        """
        Initialize schematic optimizer.

        Args:
            config: Gaming AI configuration
            llm_client: LLM client for guided mutations
            validation_callback: Async callback for schematic validation (ERC)
        """
        self.config = config or get_schematic_config()
        self.validation_callback = validation_callback

        # Initialize LLM client
        if llm_client is None and self.config.llm_api_key:
            encoder, estimator, generator, _ = get_llm_backends(
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
            )
            self.llm_client = LLMClient(
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
            )
        else:
            self.llm_client = llm_client

        # Components (initialized on first use)
        self.archive: Optional[SchematicMAPElitesArchive] = None
        self.mutator: Optional[SchematicMutator] = None
        self.evolver: Optional[SchematicRedQueenEvolver] = None

        logger.info(f"Initialized SchematicOptimizer with mode={self.config.mode}")

    async def optimize(
        self,
        request: SchematicOptimizationRequest
    ) -> SchematicOptimizationResponse:
        """
        Run schematic optimization.

        Args:
            request: Optimization request

        Returns:
            SchematicOptimizationResponse
        """
        logger.info(f"Starting optimization for project {request.project_id}")

        # Initialize components
        self._init_components()

        # Route based on mode
        if request.mode == SchematicOptimizationMode.STANDARD:
            return await self._optimize_standard(request)
        elif request.mode == SchematicOptimizationMode.GAMING_AI:
            return await self._optimize_full_gaming_ai(request)
        elif request.mode == SchematicOptimizationMode.HYBRID:
            return await self._optimize_hybrid(request)
        elif request.mode == SchematicOptimizationMode.FAST:
            return await self._optimize_fast(request)
        else:
            raise ValueError(f"Unknown mode: {request.mode}")

    def _init_components(self) -> None:
        """Initialize optimization components."""
        if self.archive is None:
            self.archive = SchematicMAPElitesArchive(config=self.config.archive)

        if self.mutator is None:
            self.mutator = SchematicMutator(
                llm_client=self.llm_client,
                config=self.config.mutation,
            )

        if self.evolver is None:
            self.evolver = SchematicRedQueenEvolver(
                archive=self.archive,
                mutator=self.mutator,
                config=self.config.evolution,
                llm_client=self.llm_client,
            )

    async def _optimize_standard(
        self,
        request: SchematicOptimizationRequest
    ) -> SchematicOptimizationResponse:
        """Standard optimization (no gaming AI)."""
        # Just compute fitness and return
        fitness = compute_schematic_fitness(
            request.schematic,
            weights=request.custom_weights,
        )
        descriptor = compute_schematic_descriptor(request.schematic)

        return SchematicOptimizationResponse(
            success=fitness.is_passing(),
            optimized_schematic=request.schematic,
            fitness=fitness,
            descriptor=descriptor,
            iterations_run=0,
            completion_reason="standard_mode",
            duration_seconds=0.0,
            fitness_history=[fitness.total],
        )

    async def _optimize_full_gaming_ai(
        self,
        request: SchematicOptimizationRequest
    ) -> SchematicOptimizationResponse:
        """Full gaming AI optimization (MAP-Elites + Red Queen + Ralph Wiggum)."""
        # Create Ralph Wiggum optimizer
        ralph = SchematicRalphWiggumOptimizer(
            project_id=request.project_id,
            config=self.config.ralph_wiggum,
            llm_client=self.llm_client,
            validation_callback=self.validation_callback,
        )

        # Override config from request
        ralph.criteria.target_fitness = request.target_fitness
        ralph.criteria.max_iterations = request.max_iterations
        ralph.config.total_timeout = request.timeout_seconds

        # Run optimization
        result = await ralph.optimize(request.schematic)

        # Get archive statistics
        archive_stats = ralph.archive.get_statistics()
        diverse = ralph.get_diverse_solutions(n=5)

        return SchematicOptimizationResponse(
            success=result.success,
            optimized_schematic=result.schematic,
            fitness=result.fitness,
            descriptor=result.descriptor,
            iterations_run=result.iterations,
            completion_reason=result.completion_reason,
            duration_seconds=result.duration_seconds,
            fitness_history=result.fitness_history,
            archive_statistics=archive_stats,
            diverse_solutions=diverse,
        )

    async def _optimize_hybrid(
        self,
        request: SchematicOptimizationRequest
    ) -> SchematicOptimizationResponse:
        """Hybrid optimization (LLM-guided with selective gaming AI)."""
        import time
        start_time = time.time()

        # Compute initial fitness
        initial_fitness = compute_schematic_fitness(
            request.schematic,
            weights=request.custom_weights,
        )

        # Quick check - if already excellent, return early
        if initial_fitness.is_excellent():
            descriptor = compute_schematic_descriptor(request.schematic)
            return SchematicOptimizationResponse(
                success=True,
                optimized_schematic=request.schematic,
                fitness=initial_fitness,
                descriptor=descriptor,
                iterations_run=0,
                completion_reason="already_excellent",
                duration_seconds=time.time() - start_time,
                fitness_history=[initial_fitness.total],
            )

        # Add to archive
        self.archive.add(request.schematic, initial_fitness)

        # Run LLM-guided mutations first (fast)
        best_schematic = request.schematic
        best_fitness = initial_fitness
        fitness_history = [initial_fitness.total]
        iterations = 0

        # Phase 1: Quick LLM-guided iterations
        max_quick_iterations = min(50, request.max_iterations // 2)

        for i in range(max_quick_iterations):
            iterations += 1

            # Mutate
            mutated, mutation_result = await self.mutator.mutate(
                best_schematic,
                stagnation_count=max(0, i - 10),
            )

            if not mutation_result.success:
                continue

            # Validate if callback provided
            if self.validation_callback:
                try:
                    validation = await self.validation_callback(mutated)
                    mutated["validation_results"] = validation
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")

            # Compute fitness
            fitness = compute_schematic_fitness(mutated, weights=request.custom_weights)
            fitness_history.append(fitness.total)

            # Update best
            if fitness.total > best_fitness.total:
                best_schematic = mutated
                best_fitness = fitness
                self.archive.add(mutated, fitness)

            # Early exit if target reached
            if best_fitness.total >= request.target_fitness:
                break

            # Timeout check
            if time.time() - start_time > request.timeout_seconds:
                break

        # Phase 2: Red Queen if not yet at target (optional)
        if best_fitness.total < request.target_fitness and "red_queen" in request.algorithms:
            # Initialize evolver with current population
            await self.evolver.initialize_population([best_schematic])

            # Run a few generations
            remaining_iterations = request.max_iterations - iterations
            max_generations = min(10, remaining_iterations // 10)

            for gen in range(max_generations):
                champion, rounds = await self.evolver.evolve_generation(
                    validation_callback=self.validation_callback
                )

                if champion.fitness.total > best_fitness.total:
                    best_schematic = champion.schematic
                    best_fitness = champion.fitness

                fitness_history.append(best_fitness.total)
                iterations += len(rounds)

                if best_fitness.total >= request.target_fitness:
                    break

                if time.time() - start_time > request.timeout_seconds:
                    break

        # Finalize
        duration = time.time() - start_time
        descriptor = compute_schematic_descriptor(best_schematic)

        # Determine completion reason
        if best_fitness.total >= request.target_fitness:
            completion_reason = "target_reached"
            success = True
        elif time.time() - start_time > request.timeout_seconds:
            completion_reason = "timeout"
            success = best_fitness.is_passing()
        else:
            completion_reason = "iterations_complete"
            success = best_fitness.is_passing()

        return SchematicOptimizationResponse(
            success=success,
            optimized_schematic=best_schematic,
            fitness=best_fitness,
            descriptor=descriptor,
            iterations_run=iterations,
            completion_reason=completion_reason,
            duration_seconds=duration,
            fitness_history=fitness_history,
            archive_statistics=self.archive.get_statistics(),
            diverse_solutions=self.archive.get_diverse_sample(5),
        )

    async def _optimize_fast(
        self,
        request: SchematicOptimizationRequest
    ) -> SchematicOptimizationResponse:
        """Fast optimization (quick iterations, fewer mutations)."""
        import time
        start_time = time.time()

        # Compute initial fitness
        initial_fitness = compute_schematic_fitness(
            request.schematic,
            weights=request.custom_weights,
        )

        best_schematic = request.schematic
        best_fitness = initial_fitness
        fitness_history = [initial_fitness.total]

        # Quick iterations (max 20)
        max_iterations = min(20, request.max_iterations)

        for i in range(max_iterations):
            mutated, mutation_result = await self.mutator.mutate(best_schematic)

            if not mutation_result.success:
                continue

            fitness = compute_schematic_fitness(mutated, weights=request.custom_weights)
            fitness_history.append(fitness.total)

            if fitness.total > best_fitness.total:
                best_schematic = mutated
                best_fitness = fitness

            if best_fitness.total >= request.target_fitness:
                break

        duration = time.time() - start_time
        descriptor = compute_schematic_descriptor(best_schematic)

        return SchematicOptimizationResponse(
            success=best_fitness.is_passing(),
            optimized_schematic=best_schematic,
            fitness=best_fitness,
            descriptor=descriptor,
            iterations_run=len(fitness_history) - 1,
            completion_reason="fast_mode",
            duration_seconds=duration,
            fitness_history=fitness_history,
        )


async def optimize_schematic(
    project_id: str,
    schematic: Dict[str, Any],
    target_fitness: float = 0.95,
    max_iterations: int = 500,
    mode: Union[str, SchematicOptimizationMode] = "hybrid",
    validation_callback: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
) -> SchematicOptimizationResponse:
    """
    Convenience function to optimize a schematic.

    Args:
        project_id: Project identifier
        schematic: Schematic to optimize
        target_fitness: Target fitness score (0-1)
        max_iterations: Maximum iterations
        mode: Optimization mode ("standard", "gaming_ai", "hybrid", "fast")
        validation_callback: Optional ERC validation callback

    Returns:
        SchematicOptimizationResponse
    """
    if isinstance(mode, str):
        mode = SchematicOptimizationMode(mode)

    request = SchematicOptimizationRequest(
        project_id=project_id,
        schematic=schematic,
        target_fitness=target_fitness,
        max_iterations=max_iterations,
        mode=mode,
    )

    optimizer = SchematicOptimizer(validation_callback=validation_callback)
    return await optimizer.optimize(request)
