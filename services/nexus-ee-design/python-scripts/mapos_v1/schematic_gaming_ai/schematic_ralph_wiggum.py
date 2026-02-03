"""
Schematic Ralph Wiggum Optimizer

Persistent iteration loop that continues optimizing a schematic
until target fitness is achieved or max iterations reached.

"Me fail optimization? That's unpossible!"
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union
from datetime import datetime
from enum import Enum

import numpy as np

from ..gaming_ai.ralph_wiggum_optimizer import file_lock, atomic_write_json
from .config import get_schematic_config, RalphWiggumConfig
from .behavior_descriptor import SchematicBehaviorDescriptor, compute_schematic_descriptor
from .fitness_function import SchematicFitness, compute_schematic_fitness
from .mutation_operators import SchematicMutator, MutationResult
from .schematic_map_elites import SchematicMAPElitesArchive
from .schematic_red_queen import SchematicRedQueenEvolver

logger = logging.getLogger(__name__)


class SchematicOptimizationStatus(Enum):
    """Status of schematic optimization."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SchematicCompletionCriteria:
    """Criteria for completing optimization."""
    target_fitness: float = 0.95
    max_iterations: int = 500
    max_stagnation: int = 50
    min_improvement: float = 0.001  # Minimum improvement to reset stagnation

    def is_satisfied(
        self,
        fitness: float,
        iteration: int,
        stagnation: int
    ) -> tuple[bool, str]:
        """Check if criteria are satisfied."""
        if fitness >= self.target_fitness:
            return True, "target_fitness_reached"
        if iteration >= self.max_iterations:
            return True, "max_iterations"
        if stagnation >= self.max_stagnation:
            return True, "stagnation"
        return False, "in_progress"


@dataclass
class SchematicOptimizationState:
    """Persistent state for schematic optimization."""
    project_id: str
    iteration: int = 0
    best_fitness: float = 0.0
    current_fitness: float = 0.0
    stagnation_count: int = 0
    status: SchematicOptimizationStatus = SchematicOptimizationStatus.NOT_STARTED
    best_schematic: Optional[Dict[str, Any]] = None
    fitness_history: List[float] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completion_reason: str = ""

    def save(self, state_dir: Path) -> None:
        """Save state to file with locking."""
        state_file = state_dir / "state.json"
        state_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "project_id": self.project_id,
            "iteration": self.iteration,
            "best_fitness": self.best_fitness,
            "current_fitness": self.current_fitness,
            "stagnation_count": self.stagnation_count,
            "status": self.status.value,
            "fitness_history": self.fitness_history,
            "mutation_history": self.mutation_history[-100:],  # Keep last 100
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "completion_reason": self.completion_reason,
        }

        with file_lock(state_file):
            atomic_write_json(state_file, data)

        # Save best schematic separately
        if self.best_schematic:
            schematic_file = state_dir / "best_schematic.json"
            with file_lock(schematic_file):
                atomic_write_json(schematic_file, self.best_schematic)

    @classmethod
    def load(cls, state_dir: Path, project_id: str) -> "SchematicOptimizationState":
        """Load state from file."""
        state_file = state_dir / "state.json"

        if not state_file.exists():
            return cls(project_id=project_id)

        with file_lock(state_file):
            with open(state_file) as f:
                data = json.load(f)

        # Load schematic
        schematic_file = state_dir / "best_schematic.json"
        best_schematic = None
        if schematic_file.exists():
            with open(schematic_file) as f:
                best_schematic = json.load(f)

        return cls(
            project_id=data.get("project_id", project_id),
            iteration=data.get("iteration", 0),
            best_fitness=data.get("best_fitness", 0.0),
            current_fitness=data.get("current_fitness", 0.0),
            stagnation_count=data.get("stagnation_count", 0),
            status=SchematicOptimizationStatus(data.get("status", "not_started")),
            best_schematic=best_schematic,
            fitness_history=data.get("fitness_history", []),
            mutation_history=data.get("mutation_history", []),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            completion_reason=data.get("completion_reason", ""),
        )


@dataclass
class SchematicOptimizationResult:
    """Result of schematic optimization."""
    success: bool
    schematic: Dict[str, Any]
    fitness: SchematicFitness
    descriptor: SchematicBehaviorDescriptor
    iterations: int
    completion_reason: str
    duration_seconds: float
    fitness_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "fitness": self.fitness.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "iterations": self.iterations,
            "completion_reason": self.completion_reason,
            "duration_seconds": self.duration_seconds,
            "fitness_history": self.fitness_history,
        }


class SchematicRalphWiggumOptimizer:
    """
    Ralph Wiggum persistent optimization loop for schematics.

    Continues iterating until target fitness is achieved,
    using file-based state persistence for fault tolerance.
    """

    def __init__(
        self,
        project_id: str,
        state_dir: Optional[Path] = None,
        config: Optional[RalphWiggumConfig] = None,
        llm_client: Optional[Any] = None,
        validation_callback: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
    ):
        """
        Initialize Ralph Wiggum optimizer.

        Args:
            project_id: Unique project identifier
            state_dir: Directory for state persistence
            config: Ralph Wiggum configuration
            llm_client: LLM client for guided mutations
            validation_callback: Async callback for schematic validation
        """
        self.project_id = project_id
        self.config = config or get_schematic_config().ralph_wiggum
        self.llm_client = llm_client
        self.validation_callback = validation_callback

        # State directory
        if state_dir is None:
            state_dir = Path.home() / ".schematic_ralph" / project_id
        self.state_dir = state_dir

        # Components
        self.mutator = SchematicMutator(llm_client=llm_client)
        self.archive = SchematicMAPElitesArchive()

        # Completion criteria
        self.criteria = SchematicCompletionCriteria(
            target_fitness=self.config.target_fitness,
            max_iterations=self.config.max_iterations,
            max_stagnation=self.config.stagnation_threshold,
        )

        # Load or create state
        self.state = SchematicOptimizationState.load(self.state_dir, project_id)

        # Cancellation flag
        self._cancelled = False

        logger.info(f"Initialized Ralph Wiggum optimizer for {project_id}")

    async def optimize(
        self,
        initial_schematic: Dict[str, Any]
    ) -> SchematicOptimizationResult:
        """
        Run optimization loop.

        Args:
            initial_schematic: Starting schematic

        Returns:
            SchematicOptimizationResult
        """
        start_time = datetime.now()
        self.state.started_at = start_time
        self.state.status = SchematicOptimizationStatus.IN_PROGRESS

        # Initialize with starting schematic if no best exists
        if self.state.best_schematic is None:
            fitness = compute_schematic_fitness(initial_schematic)
            self.state.best_schematic = initial_schematic
            self.state.best_fitness = fitness.total
            self.archive.add(initial_schematic, fitness)

        logger.info(
            f"Starting optimization from iteration {self.state.iteration}, "
            f"best fitness = {self.state.best_fitness:.3f}"
        )

        try:
            while not self._cancelled:
                self.state.iteration += 1

                # Check completion
                done, reason = self.criteria.is_satisfied(
                    self.state.best_fitness,
                    self.state.iteration,
                    self.state.stagnation_count
                )

                if done:
                    self.state.completion_reason = reason
                    if reason == "target_fitness_reached":
                        self.state.status = SchematicOptimizationStatus.SUCCEEDED
                    else:
                        self.state.status = SchematicOptimizationStatus.FAILED
                    break

                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > self.config.total_timeout:
                    self.state.completion_reason = "timeout"
                    self.state.status = SchematicOptimizationStatus.TIMEOUT
                    break

                # Run iteration
                await self._run_iteration()

                # Checkpoint
                if self.state.iteration % self.config.checkpoint_interval == 0:
                    self.state.save(self.state_dir)
                    logger.debug(
                        f"Checkpoint at iteration {self.state.iteration}, "
                        f"fitness = {self.state.best_fitness:.3f}"
                    )

                # Brief pause
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            self.state.status = SchematicOptimizationStatus.CANCELLED
            self.state.completion_reason = "cancelled"
            raise

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.state.status = SchematicOptimizationStatus.FAILED
            self.state.completion_reason = f"error: {str(e)}"

        # Finalize
        self.state.completed_at = datetime.now()
        self.state.save(self.state_dir)

        duration = (self.state.completed_at - start_time).total_seconds()

        # Get final results
        final_fitness = compute_schematic_fitness(self.state.best_schematic)
        final_descriptor = compute_schematic_descriptor(self.state.best_schematic)

        logger.info(
            f"Optimization complete: {self.state.status.value}, "
            f"fitness = {final_fitness.total:.3f}, "
            f"iterations = {self.state.iteration}, "
            f"duration = {duration:.1f}s"
        )

        return SchematicOptimizationResult(
            success=self.state.status == SchematicOptimizationStatus.SUCCEEDED,
            schematic=self.state.best_schematic,
            fitness=final_fitness,
            descriptor=final_descriptor,
            iterations=self.state.iteration,
            completion_reason=self.state.completion_reason,
            duration_seconds=duration,
            fitness_history=self.state.fitness_history,
        )

    async def _run_iteration(self) -> None:
        """Run single optimization iteration."""
        # Adjust mutation rate based on stagnation
        escalation_factor = 1.0
        if self.state.stagnation_count > self.config.stagnation_threshold // 2:
            escalation_factor = self.config.escalation_multiplier

        # Mutate best schematic
        try:
            mutated, mutation_result = await self.mutator.mutate(
                self.state.best_schematic,
                stagnation_count=self.state.stagnation_count,
            )
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            self.state.stagnation_count += 1
            return

        if not mutation_result.success:
            self.state.stagnation_count += 1
            return

        # Validate if callback provided
        if self.validation_callback:
            try:
                validation = await asyncio.wait_for(
                    self.validation_callback(mutated),
                    timeout=self.config.iteration_timeout
                )
                mutated["validation_results"] = validation
            except asyncio.TimeoutError:
                logger.warning("Validation timeout")
            except Exception as e:
                logger.warning(f"Validation failed: {e}")

        # Compute fitness
        fitness = compute_schematic_fitness(mutated)
        self.state.current_fitness = fitness.total
        self.state.fitness_history.append(fitness.total)

        # Check improvement
        improvement = fitness.total - self.state.best_fitness

        if improvement > self.criteria.min_improvement:
            # Improvement found!
            self.state.best_schematic = mutated
            self.state.best_fitness = fitness.total
            self.state.stagnation_count = 0

            # Add to archive
            descriptor = compute_schematic_descriptor(mutated)
            self.archive.add(mutated, fitness, descriptor)

            logger.debug(
                f"Iteration {self.state.iteration}: "
                f"improvement +{improvement:.4f} -> {fitness.total:.3f}"
            )

            # Record mutation
            self.state.mutation_history.append({
                "iteration": self.state.iteration,
                "strategy": mutation_result.strategy.value,
                "improvement": improvement,
                "fitness": fitness.total,
            })
        else:
            self.state.stagnation_count += 1

            # Still add to archive for diversity
            if improvement > -0.01:  # Not too much worse
                descriptor = compute_schematic_descriptor(mutated)
                self.archive.add(mutated, fitness, descriptor)

    def cancel(self) -> None:
        """Cancel optimization."""
        self._cancelled = True
        logger.info("Optimization cancelled")

    def get_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "project_id": self.project_id,
            "status": self.state.status.value,
            "iteration": self.state.iteration,
            "best_fitness": self.state.best_fitness,
            "current_fitness": self.state.current_fitness,
            "stagnation_count": self.state.stagnation_count,
            "target_fitness": self.criteria.target_fitness,
            "max_iterations": self.criteria.max_iterations,
            "archive_size": len(self.archive.archive),
        }

    def get_diverse_solutions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n diverse solutions from archive."""
        cells = self.archive.get_diverse_sample(n)
        return [c.schematic for c in cells]
