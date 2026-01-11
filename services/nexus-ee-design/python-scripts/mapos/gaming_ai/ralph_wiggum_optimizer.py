"""
Ralph Wiggum Optimizer - Persistent Iteration Until Success

This module implements the Ralph Wiggum technique: an infinite improvement
loop that persists progress in files rather than context, enabling
indefinite optimization until success criteria are met.

Core Philosophy: "Iteration beats perfection when you have clear goals
and automatic verification."

Key Mechanisms:
1. File-Based Persistence: Progress stored in PCB files and state JSON
2. Completion Criteria: Configurable success conditions
3. Stagnation Detection: Escalation when progress stalls
4. Git Integration: Version control for optimization history
5. File Locking: Safe concurrent access to state files
6. Atomic Writes: Prevent corruption on interruption

References:
- Ralph Wiggum Technique: https://awesomeclaude.ai/ralph-wiggum
- Iterative Optimization: https://en.wikipedia.org/wiki/Iterative_method
"""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from enum import Enum, auto
from contextlib import contextmanager

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# File locking support (Unix/macOS)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    logger.warning("fcntl not available - file locking disabled")

# Local imports
from .red_queen_evolver import RedQueenEvolver, Champion
from .map_elites import BehavioralDescriptor


@contextmanager
def file_lock(filepath: Path, timeout: float = 30.0):
    """
    Context manager for file locking.

    Prevents concurrent access to state files in multi-pod K8s deployments.
    Falls back to no locking if fcntl not available.
    """
    lock_path = Path(str(filepath) + ".lock")

    if not HAS_FCNTL:
        yield
        return

    lock_file = None
    try:
        lock_file = open(lock_path, 'w')
        # Non-blocking lock with timeout
        start = datetime.now()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                elapsed = (datetime.now() - start).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Could not acquire lock on {filepath} within {timeout}s")
                import time
                time.sleep(0.1)

        yield

    finally:
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass


def atomic_write_json(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON atomically to prevent corruption.

    Writes to temp file then renames for atomicity.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=filepath.stem + "_",
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        # Atomic rename
        os.replace(temp_path, filepath)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


class OptimizationStatus(Enum):
    """Status of the optimization run."""
    RUNNING = auto()
    SUCCESS = auto()
    STAGNATED = auto()
    MAX_ITERATIONS = auto()
    FAILED = auto()
    INTERRUPTED = auto()


class EscalationStrategy(Enum):
    """Strategies when optimization stagnates."""
    INCREASE_MUTATION = auto()      # Increase mutation rate
    RESET_POPULATION = auto()        # Reset to diverse starting points
    SWITCH_AGENTS = auto()           # Change agent focus
    EXPAND_SEARCH = auto()           # Explore more behavioral dimensions
    CALL_FOR_HELP = auto()           # Request human intervention


@dataclass
class CompletionCriteria:
    """
    Configurable criteria for optimization completion.

    Success is achieved when ALL criteria are met.
    """
    target_violations: int = 50           # Maximum acceptable DRC violations
    target_fitness: float = 0.9           # Minimum fitness score
    max_iterations: int = 100             # Maximum iterations before stopping
    max_stagnation: int = 15              # Iterations without improvement
    min_improvement_rate: float = 0.01    # Minimum improvement per iteration
    target_generality: float = 0.8        # Required generality score
    max_duration_hours: float = 24.0      # Maximum run duration

    def is_success(
        self,
        violations: int,
        fitness: float,
        generality: float = 1.0,
    ) -> bool:
        """Check if optimization has succeeded."""
        return (
            violations <= self.target_violations and
            fitness >= self.target_fitness and
            generality >= self.target_generality
        )

    def should_stop(
        self,
        iteration: int,
        stagnation_count: int,
        duration_hours: float,
    ) -> Tuple[bool, str]:
        """
        Check if optimization should stop.

        Returns:
            (should_stop, reason)
        """
        if iteration >= self.max_iterations:
            return True, "max_iterations"
        if stagnation_count >= self.max_stagnation:
            return True, "stagnation"
        if duration_hours >= self.max_duration_hours:
            return True, "max_duration"
        return False, ""


@dataclass
class OptimizationState:
    """
    Persistent state for the optimization run.

    This state is saved to disk after each iteration,
    enabling resume from any point.
    """
    pcb_path: str
    iteration: int = 0
    best_violations: int = 999999
    best_fitness: float = 0.0
    best_generality: float = 0.0
    stagnation_count: int = 0
    violations_history: List[int] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    escalation_count: int = 0
    current_strategy: str = "normal"
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    red_queen_rounds: int = 0
    champions_count: int = 0
    status: str = "running"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pcb_path': self.pcb_path,
            'iteration': self.iteration,
            'best_violations': self.best_violations,
            'best_fitness': self.best_fitness,
            'best_generality': self.best_generality,
            'stagnation_count': self.stagnation_count,
            'violations_history': self.violations_history,
            'fitness_history': self.fitness_history,
            'escalation_count': self.escalation_count,
            'current_strategy': self.current_strategy,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'red_queen_rounds': self.red_queen_rounds,
            'champions_count': self.champions_count,
            'status': self.status,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationState':
        return cls(
            pcb_path=data['pcb_path'],
            iteration=data.get('iteration', 0),
            best_violations=data.get('best_violations', 999999),
            best_fitness=data.get('best_fitness', 0.0),
            best_generality=data.get('best_generality', 0.0),
            stagnation_count=data.get('stagnation_count', 0),
            violations_history=data.get('violations_history', []),
            fitness_history=data.get('fitness_history', []),
            escalation_count=data.get('escalation_count', 0),
            current_strategy=data.get('current_strategy', 'normal'),
            start_time=data.get('start_time', datetime.now().isoformat()),
            last_update=data.get('last_update', datetime.now().isoformat()),
            red_queen_rounds=data.get('red_queen_rounds', 0),
            champions_count=data.get('champions_count', 0),
            status=data.get('status', 'running'),
            metadata=data.get('metadata', {}),
        )

    def save(self, path: Path) -> None:
        """
        Save state to JSON file with atomic write and file locking.

        Uses atomic writes to prevent corruption if interrupted.
        Uses file locking to prevent concurrent access in K8s.
        """
        with file_lock(path):
            atomic_write_json(path, self.to_dict())
        logger.debug(f"Saved optimization state to {path}")

    @classmethod
    def load(cls, path: Path) -> 'OptimizationState':
        """
        Load state from JSON file with validation.

        Uses file locking for safe concurrent access.
        """
        with file_lock(path):
            try:
                with open(path) as f:
                    data = json.load(f)

                # Validate required fields
                if 'pcb_path' not in data:
                    raise ValueError("Invalid state file: missing pcb_path")

                return cls.from_dict(data)

            except json.JSONDecodeError as e:
                logger.error(f"Corrupted state file {path}: {e}")
                raise ValueError(f"Corrupted optimization state: {e}")
            except Exception as e:
                logger.error(f"Failed to load state from {path}: {e}")
                raise


@dataclass
class OptimizationResult:
    """Final result of the optimization run."""
    status: OptimizationStatus
    final_violations: int
    final_fitness: float
    final_generality: float
    total_iterations: int
    total_duration_seconds: float
    improvement: float                    # Reduction in violations
    best_solution_path: Optional[Path]
    history: OptimizationState
    champions: List[Champion]


class RalphWiggumOptimizer:
    """
    Persistent iteration optimizer using Ralph Wiggum technique.

    This optimizer:
    1. Persists all progress to disk
    2. Can resume from any interruption
    3. Escalates strategy when stuck
    4. Integrates with Red Queen evolution
    5. Commits progress to git for history
    """

    def __init__(
        self,
        pcb_path: Path,
        output_dir: Optional[Path] = None,
        criteria: Optional[CompletionCriteria] = None,
        red_queen_evolver: Optional[RedQueenEvolver] = None,
        use_git: bool = True,
        fitness_fn: Optional[Callable] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize Ralph Wiggum Optimizer.

        Args:
            pcb_path: Path to the PCB file to optimize
            output_dir: Directory for outputs (default: next to PCB)
            criteria: Completion criteria
            red_queen_evolver: Red Queen evolver instance
            use_git: Whether to commit progress to git
            fitness_fn: Custom fitness function
            llm_client: LLM client for guided optimization
        """
        self.pcb_path = Path(pcb_path)
        self.output_dir = output_dir or self.pcb_path.parent / 'ralph_output'
        self.criteria = criteria or CompletionCriteria()
        self.use_git = use_git
        self.fitness_fn = fitness_fn
        self.llm_client = llm_client

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State file
        self.state_file = self.output_dir / 'optimization_state.json'

        # Red Queen evolver
        self.red_queen = red_queen_evolver or RedQueenEvolver(
            fitness_fn=fitness_fn,
            llm_client=llm_client,
        )

        # Load or create state
        if self.state_file.exists():
            self.state = OptimizationState.load(self.state_file)
            print(f"Resuming from iteration {self.state.iteration}")
        else:
            self.state = OptimizationState(pcb_path=str(self.pcb_path))

    def _load_pcb_state(self) -> Any:
        """Load PCBState from file."""
        import sys
        mapos_dir = Path(__file__).parent.parent
        if str(mapos_dir) not in sys.path:
            sys.path.insert(0, str(mapos_dir))

        from pcb_state import PCBState
        return PCBState.from_file(str(self.pcb_path))

    def _save_pcb_state(self, pcb_state: Any, suffix: str = '') -> Path:
        """Save PCBState to output directory."""
        output_name = f"{self.pcb_path.stem}{suffix}.kicad_pcb"
        output_path = self.output_dir / output_name

        if hasattr(pcb_state, 'save_to_file'):
            pcb_state.save_to_file(str(output_path))
        else:
            # Copy original if we can't save modified
            shutil.copy(self.pcb_path, output_path)

        return output_path

    def _evaluate_state(self, pcb_state: Any) -> Tuple[int, float, float]:
        """
        Evaluate a PCB state.

        Returns:
            (violations, fitness, generality)
        """
        # Run DRC
        if hasattr(pcb_state, 'run_drc'):
            drc = pcb_state.run_drc()
            violations = drc.total_violations
        else:
            violations = 9999

        # Compute fitness
        if self.fitness_fn:
            fitness = self.fitness_fn(pcb_state)
        else:
            fitness = 1.0 / (1.0 + violations / 100.0)

        # Compute generality from Red Queen
        if self.red_queen.champions_history:
            descriptor = BehavioralDescriptor.from_pcb_state(pcb_state)
            gen_score = self.red_queen._evaluate_generality(
                pcb_state, fitness, descriptor
            )
            generality = gen_score.generality
        else:
            generality = 1.0

        return violations, fitness, generality

    def _detect_stagnation(self) -> bool:
        """Check if optimization has stagnated."""
        if len(self.state.violations_history) < self.criteria.max_stagnation:
            return False

        recent = self.state.violations_history[-self.criteria.max_stagnation:]
        improvement = max(recent) - min(recent)

        return improvement < 5  # Less than 5 violation improvement

    def _escalate_strategy(self) -> EscalationStrategy:
        """
        Choose escalation strategy when stuck.

        Returns the next strategy to try based on escalation count.
        """
        strategies = [
            EscalationStrategy.INCREASE_MUTATION,
            EscalationStrategy.SWITCH_AGENTS,
            EscalationStrategy.EXPAND_SEARCH,
            EscalationStrategy.RESET_POPULATION,
            EscalationStrategy.CALL_FOR_HELP,
        ]

        idx = self.state.escalation_count % len(strategies)
        return strategies[idx]

    async def _apply_escalation(self, strategy: EscalationStrategy) -> None:
        """Apply escalation strategy."""
        print(f"\n!!! ESCALATING: {strategy.name} !!!")

        if strategy == EscalationStrategy.INCREASE_MUTATION:
            self.red_queen.mutation_rate = min(0.95, self.red_queen.mutation_rate + 0.1)
            print(f"  Mutation rate increased to {self.red_queen.mutation_rate:.2f}")

        elif strategy == EscalationStrategy.SWITCH_AGENTS:
            # Rotate agent priorities (implementation depends on agent system)
            print("  Switching agent focus")
            self.state.current_strategy = "agent_rotation"

        elif strategy == EscalationStrategy.EXPAND_SEARCH:
            # Add more behavioral dimensions
            print("  Expanding behavioral search space")
            self.red_queen.iterations_per_round = int(
                self.red_queen.iterations_per_round * 1.5
            )

        elif strategy == EscalationStrategy.RESET_POPULATION:
            print("  Resetting to diverse starting points")
            self.state.current_strategy = "reset"

        elif strategy == EscalationStrategy.CALL_FOR_HELP:
            print("  *** HUMAN INTERVENTION REQUESTED ***")
            print("  Optimization is stuck. Review state file and PCB.")
            self.state.status = "needs_help"
            self.state.save(self.state_file)

        self.state.escalation_count += 1

    def _git_commit(self, message: str) -> bool:
        """Commit current state to git."""
        if not self.use_git:
            return False

        try:
            # Add state file
            subprocess.run(
                ['git', 'add', str(self.state_file)],
                cwd=self.output_dir,
                capture_output=True,
            )

            # Commit
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.output_dir,
                capture_output=True,
            )
            return True
        except Exception:
            return False

    async def run(self) -> OptimizationResult:
        """
        Main Ralph Wiggum loop.

        Iterates until success criteria are met or termination
        conditions are reached.
        """
        print("\n" + "=" * 60)
        print("RALPH WIGGUM OPTIMIZER")
        print("=" * 60)
        print(f"PCB: {self.pcb_path.name}")
        print(f"Target: <= {self.criteria.target_violations} violations")
        print(f"Max iterations: {self.criteria.max_iterations}")
        print("=" * 60)

        start_time = datetime.now()

        # Load initial state
        pcb_state = self._load_pcb_state()
        initial_violations, initial_fitness, _ = self._evaluate_state(pcb_state)

        print(f"\nInitial state: {initial_violations} violations, fitness={initial_fitness:.4f}")

        # Initialize if first run
        if self.state.iteration == 0:
            self.state.violations_history.append(initial_violations)
            self.state.fitness_history.append(initial_fitness)
            self.state.best_violations = initial_violations
            self.state.best_fitness = initial_fitness
            self.state.save(self.state_file)

        # Main loop
        while True:
            self.state.iteration += 1
            self.state.last_update = datetime.now().isoformat()

            # Check termination conditions
            duration_hours = (datetime.now() - datetime.fromisoformat(
                self.state.start_time
            )).total_seconds() / 3600

            should_stop, stop_reason = self.criteria.should_stop(
                self.state.iteration,
                self.state.stagnation_count,
                duration_hours,
            )

            if should_stop:
                print(f"\nStopping: {stop_reason}")
                self.state.status = stop_reason
                break

            print(f"\n--- Iteration {self.state.iteration} ---")

            # Run Red Queen round
            current_solutions = [pcb_state]

            # Add mutated versions for diversity
            for _ in range(4):
                mutated = self.red_queen._random_mutate(pcb_state)
                if mutated:
                    current_solutions.append(mutated)

            round_result = await self.red_queen.run_round(
                current_solutions,
                self.state.red_queen_rounds,
            )
            self.state.red_queen_rounds += 1

            # Get best from round
            if round_result.champions:
                best_champion = max(round_result.champions, key=lambda c: c.fitness)
                pcb_state = best_champion.solution
                violations, fitness, generality = self._evaluate_state(pcb_state)
            else:
                violations = self.state.violations_history[-1] if self.state.violations_history else initial_violations
                fitness = self.state.fitness_history[-1] if self.state.fitness_history else initial_fitness
                generality = 0.0

            # Update history
            self.state.violations_history.append(violations)
            self.state.fitness_history.append(fitness)
            self.state.champions_count = sum(
                len(c) for c in self.red_queen.champions_history
            )

            # Check for improvement
            if violations < self.state.best_violations:
                improvement = self.state.best_violations - violations
                print(f"  IMPROVED: {self.state.best_violations} -> {violations} "
                      f"(-{improvement} violations)")

                self.state.best_violations = violations
                self.state.best_fitness = max(self.state.best_fitness, fitness)
                self.state.best_generality = max(self.state.best_generality, generality)
                self.state.stagnation_count = 0

                # Save best PCB
                self._save_pcb_state(pcb_state, '_best')

                # Git commit
                self._git_commit(f"Iteration {self.state.iteration}: {violations} violations")

            else:
                self.state.stagnation_count += 1
                print(f"  No improvement (stagnation: {self.state.stagnation_count}/"
                      f"{self.criteria.max_stagnation})")

            # Check success
            if self.criteria.is_success(violations, fitness, generality):
                print(f"\n*** SUCCESS! ***")
                print(f"Achieved {violations} violations with fitness {fitness:.4f}")
                self.state.status = "success"
                break

            # Check stagnation and escalate
            if self._detect_stagnation():
                strategy = self._escalate_strategy()
                await self._apply_escalation(strategy)
                self.state.stagnation_count = 0

            # Save state (persistence!)
            self.state.save(self.state_file)

            print(f"  Current: {violations} violations, fitness={fitness:.4f}, "
                  f"generality={generality:.3f}")

        # Finalize
        total_duration = (datetime.now() - start_time).total_seconds()
        self.state.save(self.state_file)

        # Determine final status
        if self.state.status == "success":
            status = OptimizationStatus.SUCCESS
        elif self.state.status == "stagnation":
            status = OptimizationStatus.STAGNATED
        elif self.state.status == "max_iterations":
            status = OptimizationStatus.MAX_ITERATIONS
        else:
            status = OptimizationStatus.FAILED

        # Save final PCB
        final_path = self._save_pcb_state(pcb_state, '_final')

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Status: {status.name}")
        print(f"Final violations: {self.state.best_violations}")
        print(f"Final fitness: {self.state.best_fitness:.4f}")
        print(f"Total iterations: {self.state.iteration}")
        print(f"Total duration: {total_duration / 60:.1f} minutes")
        print(f"Improvement: {initial_violations - self.state.best_violations} violations")
        print(f"Output: {final_path}")

        return OptimizationResult(
            status=status,
            final_violations=self.state.best_violations,
            final_fitness=self.state.best_fitness,
            final_generality=self.state.best_generality,
            total_iterations=self.state.iteration,
            total_duration_seconds=total_duration,
            improvement=initial_violations - self.state.best_violations,
            best_solution_path=final_path,
            history=self.state,
            champions=self.red_queen.get_all_champions(),
        )

    async def run_indefinitely(self) -> None:
        """
        Run forever until manually stopped or success.

        This is the true Ralph Wiggum mode - never gives up.
        """
        self.criteria.max_iterations = 999999
        self.criteria.max_duration_hours = 999999

        while True:
            result = await self.run()

            if result.status == OptimizationStatus.SUCCESS:
                break

            if result.status == OptimizationStatus.STAGNATED:
                print("\n*** RESTARTING after stagnation ***")
                self.state.stagnation_count = 0
                continue

            # Any other status, continue
            print("\n*** CONTINUING optimization ***")


if __name__ == '__main__':
    import asyncio
    from pathlib import Path
    import tempfile

    print("Ralph Wiggum Optimizer Test")
    print("=" * 60)

    # Create a mock PCB file for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pcb_file = tmpdir / 'test.kicad_pcb'

        # Write minimal PCB content
        pcb_file.write_text("""
(kicad_pcb (version 20211014) (generator pcbnew)
  (general
    (thickness 1.6)
  )
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
)
""")

        # Create optimizer with relaxed criteria for testing
        criteria = CompletionCriteria(
            target_violations=100,
            max_iterations=5,
            max_stagnation=3,
        )

        optimizer = RalphWiggumOptimizer(
            pcb_path=pcb_file,
            output_dir=tmpdir / 'output',
            criteria=criteria,
            use_git=False,
        )

        # Run a few iterations
        async def main():
            result = await optimizer.run()
            print(f"\nResult: {result.status.name}")
            print(f"Iterations: {result.total_iterations}")
            print(f"Duration: {result.total_duration_seconds:.1f}s")

        asyncio.run(main())
