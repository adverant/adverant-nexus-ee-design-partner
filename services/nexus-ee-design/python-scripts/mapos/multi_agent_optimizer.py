#!/usr/bin/env python3
"""
Multi-Agent PCB Optimization System (MAPOS) - Main Orchestration.

This is the main entry point that orchestrates all optimization components:
1. Pre-DRC structural fixes (from pcb_file_fixer)
2. Multi-agent generation (from generator_agents)
3. MCTS exploration (from mcts_optimizer)
4. Evolutionary optimization (from evolutionary_optimizer)
5. Tournament selection (from tournament_judge)
6. AlphaFold-style refinement (from refinement_loop)

The system is designed to automatically reduce DRC violations from 1,000+ to <100.
"""

import os
import sys
import json
import asyncio
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pcb_state import PCBState, DRCResult
from generator_agents import AgentPool, create_default_agent_pool
from mcts_optimizer import MCTSOptimizer, ProgressiveMCTS
from evolutionary_optimizer import EvolutionaryOptimizer, IslandEvolution
from tournament_judge import TournamentJudge, Tournament, SwissTournament
from refinement_loop import RefinementLoop, IterativeRefinementOptimizer


@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""
    # General
    target_violations: int = 100
    max_time_minutes: int = 30

    # MCTS
    mcts_iterations: int = 50
    mcts_exploration: float = 1.414

    # Evolution
    evolution_population: int = 15
    evolution_generations: int = 30
    evolution_mutation_rate: float = 0.15

    # Tournament
    tournament_rounds: int = 5

    # Refinement
    refinement_cycles: int = 10
    refinement_threshold: float = 0.01

    # Parallelism
    parallel_evaluations: int = 4

    # Output
    save_checkpoints: bool = True
    checkpoint_interval: int = 10


@dataclass
class OptimizationResult:
    """Result from the full optimization pipeline."""
    initial_violations: int
    final_violations: int
    total_improvement: int
    improvement_percent: float
    target_reached: bool
    phases_completed: List[str]
    total_time_seconds: float
    best_state: PCBState
    phase_results: Dict[str, Any]


class MultiAgentOptimizer:
    """
    Main orchestration system for multi-agent PCB optimization.

    Coordinates multiple optimization strategies:
    1. Pre-DRC fixes (deterministic structural fixes)
    2. MCTS exploration (tree search with LLM expansion)
    3. Evolutionary optimization (population-based search)
    4. Tournament selection (competitive ranking)
    5. AlphaFold refinement (iterative improvement)

    The optimizer can run phases sequentially or select based on progress.
    """

    def __init__(
        self,
        pcb_path: str,
        config: Optional[OptimizationConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize multi-agent optimizer.

        Args:
            pcb_path: Path to PCB file
            config: Optimization configuration
            output_dir: Directory for output files
        """
        self.pcb_path = Path(pcb_path)
        self.config = config or OptimizationConfig()
        self.output_dir = Path(output_dir) if output_dir else self.pcb_path.parent / "optimization_output"

        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize agent pool
        self.agent_pool = create_default_agent_pool()

        # State tracking
        self.current_state: Optional[PCBState] = None
        self.best_state: Optional[PCBState] = None
        self.best_violations = float('inf')
        self.phase_results: Dict[str, Any] = {}

        # Timing
        self.start_time: Optional[float] = None

    async def optimize(self) -> OptimizationResult:
        """
        Run the full optimization pipeline.

        Returns:
            OptimizationResult with final state and statistics
        """
        self.start_time = time.time()

        print("\n" + "="*70)
        print("MULTI-AGENT PCB OPTIMIZATION SYSTEM (MAPOS)")
        print("="*70)
        print(f"PCB: {self.pcb_path.name}")
        print(f"Target: {self.config.target_violations} violations")
        print(f"Max time: {self.config.max_time_minutes} minutes")
        print(f"Output: {self.output_dir}")
        print("="*70)

        # Phase 0: Load and analyze initial state
        print("\n[PHASE 0] Initial Analysis")
        print("-"*40)

        self.current_state = PCBState.from_file(str(self.pcb_path))
        initial_drc = self.current_state.run_drc()
        initial_violations = initial_drc.total_violations

        print(f"Initial violations: {initial_violations}")
        print(f"  Errors: {initial_drc.errors}")
        print(f"  Warnings: {initial_drc.warnings}")
        print(f"  Unconnected: {initial_drc.unconnected}")

        self.best_state = self.current_state
        self.best_violations = initial_violations

        phases_completed = []

        # Phase 1: Pre-DRC structural fixes
        if not self._time_exceeded():
            try:
                await self._run_pre_drc_fixes()
                phases_completed.append("pre_drc")
            except Exception as e:
                print(f"  Pre-DRC fixes failed: {e}")

        # Check if target already reached
        if self.best_violations <= self.config.target_violations:
            return self._create_result(initial_violations, phases_completed)

        # Phase 2: MCTS exploration
        if not self._time_exceeded():
            try:
                await self._run_mcts_phase()
                phases_completed.append("mcts")
            except Exception as e:
                print(f"  MCTS phase failed: {e}")

        # Check progress
        if self.best_violations <= self.config.target_violations:
            return self._create_result(initial_violations, phases_completed)

        # Phase 3: Evolutionary optimization
        if not self._time_exceeded():
            try:
                await self._run_evolution_phase()
                phases_completed.append("evolution")
            except Exception as e:
                print(f"  Evolution phase failed: {e}")

        # Check progress
        if self.best_violations <= self.config.target_violations:
            return self._create_result(initial_violations, phases_completed)

        # Phase 4: Tournament selection (if we have multiple candidates)
        if not self._time_exceeded():
            try:
                await self._run_tournament_phase()
                phases_completed.append("tournament")
            except Exception as e:
                print(f"  Tournament phase failed: {e}")

        # Phase 5: AlphaFold-style refinement
        if not self._time_exceeded():
            try:
                await self._run_refinement_phase()
                phases_completed.append("refinement")
            except Exception as e:
                print(f"  Refinement phase failed: {e}")

        return self._create_result(initial_violations, phases_completed)

    async def _run_pre_drc_fixes(self) -> None:
        """Run pre-DRC structural fixes."""
        print("\n[PHASE 1] Pre-DRC Structural Fixes")
        print("-"*40)

        try:
            from pcb_file_fixer import PCBFileFixer

            # Work on a copy
            work_path = self.output_dir / "working_copy.kicad_pcb"
            import shutil
            shutil.copy2(self.pcb_path, work_path)

            fixer = PCBFileFixer(str(work_path), backup=False)
            results = fixer.fix_all()

            print(f"  Zone nets fixed: {results['zone_nets_fixed']}")
            print(f"  Boundaries fixed: {results['boundaries_fixed']}")
            print(f"  Dangling vias removed: {results['vias_removed']}")

            # Update state
            new_state = PCBState.from_file(str(work_path))
            new_drc = new_state.run_drc()

            if new_drc.total_violations < self.best_violations:
                self.best_state = new_state
                self.best_violations = new_drc.total_violations
                self.current_state = new_state
                print(f"  New best: {self.best_violations} violations")

            self.phase_results['pre_drc'] = {
                'fixes_applied': sum(results.values()),
                'violations_after': new_drc.total_violations
            }

        except ImportError:
            print("  pcb_file_fixer not available - skipping")

    async def _run_mcts_phase(self) -> None:
        """Run MCTS exploration phase."""
        print("\n[PHASE 2] MCTS Exploration")
        print("-"*40)

        # Use current best state as starting point
        start_path = self.output_dir / "mcts_start.kicad_pcb"
        if self.current_state:
            self.current_state.save_to_file(str(start_path))
        else:
            import shutil
            shutil.copy2(self.pcb_path, start_path)

        optimizer = ProgressiveMCTS(
            pcb_path=str(start_path),
            agent_pool=self.agent_pool,
            initial_iterations=20,
            max_iterations=self.config.mcts_iterations,
            target_violations=self.config.target_violations
        )

        best_state = await optimizer.progressive_search()

        if best_state._drc_result.total_violations < self.best_violations:
            self.best_state = best_state
            self.best_violations = best_state._drc_result.total_violations
            self.current_state = best_state
            print(f"  New best: {self.best_violations} violations")

        self.phase_results['mcts'] = optimizer.get_statistics()

    async def _run_evolution_phase(self) -> None:
        """Run evolutionary optimization phase."""
        print("\n[PHASE 3] Evolutionary Optimization")
        print("-"*40)

        # Save current best as starting point
        start_path = self.output_dir / "evolution_start.kicad_pcb"
        if self.current_state:
            self.current_state.save_to_file(str(start_path))
        else:
            import shutil
            shutil.copy2(self.pcb_path, start_path)

        optimizer = EvolutionaryOptimizer(
            pcb_path=str(start_path),
            population_size=self.config.evolution_population,
            generations=self.config.evolution_generations,
            mutation_rate=self.config.evolution_mutation_rate,
            target_violations=self.config.target_violations,
            agent_pool=self.agent_pool
        )

        best_individual = await optimizer.evolve()

        if best_individual.state._drc_result.total_violations < self.best_violations:
            self.best_state = best_individual.state
            self.best_violations = best_individual.state._drc_result.total_violations
            self.current_state = best_individual.state
            print(f"  New best: {self.best_violations} violations")

        self.phase_results['evolution'] = optimizer.get_statistics()

    async def _run_tournament_phase(self) -> None:
        """Run tournament selection phase."""
        print("\n[PHASE 4] Tournament Selection")
        print("-"*40)

        # Collect all candidate states from previous phases
        candidates = []

        if self.best_state:
            candidates.append(self.best_state)

        # Generate additional variants
        if self.current_state:
            from pcb_state import create_random_modification

            for i in range(7):  # Create 7 more for total of 8
                variant = self.current_state.copy()
                for _ in range(2):
                    mod = create_random_modification(variant)
                    variant = variant.apply_modification(mod)
                variant.run_drc()
                candidates.append(variant)

        if len(candidates) < 2:
            print("  Not enough candidates for tournament")
            return

        judge = TournamentJudge(k_factor=32)
        tournament = Tournament(judge, candidates)

        winner = await tournament.run()

        if winner._drc_result.total_violations < self.best_violations:
            self.best_state = winner
            self.best_violations = winner._drc_result.total_violations
            print(f"  New best: {self.best_violations} violations")

        self.phase_results['tournament'] = {
            'candidates': len(candidates),
            'winner_violations': winner._drc_result.total_violations,
            'elo_rankings': judge.get_rankings()[:5]
        }

    async def _run_refinement_phase(self) -> None:
        """Run AlphaFold-style refinement phase."""
        print("\n[PHASE 5] AlphaFold-Style Refinement")
        print("-"*40)

        if not self.best_state:
            print("  No state to refine")
            return

        refinement = RefinementLoop(
            max_cycles=self.config.refinement_cycles,
            convergence_threshold=self.config.refinement_threshold,
            target_violations=self.config.target_violations,
            agent_pool=self.agent_pool
        )

        result = await refinement.refine(self.best_state)

        if result.final_violations < self.best_violations:
            self.best_state = result.final_state
            self.best_violations = result.final_violations
            print(f"  New best: {self.best_violations} violations")

        self.phase_results['refinement'] = {
            'cycles': result.cycles_completed,
            'converged': result.converged,
            'improvement': result.total_improvement
        }

    def _time_exceeded(self) -> bool:
        """Check if max time has been exceeded."""
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        max_seconds = self.config.max_time_minutes * 60
        return elapsed > max_seconds

    def _create_result(
        self,
        initial_violations: int,
        phases_completed: List[str]
    ) -> OptimizationResult:
        """Create final optimization result."""
        total_time = time.time() - self.start_time if self.start_time else 0

        improvement = initial_violations - self.best_violations
        improvement_pct = 100 * improvement / max(1, initial_violations)

        # Final summary
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Initial violations: {initial_violations}")
        print(f"Final violations: {self.best_violations}")
        print(f"Total improvement: {improvement} ({improvement_pct:.1f}%)")
        print(f"Target reached: {self.best_violations <= self.config.target_violations}")
        print(f"Phases completed: {', '.join(phases_completed)}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print("="*70)

        # Save results
        self._save_results(initial_violations, phases_completed, total_time)

        return OptimizationResult(
            initial_violations=initial_violations,
            final_violations=self.best_violations,
            total_improvement=improvement,
            improvement_percent=improvement_pct,
            target_reached=self.best_violations <= self.config.target_violations,
            phases_completed=phases_completed,
            total_time_seconds=total_time,
            best_state=self.best_state,
            phase_results=self.phase_results
        )

    def _save_results(
        self,
        initial_violations: int,
        phases_completed: List[str],
        total_time: float
    ) -> None:
        """Save optimization results to files."""
        # Save JSON summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'pcb_file': str(self.pcb_path),
            'initial_violations': initial_violations,
            'final_violations': self.best_violations,
            'improvement': initial_violations - self.best_violations,
            'improvement_percent': 100 * (initial_violations - self.best_violations) / max(1, initial_violations),
            'target': self.config.target_violations,
            'target_reached': self.best_violations <= self.config.target_violations,
            'phases_completed': phases_completed,
            'total_time_seconds': total_time,
            'phase_results': self.phase_results
        }

        summary_path = self.output_dir / "optimization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")

        # Save best state modifications
        if self.best_state and self.best_state.modifications:
            mods_path = self.output_dir / "best_modifications.json"
            mods = [m.to_dict() for m in self.best_state.modifications]
            with open(mods_path, 'w') as f:
                json.dump(mods, f, indent=2)


async def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent PCB Optimization System (MAPOS)"
    )
    parser.add_argument(
        "pcb_path",
        help="Path to KiCad PCB file"
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=100,
        help="Target violation count (default: 100)"
    )
    parser.add_argument(
        "--max-time", "-m",
        type=int,
        default=30,
        help="Maximum optimization time in minutes (default: 30)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: pcb_dir/optimization_output)"
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=50,
        help="MCTS iterations (default: 50)"
    )
    parser.add_argument(
        "--evolution-generations",
        type=int,
        default=30,
        help="Evolution generations (default: 30)"
    )
    parser.add_argument(
        "--refinement-cycles",
        type=int,
        default=10,
        help="Refinement cycles (default: 10)"
    )

    args = parser.parse_args()

    config = OptimizationConfig(
        target_violations=args.target,
        max_time_minutes=args.max_time,
        mcts_iterations=args.mcts_iterations,
        evolution_generations=args.evolution_generations,
        refinement_cycles=args.refinement_cycles
    )

    optimizer = MultiAgentOptimizer(
        pcb_path=args.pcb_path,
        config=config,
        output_dir=args.output
    )

    result = await optimizer.optimize()

    # Exit with appropriate code
    sys.exit(0 if result.target_reached else 1)


if __name__ == '__main__':
    asyncio.run(main())
