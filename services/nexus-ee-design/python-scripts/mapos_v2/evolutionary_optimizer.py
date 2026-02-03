#!/usr/bin/env python3
"""
Evolutionary Optimizer - Genetic Algorithm with LLM-guided operations.

This module implements an evolutionary algorithm for PCB optimization:
1. Population: Pool of PCB configurations
2. Selection: Tournament selection based on DRC fitness
3. Crossover: LLM-guided combination of configurations
4. Mutation: LLM-guided modifications

Inspired by neuroevolution and LLM-guided program synthesis.
"""

import random
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcb_state import (
    PCBState, PCBModification, ModificationType,
    ParameterSpace, DRCResult, create_random_modification
)
from generator_agents import AgentPool, GenerationResult


@dataclass
class Individual:
    """
    An individual in the evolutionary population.

    Represents a complete PCB configuration with fitness.
    """
    state: PCBState
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.state.state_id

    def __repr__(self) -> str:
        return f"Individual(id={self.id[:8]}, fitness={self.fitness:.4f}, gen={self.generation})"


@dataclass
class EvolutionStats:
    """Statistics for evolutionary optimization."""
    generation: int = 0
    best_fitness: float = 0.0
    best_violations: int = 9999
    avg_fitness: float = 0.0
    population_diversity: float = 0.0
    crossovers: int = 0
    mutations: int = 0
    selections: int = 0


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer for PCB layouts.

    Uses genetic algorithm with:
    - Tournament selection
    - LLM-guided crossover
    - LLM-guided mutation
    - Elitism for best individuals
    """

    def __init__(
        self,
        pcb_path: str,
        population_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
        tournament_size: int = 3,
        target_violations: int = 100,
        agent_pool: Optional[AgentPool] = None
    ):
        """
        Initialize evolutionary optimizer.

        Args:
            pcb_path: Path to PCB file
            population_size: Size of population
            generations: Maximum generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_count: Number of elites to preserve
            tournament_size: Size of tournament for selection
            target_violations: Target violation count
            agent_pool: Pool of LLM agents for guided operations
        """
        self.pcb_path = Path(pcb_path)
        self.population_size = population_size
        self.max_generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.target_violations = target_violations
        self.agent_pool = agent_pool or AgentPool()

        # Population
        self.population: List[Individual] = []
        self.generation = 0

        # Statistics
        self.stats_history: List[EvolutionStats] = []
        self.best_individual: Optional[Individual] = None

        # Seen states for diversity
        self.seen_hashes: Set[str] = set()

    def _create_initial_population(self) -> List[Individual]:
        """Create initial population with random variations."""
        print(f"Creating initial population of {self.population_size} individuals...")

        population = []

        # First individual is the original PCB
        original_state = PCBState.from_file(str(self.pcb_path))
        original_drc = original_state.run_drc()
        population.append(Individual(
            state=original_state,
            fitness=original_drc.fitness_score,
            generation=0
        ))
        self.seen_hashes.add(original_state.get_hash())

        print(f"  Original: {original_drc.total_violations} violations, fitness={original_drc.fitness_score:.4f}")

        # Rest are random variations
        for i in range(1, self.population_size):
            # Create variant with random modifications
            state = original_state.copy()

            # Apply 1-3 random modifications
            num_mods = random.randint(1, 3)
            for _ in range(num_mods):
                mod = create_random_modification(state)
                state = state.apply_modification(mod)

            # Ensure unique
            state_hash = state.get_hash()
            attempts = 0
            while state_hash in self.seen_hashes and attempts < 10:
                mod = create_random_modification(state)
                state = state.apply_modification(mod)
                state_hash = state.get_hash()
                attempts += 1

            self.seen_hashes.add(state_hash)

            # Evaluate fitness
            drc = state.run_drc()
            population.append(Individual(
                state=state,
                fitness=drc.fitness_score,
                generation=0
            ))

            if (i + 1) % 5 == 0:
                print(f"  Created {i + 1}/{self.population_size} individuals")

        return population

    def _evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness of an individual."""
        drc = individual.state.run_drc()

        # Base fitness from DRC score
        base_fitness = drc.fitness_score

        # Diversity bonus (reward unique configurations)
        hash_diversity = len(self.seen_hashes) / max(1, self.population_size * self.generation)
        diversity_bonus = 0.05 * min(1.0, hash_diversity)

        # Improvement bonus over initial
        if self.best_individual and drc.total_violations < self.best_individual.state._drc_result.total_violations:
            improvement_bonus = 0.1
        else:
            improvement_bonus = 0.0

        total_fitness = base_fitness + diversity_bonus + improvement_bonus
        individual.fitness = min(1.0, total_fitness)

        return individual.fitness

    def _tournament_select(self) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner

    async def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Create offspring through LLM-guided crossover.

        Combines parameters from both parents intelligently.
        """
        # Start with copy of fitter parent
        if parent1.fitness >= parent2.fitness:
            base_state = parent1.state.copy()
            other_state = parent2.state
        else:
            base_state = parent2.state.copy()
            other_state = parent1.state

        # Combine parameters
        for param_name in base_state.parameters:
            if param_name in other_state.parameters:
                # Blend parameters (weighted by fitness)
                w1 = parent1.fitness / (parent1.fitness + parent2.fitness + 0.001)
                w2 = 1 - w1

                p1_val = parent1.state.parameters.get(param_name, 0)
                p2_val = parent2.state.parameters.get(param_name, 0)

                blended = w1 * p1_val + w2 * p2_val
                base_state.parameters[param_name] = blended

        # Create crossover modification
        crossover_mod = PCBModification(
            mod_type=ModificationType.ADJUST_CLEARANCE,
            target="crossover",
            parameters={'blended_from': [parent1.id, parent2.id]},
            description=f"Crossover of {parent1.id[:8]} and {parent2.id[:8]}",
            source_agent="crossover"
        )

        base_state.modifications.append(crossover_mod)

        child = Individual(
            state=base_state,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )

        return child

    async def _mutate(self, individual: Individual) -> Individual:
        """
        Apply LLM-guided mutation to individual.

        Uses agent pool to generate intelligent mutations.
        """
        state = individual.state.copy()

        # Get DRC for mutation guidance
        drc = state.run_drc()

        # Try to get agent-guided mutation
        try:
            results = await self.agent_pool.generate_all(state, drc, max_modifications_per_agent=2)

            # Pick a random modification from agent results
            all_mods = []
            for result in results:
                all_mods.extend(result.modifications)

            if all_mods:
                mod = random.choice(all_mods)
                state = state.apply_modification(mod)
                mutation_desc = f"Agent-guided: {mod.description}"
            else:
                # Fallback to random mutation
                mod = create_random_modification(state)
                state = state.apply_modification(mod)
                mutation_desc = f"Random: {mod.description}"
        except Exception as e:
            # Fallback to random mutation
            mod = create_random_modification(state)
            state = state.apply_modification(mod)
            mutation_desc = f"Random (fallback): {mod.description}"

        mutated = Individual(
            state=state,
            generation=self.generation + 1,
            parent_ids=[individual.id],
            mutation_history=individual.mutation_history + [mutation_desc]
        )

        return mutated

    def _get_elites(self) -> List[Individual]:
        """Get top individuals for elitism."""
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        return sorted_pop[:self.elite_count]

    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on parameter variance."""
        if len(self.population) < 2:
            return 0.0

        # Calculate variance of fitness
        fitnesses = [ind.fitness for ind in self.population]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)

        # Normalize to 0-1 range
        return min(1.0, variance * 10)

    async def evolve(self) -> Individual:
        """
        Run evolutionary optimization.

        Returns:
            Best individual found
        """
        print(f"\n{'='*60}")
        print("EVOLUTIONARY PCB OPTIMIZER")
        print(f"{'='*60}")
        print(f"Population: {self.population_size}, Generations: {self.max_generations}")
        print(f"Target: {self.target_violations} violations")

        start_time = time.time()

        # Initialize population
        self.population = self._create_initial_population()

        # Track best
        self.best_individual = max(self.population, key=lambda ind: ind.fitness)
        initial_violations = self.best_individual.state._drc_result.total_violations

        print(f"\nInitial best: {initial_violations} violations")

        # Evolution loop
        for gen in range(self.max_generations):
            self.generation = gen

            # Create next generation
            next_population = []

            # Elitism: preserve top individuals
            elites = self._get_elites()
            for elite in elites:
                elite_copy = Individual(
                    state=elite.state.copy(),
                    fitness=elite.fitness,
                    generation=gen + 1,
                    parent_ids=[elite.id]
                )
                next_population.append(elite_copy)

            # Fill rest with offspring
            while len(next_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()

                # Crossover
                if random.random() < self.crossover_rate:
                    child = await self._crossover(parent1, parent2)
                else:
                    child = Individual(
                        state=parent1.state.copy(),
                        generation=gen + 1,
                        parent_ids=[parent1.id]
                    )

                # Mutation
                if random.random() < self.mutation_rate:
                    child = await self._mutate(child)

                # Ensure unique
                state_hash = child.state.get_hash()
                if state_hash not in self.seen_hashes:
                    self.seen_hashes.add(state_hash)
                    next_population.append(child)

            # Evaluate fitness
            for ind in next_population:
                if ind.fitness == 0.0:
                    self._evaluate_fitness(ind)

            # Update population
            self.population = next_population

            # Update best
            gen_best = max(self.population, key=lambda ind: ind.fitness)
            if gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best

            # Record stats
            current_violations = self.best_individual.state._drc_result.total_violations
            stats = EvolutionStats(
                generation=gen,
                best_fitness=self.best_individual.fitness,
                best_violations=current_violations,
                avg_fitness=sum(ind.fitness for ind in self.population) / len(self.population),
                population_diversity=self._calculate_diversity()
            )
            self.stats_history.append(stats)

            # Progress report
            if (gen + 1) % 5 == 0 or current_violations <= self.target_violations:
                elapsed = time.time() - start_time
                print(f"Gen {gen+1}: best_violations={current_violations}, "
                      f"best_fitness={self.best_individual.fitness:.4f}, "
                      f"avg_fitness={stats.avg_fitness:.4f}, "
                      f"diversity={stats.population_diversity:.3f}, "
                      f"time={elapsed:.1f}s")

            # Early termination
            if current_violations <= self.target_violations:
                print(f"\n  Target reached at generation {gen+1}!")
                break

        total_time = time.time() - start_time
        final_violations = self.best_individual.state._drc_result.total_violations

        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Generations: {self.generation + 1}")
        print(f"Time: {total_time:.1f}s")
        print(f"Initial violations: {initial_violations}")
        print(f"Final violations: {final_violations}")
        print(f"Improvement: {initial_violations - final_violations} ({100*(initial_violations-final_violations)/initial_violations:.1f}%)")

        return self.best_individual

    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        return {
            'generations': self.generation + 1,
            'population_size': self.population_size,
            'best_fitness': self.best_individual.fitness if self.best_individual else 0,
            'best_violations': self.best_individual.state._drc_result.total_violations if self.best_individual else 9999,
            'final_diversity': self._calculate_diversity(),
            'unique_states': len(self.seen_hashes),
            'history': [
                {
                    'generation': s.generation,
                    'best_fitness': s.best_fitness,
                    'best_violations': s.best_violations,
                    'avg_fitness': s.avg_fitness,
                    'diversity': s.population_diversity
                }
                for s in self.stats_history
            ]
        }

    def save_results(self, output_path: str) -> None:
        """Save optimization results to file."""
        results = {
            'pcb_path': str(self.pcb_path),
            'statistics': self.get_statistics(),
            'best_modifications': [m.to_dict() for m in self.best_individual.state.modifications] if self.best_individual else [],
            'best_state': self.best_individual.state.to_dict() if self.best_individual else None
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {output_path}")


class IslandEvolution(EvolutionaryOptimizer):
    """
    Island model evolutionary optimizer.

    Maintains multiple sub-populations that evolve independently,
    with periodic migration between islands.
    """

    def __init__(
        self,
        pcb_path: str,
        num_islands: int = 3,
        island_population: int = 10,
        migration_interval: int = 5,
        migration_count: int = 2,
        **kwargs
    ):
        # Total population is spread across islands
        total_pop = num_islands * island_population
        super().__init__(pcb_path, population_size=total_pop, **kwargs)

        self.num_islands = num_islands
        self.island_population = island_population
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.islands: List[List[Individual]] = []

    async def evolve(self) -> Individual:
        """Run island model evolution."""
        print(f"\n{'='*60}")
        print("ISLAND MODEL EVOLUTIONARY OPTIMIZER")
        print(f"{'='*60}")
        print(f"Islands: {self.num_islands}, Population per island: {self.island_population}")

        # Initialize all populations
        all_individuals = self._create_initial_population()

        # Distribute to islands
        random.shuffle(all_individuals)
        self.islands = [
            all_individuals[i*self.island_population:(i+1)*self.island_population]
            for i in range(self.num_islands)
        ]

        # Track best across all islands
        self.best_individual = max(all_individuals, key=lambda ind: ind.fitness)

        for gen in range(self.max_generations):
            self.generation = gen

            # Evolve each island independently
            for island_idx, island in enumerate(self.islands):
                self.population = island

                # One generation of evolution
                next_pop = []

                # Elites
                elites = self._get_elites()
                for elite in elites:
                    elite_copy = Individual(
                        state=elite.state.copy(),
                        fitness=elite.fitness,
                        generation=gen + 1
                    )
                    next_pop.append(elite_copy)

                # Fill with offspring
                while len(next_pop) < self.island_population:
                    parent1 = self._tournament_select()
                    parent2 = self._tournament_select()

                    if random.random() < self.crossover_rate:
                        child = await self._crossover(parent1, parent2)
                    else:
                        child = Individual(state=parent1.state.copy(), generation=gen+1)

                    if random.random() < self.mutation_rate:
                        child = await self._mutate(child)

                    self._evaluate_fitness(child)
                    next_pop.append(child)

                self.islands[island_idx] = next_pop

                # Update global best
                island_best = max(next_pop, key=lambda ind: ind.fitness)
                if island_best.fitness > self.best_individual.fitness:
                    self.best_individual = island_best

            # Migration
            if (gen + 1) % self.migration_interval == 0:
                self._migrate()
                print(f"  Gen {gen+1}: Migration between islands")

            # Check termination
            current_violations = self.best_individual.state._drc_result.total_violations
            if current_violations <= self.target_violations:
                print(f"\n  Target reached at generation {gen+1}!")
                break

            if (gen + 1) % 10 == 0:
                print(f"  Gen {gen+1}: best_violations={current_violations}")

        return self.best_individual

    def _migrate(self) -> None:
        """Migrate best individuals between islands."""
        # Collect best from each island
        island_bests = []
        for island in self.islands:
            sorted_island = sorted(island, key=lambda ind: ind.fitness, reverse=True)
            island_bests.append(sorted_island[:self.migration_count])

        # Rotate migrants between islands
        for i in range(self.num_islands):
            source_idx = (i + 1) % self.num_islands
            migrants = island_bests[source_idx]

            # Replace worst in target island
            self.islands[i].sort(key=lambda ind: ind.fitness)
            for j, migrant in enumerate(migrants):
                if j < len(self.islands[i]):
                    self.islands[i][j] = Individual(
                        state=migrant.state.copy(),
                        fitness=migrant.fitness,
                        generation=self.generation
                    )


async def run_evolutionary_optimization(
    pcb_path: str,
    population_size: int = 20,
    generations: int = 30,
    target_violations: int = 100,
    output_json: Optional[str] = None
) -> Individual:
    """
    Run evolutionary optimization on a PCB file.

    Args:
        pcb_path: Path to PCB file
        population_size: Size of population
        generations: Maximum generations
        target_violations: Target violation count
        output_json: Optional path to save results

    Returns:
        Best Individual found
    """
    optimizer = EvolutionaryOptimizer(
        pcb_path=pcb_path,
        population_size=population_size,
        generations=generations,
        target_violations=target_violations
    )

    best = await optimizer.evolve()

    print(f"\nBest Individual:")
    print(f"  ID: {best.id}")
    print(f"  Fitness: {best.fitness:.4f}")
    print(f"  Violations: {best.state._drc_result.total_violations}")
    print(f"  Modifications: {len(best.state.modifications)}")

    if output_json:
        optimizer.save_results(output_json)

    return best


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evolutionary_optimizer.py <path_to.kicad_pcb> [population] [generations] [target]")
        sys.exit(1)

    pcb_path = sys.argv[1]
    population = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    target = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    asyncio.run(run_evolutionary_optimization(
        pcb_path,
        population_size=population,
        generations=generations,
        target_violations=target,
        output_json=f"{pcb_path.rsplit('.', 1)[0]}_evolution_results.json"
    ))
