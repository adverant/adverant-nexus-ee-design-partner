"""
Red Queen Evolver - Adversarial Co-Evolution for PCB Optimization

This module implements the Digital Red Queen (DRQ) algorithm, which
evolves PCB designs through adversarial competition against historical
champions.

Core concept: Each round evolves designs that must beat ALL previous
champions, creating increasingly general and robust solutions.

Key mechanisms:
1. MAP-Elites Archive: Maintains diverse solutions per round
2. LLM Mutation: Uses Claude to intelligently modify designs
3. Generality Scoring: Fitness based on beating historical champions
4. Convergent Evolution Tracking: Monitors phenotype/genotype convergence

References:
- Digital Red Queen: https://sakana.ai/drq/
- Red Queen Hypothesis: https://en.wikipedia.org/wiki/Red_Queen_hypothesis
- MAP-Elites: https://arxiv.org/abs/1504.04909
"""

import asyncio
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from datetime import datetime
from enum import Enum, auto
import math

import numpy as np

# Local imports
from .map_elites import MAPElitesArchive, BehavioralDescriptor, ArchiveStatistics


class MutationStrategy(Enum):
    """Strategies for mutating PCB designs."""
    LLM_GUIDED = auto()      # Use LLM to suggest modifications
    RANDOM = auto()           # Random parameter perturbation
    CROSSOVER = auto()        # Combine two parent designs
    TARGETED = auto()         # Target specific violation types
    EXPLORATORY = auto()      # Explore novel behavioral regions


@dataclass
class Champion:
    """A champion design from a previous round."""
    solution: Any                     # PCBState or similar
    fitness: float                    # Fitness when it became champion
    descriptor: BehavioralDescriptor  # Behavioral characteristics
    round_number: int                 # Which round this champion is from
    phenotype_hash: str               # Hash of behavioral vector
    genotype_hash: str                # Hash of design parameters
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionRound:
    """Results from a single evolution round."""
    round_number: int
    champions: List[Champion]
    archive_stats: ArchiveStatistics
    total_evaluations: int
    best_fitness: float
    best_generality: float             # Fraction of historical champions beaten
    convergence_metrics: Dict[str, float]
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GeneralityScore:
    """Measures how well a design beats historical champions."""
    wins: int                          # Number of champions beaten
    ties: int                          # Number of ties
    losses: int                        # Number of losses
    total_champions: int               # Total champions competed against
    generality: float                  # Fraction won or tied
    win_margin: float                  # Average margin of victory
    per_round_scores: Dict[int, float] # Score against each round's champions


class RedQueenEvolver:
    """
    Digital Red Queen evolution engine.

    Evolves PCB designs through adversarial rounds where each new
    generation must beat all previous champions.
    """

    def __init__(
        self,
        archive_dimensions: Optional[List[Tuple[str, float, float, int]]] = None,
        population_size: int = 50,
        iterations_per_round: int = 100,
        elite_count: int = 5,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.2,
        llm_client: Optional[Any] = None,
        fitness_fn: Optional[Callable] = None,
        descriptor_fn: Optional[Callable] = None,
    ):
        """
        Initialize Red Queen Evolver.

        Args:
            archive_dimensions: Behavioral dimensions for MAP-Elites
            population_size: Number of solutions to maintain
            iterations_per_round: Iterations per evolution round
            elite_count: Number of elites to extract as champions
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            llm_client: LLM client for guided mutation (optional)
            fitness_fn: Custom fitness function
            descriptor_fn: Custom descriptor extraction function
        """
        self.archive_dimensions = archive_dimensions
        self.population_size = population_size
        self.iterations_per_round = iterations_per_round
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.llm_client = llm_client
        self.fitness_fn = fitness_fn
        self.descriptor_fn = descriptor_fn or BehavioralDescriptor.from_pcb_state

        # Historical champions from all rounds
        self.champions_history: List[List[Champion]] = []

        # Evolution history
        self.rounds_history: List[EvolutionRound] = []

        # Convergence tracking
        self.phenotype_history: List[np.ndarray] = []  # Average phenotype per round
        self.genotype_diversity: List[float] = []       # Genotype diversity per round

    def _compute_phenotype_hash(self, descriptor: BehavioralDescriptor) -> str:
        """Compute hash of behavioral phenotype."""
        vector = descriptor.to_vector()
        # Discretize to reduce noise
        discretized = np.round(vector * 10) / 10
        return hashlib.md5(discretized.tobytes()).hexdigest()[:12]

    def _compute_genotype_hash(self, solution: Any) -> str:
        """Compute hash of design genotype (parameters)."""
        if hasattr(solution, 'get_hash'):
            return solution.get_hash()
        elif hasattr(solution, 'parameters'):
            param_str = json.dumps(solution.parameters, sort_keys=True)
            return hashlib.md5(param_str.encode()).hexdigest()[:12]
        else:
            return hashlib.md5(str(id(solution)).encode()).hexdigest()[:12]

    def _evaluate_generality(
        self,
        solution: Any,
        fitness: float,
        descriptor: BehavioralDescriptor,
    ) -> GeneralityScore:
        """
        Evaluate how well a solution beats historical champions.

        A solution "beats" a champion if it has higher fitness.
        Generality measures the fraction of champions beaten.
        """
        if not self.champions_history:
            # First round - no champions to beat
            return GeneralityScore(
                wins=0, ties=0, losses=0, total_champions=0,
                generality=1.0, win_margin=0.0, per_round_scores={}
            )

        wins = 0
        ties = 0
        losses = 0
        win_margins = []
        per_round_scores = {}

        for round_idx, round_champions in enumerate(self.champions_history):
            round_wins = 0
            round_total = len(round_champions)

            for champion in round_champions:
                margin = fitness - champion.fitness

                if margin > 0.01:  # Win threshold
                    wins += 1
                    round_wins += 1
                    win_margins.append(margin)
                elif margin > -0.01:  # Tie threshold
                    ties += 1
                else:
                    losses += 1

            per_round_scores[round_idx] = round_wins / max(1, round_total)

        total_champions = wins + ties + losses
        generality = (wins + ties * 0.5) / max(1, total_champions)
        avg_margin = np.mean(win_margins) if win_margins else 0.0

        return GeneralityScore(
            wins=wins,
            ties=ties,
            losses=losses,
            total_champions=total_champions,
            generality=generality,
            win_margin=avg_margin,
            per_round_scores=per_round_scores,
        )

    def _compute_combined_fitness(
        self,
        solution: Any,
        base_fitness: float,
        generality: GeneralityScore,
        fitness_weight: float = 0.5,
        generality_weight: float = 0.5,
    ) -> float:
        """
        Compute combined fitness from base fitness and generality.

        In early rounds, base fitness dominates.
        In later rounds, generality becomes more important.
        """
        round_number = len(self.champions_history)

        # Adaptive weighting based on round
        if round_number == 0:
            # First round: pure fitness
            return base_fitness
        else:
            # Increase generality importance over rounds
            gen_weight = min(0.7, generality_weight + round_number * 0.05)
            fit_weight = 1.0 - gen_weight

            return fit_weight * base_fitness + gen_weight * generality.generality

    async def _llm_mutate(
        self,
        solution: Any,
        descriptor: BehavioralDescriptor,
        round_number: int,
        target_region: Optional[Tuple[int, ...]] = None,
    ) -> Optional[Any]:
        """
        Use LLM to suggest intelligent mutations.

        Args:
            solution: Current solution to mutate
            descriptor: Behavioral characteristics
            round_number: Current round number
            target_region: Target behavioral region (optional)

        Returns:
            Mutated solution or None if mutation failed
        """
        if self.llm_client is None:
            return self._random_mutate(solution)

        # Build context for LLM
        context = {
            'round': round_number,
            'current_fitness': self.fitness_fn(solution) if self.fitness_fn else 0.0,
            'violations': getattr(solution, 'run_drc', lambda: None)(),
            'parameters': getattr(solution, 'parameters', {}),
            'descriptor': descriptor.to_dict(),
            'champions_to_beat': len(sum(self.champions_history, [])),
        }

        # Build prompt
        prompt = self._build_mutation_prompt(context, target_region)

        try:
            # Call LLM
            response = await self.llm_client.generate(prompt)

            # Parse response and apply mutation
            mutation = self._parse_mutation_response(response)

            if mutation and hasattr(solution, 'apply_modification'):
                return solution.apply_modification(mutation)

        except Exception as e:
            print(f"LLM mutation failed: {e}")

        # Fallback to random mutation
        return self._random_mutate(solution)

    def _build_mutation_prompt(
        self,
        context: Dict[str, Any],
        target_region: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """Build prompt for LLM-guided mutation."""
        violations = context.get('violations')
        violation_summary = ""
        if violations and hasattr(violations, 'violations_by_type'):
            top_violations = sorted(
                violations.violations_by_type.items(),
                key=lambda x: -x[1]
            )[:5]
            violation_summary = ", ".join(f"{k}: {v}" for k, v in top_violations)

        prompt = f"""
You are optimizing a PCB design in round {context['round']} of Red Queen evolution.

Current State:
- Fitness: {context['current_fitness']:.4f}
- Total violations: {violations.total_violations if violations else 'N/A'}
- Top violations: {violation_summary or 'N/A'}
- Champions to beat: {context['champions_to_beat']}

Current parameters: {json.dumps(context['parameters'], indent=2)}

Behavioral profile: {json.dumps(context['descriptor'], indent=2)}

{'Target behavioral region: ' + str(target_region) if target_region else ''}

Suggest ONE modification that will:
1. Reduce the most impactful violations
2. Maintain or improve generality against previous champions
3. Explore a {'novel behavioral region' if target_region else 'promising direction'}

Return JSON with format:
{{
    "mod_type": "ADJUST_CLEARANCE|ADJUST_TRACE_WIDTH|ADJUST_VIA_SIZE|ADJUST_ZONE|...",
    "target": "parameter_or_component_name",
    "parameters": {{"param_name": value}},
    "reasoning": "brief explanation"
}}
"""
        return prompt

    def _parse_mutation_response(self, response: str) -> Optional[Any]:
        """Parse LLM response into a PCBModification."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                # Import here to avoid circular dependency
                import sys
                mapos_dir = Path(__file__).parent.parent
                if str(mapos_dir) not in sys.path:
                    sys.path.insert(0, str(mapos_dir))

                from pcb_state import PCBModification, ModificationType

                mod_type = ModificationType[data.get('mod_type', 'ADJUST_CLEARANCE')]

                return PCBModification(
                    mod_type=mod_type,
                    target=data.get('target', ''),
                    parameters=data.get('parameters', {}),
                    description=data.get('reasoning', 'LLM-guided mutation'),
                    source_agent='RedQueenEvolver',
                    confidence=0.8,
                )
        except Exception as e:
            print(f"Failed to parse mutation response: {e}")

        return None

    def _random_mutate(self, solution: Any) -> Optional[Any]:
        """Apply random mutation to solution."""
        if not hasattr(solution, 'copy') or not hasattr(solution, 'parameters'):
            return None

        mutated = solution.copy()

        # Randomly perturb 1-3 parameters
        params = list(mutated.parameters.keys())
        if not params:
            return mutated

        num_mutations = np.random.randint(1, min(4, len(params) + 1))
        params_to_mutate = np.random.choice(params, size=num_mutations, replace=False)

        for param in params_to_mutate:
            current = mutated.parameters[param]

            # Apply Gaussian perturbation
            std = abs(current) * 0.1 + 0.01  # 10% of value + small constant
            mutated.parameters[param] = current + np.random.normal(0, std)

        return mutated

    def _crossover(self, parent1: Any, parent2: Any) -> Optional[Any]:
        """Combine two parent solutions."""
        if not hasattr(parent1, 'copy') or not hasattr(parent1, 'parameters'):
            return None

        child = parent1.copy()

        # Blend parameters
        for param in child.parameters:
            if param in getattr(parent2, 'parameters', {}):
                alpha = np.random.random()
                child.parameters[param] = (
                    alpha * parent1.parameters[param] +
                    (1 - alpha) * parent2.parameters[param]
                )

        return child

    async def run_round(
        self,
        initial_solutions: List[Any],
        round_number: Optional[int] = None,
    ) -> EvolutionRound:
        """
        Run one round of Red Queen evolution.

        Args:
            initial_solutions: Starting population
            round_number: Override round number (default: auto-increment)

        Returns:
            EvolutionRound with results
        """
        start_time = datetime.now()

        if round_number is None:
            round_number = len(self.champions_history)

        print(f"\n{'='*60}")
        print(f"Red Queen Evolution - Round {round_number}")
        print(f"{'='*60}")
        print(f"Champions to beat: {sum(len(c) for c in self.champions_history)}")

        # Initialize MAP-Elites archive
        archive = MAPElitesArchive(
            dimensions=self.archive_dimensions,
            fitness_fn=self.fitness_fn,
            descriptor_fn=self.descriptor_fn,
        )

        # Seed archive with initial solutions
        for solution in initial_solutions:
            base_fitness = self.fitness_fn(solution) if self.fitness_fn else 0.5
            descriptor = self.descriptor_fn(solution)
            generality = self._evaluate_generality(solution, base_fitness, descriptor)
            combined_fitness = self._compute_combined_fitness(
                solution, base_fitness, generality
            )
            archive.add(solution, combined_fitness, descriptor)

        total_evaluations = len(initial_solutions)
        best_generality = 0.0

        # Evolution loop
        for iteration in range(self.iterations_per_round):
            # Sample parent from archive
            parent = archive.sample('curiosity')
            if parent is None:
                parent = initial_solutions[0] if initial_solutions else None

            if parent is None:
                continue

            # Generate offspring
            if np.random.random() < self.crossover_rate and archive.get_statistics().filled_cells > 1:
                # Crossover with another elite
                parent2 = archive.sample('fitness_weighted')
                offspring = self._crossover(parent, parent2)
            else:
                # Mutation
                descriptor = self.descriptor_fn(parent)
                unexplored = archive.get_unexplored_regions()
                target_region = unexplored[0] if unexplored and np.random.random() < 0.3 else None

                if np.random.random() < 0.5 and self.llm_client:
                    offspring = await self._llm_mutate(
                        parent, descriptor, round_number, target_region
                    )
                else:
                    offspring = self._random_mutate(parent)

            if offspring is None:
                continue

            # Evaluate offspring
            base_fitness = self.fitness_fn(offspring) if self.fitness_fn else 0.5
            descriptor = self.descriptor_fn(offspring)
            generality = self._evaluate_generality(offspring, base_fitness, descriptor)
            combined_fitness = self._compute_combined_fitness(
                offspring, base_fitness, generality
            )

            # Try to add to archive
            archive.add(offspring, combined_fitness, descriptor)
            total_evaluations += 1

            # Track best generality
            if generality.generality > best_generality:
                best_generality = generality.generality

            # Progress logging
            if (iteration + 1) % 20 == 0:
                stats = archive.get_statistics()
                print(f"  Iteration {iteration + 1}/{self.iterations_per_round}: "
                      f"filled={stats.filled_cells}, best={stats.max_fitness:.4f}, "
                      f"generality={best_generality:.3f}")

        # Extract champions
        elites = archive.get_all_elites()
        elites_with_fitness = [
            (e, self.fitness_fn(e) if self.fitness_fn else 0.5)
            for e in elites
        ]
        elites_with_fitness.sort(key=lambda x: -x[1])

        champions = []
        for solution, fitness in elites_with_fitness[:self.elite_count]:
            descriptor = self.descriptor_fn(solution)
            champions.append(Champion(
                solution=solution,
                fitness=fitness,
                descriptor=descriptor,
                round_number=round_number,
                phenotype_hash=self._compute_phenotype_hash(descriptor),
                genotype_hash=self._compute_genotype_hash(solution),
            ))

        # Add to history
        self.champions_history.append(champions)

        # Compute convergence metrics
        if champions:
            phenotypes = np.stack([c.descriptor.to_vector() for c in champions])
            avg_phenotype = phenotypes.mean(axis=0)
            self.phenotype_history.append(avg_phenotype)

            genotype_hashes = set(c.genotype_hash for c in champions)
            self.genotype_diversity.append(len(genotype_hashes) / len(champions))

        convergence_metrics = self._compute_convergence_metrics()

        # Build round result
        duration = (datetime.now() - start_time).total_seconds()
        stats = archive.get_statistics()

        round_result = EvolutionRound(
            round_number=round_number,
            champions=champions,
            archive_stats=stats,
            total_evaluations=total_evaluations,
            best_fitness=stats.max_fitness,
            best_generality=best_generality,
            convergence_metrics=convergence_metrics,
            duration_seconds=duration,
        )

        self.rounds_history.append(round_result)

        print(f"\nRound {round_number} complete:")
        print(f"  Champions: {len(champions)}")
        print(f"  Best fitness: {stats.max_fitness:.4f}")
        print(f"  Best generality: {best_generality:.3f}")
        print(f"  Archive coverage: {stats.coverage:.1%}")
        print(f"  Duration: {duration:.1f}s")

        return round_result

    def _compute_convergence_metrics(self) -> Dict[str, float]:
        """
        Compute convergence metrics tracking phenotype/genotype evolution.

        Key insight from DRQ: Phenotypes converge while genotypes remain diverse,
        similar to convergent evolution in biology.
        """
        metrics = {}

        if len(self.phenotype_history) >= 2:
            # Phenotype variance over rounds (should decrease = convergence)
            phenotypes = np.stack(self.phenotype_history)
            metrics['phenotype_variance'] = float(phenotypes.var(axis=0).mean())

            # Phenotype stability (change between last two rounds)
            delta = np.linalg.norm(self.phenotype_history[-1] - self.phenotype_history[-2])
            metrics['phenotype_stability'] = float(1.0 / (1.0 + delta))

        if len(self.genotype_diversity) >= 2:
            # Genotype diversity (should remain high = different implementations)
            metrics['genotype_diversity'] = float(np.mean(self.genotype_diversity))
            metrics['genotype_diversity_trend'] = float(
                self.genotype_diversity[-1] - self.genotype_diversity[0]
            )

        if len(self.rounds_history) >= 2:
            # Fitness progression
            fitnesses = [r.best_fitness for r in self.rounds_history]
            metrics['fitness_progression'] = float(fitnesses[-1] - fitnesses[0])

            # Generality progression
            generalities = [r.best_generality for r in self.rounds_history]
            metrics['generality_progression'] = float(generalities[-1] - generalities[0])

        return metrics

    async def evolve(
        self,
        initial_solutions: List[Any],
        num_rounds: int = 10,
        target_generality: float = 0.9,
    ) -> List[Champion]:
        """
        Run multiple rounds of Red Queen evolution.

        Args:
            initial_solutions: Starting population
            num_rounds: Maximum number of rounds
            target_generality: Stop early if this generality is achieved

        Returns:
            Final champions from the last round
        """
        print("\n" + "=" * 60)
        print("DIGITAL RED QUEEN EVOLUTION")
        print("=" * 60)
        print(f"Starting solutions: {len(initial_solutions)}")
        print(f"Max rounds: {num_rounds}")
        print(f"Target generality: {target_generality}")

        current_solutions = initial_solutions

        for round_num in range(num_rounds):
            result = await self.run_round(current_solutions, round_num)

            # Check for early termination
            if result.best_generality >= target_generality:
                print(f"\nTarget generality {target_generality} achieved!")
                break

            # Use champions as seeds for next round
            current_solutions = [c.solution for c in result.champions]

            # Add some random mutations for diversity
            if len(current_solutions) < self.population_size // 2:
                for _ in range(self.population_size // 4):
                    parent = np.random.choice(current_solutions)
                    mutated = self._random_mutate(parent)
                    if mutated:
                        current_solutions.append(mutated)

        # Return final champions
        if self.champions_history:
            return self.champions_history[-1]
        return []

    def get_all_champions(self) -> List[Champion]:
        """Get all champions from all rounds."""
        return sum(self.champions_history, [])

    def save(self, path: Path) -> None:
        """Save evolution state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save rounds history
        rounds_data = []
        for round_result in self.rounds_history:
            rounds_data.append({
                'round_number': round_result.round_number,
                'champion_count': len(round_result.champions),
                'best_fitness': round_result.best_fitness,
                'best_generality': round_result.best_generality,
                'total_evaluations': round_result.total_evaluations,
                'convergence_metrics': round_result.convergence_metrics,
                'duration_seconds': round_result.duration_seconds,
                'timestamp': round_result.timestamp,
            })

        with open(path / 'evolution_history.json', 'w') as f:
            json.dump({
                'rounds': rounds_data,
                'total_rounds': len(self.rounds_history),
                'total_champions': sum(len(c) for c in self.champions_history),
                'phenotype_convergence': self.phenotype_history[-1].tolist()
                    if self.phenotype_history else [],
                'genotype_diversity': self.genotype_diversity,
            }, f, indent=2)


if __name__ == '__main__':
    import asyncio

    print("Red Queen Evolver Test")
    print("=" * 60)

    # Create mock solutions for testing
    class MockSolution:
        def __init__(self, params=None):
            self.parameters = params or {
                'signal_clearance': np.random.uniform(0.1, 0.3),
                'via_diameter': np.random.uniform(0.6, 1.2),
                'zone_coverage': np.random.uniform(0.3, 0.8),
            }
            self._violations = np.random.randint(50, 500)

        def copy(self):
            return MockSolution(dict(self.parameters))

        def run_drc(self):
            class DRC:
                def __init__(self, v):
                    self.total_violations = v
                    self.violations_by_type = {'clearance': v // 2}
            return DRC(self._violations)

    def mock_fitness(solution):
        return 1.0 / (1.0 + solution._violations / 100.0)

    def mock_descriptor(solution):
        return BehavioralDescriptor(
            routing_density=solution.parameters.get('zone_coverage', 0.5),
            via_count=int(solution.parameters.get('via_diameter', 0.8) * 100),
            layer_utilization=0.5,
            zone_coverage=solution.parameters.get('zone_coverage', 0.5),
            thermal_spread=0.3,
            signal_length_variance=0.2,
            component_clustering=0.5,
            power_path_directness=0.6,
            min_clearance_ratio=solution.parameters.get('signal_clearance', 0.15) / 0.1,
            silk_density=0.3,
        )

    # Create evolver
    evolver = RedQueenEvolver(
        population_size=20,
        iterations_per_round=50,
        elite_count=3,
        fitness_fn=mock_fitness,
        descriptor_fn=mock_descriptor,
    )

    # Create initial solutions
    initial = [MockSolution() for _ in range(10)]

    # Run evolution
    async def main():
        champions = await evolver.evolve(initial, num_rounds=3, target_generality=0.8)
        print(f"\nFinal champions: {len(champions)}")
        for i, champ in enumerate(champions):
            print(f"  Champion {i+1}: fitness={champ.fitness:.4f}, "
                  f"phenotype={champ.phenotype_hash}")

    asyncio.run(main())
