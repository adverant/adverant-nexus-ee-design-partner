"""
Schematic Red Queen Evolver

Adversarial co-evolution for schematic optimization.
Solutions compete against each other, driving continuous improvement.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import random

import numpy as np

from .config import get_schematic_config, EvolutionConfig
from .behavior_descriptor import SchematicBehaviorDescriptor, compute_schematic_descriptor
from .fitness_function import SchematicFitness, compute_schematic_fitness
from .mutation_operators import SchematicMutator, MutationStrategy, MutationResult
from .schematic_map_elites import SchematicMAPElitesArchive, SchematicArchiveCell

logger = logging.getLogger(__name__)


class CompetitionOutcome(Enum):
    """Result of a head-to-head competition."""
    CHALLENGER_WINS = "challenger_wins"
    CHAMPION_WINS = "champion_wins"
    TIE = "tie"


@dataclass
class SchematicChampion:
    """A champion schematic in the evolution."""
    schematic: Dict[str, Any]
    fitness: SchematicFitness
    descriptor: SchematicBehaviorDescriptor
    wins: int = 0
    losses: int = 0
    ties: int = 0
    generation_created: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses + self.ties
        if total == 0:
            return 0.0
        return self.wins / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schematic": self.schematic,
            "fitness": self.fitness.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "generation_created": self.generation_created,
            "win_rate": self.win_rate,
        }


@dataclass
class SchematicEvolutionRound:
    """Results of an evolution round."""
    round_number: int
    champion: SchematicChampion
    challenger: SchematicChampion
    outcome: CompetitionOutcome
    fitness_improvement: float
    mutation_strategy: MutationStrategy
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round_number": self.round_number,
            "champion_fitness": self.champion.fitness.total,
            "challenger_fitness": self.challenger.fitness.total,
            "outcome": self.outcome.value,
            "fitness_improvement": self.fitness_improvement,
            "mutation_strategy": self.mutation_strategy.value,
            "timestamp": self.timestamp.isoformat(),
        }


class SchematicRedQueenEvolver:
    """
    Red Queen adversarial evolution for schematics.

    Implements a competitive evolution strategy where schematics
    compete against each other, driving continuous improvement
    through the "Red Queen" effect (running to stay in place).
    """

    def __init__(
        self,
        archive: Optional[SchematicMAPElitesArchive] = None,
        mutator: Optional[SchematicMutator] = None,
        config: Optional[EvolutionConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize Red Queen evolver.

        Args:
            archive: MAP-Elites archive for diversity
            mutator: Schematic mutator for generating challengers
            config: Evolution configuration
            llm_client: LLM client for guided mutations
        """
        self.config = config or get_schematic_config().evolution
        self.archive = archive or SchematicMAPElitesArchive()
        self.mutator = mutator or SchematicMutator(llm_client=llm_client)

        # Evolution state
        self.generation = 0
        self.round_count = 0
        self.history: List[SchematicEvolutionRound] = []

        # Population of champions
        self.champions: List[SchematicChampion] = []

        logger.info("Initialized Schematic Red Queen Evolver")

    async def initialize_population(
        self,
        seed_schematics: List[Dict[str, Any]]
    ) -> None:
        """
        Initialize champion population from seed schematics.

        Args:
            seed_schematics: Initial schematics to form population
        """
        self.champions = []

        for schematic in seed_schematics[:self.config.population_size]:
            fitness = compute_schematic_fitness(schematic)
            descriptor = compute_schematic_descriptor(schematic)

            champion = SchematicChampion(
                schematic=schematic,
                fitness=fitness,
                descriptor=descriptor,
                generation_created=self.generation,
            )
            self.champions.append(champion)

            # Add to archive
            self.archive.add(schematic, fitness, descriptor)

        logger.info(f"Initialized population with {len(self.champions)} champions")

    async def evolve_generation(
        self,
        validation_callback: Optional[Any] = None
    ) -> Tuple[SchematicChampion, List[SchematicEvolutionRound]]:
        """
        Run one generation of evolution.

        Args:
            validation_callback: Optional callback for validating schematics

        Returns:
            (best_champion, evolution_rounds)
        """
        self.generation += 1
        rounds: List[SchematicEvolutionRound] = []

        for round_idx in range(self.config.rounds_per_generation):
            # Select champion and create challenger
            champion = self._select_champion()
            challenger, mutation_result = await self._create_challenger(champion)

            if challenger is None:
                continue

            # Validate challenger if callback provided
            if validation_callback:
                try:
                    validation = await validation_callback(challenger.schematic)
                    # Update fitness based on validation
                    challenger.fitness = compute_schematic_fitness(
                        {**challenger.schematic, "validation_results": validation}
                    )
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")

            # Competition
            outcome, improvement = self._compete(champion, challenger)
            self.round_count += 1

            # Update records
            if outcome == CompetitionOutcome.CHALLENGER_WINS:
                champion.losses += 1
                challenger.wins += 1
                # Add challenger to population
                self._update_population(challenger)
            elif outcome == CompetitionOutcome.CHAMPION_WINS:
                champion.wins += 1
                challenger.losses += 1
            else:  # TIE
                champion.ties += 1
                challenger.ties += 1
                # Add challenger to archive anyway (diversity)
                self.archive.add(
                    challenger.schematic,
                    challenger.fitness,
                    challenger.descriptor
                )

            # Record round
            round_record = SchematicEvolutionRound(
                round_number=self.round_count,
                champion=champion,
                challenger=challenger,
                outcome=outcome,
                fitness_improvement=improvement,
                mutation_strategy=mutation_result.strategy if mutation_result else MutationStrategy.LLM_GUIDED,
            )
            rounds.append(round_record)
            self.history.append(round_record)

        # Update archive generation
        self.archive.increment_generation()

        # Get best champion
        best = self._get_best_champion()

        logger.info(
            f"Generation {self.generation} complete: "
            f"best fitness = {best.fitness.total:.3f}"
        )

        return best, rounds

    def _select_champion(self) -> SchematicChampion:
        """Select a champion for competition using tournament selection."""
        if not self.champions:
            raise ValueError("No champions in population")

        # Tournament selection
        tournament_size = min(3, len(self.champions))
        tournament = random.sample(self.champions, tournament_size)

        # Select based on fitness with some randomness
        weights = [c.fitness.total ** 2 for c in tournament]  # Square for selection pressure
        total = sum(weights)
        if total > 0:
            probs = [w / total for w in weights]
            return random.choices(tournament, weights=probs, k=1)[0]

        return random.choice(tournament)

    async def _create_challenger(
        self,
        champion: SchematicChampion
    ) -> Tuple[Optional[SchematicChampion], Optional[MutationResult]]:
        """Create a challenger by mutating the champion."""
        try:
            # Calculate stagnation based on champion's recent performance
            stagnation = max(0, champion.losses - champion.wins)

            # Mutate champion's schematic
            mutated, mutation_result = await self.mutator.mutate(
                champion.schematic,
                stagnation_count=stagnation,
            )

            if not mutation_result.success:
                return None, mutation_result

            # Compute fitness and descriptor
            fitness = compute_schematic_fitness(mutated)
            descriptor = compute_schematic_descriptor(mutated)

            challenger = SchematicChampion(
                schematic=mutated,
                fitness=fitness,
                descriptor=descriptor,
                generation_created=self.generation,
            )

            return challenger, mutation_result

        except Exception as e:
            logger.error(f"Failed to create challenger: {e}")
            return None, None

    def _compete(
        self,
        champion: SchematicChampion,
        challenger: SchematicChampion
    ) -> Tuple[CompetitionOutcome, float]:
        """
        Run competition between champion and challenger.

        Returns:
            (outcome, fitness_improvement)
        """
        improvement = challenger.fitness.total - champion.fitness.total

        if improvement > self.config.win_margin:
            return CompetitionOutcome.CHALLENGER_WINS, improvement
        elif improvement < -self.config.win_margin:
            return CompetitionOutcome.CHAMPION_WINS, improvement
        else:
            # Within tie margin - use secondary criteria
            # Compare diversity contribution
            if abs(improvement) <= self.config.tie_margin:
                return CompetitionOutcome.TIE, improvement

            # Slight advantage to one
            if improvement > 0:
                return CompetitionOutcome.CHALLENGER_WINS, improvement
            else:
                return CompetitionOutcome.CHAMPION_WINS, improvement

    def _update_population(self, challenger: SchematicChampion) -> None:
        """Update population with new challenger."""
        # Add challenger to archive
        self.archive.add(
            challenger.schematic,
            challenger.fitness,
            challenger.descriptor
        )

        # Add to champions if better than worst
        if len(self.champions) < self.config.population_size:
            self.champions.append(challenger)
        else:
            # Replace worst champion
            worst_idx = min(
                range(len(self.champions)),
                key=lambda i: self.champions[i].fitness.total
            )
            if challenger.fitness.total > self.champions[worst_idx].fitness.total:
                self.champions[worst_idx] = challenger

        # Maintain elite count
        self.champions.sort(key=lambda c: c.fitness.total, reverse=True)
        self.champions = self.champions[:self.config.population_size]

    def _get_best_champion(self) -> SchematicChampion:
        """Get the best champion from population."""
        if not self.champions:
            raise ValueError("No champions in population")
        return max(self.champions, key=lambda c: c.fitness.total)

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        if not self.champions:
            return {
                "generation": self.generation,
                "rounds": self.round_count,
                "champions": 0,
            }

        fitnesses = [c.fitness.total for c in self.champions]
        win_rates = [c.win_rate for c in self.champions]

        return {
            "generation": self.generation,
            "rounds": self.round_count,
            "champions": len(self.champions),
            "best_fitness": max(fitnesses),
            "average_fitness": np.mean(fitnesses),
            "best_win_rate": max(win_rates),
            "archive_statistics": self.archive.get_statistics().to_dict(),
        }

    def get_diverse_elites(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n diverse elite schematics."""
        cells = self.archive.get_diverse_sample(n)
        return [c.schematic for c in cells]
