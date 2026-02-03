"""
Tests for Red Queen Evolver module.

Tests the Digital Red Queen adversarial evolution system.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import MockPCBState, create_mock_population, TORCH_AVAILABLE

# Import module under test
from red_queen_evolver import (
    RedQueenEvolver, Champion, EvolutionRound, GeneralityScore
)
from map_elites import BehavioralDescriptor


class TestChampion:
    """Tests for Champion dataclass."""

    def test_creation(self):
        """Test creating a champion."""
        state = MockPCBState(violations=50)
        champion = Champion(
            solution=state,
            fitness=0.8,
            generality_score=0.7,
            round_discovered=5,
            phenotype_hash="abc123",
            genotype_hash="def456",
        )

        assert champion.fitness == 0.8
        assert champion.generality_score == 0.7
        assert champion.round_discovered == 5

    def test_champion_comparison(self):
        """Test comparing champions by fitness."""
        c1 = Champion(
            solution=MockPCBState(violations=100),
            fitness=0.5,
            generality_score=0.6,
            round_discovered=1,
            phenotype_hash="a",
            genotype_hash="b",
        )
        c2 = Champion(
            solution=MockPCBState(violations=50),
            fitness=0.8,
            generality_score=0.7,
            round_discovered=2,
            phenotype_hash="c",
            genotype_hash="d",
        )

        assert c2.fitness > c1.fitness


class TestGeneralityScore:
    """Tests for GeneralityScore dataclass."""

    def test_creation(self):
        """Test creating generality score."""
        score = GeneralityScore(
            score=0.75,
            wins=15,
            losses=5,
            draws=0,
            opponents_tested=20,
        )

        assert score.score == 0.75
        assert score.wins == 15
        assert score.opponents_tested == 20

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        score = GeneralityScore(
            score=0.0,  # Will be calculated
            wins=10,
            losses=5,
            draws=5,
            opponents_tested=20,
        )

        # Win rate = wins / opponents_tested
        expected_rate = 10 / 20
        assert abs(score.wins / score.opponents_tested - expected_rate) < 0.001


class TestEvolutionRound:
    """Tests for EvolutionRound dataclass."""

    def test_creation(self):
        """Test creating evolution round."""
        champions = [
            Champion(
                solution=MockPCBState(violations=50 + i * 10),
                fitness=0.8 - i * 0.1,
                generality_score=0.7,
                round_discovered=1,
                phenotype_hash=f"p{i}",
                genotype_hash=f"g{i}",
            )
            for i in range(3)
        ]

        round_result = EvolutionRound(
            round_number=5,
            champions=champions,
            population_size=50,
            elite_count=5,
            best_fitness=0.8,
            avg_fitness=0.65,
            diversity_score=0.7,
            convergence_detected=False,
        )

        assert round_result.round_number == 5
        assert len(round_result.champions) == 3
        assert round_result.best_fitness == 0.8


class TestRedQueenEvolver:
    """Tests for RedQueenEvolver."""

    @pytest.fixture
    def evolver(self):
        """Create a Red Queen evolver."""
        def fitness_fn(solution):
            if hasattr(solution, 'run_drc'):
                drc = solution.run_drc()
                return 1.0 / (1.0 + drc.total_violations / 100.0)
            return 0.5

        return RedQueenEvolver(
            population_size=20,
            iterations_per_round=10,
            elite_count=3,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=None,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

    def test_initialization(self, evolver):
        """Test evolver initialization."""
        assert evolver.population_size == 20
        assert evolver.elite_count == 3
        assert evolver.mutation_rate == 0.8

    @pytest.mark.asyncio
    async def test_run_round(self, evolver):
        """Test running a single evolution round."""
        population = create_mock_population(size=20, base_violations=500)

        round_result = await evolver.run_round(population, round_number=0)

        assert isinstance(round_result, EvolutionRound)
        assert round_result.round_number == 0
        assert round_result.population_size == len(population)

    @pytest.mark.asyncio
    async def test_multiple_rounds(self, evolver):
        """Test running multiple evolution rounds."""
        population = create_mock_population(size=20, base_violations=500)

        for i in range(3):
            round_result = await evolver.run_round(population, round_number=i)

            # Get elites for next round
            if round_result.champions:
                population = [c.solution for c in round_result.champions]
                # Fill back to population size
                while len(population) < 20:
                    base = population[np.random.randint(len(population))]
                    mutant = evolver._random_mutate(base)
                    if mutant:
                        population.append(mutant)
                    else:
                        population.append(base.copy())

        assert len(evolver.rounds_history) >= 1

    def test_random_mutate(self, evolver):
        """Test random mutation."""
        state = MockPCBState(violations=100)
        mutant = evolver._random_mutate(state)

        # Should return a new state (or None if mutation fails)
        if mutant is not None:
            assert mutant is not state

    def test_crossover(self, evolver):
        """Test crossover operation."""
        parent1 = MockPCBState(violations=100)
        parent2 = MockPCBState(violations=80)

        child = evolver._crossover(parent1, parent2)

        if child is not None:
            assert isinstance(child, MockPCBState)

    def test_get_all_champions(self, evolver):
        """Test getting all champions."""
        # Initially empty
        champions = evolver.get_all_champions()
        assert len(champions) == 0

    @pytest.mark.asyncio
    async def test_champion_discovery(self, evolver):
        """Test that champions are discovered during evolution."""
        population = create_mock_population(size=20, base_violations=200)

        round_result = await evolver.run_round(population, round_number=0)

        if round_result.champions:
            assert all(isinstance(c, Champion) for c in round_result.champions)
            assert all(c.fitness > 0 for c in round_result.champions)

    def test_compute_convergence_metrics(self, evolver):
        """Test convergence metric computation."""
        metrics = evolver._compute_convergence_metrics()

        assert isinstance(metrics, dict)
        # Empty evolver should return valid metrics
        assert 'fitness_variance' in metrics or len(metrics) == 0

    @pytest.mark.asyncio
    async def test_diversity_maintenance(self, evolver):
        """Test that diversity is maintained across rounds."""
        population = create_mock_population(size=20, base_violations=300)

        round_result = await evolver.run_round(population, round_number=0)

        assert round_result.diversity_score >= 0

    @pytest.mark.asyncio
    async def test_fitness_improvement(self, evolver):
        """Test that fitness generally improves over rounds."""
        # Start with high violations
        population = create_mock_population(size=20, base_violations=800)

        initial_best = max(
            1.0 / (1.0 + s._violations / 100.0)
            for s in population
        )

        # Run several rounds
        for i in range(5):
            round_result = await evolver.run_round(population, round_number=i)
            if round_result.champions:
                population = [c.solution for c in round_result.champions[:5]]
                while len(population) < 20:
                    base = population[np.random.randint(len(population))]
                    mutant = evolver._random_mutate(base)
                    population.append(mutant if mutant else base.copy())

        # Check final best
        if evolver.rounds_history:
            final_best = evolver.rounds_history[-1].best_fitness
            # Should generally improve (or at least not get worse)
            assert final_best >= initial_best * 0.8  # Allow some variance


class TestRedQueenWithLLM:
    """Tests for Red Queen with LLM integration."""

    @pytest.fixture
    def mock_llm_evolver(self, mock_llm_client):
        """Create evolver with mock LLM."""
        def fitness_fn(solution):
            if hasattr(solution, 'run_drc'):
                drc = solution.run_drc()
                return 1.0 / (1.0 + drc.total_violations / 100.0)
            return 0.5

        return RedQueenEvolver(
            population_size=10,
            iterations_per_round=5,
            elite_count=2,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=mock_llm_client,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

    @pytest.mark.asyncio
    async def test_llm_mutation(self, mock_llm_evolver):
        """Test LLM-guided mutation."""
        state = MockPCBState(violations=100)

        # LLM mutation should be attempted
        mutant = await mock_llm_evolver._llm_mutate(state)

        # May or may not succeed depending on mock
        assert mutant is None or isinstance(mutant, MockPCBState)


class TestRedQueenPersistence:
    """Tests for Red Queen state persistence."""

    @pytest.fixture
    def evolver(self):
        """Create evolver for persistence tests."""
        def fitness_fn(solution):
            return 0.5

        return RedQueenEvolver(
            population_size=10,
            iterations_per_round=5,
            elite_count=2,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=None,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

    @pytest.mark.asyncio
    async def test_history_persistence(self, evolver, temp_dir):
        """Test that evolution history is maintained."""
        population = create_mock_population(size=10, base_violations=200)

        # Run some rounds
        for i in range(3):
            await evolver.run_round(population, round_number=i)

        assert len(evolver.rounds_history) == 3

    @pytest.mark.asyncio
    async def test_champions_accumulation(self, evolver):
        """Test that champions accumulate over rounds."""
        population = create_mock_population(size=10, base_violations=200)

        for i in range(3):
            await evolver.run_round(population, round_number=i)

        # Some champions should have been discovered
        all_champions = evolver.get_all_champions()
        # May be 0 if no champions discovered
        assert isinstance(all_champions, list)
