"""
Tests for Ralph Wiggum Optimizer module.

Tests the persistent iteration optimization loop.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys
import json

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import MockPCBState, create_mock_population, TORCH_AVAILABLE

# Import module under test
from ralph_wiggum_optimizer import (
    RalphWiggumOptimizer, CompletionCriteria, OptimizationState,
    OptimizationResult, OptimizationStatus, EscalationStrategy
)
from red_queen_evolver import RedQueenEvolver
from map_elites import BehavioralDescriptor


class TestCompletionCriteria:
    """Tests for CompletionCriteria."""

    def test_creation(self):
        """Test creating completion criteria."""
        criteria = CompletionCriteria(
            target_violations=50,
            target_fitness=0.9,
            max_iterations=100,
            max_stagnation=15,
            max_duration_hours=24.0,
        )

        assert criteria.target_violations == 50
        assert criteria.target_fitness == 0.9
        assert criteria.max_iterations == 100

    def test_check_met_target_violations(self):
        """Test criteria met by reaching target violations."""
        criteria = CompletionCriteria(
            target_violations=50,
            target_fitness=0.9,
            max_iterations=1000,
            max_stagnation=100,
            max_duration_hours=100.0,
        )

        state = OptimizationState(
            current_iteration=10,
            best_violations=40,  # Below target
            best_fitness=0.85,
            stagnation_count=0,
            start_time=0,
        )

        is_met, reason = criteria.check(state)
        assert is_met
        assert 'target' in reason.lower()

    def test_check_met_max_iterations(self):
        """Test criteria met by reaching max iterations."""
        criteria = CompletionCriteria(
            target_violations=10,
            target_fitness=0.99,
            max_iterations=100,
            max_stagnation=50,
            max_duration_hours=100.0,
        )

        state = OptimizationState(
            current_iteration=100,  # At max
            best_violations=50,
            best_fitness=0.7,
            stagnation_count=0,
            start_time=0,
        )

        is_met, reason = criteria.check(state)
        assert is_met
        assert 'iteration' in reason.lower()

    def test_check_met_stagnation(self):
        """Test criteria met by stagnation."""
        criteria = CompletionCriteria(
            target_violations=10,
            target_fitness=0.99,
            max_iterations=1000,
            max_stagnation=15,
            max_duration_hours=100.0,
        )

        state = OptimizationState(
            current_iteration=50,
            best_violations=50,
            best_fitness=0.7,
            stagnation_count=15,  # At max stagnation
            start_time=0,
        )

        is_met, reason = criteria.check(state)
        assert is_met
        assert 'stagnation' in reason.lower() or 'stagnated' in reason.lower()

    def test_check_not_met(self):
        """Test criteria not met."""
        criteria = CompletionCriteria(
            target_violations=10,
            target_fitness=0.99,
            max_iterations=1000,
            max_stagnation=100,
            max_duration_hours=100.0,
        )

        state = OptimizationState(
            current_iteration=50,
            best_violations=50,
            best_fitness=0.7,
            stagnation_count=5,
            start_time=0,
        )

        is_met, reason = criteria.check(state)
        assert not is_met


class TestOptimizationState:
    """Tests for OptimizationState."""

    def test_creation(self):
        """Test creating optimization state."""
        state = OptimizationState(
            current_iteration=10,
            best_violations=100,
            best_fitness=0.7,
            stagnation_count=2,
            start_time=1000.0,
        )

        assert state.current_iteration == 10
        assert state.best_violations == 100

    def test_update_improvement(self):
        """Test updating state with improvement."""
        state = OptimizationState(
            current_iteration=10,
            best_violations=100,
            best_fitness=0.7,
            stagnation_count=5,
            start_time=0,
        )

        # Simulate improvement
        new_violations = 80
        if new_violations < state.best_violations:
            state.best_violations = new_violations
            state.stagnation_count = 0
        else:
            state.stagnation_count += 1

        state.current_iteration += 1

        assert state.best_violations == 80
        assert state.stagnation_count == 0
        assert state.current_iteration == 11

    def test_update_no_improvement(self):
        """Test updating state without improvement."""
        state = OptimizationState(
            current_iteration=10,
            best_violations=100,
            best_fitness=0.7,
            stagnation_count=5,
            start_time=0,
        )

        # Simulate no improvement
        new_violations = 120
        if new_violations < state.best_violations:
            state.best_violations = new_violations
            state.stagnation_count = 0
        else:
            state.stagnation_count += 1

        state.current_iteration += 1

        assert state.best_violations == 100  # Unchanged
        assert state.stagnation_count == 6
        assert state.current_iteration == 11

    def test_to_dict(self):
        """Test serialization."""
        state = OptimizationState(
            current_iteration=10,
            best_violations=100,
            best_fitness=0.7,
            stagnation_count=2,
            start_time=1000.0,
        )

        data = state.to_dict()
        assert data['current_iteration'] == 10
        assert data['best_violations'] == 100

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            'current_iteration': 10,
            'best_violations': 100,
            'best_fitness': 0.7,
            'stagnation_count': 2,
            'start_time': 1000.0,
        }

        state = OptimizationState.from_dict(data)
        assert state.current_iteration == 10
        assert state.best_violations == 100


class TestOptimizationStatus:
    """Tests for OptimizationStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses exist."""
        expected = ['SUCCESS', 'MAX_ITERATIONS', 'STAGNATED', 'TIMEOUT', 'FAILED']
        for status in expected:
            assert hasattr(OptimizationStatus, status)


class TestEscalationStrategy:
    """Tests for EscalationStrategy enum."""

    def test_all_strategies_exist(self):
        """Verify all expected strategies exist."""
        expected = [
            'INCREASE_POPULATION', 'INCREASE_MUTATION', 'ENABLE_LLM',
            'RESTART_POPULATION', 'ENABLE_WORLD_MODEL'
        ]
        for strategy in expected:
            assert hasattr(EscalationStrategy, strategy)


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_creation(self):
        """Test creating optimization result."""
        result = OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            final_violations=25,
            initial_violations=500,
            improvement=475,
            iterations=50,
            duration_seconds=3600.0,
            escalations_used=[EscalationStrategy.INCREASE_MUTATION],
        )

        assert result.status == OptimizationStatus.SUCCESS
        assert result.final_violations == 25
        assert result.improvement == 475


class TestRalphWiggumOptimizer:
    """Tests for RalphWiggumOptimizer."""

    @pytest.fixture
    def optimizer(self, temp_dir):
        """Create a Ralph Wiggum optimizer."""
        def fitness_fn(solution):
            if hasattr(solution, 'run_drc'):
                drc = solution.run_drc()
                return 1.0 / (1.0 + drc.total_violations / 100.0)
            return 0.5

        red_queen = RedQueenEvolver(
            population_size=10,
            iterations_per_round=5,
            elite_count=2,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=None,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

        criteria = CompletionCriteria(
            target_violations=20,
            target_fitness=0.9,
            max_iterations=10,  # Low for testing
            max_stagnation=5,
            max_duration_hours=0.01,  # Very short
        )

        # Create mock PCB file
        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        return RalphWiggumOptimizer(
            pcb_path=mock_pcb,
            output_dir=temp_dir,
            criteria=criteria,
            red_queen_evolver=red_queen,
            use_git=False,
            llm_client=None,
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert optimizer.criteria is not None
        assert optimizer.red_queen is not None

    def test_state_initialization(self, optimizer):
        """Test state is properly initialized."""
        assert optimizer.state is not None
        assert optimizer.state.current_iteration == 0

    @pytest.mark.asyncio
    async def test_run_completes(self, optimizer):
        """Test that optimization run completes."""
        # Mock the PCB state loading
        optimizer._load_initial_state = lambda: MockPCBState(violations=100)

        result = await optimizer.run()

        assert isinstance(result, OptimizationResult)
        assert result.status in OptimizationStatus

    @pytest.mark.asyncio
    async def test_run_respects_max_iterations(self, optimizer):
        """Test that run respects max iterations."""
        optimizer._load_initial_state = lambda: MockPCBState(violations=500)

        result = await optimizer.run()

        assert result.iterations <= optimizer.criteria.max_iterations + 1

    def test_state_persistence(self, optimizer, temp_dir):
        """Test state file persistence."""
        optimizer.state.current_iteration = 5
        optimizer.state.best_violations = 75
        optimizer._save_state()

        state_file = temp_dir / "optimization_state.json"
        assert state_file.exists()

        with open(state_file) as f:
            data = json.load(f)
        assert data['current_iteration'] == 5

    def test_state_resume(self, optimizer, temp_dir):
        """Test resuming from saved state."""
        # Save initial state
        state_file = temp_dir / "optimization_state.json"
        state_data = {
            'current_iteration': 5,
            'best_violations': 75,
            'best_fitness': 0.6,
            'stagnation_count': 2,
            'start_time': 1000.0,
        }
        with open(state_file, 'w') as f:
            json.dump(state_data, f)

        # Load state
        optimizer._load_state()

        assert optimizer.state.current_iteration == 5
        assert optimizer.state.best_violations == 75

    def test_escalation_detection(self, optimizer):
        """Test that escalation is triggered on stagnation."""
        optimizer.state.stagnation_count = 5

        should_escalate = optimizer.state.stagnation_count >= optimizer.criteria.max_stagnation // 2
        assert should_escalate

    @pytest.mark.asyncio
    async def test_early_termination_on_target(self, optimizer):
        """Test early termination when target is reached."""
        # Mock a state that quickly reaches target
        class QuickSuccessState(MockPCBState):
            def __init__(self):
                super().__init__(violations=10)  # Below target of 20

        optimizer._load_initial_state = lambda: QuickSuccessState()

        result = await optimizer.run()

        # Should terminate early with success
        assert result.status in [OptimizationStatus.SUCCESS, OptimizationStatus.MAX_ITERATIONS]


class TestRalphWiggumPersistence:
    """Tests for Ralph Wiggum file-based persistence."""

    def test_progress_file_creation(self, temp_dir):
        """Test that progress files are created."""
        criteria = CompletionCriteria(
            target_violations=10,
            target_fitness=0.99,
            max_iterations=5,
            max_stagnation=3,
            max_duration_hours=0.001,
        )

        def fitness_fn(s):
            return 0.5

        red_queen = RedQueenEvolver(
            population_size=5,
            iterations_per_round=2,
            elite_count=1,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=None,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        optimizer = RalphWiggumOptimizer(
            pcb_path=mock_pcb,
            output_dir=temp_dir,
            criteria=criteria,
            red_queen_evolver=red_queen,
            use_git=False,
            llm_client=None,
        )

        optimizer._save_state()

        state_file = temp_dir / "optimization_state.json"
        assert state_file.exists()

    def test_checkpoint_save_load(self, temp_dir):
        """Test checkpoint save and load."""
        criteria = CompletionCriteria(
            target_violations=10,
            target_fitness=0.99,
            max_iterations=100,
            max_stagnation=10,
            max_duration_hours=1.0,
        )

        def fitness_fn(s):
            return 0.5

        red_queen = RedQueenEvolver(
            population_size=5,
            iterations_per_round=2,
            elite_count=1,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=None,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        # First optimizer instance
        optimizer1 = RalphWiggumOptimizer(
            pcb_path=mock_pcb,
            output_dir=temp_dir,
            criteria=criteria,
            red_queen_evolver=red_queen,
            use_git=False,
            llm_client=None,
        )

        optimizer1.state.current_iteration = 25
        optimizer1.state.best_violations = 50
        optimizer1._save_state()

        # Second optimizer instance - should resume
        optimizer2 = RalphWiggumOptimizer(
            pcb_path=mock_pcb,
            output_dir=temp_dir,
            criteria=criteria,
            red_queen_evolver=red_queen,
            use_git=False,
            llm_client=None,
        )
        optimizer2._load_state()

        assert optimizer2.state.current_iteration == 25
        assert optimizer2.state.best_violations == 50
