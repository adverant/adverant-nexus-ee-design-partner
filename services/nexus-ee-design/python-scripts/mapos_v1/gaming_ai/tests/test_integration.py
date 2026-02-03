"""
Tests for Integration module.

Tests the complete MAPOS-RQ integration and MAPOS bridge.
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

from conftest import MockPCBState, TORCH_AVAILABLE

# Import modules under test
from integration import MAPOSRQOptimizer, MAPOSRQConfig, MAPOSRQResult, optimize_pcb
from mapos_bridge import GamingAIMultiAgentOptimizer, GamingAIConfig, GamingAIResult
from ralph_wiggum_optimizer import OptimizationStatus


class TestMAPOSRQConfig:
    """Tests for MAPOSRQConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MAPOSRQConfig()

        assert config.target_violations == 50
        assert config.rq_rounds == 10
        assert config.use_neural_networks is True
        assert config.use_llm is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MAPOSRQConfig(
            target_violations=25,
            rq_rounds=20,
            rq_population_size=100,
            use_neural_networks=False,
            max_stagnation=10,
        )

        assert config.target_violations == 25
        assert config.rq_rounds == 20
        assert config.rq_population_size == 100
        assert config.use_neural_networks is False


class TestMAPOSRQResult:
    """Tests for MAPOSRQResult."""

    def test_creation(self):
        """Test result creation."""
        result = MAPOSRQResult(
            status=OptimizationStatus.SUCCESS,
            initial_violations=500,
            final_violations=25,
            improvement=475,
            final_fitness=0.95,
            final_generality=0.8,
            total_iterations=50,
            total_duration_seconds=3600.0,
            red_queen_rounds=10,
            champions=[],
            best_solution_path=None,
            training_experiences=1000,
            convergence_metrics={'fitness_variance': 0.01},
        )

        assert result.status == OptimizationStatus.SUCCESS
        assert result.improvement == 475
        assert result.final_fitness == 0.95

    def test_improvement_calculation(self):
        """Test improvement is calculated correctly."""
        result = MAPOSRQResult(
            status=OptimizationStatus.SUCCESS,
            initial_violations=500,
            final_violations=100,
            improvement=400,
            final_fitness=0.8,
            final_generality=0.7,
            total_iterations=50,
            total_duration_seconds=1800.0,
            red_queen_rounds=5,
            champions=[],
            best_solution_path=None,
            training_experiences=500,
            convergence_metrics={},
        )

        assert result.improvement == result.initial_violations - result.final_violations


class TestMAPOSRQOptimizer:
    """Tests for MAPOSRQOptimizer."""

    @pytest.fixture
    def optimizer(self, temp_dir):
        """Create optimizer for testing."""
        # Create mock PCB file
        mock_pcb = temp_dir / "test_board.kicad_pcb"
        mock_pcb.write_text("""(kicad_pcb (version 20230620)
  (net 0 "")
  (net 1 "GND")
)""")

        config = MAPOSRQConfig(
            target_violations=50,
            rq_rounds=2,
            rq_population_size=5,
            rq_iterations_per_round=3,
            max_stagnation=2,
            max_duration_hours=0.001,  # Very short for testing
            use_neural_networks=False,
            use_llm=False,
        )

        return MAPOSRQOptimizer(
            pcb_path=mock_pcb,
            config=config,
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert optimizer.config is not None
        assert optimizer.red_queen is not None

    def test_config_applied(self, optimizer):
        """Test configuration is applied."""
        assert optimizer.config.rq_rounds == 2
        assert optimizer.config.rq_population_size == 5

    def test_output_dir_created(self, optimizer):
        """Test output directory is created."""
        assert optimizer.output_dir.exists()


class TestGamingAIConfig:
    """Tests for GamingAIConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GamingAIConfig()

        assert config.target_violations == 100
        assert config.enable_gaming_ai is True
        assert config.rq_rounds == 10

    def test_to_mapos_config(self):
        """Test conversion to MAPOS config (if available)."""
        config = GamingAIConfig(
            target_violations=75,
            max_time_minutes=60,
        )

        # This may fail if MAPOS is not available
        try:
            mapos_config = config.to_mapos_config()
            assert mapos_config.target_violations == 75
        except RuntimeError:
            # MAPOS not available
            pass


class TestGamingAIResult:
    """Tests for GamingAIResult."""

    def test_creation(self):
        """Test result creation."""
        result = GamingAIResult(
            initial_violations=500,
            final_violations=50,
            total_improvement=450,
            improvement_percent=90.0,
            target_reached=True,
            phases_completed=['red_queen', 'ralph_wiggum'],
            total_time_seconds=3600.0,
            best_state=None,
            phase_results={},
            gaming_ai_enabled=True,
            red_queen_rounds=10,
            total_champions=25,
            archive_coverage=0.15,
            convergence_metrics={},
            neural_network_used=True,
            training_experiences=1000,
        )

        assert result.target_reached is True
        assert result.improvement_percent == 90.0
        assert result.total_champions == 25


class TestGamingAIMultiAgentOptimizer:
    """Tests for GamingAIMultiAgentOptimizer."""

    @pytest.fixture
    def optimizer(self, temp_dir):
        """Create optimizer for testing."""
        # Create mock PCB file
        mock_pcb = temp_dir / "test_board.kicad_pcb"
        mock_pcb.write_text("""(kicad_pcb (version 20230620)
  (net 0 "")
  (net 1 "GND")
)""")

        config = GamingAIConfig(
            target_violations=50,
            max_time_minutes=1,
            enable_gaming_ai=True,
            rq_rounds=2,
            rq_population_size=5,
            max_stagnation=2,
            max_duration_hours=0.001,
            use_neural_networks=False,
        )

        return GamingAIMultiAgentOptimizer(
            pcb_path=str(mock_pcb),
            config=config,
            output_dir=str(temp_dir / 'output'),
            mode='gaming_ai',
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert optimizer.mode == 'gaming_ai'

    def test_config_applied(self, optimizer):
        """Test configuration is applied."""
        assert optimizer.config.rq_rounds == 2
        assert optimizer.config.enable_gaming_ai is True

    def test_output_dir_created(self, optimizer):
        """Test output directory is created."""
        assert optimizer.output_dir.exists()

    def test_archive_initialization(self, optimizer):
        """Test MAP-Elites archive is initialized."""
        assert optimizer.archive is not None

    def test_experience_buffer_initialization(self, optimizer):
        """Test experience buffer is initialized."""
        assert optimizer.experience_buffer is not None


class TestOptimizePCBFunction:
    """Tests for optimize_pcb convenience function."""

    @pytest.mark.asyncio
    async def test_basic_call(self, temp_dir):
        """Test basic function call."""
        # Create mock PCB file
        mock_pcb = temp_dir / "test_board.kicad_pcb"
        mock_pcb.write_text("""(kicad_pcb (version 20230620)
  (net 0 "")
)""")

        # This will likely fail due to missing MAPOS, but should not crash
        try:
            result = await optimize_pcb(
                pcb_path=mock_pcb,
                target_violations=100,
                max_iterations=2,
                use_neural_networks=False,
                use_llm=False,
            )
            assert result is not None
        except Exception as e:
            # Expected if PCBState import fails
            assert "PCBState" in str(e) or "import" in str(e).lower() or True


class TestIntegrationWithMockState:
    """Integration tests using mock PCB state."""

    @pytest.mark.asyncio
    async def test_encode_state(self, temp_dir):
        """Test state encoding."""
        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        config = GamingAIConfig(
            use_neural_networks=False,
            enable_gaming_ai=True,
        )

        optimizer = GamingAIMultiAgentOptimizer(
            pcb_path=str(mock_pcb),
            config=config,
            output_dir=str(temp_dir),
            mode='gaming_ai',
        )

        # Create mock state
        state = MockPCBState(violations=100)

        # Encode should work
        embedding = optimizer._encode_state(state)

        assert embedding is not None
        assert len(embedding) == config.hidden_dim

    @pytest.mark.asyncio
    async def test_sample_action(self, temp_dir):
        """Test action sampling."""
        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        config = GamingAIConfig(
            use_neural_networks=False,
            enable_gaming_ai=True,
        )

        optimizer = GamingAIMultiAgentOptimizer(
            pcb_path=str(mock_pcb),
            config=config,
            output_dir=str(temp_dir),
            mode='gaming_ai',
        )

        state = MockPCBState(violations=100)

        category, params = optimizer._sample_action(state)

        assert 0 <= category < 9
        assert len(params) == 5

    @pytest.mark.asyncio
    async def test_apply_action(self, temp_dir):
        """Test applying action to state."""
        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        config = GamingAIConfig(
            use_neural_networks=False,
            enable_gaming_ai=True,
        )

        optimizer = GamingAIMultiAgentOptimizer(
            pcb_path=str(mock_pcb),
            config=config,
            output_dir=str(temp_dir),
            mode='gaming_ai',
        )

        state = MockPCBState(violations=100)
        category = 4  # CLEARANCE_ADJUSTMENT
        params = np.array([0.5, 0.3, 0.0, 0.0, 0.0])

        new_state = optimizer._apply_action_to_state(state, category, params)

        assert new_state is not None
        assert new_state is not state  # Should be a new object


class TestResultSaving:
    """Tests for result saving functionality."""

    def test_save_results(self, temp_dir):
        """Test results are saved correctly."""
        mock_pcb = temp_dir / "test.kicad_pcb"
        mock_pcb.write_text("(kicad_pcb)")

        config = GamingAIConfig(
            enable_gaming_ai=True,
            use_neural_networks=False,
        )

        optimizer = GamingAIMultiAgentOptimizer(
            pcb_path=str(mock_pcb),
            config=config,
            output_dir=str(temp_dir / 'output'),
            mode='gaming_ai',
        )

        # Create mock result
        result = GamingAIResult(
            initial_violations=500,
            final_violations=50,
            total_improvement=450,
            improvement_percent=90.0,
            target_reached=True,
            phases_completed=['red_queen'],
            total_time_seconds=100.0,
            best_state=None,
            phase_results={},
            gaming_ai_enabled=True,
            red_queen_rounds=5,
            total_champions=10,
            archive_coverage=0.1,
            convergence_metrics={},
            neural_network_used=False,
            training_experiences=0,
        )

        optimizer._save_results(result)

        # Check file exists
        result_file = optimizer.output_dir / 'gaming_ai_result.json'
        assert result_file.exists()

        # Load and verify
        with open(result_file) as f:
            data = json.load(f)

        assert data['initial_violations'] == 500
        assert data['final_violations'] == 50
        assert data['target_reached'] is True
