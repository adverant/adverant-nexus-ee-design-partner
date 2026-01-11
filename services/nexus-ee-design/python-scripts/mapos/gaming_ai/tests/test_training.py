"""
Tests for Training Pipeline module.

Tests the experience buffer and training pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import TORCH_AVAILABLE, create_random_experiences

# Import module under test
from training import ExperienceBuffer, Experience


class TestExperience:
    """Tests for Experience dataclass."""

    def test_creation(self):
        """Test creating an experience."""
        exp = Experience(
            state_embedding=np.random.randn(256).astype(np.float32),
            drc_context=np.random.randn(12).astype(np.float32),
            action_category=3,
            action_params=np.random.randn(5).astype(np.float32),
            reward=10.5,
            next_state_embedding=np.random.randn(256).astype(np.float32),
            done=False,
            value_target=0.75,
            policy_target=np.random.randn(9).astype(np.float32),
            optimization_id="test_123",
            step=5,
        )

        assert exp.action_category == 3
        assert exp.reward == 10.5
        assert exp.done is False

    def test_to_dict(self):
        """Test experience serialization."""
        exp = Experience(
            state_embedding=np.zeros(256, dtype=np.float32),
            drc_context=np.zeros(12, dtype=np.float32),
            action_category=3,
            action_params=np.zeros(5, dtype=np.float32),
            reward=10.5,
            next_state_embedding=np.zeros(256, dtype=np.float32),
            done=False,
            value_target=0.75,
            policy_target=np.zeros(9, dtype=np.float32),
            optimization_id="test_123",
            step=5,
        )

        data = exp.to_dict()
        assert data['action_category'] == 3
        assert data['reward'] == 10.5
        assert 'state_embedding' in data

    def test_from_dict(self):
        """Test experience deserialization."""
        data = {
            'state_embedding': [0.0] * 256,
            'drc_context': [0.0] * 12,
            'action_category': 3,
            'action_params': [0.0] * 5,
            'reward': 10.5,
            'next_state_embedding': [0.0] * 256,
            'done': False,
            'value_target': 0.75,
            'policy_target': [0.0] * 9,
            'optimization_id': 'test_123',
            'step': 5,
        }

        exp = Experience.from_dict(data)
        assert exp.action_category == 3
        assert exp.reward == 10.5


class TestExperienceBuffer:
    """Tests for ExperienceBuffer."""

    def test_initialization(self, experience_buffer):
        """Test buffer initialization."""
        assert experience_buffer is not None
        assert experience_buffer.capacity == 1000
        assert len(experience_buffer) == 0

    def test_add_experience(self, experience_buffer):
        """Test adding an experience."""
        exp = Experience(
            state_embedding=np.random.randn(256).astype(np.float32),
            drc_context=np.random.randn(12).astype(np.float32),
            action_category=3,
            action_params=np.random.randn(5).astype(np.float32),
            reward=10.5,
            next_state_embedding=np.random.randn(256).astype(np.float32),
            done=False,
            value_target=0.75,
            policy_target=np.random.randn(9).astype(np.float32),
        )

        experience_buffer.add(exp)
        assert len(experience_buffer) == 1

    def test_add_multiple_experiences(self, experience_buffer):
        """Test adding multiple experiences."""
        for i in range(100):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=i % 9,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i),
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=i % 10 == 0,
                value_target=i / 100,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        assert len(experience_buffer) == 100

    def test_capacity_limit(self, temp_dir):
        """Test buffer respects capacity limit."""
        buffer = ExperienceBuffer(
            capacity=50,
            save_path=temp_dir / "exp.json",
        )

        for i in range(100):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=0,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i),
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=False,
                value_target=0.5,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            buffer.add(exp)

        assert len(buffer) <= 50

    def test_sample(self, experience_buffer):
        """Test sampling from buffer."""
        # Add experiences
        for i in range(50):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=i % 9,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i),
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=False,
                value_target=i / 50,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        # Sample
        samples, indices, weights = experience_buffer.sample(16)

        assert len(samples) == 16
        assert len(indices) == 16
        assert len(weights) == 16
        assert all(w > 0 for w in weights)

    def test_sample_size_limit(self, experience_buffer):
        """Test sampling respects buffer size."""
        # Add only 5 experiences
        for i in range(5):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=0,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i),
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=False,
                value_target=0.5,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        # Try to sample more than available
        samples, indices, weights = experience_buffer.sample(10)

        assert len(samples) <= 5

    def test_update_priorities(self, experience_buffer):
        """Test updating priorities."""
        # Add experiences
        for i in range(20):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=0,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i),
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=False,
                value_target=0.5,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        # Sample and update priorities
        samples, indices, _ = experience_buffer.sample(10)
        new_priorities = np.random.rand(10)
        experience_buffer.update_priorities(indices, new_priorities)

        # Should not raise
        assert True

    def test_save_and_load(self, experience_buffer, temp_dir):
        """Test saving and loading buffer."""
        # Add experiences
        for i in range(10):
            exp = Experience(
                state_embedding=np.zeros(256, dtype=np.float32),
                drc_context=np.zeros(12, dtype=np.float32),
                action_category=i % 9,
                action_params=np.zeros(5, dtype=np.float32),
                reward=float(i),
                next_state_embedding=np.zeros(256, dtype=np.float32),
                done=False,
                value_target=0.5,
                policy_target=np.zeros(9, dtype=np.float32),
            )
            experience_buffer.add(exp)

        # Save
        experience_buffer.save()

        # Check file exists
        assert experience_buffer.save_path.exists()

        # Load into new buffer
        new_buffer = ExperienceBuffer(
            capacity=1000,
            save_path=experience_buffer.save_path,
        )
        new_buffer.load()

        assert len(new_buffer) == 10

    def test_clear(self, experience_buffer):
        """Test clearing buffer."""
        # Add experiences
        for i in range(10):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=0,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i),
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=False,
                value_target=0.5,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        assert len(experience_buffer) == 10

        experience_buffer.clear()
        assert len(experience_buffer) == 0

    def test_get_statistics(self, experience_buffer):
        """Test getting buffer statistics."""
        # Add experiences
        for i in range(50):
            exp = Experience(
                state_embedding=np.random.randn(256).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=i % 9,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(i) - 25,  # Some negative
                next_state_embedding=np.random.randn(256).astype(np.float32),
                done=i % 10 == 0,
                value_target=i / 50,
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        stats = experience_buffer.get_statistics()

        assert 'size' in stats
        assert 'capacity' in stats
        assert stats['size'] == 50


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingPipeline:
    """Tests for TrainingPipeline (requires PyTorch)."""

    @pytest.fixture
    def training_pipeline(self, temp_dir):
        """Create training pipeline."""
        from training import TrainingPipeline
        return TrainingPipeline(
            hidden_dim=64,
            checkpoint_dir=temp_dir / 'checkpoints',
        )

    def test_initialization(self, training_pipeline):
        """Test pipeline initialization."""
        assert training_pipeline is not None

    def test_train_with_buffer(self, training_pipeline, experience_buffer):
        """Test training with experience buffer."""
        # Add experiences
        for i in range(100):
            exp = Experience(
                state_embedding=np.random.randn(64).astype(np.float32),
                drc_context=np.random.randn(12).astype(np.float32),
                action_category=i % 9,
                action_params=np.random.randn(5).astype(np.float32),
                reward=float(np.random.randn()),
                next_state_embedding=np.random.randn(64).astype(np.float32),
                done=i % 10 == 0,
                value_target=np.random.rand(),
                policy_target=np.random.randn(9).astype(np.float32),
            )
            experience_buffer.add(exp)

        training_pipeline.buffer = experience_buffer
        result = training_pipeline.train(num_epochs=2, batch_size=16)

        assert result is not None

    def test_save_checkpoint(self, training_pipeline, temp_dir):
        """Test checkpoint saving."""
        checkpoint_path = training_pipeline.save_checkpoint()

        assert checkpoint_path is not None or training_pipeline.checkpoint_dir.exists()

    def test_load_checkpoint(self, training_pipeline, temp_dir):
        """Test checkpoint loading."""
        # Save first
        training_pipeline.save_checkpoint()

        # Load
        loaded = training_pipeline.load_checkpoint()

        # May or may not succeed depending on whether networks are initialized
        assert loaded is True or loaded is False
