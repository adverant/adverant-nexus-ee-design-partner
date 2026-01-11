"""
Tests for Dynamics Network (World Model) module.

Tests the MuZero-style learned world model.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWorldModel:
    """Tests for WorldModel."""

    def test_initialization(self, world_model):
        """Test world model initializes correctly."""
        import torch
        assert isinstance(world_model, torch.nn.Module)
        assert hasattr(world_model, 'representation')
        assert hasattr(world_model, 'dynamics')
        assert hasattr(world_model, 'prediction')

    def test_represent(self, world_model):
        """Test representation function."""
        import torch

        observation = torch.randn(8, 64)
        hidden_state = world_model.represent(observation)

        assert hidden_state.shape == (8, 64)

    def test_dynamics_step(self, world_model):
        """Test dynamics step."""
        import torch

        hidden_state = torch.randn(8, 64)
        action = torch.randn(8, 5)

        next_hidden, reward = world_model.dynamics_step(hidden_state, action)

        assert next_hidden.shape == (8, 64)
        assert reward.shape == (8,)

    def test_predict(self, world_model):
        """Test prediction from hidden state."""
        import torch

        hidden_state = torch.randn(8, 64)
        value, policy = world_model.predict(hidden_state)

        assert value.shape == (8,)
        assert policy.shape[0] == 8

    def test_imagine_trajectory(self, world_model):
        """Test trajectory imagination."""
        import torch

        observation = torch.randn(1, 64)
        actions = [torch.randn(1, 5) for _ in range(5)]

        trajectory = world_model.imagine_trajectory(observation, actions)

        assert len(trajectory) == 5
        for pred in trajectory:
            assert hasattr(pred, 'next_hidden_state')
            assert hasattr(pred, 'predicted_reward')
            assert hasattr(pred, 'predicted_value')
            assert hasattr(pred, 'predicted_policy')

    def test_gradient_flow(self, world_model):
        """Test gradients flow through all components."""
        import torch

        observation = torch.randn(4, 64, requires_grad=True)
        action = torch.randn(4, 5)

        hidden = world_model.represent(observation)
        next_hidden, reward = world_model.dynamics_step(hidden, action)
        value, policy = world_model.predict(next_hidden)

        loss = reward.mean() + value.mean()
        loss.backward()

        for param in world_model.parameters():
            if param.requires_grad:
                assert param.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDynamicsNetwork:
    """Tests for standalone DynamicsNetwork."""

    def test_initialization(self):
        """Test dynamics network initialization."""
        import torch
        from dynamics_network import DynamicsNetwork

        network = DynamicsNetwork(hidden_dim=64, action_dim=5)
        assert isinstance(network, torch.nn.Module)

    def test_forward(self):
        """Test forward pass."""
        import torch
        from dynamics_network import DynamicsNetwork

        network = DynamicsNetwork(hidden_dim=64, action_dim=5)
        hidden = torch.randn(8, 64)
        action = torch.randn(8, 5)

        next_hidden, reward = network(hidden, action)

        assert next_hidden.shape == (8, 64)
        assert reward.shape == (8,)

    def test_reward_prediction_range(self):
        """Test reward predictions are reasonable."""
        import torch
        from dynamics_network import DynamicsNetwork

        network = DynamicsNetwork(hidden_dim=64, action_dim=5)
        network.eval()

        hidden = torch.randn(100, 64)
        action = torch.randn(100, 5)

        with torch.no_grad():
            _, reward = network(hidden, action)

        # Rewards should be bounded
        assert torch.all(torch.isfinite(reward))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWorldModelTraining:
    """Tests for world model training."""

    def test_training_step(self, world_model):
        """Test single training step."""
        import torch
        import torch.optim as optim

        optimizer = optim.Adam(world_model.parameters(), lr=1e-3)

        # Training data
        observation = torch.randn(16, 64)
        action = torch.randn(16, 5)
        target_reward = torch.randn(16)
        target_value = torch.rand(16)

        world_model.train()
        optimizer.zero_grad()

        hidden = world_model.represent(observation)
        next_hidden, pred_reward = world_model.dynamics_step(hidden, action)
        pred_value, _ = world_model.predict(next_hidden)

        loss = (
            torch.nn.functional.mse_loss(pred_reward, target_reward) +
            torch.nn.functional.mse_loss(pred_value, target_value)
        )
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)

    def test_latent_consistency(self, world_model):
        """Test latent representations are consistent."""
        import torch

        world_model.eval()
        observation = torch.randn(1, 64)

        with torch.no_grad():
            hidden1 = world_model.represent(observation)
            hidden2 = world_model.represent(observation)

        assert torch.allclose(hidden1, hidden2)

    def test_save_load(self, world_model, temp_dir):
        """Test saving and loading world model."""
        import torch
        from dynamics_network import WorldModel

        save_path = temp_dir / "world_model.pt"
        torch.save(world_model.state_dict(), save_path)

        new_model = WorldModel(observation_dim=64, latent_dim=64)
        new_model.load_state_dict(torch.load(save_path))

        world_model.eval()
        new_model.eval()

        test_obs = torch.randn(1, 64)
        with torch.no_grad():
            h1 = world_model.represent(test_obs)
            h2 = new_model.represent(test_obs)

        assert torch.allclose(h1, h2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDynamicsPrediction:
    """Tests for DynamicsPrediction dataclass."""

    def test_creation(self):
        """Test creating DynamicsPrediction."""
        import torch
        from dynamics_network import DynamicsPrediction

        pred = DynamicsPrediction(
            next_hidden_state=torch.randn(1, 64),
            predicted_reward=0.5,
            predicted_value=0.7,
            predicted_policy=torch.randn(1, 9),
        )

        assert pred.predicted_reward == 0.5
        assert pred.predicted_value == 0.7


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWorldModelPlanning:
    """Tests for using world model for planning."""

    def test_multi_step_planning(self, world_model):
        """Test multi-step planning in latent space."""
        import torch

        world_model.eval()
        observation = torch.randn(1, 64)

        # Plan 10 steps ahead
        actions = [torch.randn(1, 5) for _ in range(10)]

        with torch.no_grad():
            trajectory = world_model.imagine_trajectory(observation, actions)

        # All predictions should be valid
        for pred in trajectory:
            assert torch.isfinite(pred.next_hidden_state).all()
            assert np.isfinite(pred.predicted_reward)
            assert np.isfinite(pred.predicted_value)

    def test_action_effect(self, world_model):
        """Test different actions produce different outcomes."""
        import torch

        world_model.eval()
        observation = torch.randn(1, 64)

        action1 = torch.zeros(1, 5)
        action2 = torch.ones(1, 5) * 2

        with torch.no_grad():
            hidden = world_model.represent(observation)
            next1, reward1 = world_model.dynamics_step(hidden, action1)
            next2, reward2 = world_model.dynamics_step(hidden, action2)

        # Different actions should produce different next states
        assert not torch.allclose(next1, next2)

    def test_batch_planning(self, world_model):
        """Test planning with batch of observations."""
        import torch

        world_model.eval()
        batch_size = 4
        observations = torch.randn(batch_size, 64)
        actions = [torch.randn(batch_size, 5) for _ in range(5)]

        with torch.no_grad():
            trajectories = world_model.imagine_trajectory(observations, actions)

        # Should handle batch correctly
        for pred in trajectories:
            assert pred.next_hidden_state.shape[0] == batch_size
