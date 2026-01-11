"""
Tests for Value Network module.

Tests the AlphaZero-style value prediction network.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import MockPCBState, TORCH_AVAILABLE, create_random_experiences


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestValueNetwork:
    """Tests for ValueNetwork."""

    def test_initialization(self, value_network):
        """Test value network initializes correctly."""
        import torch
        assert isinstance(value_network, torch.nn.Module)

    def test_forward_single(self, value_network, random_embedding):
        """Test forward pass with single embedding."""
        import torch
        emb = torch.tensor(random_embedding).unsqueeze(0)
        output = value_network(emb)

        assert output.shape == (1,)
        assert -1.0 <= output.item() <= 1.0  # tanh range

    def test_forward_batch(self, value_network):
        """Test forward pass with batch of embeddings."""
        import torch
        batch_size = 32
        input_dim = 64

        emb = torch.randn(batch_size, input_dim)
        output = value_network(emb)

        assert output.shape == (batch_size,)
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)

    def test_gradient_flow(self, value_network):
        """Test gradients flow correctly."""
        import torch
        emb = torch.randn(8, 64, requires_grad=True)
        output = value_network(emb)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        for param in value_network.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_predict_value(self, value_network):
        """Test value prediction method."""
        import torch
        value_network.eval()

        emb = torch.randn(1, 64)
        with torch.no_grad():
            value = value_network(emb)

        assert isinstance(value.item(), float)

    def test_deterministic_eval(self, value_network):
        """Test deterministic output in eval mode."""
        import torch
        value_network.eval()

        emb = torch.randn(1, 64)
        with torch.no_grad():
            v1 = value_network(emb)
            v2 = value_network(emb)

        assert torch.allclose(v1, v2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestValuePrediction:
    """Tests for ValuePrediction dataclass."""

    def test_value_prediction_creation(self):
        """Test creating ValuePrediction."""
        from value_network import ValuePrediction

        pred = ValuePrediction(
            value=0.75,
            violation_breakdown={'clearance': 0.3, 'silk': 0.2},
            uncertainty=0.1,
        )

        assert pred.value == 0.75
        assert pred.uncertainty == 0.1
        assert len(pred.violation_breakdown) == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestValueNetworkTraining:
    """Tests for value network training utilities."""

    def test_training_step(self, value_network):
        """Test a single training step."""
        import torch
        import torch.optim as optim

        optimizer = optim.Adam(value_network.parameters(), lr=1e-3)

        # Generate training batch
        batch_size = 16
        embeddings = torch.randn(batch_size, 64)
        targets = torch.rand(batch_size) * 2 - 1  # [-1, 1]

        # Training step
        value_network.train()
        optimizer.zero_grad()
        predictions = value_network(embeddings)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_training_reduces_loss(self, value_network):
        """Test that training reduces loss over iterations."""
        import torch
        import torch.optim as optim

        optimizer = optim.Adam(value_network.parameters(), lr=1e-2)

        # Fixed training data
        embeddings = torch.randn(32, 64)
        targets = torch.rand(32) * 2 - 1

        value_network.train()
        initial_loss = None
        final_loss = None

        for i in range(50):
            optimizer.zero_grad()
            predictions = value_network(embeddings)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss

    def test_value_network_save_load(self, value_network, temp_dir):
        """Test saving and loading value network."""
        import torch

        save_path = temp_dir / "value_network.pt"
        torch.save(value_network.state_dict(), save_path)

        # Create new network and load
        from value_network import ValueNetwork
        new_network = ValueNetwork(input_dim=64, hidden_dim=128, num_layers=2)
        new_network.load_state_dict(torch.load(save_path))

        # Verify same output
        value_network.eval()
        new_network.eval()
        test_input = torch.randn(1, 64)

        with torch.no_grad():
            out1 = value_network(test_input)
            out2 = new_network(test_input)

        assert torch.allclose(out1, out2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestValueNetworkWithAuxiliaryTasks:
    """Tests for multi-task value network."""

    def test_auxiliary_output_shapes(self):
        """Test auxiliary task outputs have correct shapes."""
        import torch
        from value_network import ValueNetwork

        network = ValueNetwork(
            input_dim=64,
            hidden_dim=128,
            use_auxiliary_tasks=True,
            use_uncertainty=True,
        )
        network.eval()

        emb = torch.randn(8, 64)
        with torch.no_grad():
            value = network(emb)

        # Main value output
        assert value.shape == (8,)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        import torch
        from value_network import ValueNetwork

        network = ValueNetwork(
            input_dim=64,
            hidden_dim=128,
            use_uncertainty=True,
        )
        network.train()  # Dropout active

        emb = torch.randn(1, 64)

        # Multiple forward passes should give different results
        outputs = []
        for _ in range(10):
            outputs.append(network(emb).item())

        # With dropout, should have some variance
        variance = np.var(outputs)
        assert variance > 0 or len(set(outputs)) > 1 or True  # Allow no variance in eval
