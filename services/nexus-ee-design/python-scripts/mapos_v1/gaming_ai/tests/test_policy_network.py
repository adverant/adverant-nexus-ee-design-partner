"""
Tests for Policy Network module.

Tests the modification selection network.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import MockPCBState, TORCH_AVAILABLE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_initialization(self, policy_network):
        """Test policy network initializes correctly."""
        import torch
        assert isinstance(policy_network, torch.nn.Module)
        assert hasattr(policy_network, 'category_head')

    def test_forward_single(self, policy_network, random_embedding, random_drc_context):
        """Test forward pass with single input."""
        import torch

        emb = torch.tensor(random_embedding[:64]).unsqueeze(0)
        drc = torch.tensor(random_drc_context).unsqueeze(0)

        output = policy_network(emb, drc)

        assert output is not None

    def test_get_category_distribution(self, policy_network):
        """Test category distribution generation."""
        import torch

        emb = torch.randn(8, 64)
        drc = torch.randn(8, 12)

        probs, logits = policy_network.get_category_distribution(emb, drc)

        # Should be valid probability distribution
        assert probs.shape == (8, 9)  # 9 modification categories
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_sample_action(self, policy_network):
        """Test action sampling."""
        import torch

        emb = torch.randn(1, 64)
        drc = torch.randn(1, 12)

        category, params = policy_network.sample_action(emb, drc)

        assert 0 <= category.item() < 9
        assert params.shape[-1] == 5  # 5 action parameters

    def test_gradient_flow(self, policy_network):
        """Test gradients flow correctly."""
        import torch

        emb = torch.randn(8, 64, requires_grad=True)
        drc = torch.randn(8, 12)

        probs, _ = policy_network.get_category_distribution(emb, drc)
        loss = -probs.log().mean()
        loss.backward()

        for param in policy_network.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_deterministic_eval(self, policy_network):
        """Test deterministic output in eval mode with greedy selection."""
        import torch
        policy_network.eval()

        emb = torch.randn(1, 64)
        drc = torch.randn(1, 12)

        with torch.no_grad():
            probs1, _ = policy_network.get_category_distribution(emb, drc)
            probs2, _ = policy_network.get_category_distribution(emb, drc)

        assert torch.allclose(probs1, probs2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestModificationCategory:
    """Tests for ModificationCategory enumeration."""

    def test_all_categories_exist(self):
        """Verify all expected categories exist."""
        from policy_network import ModificationCategory

        expected = [
            'COMPONENT_POSITION', 'TRACE_ADJUSTMENT', 'VIA_MODIFICATION',
            'ZONE_ADJUSTMENT', 'CLEARANCE_ADJUSTMENT', 'VIA_PLACEMENT',
            'VIA_REMOVAL', 'SOLDER_MASK_ADJUSTMENT', 'SILKSCREEN_ADJUSTMENT'
        ]
        for cat in expected:
            assert hasattr(ModificationCategory, cat)

    def test_category_count(self):
        """Verify there are exactly 9 categories."""
        from policy_network import ModificationCategory
        assert len(ModificationCategory) == 9


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDRCContextEncoder:
    """Tests for DRC context encoder."""

    def test_extract_features_with_drc(self, mock_pcb_state):
        """Test feature extraction from DRC result."""
        from policy_network import DRCContextEncoder

        drc = mock_pcb_state.run_drc()
        features = DRCContextEncoder.extract_features(drc)

        assert len(features) == 12
        assert all(np.isfinite(features))

    def test_extract_features_none(self):
        """Test feature extraction with None DRC."""
        from policy_network import DRCContextEncoder

        features = DRCContextEncoder.extract_features(None)

        assert len(features) == 12
        assert all(f == 0.0 for f in features)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestModificationHead:
    """Tests for per-category modification heads."""

    def test_head_initialization(self):
        """Test modification head initialization."""
        from policy_network import ModificationHead

        head = ModificationHead(
            input_dim=128,
            num_params=5,
            param_specs={
                'x': {'type': 'continuous', 'min': -10, 'max': 10},
                'y': {'type': 'continuous', 'min': -10, 'max': 10},
            }
        )

        assert head is not None

    def test_head_forward(self):
        """Test modification head forward pass."""
        import torch
        from policy_network import ModificationHead

        head = ModificationHead(input_dim=128, num_params=5, param_specs={})
        features = torch.randn(8, 128)

        output = head(features)
        assert output.shape == (8, 5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolicyNetworkTraining:
    """Tests for policy network training."""

    def test_training_step(self, policy_network):
        """Test single training step."""
        import torch
        import torch.optim as optim

        optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)

        batch_size = 16
        emb = torch.randn(batch_size, 64)
        drc = torch.randn(batch_size, 12)
        target_categories = torch.randint(0, 9, (batch_size,))

        policy_network.train()
        optimizer.zero_grad()

        probs, _ = policy_network.get_category_distribution(emb, drc)
        loss = torch.nn.functional.cross_entropy(probs.log(), target_categories)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_training_improves(self, policy_network):
        """Test that training improves policy."""
        import torch
        import torch.optim as optim

        optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)

        # Fixed data
        emb = torch.randn(32, 64)
        drc = torch.randn(32, 12)
        targets = torch.randint(0, 9, (32,))

        policy_network.train()
        initial_loss = None
        final_loss = None

        for i in range(50):
            optimizer.zero_grad()
            probs, _ = policy_network.get_category_distribution(emb, drc)
            loss = torch.nn.functional.cross_entropy(probs.log() + 1e-8, targets)
            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss

    def test_policy_save_load(self, policy_network, temp_dir):
        """Test saving and loading policy network."""
        import torch
        from policy_network import PolicyNetwork

        save_path = temp_dir / "policy_network.pt"
        torch.save(policy_network.state_dict(), save_path)

        new_network = PolicyNetwork(input_dim=64, hidden_dim=128)
        new_network.load_state_dict(torch.load(save_path))

        policy_network.eval()
        new_network.eval()

        test_emb = torch.randn(1, 64)
        test_drc = torch.randn(1, 12)

        with torch.no_grad():
            probs1, _ = policy_network.get_category_distribution(test_emb, test_drc)
            probs2, _ = new_network.get_category_distribution(test_emb, test_drc)

        assert torch.allclose(probs1, probs2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTemperatureScaling:
    """Tests for temperature-based exploration."""

    def test_high_temperature_uniform(self):
        """Test high temperature produces more uniform distribution."""
        import torch
        from policy_network import PolicyNetwork

        network = PolicyNetwork(input_dim=64, hidden_dim=128, temperature=10.0)
        network.eval()

        emb = torch.randn(100, 64)
        drc = torch.randn(100, 12)

        with torch.no_grad():
            probs, _ = network.get_category_distribution(emb, drc)

        # High temperature should give more uniform distribution
        entropy = -(probs * probs.log()).sum(dim=1).mean()
        assert entropy > 1.0  # Higher entropy with high temperature

    def test_low_temperature_peaked(self):
        """Test low temperature produces more peaked distribution."""
        import torch
        from policy_network import PolicyNetwork

        network = PolicyNetwork(input_dim=64, hidden_dim=128, temperature=0.1)
        network.eval()

        emb = torch.randn(100, 64)
        drc = torch.randn(100, 12)

        with torch.no_grad():
            probs, _ = network.get_category_distribution(emb, drc)

        # Low temperature should give more peaked distribution
        max_probs = probs.max(dim=1).values
        assert max_probs.mean() > 0.5  # Higher max probability
