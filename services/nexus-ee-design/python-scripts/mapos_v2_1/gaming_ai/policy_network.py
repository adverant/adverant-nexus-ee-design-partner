"""
Policy Network - Learned Modification Selection

This module implements an AlphaZero-style policy network that predicts
which modifications are most likely to improve a PCB design.

Architecture:
- Input: PCB graph embedding + DRC context
- Policy Head: Probability distribution over modification types
- Parameter Heads: Continuous parameters for each modification type
- Action Composition: Combines type selection with parameters

The policy network guides MCTS expansion by prioritizing promising
modifications, dramatically reducing search space exploration.

References:
- AlphaZero Policy Network: https://arxiv.org/abs/1712.01815
- Continuous Control: https://arxiv.org/abs/1509.02971
"""

import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum, IntEnum
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModificationCategory(IntEnum):
    """High-level modification categories for policy output."""
    COMPONENT_POSITION = 0    # Move or rotate component
    TRACE_ADJUSTMENT = 1      # Adjust trace width
    VIA_MODIFICATION = 2      # Add, remove, or resize via
    CLEARANCE_CHANGE = 3      # Adjust clearance parameters
    ZONE_MODIFICATION = 4     # Adjust zone parameters
    SILKSCREEN_FIX = 5        # Fix silkscreen issues
    SOLDER_MASK_FIX = 6       # Fix solder mask issues
    THERMAL_VIA = 7           # Add thermal vias
    NO_ACTION = 8             # Do nothing (terminal)


@dataclass
class PolicyOutput:
    """Output from the policy network."""
    category_probs: np.ndarray              # [num_categories] probabilities
    selected_category: int                   # Sampled or argmax category
    parameters: Dict[str, float]            # Category-specific parameters
    log_prob: float                          # Log probability of this action
    entropy: float                           # Policy entropy (for exploration)
    confidence: float                        # Overall confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_modification_dict(self) -> Dict[str, Any]:
        """Convert to format compatible with PCBModification."""
        category_name = ModificationCategory(self.selected_category).name

        return {
            'category': category_name,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'log_prob': self.log_prob,
        }


@dataclass
class PolicyTrainingSample:
    """Training sample for policy network (from MCTS or expert)."""
    state_embedding: np.ndarray       # PCB state embedding
    drc_context: np.ndarray           # DRC violation features
    target_probs: np.ndarray          # Target category distribution (from MCTS)
    action_taken: int                 # Which action was taken
    parameters_taken: Dict[str, float] # Parameters of action taken
    reward: float                     # Resulting improvement
    optimization_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


if TORCH_AVAILABLE:

    class DRCContextEncoder(nn.Module):
        """
        Encodes DRC violation context into a fixed-size representation.

        Takes violation counts and produces an embedding that guides
        the policy toward fixing the most impactful violations.
        """

        # DRC violation types with their typical impact
        VIOLATION_FEATURES = [
            'clearance_count',
            'track_width_count',
            'via_dangling_count',
            'track_dangling_count',
            'shorting_count',
            'silk_over_copper_count',
            'solder_mask_bridge_count',
            'courtyard_overlap_count',
            'unconnected_count',
            'total_violations',
            'error_count',
            'warning_count',
        ]

        def __init__(self, output_dim: int = 64):
            super().__init__()
            self.output_dim = output_dim
            self.input_dim = len(self.VIOLATION_FEATURES)

            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )

        def forward(self, drc_features: torch.Tensor) -> torch.Tensor:
            """Encode DRC context [batch, input_dim] -> [batch, output_dim]."""
            return self.encoder(drc_features)

        @staticmethod
        def extract_features(drc_result: Any) -> np.ndarray:
            """Extract feature vector from DRCResult object."""
            features = np.zeros(len(DRCContextEncoder.VIOLATION_FEATURES), dtype=np.float32)

            if drc_result is None:
                return features

            # Extract from violations_by_type
            vbt = getattr(drc_result, 'violations_by_type', {})

            features[0] = vbt.get('clearance', 0) / 100.0  # Normalize
            features[1] = vbt.get('track_width', 0) / 100.0
            features[2] = vbt.get('via_dangling', 0) / 50.0
            features[3] = vbt.get('track_dangling', 0) / 50.0
            features[4] = vbt.get('shorting_items', 0) / 20.0
            features[5] = vbt.get('silk_over_copper', 0) / 100.0
            features[6] = vbt.get('solder_mask_bridge', 0) / 50.0
            features[7] = vbt.get('courtyards_overlap', 0) / 20.0
            features[8] = getattr(drc_result, 'unconnected', 0) / 50.0
            features[9] = getattr(drc_result, 'total_violations', 0) / 500.0
            features[10] = getattr(drc_result, 'errors', 0) / 200.0
            features[11] = getattr(drc_result, 'warnings', 0) / 300.0

            return features


    class ModificationHead(nn.Module):
        """
        Parameter prediction head for a specific modification category.

        Each category has different parameters:
        - COMPONENT_POSITION: dx, dy, rotation
        - TRACE_ADJUSTMENT: width_delta
        - VIA_MODIFICATION: diameter, drill, action (add/remove/resize)
        - CLEARANCE_CHANGE: clearance_value
        - etc.
        """

        # Parameter definitions per category
        PARAMETER_SPECS = {
            ModificationCategory.COMPONENT_POSITION: [
                ('dx', -10.0, 10.0),       # Delta X in mm
                ('dy', -10.0, 10.0),       # Delta Y in mm
                ('rotation', 0.0, 360.0),  # Rotation in degrees
            ],
            ModificationCategory.TRACE_ADJUSTMENT: [
                ('width', 0.1, 5.0),        # New width in mm
                ('layer_idx', 0.0, 10.0),   # Layer index
            ],
            ModificationCategory.VIA_MODIFICATION: [
                ('diameter', 0.4, 1.5),     # Via diameter in mm
                ('drill', 0.2, 0.8),        # Drill size in mm
                ('action', 0.0, 2.0),       # 0=add, 1=remove, 2=resize
            ],
            ModificationCategory.CLEARANCE_CHANGE: [
                ('clearance', 0.1, 1.0),    # Clearance value in mm
                ('type_idx', 0.0, 3.0),     # 0=signal, 1=power, 2=hv
            ],
            ModificationCategory.ZONE_MODIFICATION: [
                ('clearance', 0.1, 0.8),    # Zone clearance
                ('thermal_gap', 0.2, 1.0),  # Thermal gap
            ],
            ModificationCategory.SILKSCREEN_FIX: [
                ('offset_x', -3.0, 3.0),    # X offset
                ('offset_y', -3.0, 3.0),    # Y offset
                ('scale', 0.5, 1.5),        # Scale factor
            ],
            ModificationCategory.SOLDER_MASK_FIX: [
                ('expansion', 0.0, 0.15),   # Mask expansion
                ('bridge_gap', 0.0, 0.1),   # Bridge gap adjustment
            ],
            ModificationCategory.THERMAL_VIA: [
                ('count', 1.0, 20.0),       # Number of vias to add
                ('spacing', 0.5, 3.0),      # Via spacing
                ('diameter', 0.4, 1.2),     # Via diameter
            ],
            ModificationCategory.NO_ACTION: [],
        }

        def __init__(self, category: ModificationCategory, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.category = category
            self.param_specs = self.PARAMETER_SPECS[category]
            self.num_params = len(self.param_specs)

            if self.num_params > 0:
                # Mean prediction
                self.mean_head = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.num_params),
                )

                # Log standard deviation (for exploration)
                self.log_std_head = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.num_params),
                )

                # Initialize to reasonable ranges
                nn.init.zeros_(self.mean_head[-1].bias)
                nn.init.constant_(self.log_std_head[-1].bias, -1.0)  # Small std

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict parameter means and log standard deviations.

            Returns:
                means: [batch, num_params] parameter means
                log_stds: [batch, num_params] log standard deviations
            """
            if self.num_params == 0:
                batch_size = x.size(0)
                device = x.device
                return torch.zeros(batch_size, 0, device=device), torch.zeros(batch_size, 0, device=device)

            means = self.mean_head(x)
            log_stds = self.log_std_head(x).clamp(-5, 2)  # Prevent extreme values

            return means, log_stds

        def sample(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Sample parameters.

            Args:
                x: Input features
                deterministic: If True, return means instead of sampling

            Returns:
                params: [batch, num_params] sampled parameters
                log_probs: [batch] log probabilities
            """
            means, log_stds = self(x)

            if self.num_params == 0:
                batch_size = x.size(0)
                device = x.device
                return torch.zeros(batch_size, 0, device=device), torch.zeros(batch_size, device=device)

            if deterministic:
                params = means
                log_probs = torch.zeros(x.size(0), device=x.device)
            else:
                stds = log_stds.exp()
                dist = Normal(means, stds)
                params = dist.rsample()  # Reparameterized sampling
                log_probs = dist.log_prob(params).sum(dim=-1)

            # Clamp to valid ranges
            for i, (name, min_val, max_val) in enumerate(self.param_specs):
                params[:, i] = params[:, i].clamp(min_val, max_val)

            return params, log_probs

        def get_param_dict(self, params: torch.Tensor) -> Dict[str, float]:
            """Convert parameter tensor to dictionary."""
            if self.num_params == 0:
                return {}

            params_np = params.detach().cpu().numpy().flatten()
            return {
                name: float(params_np[i])
                for i, (name, _, _) in enumerate(self.param_specs)
            }


    class PolicyNetwork(nn.Module):
        """
        Policy Network for PCB Modification Selection.

        Predicts:
        1. Category distribution: Which type of modification to apply
        2. Parameters: Category-specific continuous parameters

        This guides MCTS by prioritizing promising modifications and
        generates actions for reinforcement learning.
        """

        NUM_CATEGORIES = len(ModificationCategory)

        def __init__(
            self,
            input_dim: int = 256,
            drc_context_dim: int = 64,
            hidden_dim: int = 512,
            num_layers: int = 3,
            dropout: float = 0.1,
            temperature: float = 1.0,
        ):
            """
            Initialize Policy Network.

            Args:
                input_dim: Dimension of PCB state embedding
                drc_context_dim: Dimension of DRC context encoding
                hidden_dim: Hidden layer dimension
                num_layers: Number of hidden layers
                dropout: Dropout rate
                temperature: Softmax temperature for exploration
            """
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.temperature = temperature

            # DRC context encoder
            self.drc_encoder = DRCContextEncoder(drc_context_dim)

            # Combined input dimension
            combined_dim = input_dim + drc_context_dim

            # Shared backbone
            layers = []
            in_dim = combined_dim
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)

            # Category head (policy over modification types)
            self.category_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, self.NUM_CATEGORIES),
            )

            # Parameter heads (one per category)
            self.param_heads = nn.ModuleDict({
                category.name: ModificationHead(category, hidden_dim)
                for category in ModificationCategory
            })

            # Confidence head
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            state_embedding: torch.Tensor,
            drc_features: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass returning category logits and features.

            Args:
                state_embedding: [batch, input_dim] PCB state embedding
                drc_features: [batch, drc_input_dim] DRC violation features

            Returns:
                category_logits: [batch, num_categories]
                hidden_features: [batch, hidden_dim] for parameter heads
            """
            # Encode DRC context
            drc_encoded = self.drc_encoder(drc_features)

            # Combine inputs
            combined = torch.cat([state_embedding, drc_encoded], dim=-1)

            # Backbone
            hidden = self.backbone(combined)

            # Category logits
            category_logits = self.category_head(hidden)

            return category_logits, hidden

        def get_category_distribution(
            self,
            state_embedding: torch.Tensor,
            drc_features: torch.Tensor,
        ) -> Tuple[torch.Tensor, float]:
            """
            Get probability distribution over categories.

            Returns:
                probs: [batch, num_categories] probabilities
                entropy: Scalar entropy value
            """
            logits, _ = self(state_embedding, drc_features)

            # Apply temperature
            probs = F.softmax(logits / self.temperature, dim=-1)

            # Compute entropy
            entropy = Categorical(probs).entropy().mean()

            return probs, entropy.item()

        def sample_action(
            self,
            state_embedding: torch.Tensor,
            drc_features: torch.Tensor,
            deterministic: bool = False,
        ) -> PolicyOutput:
            """
            Sample a complete action (category + parameters).

            Args:
                state_embedding: PCB state embedding
                drc_features: DRC context features
                deterministic: If True, take argmax instead of sampling

            Returns:
                PolicyOutput with sampled action
            """
            # Get category distribution
            logits, hidden = self(state_embedding, drc_features)
            probs = F.softmax(logits / self.temperature, dim=-1)

            # Sample or argmax category
            if deterministic:
                category = probs.argmax(dim=-1)
                category_log_prob = torch.log(probs.gather(1, category.unsqueeze(-1)) + 1e-8).squeeze()
            else:
                dist = Categorical(probs)
                category = dist.sample()
                category_log_prob = dist.log_prob(category)

            # Get parameters for selected category
            category_idx = category.item() if category.dim() == 0 else category[0].item()
            category_enum = ModificationCategory(category_idx)
            param_head = self.param_heads[category_enum.name]

            params, param_log_prob = param_head.sample(hidden, deterministic)
            param_dict = param_head.get_param_dict(params[0] if params.dim() > 1 else params)

            # Total log probability
            total_log_prob = category_log_prob + param_log_prob

            # Confidence
            confidence = self.confidence_head(hidden).item()

            # Entropy
            entropy = Categorical(probs).entropy().item()

            return PolicyOutput(
                category_probs=probs.detach().cpu().numpy().flatten(),
                selected_category=category_idx,
                parameters=param_dict,
                log_prob=total_log_prob.item(),
                entropy=entropy,
                confidence=confidence,
                metadata={
                    'temperature': self.temperature,
                    'deterministic': deterministic,
                }
            )

        def compute_loss(
            self,
            state_embedding: torch.Tensor,
            drc_features: torch.Tensor,
            target_probs: torch.Tensor,
            actions_taken: torch.Tensor,
            rewards: torch.Tensor,
            entropy_coef: float = 0.01,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute policy loss for training.

            Uses cross-entropy for category prediction and
            policy gradient for reward optimization.

            Args:
                state_embedding: Batch of state embeddings
                drc_features: Batch of DRC features
                target_probs: Target category distribution (from MCTS)
                actions_taken: Categories that were taken
                rewards: Resulting improvements
                entropy_coef: Entropy regularization coefficient

            Returns:
                loss: Total loss
                loss_dict: Dictionary of loss components
            """
            logits, hidden = self(state_embedding, drc_features)
            probs = F.softmax(logits / self.temperature, dim=-1)

            # Cross-entropy loss against MCTS-improved targets
            ce_loss = F.cross_entropy(logits, target_probs)

            # Policy gradient loss
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions_taken.unsqueeze(-1)).squeeze(-1)
            pg_loss = -(selected_log_probs * rewards).mean()

            # Entropy bonus (encourages exploration)
            entropy = Categorical(probs).entropy().mean()
            entropy_loss = -entropy_coef * entropy

            total_loss = ce_loss + pg_loss + entropy_loss

            return total_loss, {
                'ce_loss': ce_loss.item(),
                'pg_loss': pg_loss.item(),
                'entropy': entropy.item(),
                'total_loss': total_loss.item(),
            }

        def save(self, path: Union[str, Path]) -> None:
            """Save model weights and config."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'state_dict': self.state_dict(),
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'temperature': self.temperature,
                }
            }, path)

        @classmethod
        def load(cls, path: Union[str, Path]) -> 'PolicyNetwork':
            """Load model from file."""
            path = Path(path)
            checkpoint = torch.load(path, map_location='cpu')

            config = checkpoint['config']
            model = cls(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                temperature=config['temperature'],
            )
            model.load_state_dict(checkpoint['state_dict'])
            return model


    class PolicyNetworkTrainer:
        """Trainer for the Policy Network."""

        def __init__(
            self,
            model: PolicyNetwork,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            batch_size: int = 32,
            device: str = 'cpu',
        ):
            self.model = model.to(device)
            self.device = device
            self.batch_size = batch_size

            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            self.training_history: List[Dict[str, float]] = []

        def train_epoch(
            self,
            samples: List[PolicyTrainingSample],
        ) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            np.random.shuffle(samples)

            total_losses = {}
            num_batches = 0

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i + self.batch_size]

                # Prepare tensors
                state_embeddings = torch.tensor(
                    np.stack([s.state_embedding for s in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                drc_features = torch.tensor(
                    np.stack([s.drc_context for s in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                target_probs = torch.tensor(
                    np.stack([s.target_probs for s in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.tensor(
                    [s.action_taken for s in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                rewards = torch.tensor(
                    [s.reward for s in batch],
                    dtype=torch.float32,
                    device=self.device,
                )

                # Forward and loss
                self.optimizer.zero_grad()
                loss, loss_dict = self.model.compute_loss(
                    state_embeddings, drc_features, target_probs, actions, rewards
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for key, val in loss_dict.items():
                    total_losses[key] = total_losses.get(key, 0.0) + val
                num_batches += 1

            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
            self.training_history.append(avg_losses)
            return avg_losses


else:
    # Fallback without PyTorch
    class PolicyNetwork:
        """Fallback Policy Network (requires PyTorch)."""

        NUM_CATEGORIES = len(ModificationCategory)

        def __init__(self, **kwargs):
            import warnings
            warnings.warn("PyTorch not available. PolicyNetwork will use random sampling.")

        def sample_action(self, state_embedding: np.ndarray, drc_features: np.ndarray,
                          deterministic: bool = False) -> PolicyOutput:
            """Random action sampling."""
            probs = np.ones(self.NUM_CATEGORIES) / self.NUM_CATEGORIES
            category = np.random.choice(self.NUM_CATEGORIES)

            return PolicyOutput(
                category_probs=probs,
                selected_category=category,
                parameters={},
                log_prob=np.log(1.0 / self.NUM_CATEGORIES),
                entropy=np.log(self.NUM_CATEGORIES),
                confidence=0.5,
                metadata={'fallback': True}
            )

    class DRCContextEncoder:
        VIOLATION_FEATURES = ['total']

        @staticmethod
        def extract_features(drc_result: Any) -> np.ndarray:
            total = getattr(drc_result, 'total_violations', 0) if drc_result else 0
            return np.array([total / 500.0], dtype=np.float32)


if __name__ == '__main__':
    print("Policy Network Test")
    print("=" * 60)

    if TORCH_AVAILABLE:
        # Create model
        model = PolicyNetwork(
            input_dim=256,
            drc_context_dim=64,
            hidden_dim=512,
            num_layers=3,
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Test forward pass
        batch_size = 4
        state_emb = torch.randn(batch_size, 256)
        drc_feat = torch.randn(batch_size, 12)

        logits, hidden = model(state_emb, drc_feat)
        print(f"\nForward pass:")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Hidden shape: {hidden.shape}")

        # Test action sampling
        print("\nAction sampling:")
        for i in range(3):
            output = model.sample_action(state_emb[:1], drc_feat[:1])
            cat_name = ModificationCategory(output.selected_category).name
            print(f"  Sample {i+1}: {cat_name}, params={output.parameters}, "
                  f"log_prob={output.log_prob:.3f}, entropy={output.entropy:.3f}")

        # Test deterministic
        det_output = model.sample_action(state_emb[:1], drc_feat[:1], deterministic=True)
        print(f"  Deterministic: {ModificationCategory(det_output.selected_category).name}")

        # Test loss computation
        target_probs = F.softmax(torch.randn(batch_size, model.NUM_CATEGORIES), dim=-1)
        actions = torch.randint(0, model.NUM_CATEGORIES, (batch_size,))
        rewards = torch.randn(batch_size)

        loss, loss_dict = model.compute_loss(state_emb, drc_feat, target_probs, actions, rewards)
        print(f"\nLoss computation:")
        for key, val in loss_dict.items():
            print(f"  {key}: {val:.4f}")

    else:
        print("PyTorch not available, testing fallback")
        model = PolicyNetwork()
        output = model.sample_action(np.zeros(256), np.zeros(12))
        print(f"Sampled action: {output}")
