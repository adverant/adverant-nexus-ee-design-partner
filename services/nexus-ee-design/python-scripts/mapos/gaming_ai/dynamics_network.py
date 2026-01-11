"""
Dynamics Network - MuZero-style Learned World Model

This module implements a learned dynamics model that predicts:
1. Next latent state from (state, action)
2. Expected reward from the transition
3. Value estimate of the resulting state

This enables planning without running expensive DRC simulations,
dramatically speeding up MCTS exploration.

Architecture:
- Representation Network: PCB state -> latent state
- Dynamics Network: (latent state, action) -> (next latent state, reward)
- Prediction Networks: latent state -> (policy, value)

References:
- MuZero: https://arxiv.org/abs/1911.08265
- World Models: https://arxiv.org/abs/1803.10122
- Dreamer: https://arxiv.org/abs/1912.01603
"""

import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DynamicsPrediction:
    """Output from dynamics network prediction."""
    next_state: np.ndarray           # Predicted next latent state
    reward: float                     # Predicted immediate reward
    value: float                      # Predicted value of next state
    policy_logits: np.ndarray        # Predicted policy from next state
    uncertainty: float                # Prediction uncertainty
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldModelTrainingSample:
    """Training sample for the world model."""
    state_embedding: np.ndarray       # Current state embedding
    action_embedding: np.ndarray      # Action taken (from policy)
    next_state_embedding: np.ndarray  # Actual next state embedding
    reward: float                     # Actual reward received
    done: bool                        # Episode termination flag
    optimization_id: str
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """Residual block for dynamics network."""

        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.block = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.block(x)


    class ActionEncoder(nn.Module):
        """
        Encodes modification actions into a fixed-size embedding.

        Actions consist of:
        - Category (discrete, 9 types)
        - Parameters (continuous, varying by category)
        """

        NUM_CATEGORIES = 9
        MAX_PARAMS = 5  # Maximum parameters per category

        def __init__(self, output_dim: int = 64):
            super().__init__()
            self.output_dim = output_dim

            # Category embedding
            self.category_embedding = nn.Embedding(self.NUM_CATEGORIES, 32)

            # Parameter encoder
            self.param_encoder = nn.Sequential(
                nn.Linear(self.MAX_PARAMS, 32),
                nn.ReLU(),
            )

            # Combined projection
            self.projection = nn.Sequential(
                nn.Linear(64, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
            )

        def forward(
            self,
            category: torch.Tensor,
            parameters: torch.Tensor,
        ) -> torch.Tensor:
            """
            Encode action.

            Args:
                category: [batch] category indices
                parameters: [batch, MAX_PARAMS] parameter values (padded)

            Returns:
                action_embedding: [batch, output_dim]
            """
            cat_emb = self.category_embedding(category)  # [batch, 32]
            param_emb = self.param_encoder(parameters)    # [batch, 32]

            combined = torch.cat([cat_emb, param_emb], dim=-1)
            return self.projection(combined)

        @classmethod
        def encode_action(cls, category: int, parameters: Dict[str, float]) -> np.ndarray:
            """
            Convert action to tensor format.

            Returns:
                Tuple of (category_idx, padded_params)
            """
            params = np.zeros(cls.MAX_PARAMS, dtype=np.float32)
            for i, (key, val) in enumerate(list(parameters.items())[:cls.MAX_PARAMS]):
                params[i] = val

            return category, params


    class RepresentationNetwork(nn.Module):
        """
        Maps observation (PCB graph embedding) to latent state.

        The latent state has the same dimension as the input but is
        transformed to be more predictable by the dynamics network.
        """

        def __init__(self, input_dim: int = 256, latent_dim: int = 256, num_layers: int = 2):
            super().__init__()
            self.latent_dim = latent_dim

            layers = [nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.append(ResidualBlock(latent_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            """Map observation to latent state."""
            return self.network(observation)


    class DynamicsNetwork(nn.Module):
        """
        Predicts next latent state and reward from (state, action).

        This is the core of the learned world model, enabling
        planning in latent space without expensive simulations.
        """

        def __init__(
            self,
            latent_dim: int = 256,
            action_dim: int = 64,
            hidden_dim: int = 512,
            num_layers: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.action_dim = action_dim

            # Action encoder
            self.action_encoder = ActionEncoder(action_dim)

            # State-action combination
            combined_dim = latent_dim + action_dim

            # Dynamics backbone
            self.backbone = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                *[ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)],
            )

            # Next state prediction
            self.next_state_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

            # Reward prediction
            self.reward_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            # Uncertainty estimation
            self.uncertainty_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Softplus(),
            )

        def forward(
            self,
            latent_state: torch.Tensor,
            action_category: torch.Tensor,
            action_params: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Predict dynamics.

            Args:
                latent_state: [batch, latent_dim] current latent state
                action_category: [batch] action category indices
                action_params: [batch, MAX_PARAMS] action parameters

            Returns:
                next_state: [batch, latent_dim] predicted next state
                reward: [batch, 1] predicted reward
                uncertainty: [batch, 1] prediction uncertainty
            """
            # Encode action
            action_emb = self.action_encoder(action_category, action_params)

            # Combine state and action
            combined = torch.cat([latent_state, action_emb], dim=-1)

            # Dynamics prediction
            hidden = self.backbone(combined)

            next_state = self.next_state_head(hidden)
            reward = self.reward_head(hidden)
            uncertainty = self.uncertainty_head(hidden)

            # Residual connection for state prediction
            next_state = latent_state + next_state

            return next_state, reward, uncertainty


    class PredictionNetwork(nn.Module):
        """
        Predicts policy and value from latent state.

        Used after dynamics unrolling to evaluate imagined trajectories.
        """

        NUM_CATEGORIES = 9

        def __init__(self, latent_dim: int = 256, hidden_dim: int = 256):
            super().__init__()

            # Shared backbone
            self.backbone = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                ResidualBlock(hidden_dim),
            )

            # Policy head
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, self.NUM_CATEGORIES),
            )

            # Value head
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict policy and value from latent state.

            Returns:
                policy_logits: [batch, NUM_CATEGORIES]
                value: [batch, 1]
            """
            hidden = self.backbone(latent_state)
            policy_logits = self.policy_head(hidden)
            value = self.value_head(hidden)
            return policy_logits, value


    class WorldModel(nn.Module):
        """
        Complete MuZero-style World Model.

        Combines:
        - Representation: observation -> latent state
        - Dynamics: (latent, action) -> (next_latent, reward)
        - Prediction: latent -> (policy, value)

        Enables planning without environment interaction by imagining
        trajectories in latent space.
        """

        def __init__(
            self,
            observation_dim: int = 256,
            latent_dim: int = 256,
            action_dim: int = 64,
            hidden_dim: int = 512,
            num_dynamics_layers: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.observation_dim = observation_dim
            self.latent_dim = latent_dim

            # Component networks
            self.representation = RepresentationNetwork(observation_dim, latent_dim)
            self.dynamics = DynamicsNetwork(latent_dim, action_dim, hidden_dim,
                                            num_dynamics_layers, dropout)
            self.prediction = PredictionNetwork(latent_dim, hidden_dim)

        def initial_inference(
            self,
            observation: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Initial inference from observation.

            Args:
                observation: [batch, observation_dim] PCB graph embedding

            Returns:
                latent_state: [batch, latent_dim]
                policy_logits: [batch, NUM_CATEGORIES]
                value: [batch, 1]
            """
            latent_state = self.representation(observation)
            policy_logits, value = self.prediction(latent_state)
            return latent_state, policy_logits, value

        def recurrent_inference(
            self,
            latent_state: torch.Tensor,
            action_category: torch.Tensor,
            action_params: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Recurrent inference in latent space.

            Args:
                latent_state: Current latent state
                action_category: Action category indices
                action_params: Action parameters

            Returns:
                next_latent: Next latent state
                reward: Predicted reward
                policy_logits: Policy from next state
                value: Value of next state
                uncertainty: Prediction uncertainty
            """
            next_latent, reward, uncertainty = self.dynamics(
                latent_state, action_category, action_params
            )
            policy_logits, value = self.prediction(next_latent)

            return next_latent, reward, policy_logits, value, uncertainty

        def imagine_trajectory(
            self,
            observation: torch.Tensor,
            actions: List[Tuple[int, Dict[str, float]]],
        ) -> List[DynamicsPrediction]:
            """
            Imagine a trajectory by unrolling dynamics.

            Args:
                observation: Initial observation
                actions: List of (category, parameters) tuples

            Returns:
                List of predictions for each step
            """
            self.eval()
            predictions = []

            with torch.no_grad():
                # Initial inference
                latent, policy_logits, value = self.initial_inference(observation)

                for category_idx, params in actions:
                    # Encode action
                    category = torch.tensor([category_idx], dtype=torch.long)
                    action_params = torch.zeros(1, ActionEncoder.MAX_PARAMS)
                    for i, (k, v) in enumerate(list(params.items())[:ActionEncoder.MAX_PARAMS]):
                        action_params[0, i] = v

                    # Recurrent inference
                    latent, reward, policy_logits, value, uncertainty = self.recurrent_inference(
                        latent, category, action_params
                    )

                    predictions.append(DynamicsPrediction(
                        next_state=latent.cpu().numpy().flatten(),
                        reward=reward.item(),
                        value=value.item(),
                        policy_logits=policy_logits.cpu().numpy().flatten(),
                        uncertainty=uncertainty.item(),
                    ))

            return predictions

        def compute_loss(
            self,
            observations: torch.Tensor,
            action_categories: torch.Tensor,
            action_params: torch.Tensor,
            target_next_latents: torch.Tensor,
            target_rewards: torch.Tensor,
            target_policies: torch.Tensor,
            target_values: torch.Tensor,
            dynamics_weight: float = 1.0,
            reward_weight: float = 1.0,
            policy_weight: float = 1.0,
            value_weight: float = 0.5,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute world model training loss.

            Args:
                observations: Initial observations
                action_categories: Actions taken
                action_params: Action parameters
                target_next_latents: True next state embeddings
                target_rewards: True rewards
                target_policies: MCTS-improved policy targets
                target_values: True/estimated values

            Returns:
                total_loss: Combined loss
                loss_dict: Individual loss components
            """
            # Initial inference
            latent, pred_policy, pred_value = self.initial_inference(observations)

            # Recurrent inference
            next_latent, pred_reward, next_policy, next_value, uncertainty = self.recurrent_inference(
                latent, action_categories, action_params
            )

            # Dynamics loss: predicted next state should match actual
            # We use the representation of the actual next observation as target
            with torch.no_grad():
                target_latent = self.representation(target_next_latents)

            dynamics_loss = F.mse_loss(next_latent, target_latent)

            # Reward loss
            reward_loss = F.mse_loss(pred_reward.squeeze(-1), target_rewards)

            # Policy loss (cross-entropy with MCTS targets)
            policy_loss = F.cross_entropy(next_policy, target_policies)

            # Value loss
            value_loss = F.mse_loss(next_value.squeeze(-1), target_values)

            # Total loss
            total_loss = (
                dynamics_weight * dynamics_loss +
                reward_weight * reward_loss +
                policy_weight * policy_loss +
                value_weight * value_loss
            )

            return total_loss, {
                'dynamics_loss': dynamics_loss.item(),
                'reward_loss': reward_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item(),
            }

        def save(self, path: Union[str, Path]) -> None:
            """Save model weights and config."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'state_dict': self.state_dict(),
                'config': {
                    'observation_dim': self.observation_dim,
                    'latent_dim': self.latent_dim,
                }
            }, path)

        @classmethod
        def load(cls, path: Union[str, Path]) -> 'WorldModel':
            """Load model from file."""
            checkpoint = torch.load(path, map_location='cpu')
            config = checkpoint['config']

            model = cls(
                observation_dim=config['observation_dim'],
                latent_dim=config['latent_dim'],
            )
            model.load_state_dict(checkpoint['state_dict'])
            return model


    class WorldModelTrainer:
        """Trainer for the World Model."""

        def __init__(
            self,
            model: WorldModel,
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

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )

            self.training_history: List[Dict[str, float]] = []

        def train_epoch(
            self,
            samples: List[WorldModelTrainingSample],
        ) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            np.random.shuffle(samples)

            total_losses = {}
            num_batches = 0

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i + self.batch_size]

                # Prepare tensors
                observations = torch.tensor(
                    np.stack([s.state_embedding for s in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )

                # Parse actions
                categories = []
                params = []
                for s in batch:
                    cat, p = ActionEncoder.encode_action(
                        int(s.action_embedding[0]) if len(s.action_embedding) > 0 else 0,
                        {}
                    )
                    categories.append(cat)
                    params.append(p)

                action_categories = torch.tensor(categories, dtype=torch.long, device=self.device)
                action_params = torch.tensor(np.stack(params), dtype=torch.float32, device=self.device)

                next_observations = torch.tensor(
                    np.stack([s.next_state_embedding for s in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )

                rewards = torch.tensor(
                    [s.reward for s in batch],
                    dtype=torch.float32,
                    device=self.device,
                )

                # Use uniform policy and reward-based value as proxy targets
                target_policies = torch.ones(len(batch), 9, device=self.device) / 9
                target_values = torch.sigmoid(rewards)  # Map rewards to [0, 1]

                # Forward and loss
                self.optimizer.zero_grad()
                loss, loss_dict = self.model.compute_loss(
                    observations, action_categories, action_params,
                    next_observations, rewards, target_policies, target_values
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for key, val in loss_dict.items():
                    total_losses[key] = total_losses.get(key, 0.0) + val
                num_batches += 1

            self.scheduler.step()

            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
            self.training_history.append(avg_losses)
            return avg_losses


else:
    # Fallback without PyTorch
    class WorldModel:
        """Fallback World Model (requires PyTorch)."""

        def __init__(self, **kwargs):
            import warnings
            warnings.warn("PyTorch not available. WorldModel will use identity dynamics.")

        def imagine_trajectory(self, observation: np.ndarray,
                               actions: List) -> List[DynamicsPrediction]:
            """Identity dynamics - next state equals current state."""
            predictions = []
            for _ in actions:
                predictions.append(DynamicsPrediction(
                    next_state=observation.copy(),
                    reward=0.0,
                    value=0.5,
                    policy_logits=np.ones(9) / 9,
                    uncertainty=1.0,
                    metadata={'fallback': True}
                ))
            return predictions


    class DynamicsNetwork:
        pass


if __name__ == '__main__':
    print("World Model Test")
    print("=" * 60)

    if TORCH_AVAILABLE:
        # Create model
        model = WorldModel(
            observation_dim=256,
            latent_dim=256,
            action_dim=64,
            hidden_dim=512,
            num_dynamics_layers=3,
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Test initial inference
        batch_size = 4
        observation = torch.randn(batch_size, 256)

        latent, policy, value = model.initial_inference(observation)
        print(f"\nInitial inference:")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Value shape: {value.shape}")

        # Test recurrent inference
        category = torch.randint(0, 9, (batch_size,))
        params = torch.randn(batch_size, 5)

        next_latent, reward, next_policy, next_value, uncertainty = model.recurrent_inference(
            latent, category, params
        )
        print(f"\nRecurrent inference:")
        print(f"  Next latent shape: {next_latent.shape}")
        print(f"  Reward shape: {reward.shape}")
        print(f"  Uncertainty shape: {uncertainty.shape}")

        # Test trajectory imagination
        single_obs = torch.randn(1, 256)
        actions = [
            (0, {'dx': 1.0, 'dy': 0.5}),
            (2, {'diameter': 0.8}),
            (4, {'clearance': 0.3}),
        ]

        predictions = model.imagine_trajectory(single_obs, actions)
        print(f"\nTrajectory imagination ({len(actions)} steps):")
        for i, pred in enumerate(predictions):
            print(f"  Step {i+1}: reward={pred.reward:.4f}, value={pred.value:.4f}, "
                  f"uncertainty={pred.uncertainty:.4f}")

        # Test loss computation
        target_next = torch.randn(batch_size, 256)
        target_rewards = torch.randn(batch_size)
        target_policies = F.softmax(torch.randn(batch_size, 9), dim=-1)
        target_values = torch.rand(batch_size)

        loss, loss_dict = model.compute_loss(
            observation, category, params,
            target_next, target_rewards, target_policies, target_values
        )
        print(f"\nLoss computation:")
        for key, val in loss_dict.items():
            print(f"  {key}: {val:.4f}")

    else:
        print("PyTorch not available, testing fallback")
        model = WorldModel()
        obs = np.random.randn(256)
        actions = [(0, {}), (1, {})]
        predictions = model.imagine_trajectory(obs, actions)
        print(f"Predictions: {len(predictions)} steps")
