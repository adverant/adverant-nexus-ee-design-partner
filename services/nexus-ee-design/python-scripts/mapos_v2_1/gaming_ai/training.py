"""
Training Pipeline - Self-Play and Experience Collection

This module implements the training pipeline for gaming AI components:
1. Experience Buffer: Stores and samples optimization experiences
2. Self-Play Generator: Generates training data through optimization runs
3. Unified Trainer: Trains all networks jointly

Training Strategy:
- Collect experiences from MAPOS optimization runs
- Store (state, action, reward, next_state) transitions
- Train networks on batched experience
- Use value targets from MCTS for temporal difference learning

References:
- AlphaZero Training: https://arxiv.org/abs/1712.01815
- Experience Replay: https://arxiv.org/abs/1312.5602
"""

import asyncio
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
from collections import deque
import pickle

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class Experience:
    """Single experience from optimization."""
    state_embedding: np.ndarray           # PCB graph embedding
    drc_context: np.ndarray               # DRC features
    action_category: int                   # Modification category
    action_params: np.ndarray             # Action parameters
    reward: float                          # Immediate reward
    next_state_embedding: np.ndarray      # Next state embedding
    done: bool                             # Episode termination
    value_target: float                    # Target value from search
    policy_target: np.ndarray             # Target policy from MCTS
    optimization_id: str                   # Source optimization run
    step: int                              # Step within run
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_embedding': self.state_embedding.tolist(),
            'drc_context': self.drc_context.tolist(),
            'action_category': self.action_category,
            'action_params': self.action_params.tolist(),
            'reward': self.reward,
            'next_state_embedding': self.next_state_embedding.tolist(),
            'done': self.done,
            'value_target': self.value_target,
            'policy_target': self.policy_target.tolist(),
            'optimization_id': self.optimization_id,
            'step': self.step,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        return cls(
            state_embedding=np.array(data['state_embedding'], dtype=np.float32),
            drc_context=np.array(data['drc_context'], dtype=np.float32),
            action_category=data['action_category'],
            action_params=np.array(data['action_params'], dtype=np.float32),
            reward=data['reward'],
            next_state_embedding=np.array(data['next_state_embedding'], dtype=np.float32),
            done=data['done'],
            value_target=data['value_target'],
            policy_target=np.array(data['policy_target'], dtype=np.float32),
            optimization_id=data['optimization_id'],
            step=data['step'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
        )


class ExperienceBuffer:
    """
    Circular buffer for storing and sampling experiences.

    Features:
    - Prioritized sampling based on TD error
    - Stratified sampling across optimization runs
    - Persistent storage to disk
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Importance sampling exponent
        save_path: Optional[Path] = None,
    ):
        """
        Initialize experience buffer.

        Args:
            capacity: Maximum experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling correction (0 = no correction, 1 = full)
            save_path: Path to save buffer to disk
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.save_path = Path(save_path) if save_path else None

        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

        # Statistics
        self.total_added = 0
        self.optimization_ids: Dict[str, int] = {}  # Track samples per run

    def add(
        self,
        experience: Experience,
        priority: Optional[float] = None,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            experience: Experience to add
            priority: Priority for sampling (default: max priority)
        """
        if priority is None:
            # Use max priority for new experiences
            priority = max(self.priorities) if self.priorities else 1.0

        self.buffer.append(experience)
        self.priorities.append(priority)

        # Update statistics
        self.total_added += 1
        opt_id = experience.optimization_id
        self.optimization_ids[opt_id] = self.optimization_ids.get(opt_id, 0) + 1

    def add_batch(self, experiences: List[Experience]) -> None:
        """Add multiple experiences."""
        for exp in experiences:
            self.add(exp)

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences with prioritized replay.

        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)

        # Compute importance sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        experiences = [self.buffer[i] for i in indices]

        return experiences, indices, weights

    def sample_uniform(self, batch_size: int) -> List[Experience]:
        """Sample uniformly without prioritization."""
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def sample_stratified(self, batch_size: int) -> List[Experience]:
        """Sample stratified across optimization runs."""
        if len(self.buffer) == 0:
            return []

        # Group by optimization ID
        by_opt: Dict[str, List[int]] = {}
        for i, exp in enumerate(self.buffer):
            opt_id = exp.optimization_id
            if opt_id not in by_opt:
                by_opt[opt_id] = []
            by_opt[opt_id].append(i)

        # Sample from each group
        samples_per_group = max(1, batch_size // len(by_opt))
        indices = []

        for opt_id, group_indices in by_opt.items():
            n = min(samples_per_group, len(group_indices))
            sampled = np.random.choice(group_indices, size=n, replace=False)
            indices.extend(sampled.tolist())

        # Trim to batch size
        if len(indices) > batch_size:
            indices = np.random.choice(indices, size=batch_size, replace=False).tolist()

        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6  # Small epsilon for stability

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: Optional[Path] = None) -> None:
        """Save buffer to disk."""
        path = path or self.save_path
        if path is None:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON for portability
        data = {
            'experiences': [exp.to_dict() for exp in self.buffer],
            'priorities': list(self.priorities),
            'total_added': self.total_added,
            'optimization_ids': self.optimization_ids,
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: Optional[Path] = None) -> None:
        """Load buffer from disk."""
        path = path or self.save_path
        if path is None or not Path(path).exists():
            return

        with open(path) as f:
            data = json.load(f)

        self.buffer = deque(
            [Experience.from_dict(exp) for exp in data['experiences']],
            maxlen=self.capacity
        )
        self.priorities = deque(data['priorities'], maxlen=self.capacity)
        self.total_added = data.get('total_added', len(self.buffer))
        self.optimization_ids = data.get('optimization_ids', {})

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'num_optimizations': len(self.optimization_ids),
            'avg_reward': float(np.mean([exp.reward for exp in self.buffer])) if self.buffer else 0.0,
            'avg_priority': float(np.mean(self.priorities)) if self.priorities else 0.0,
        }


if TORCH_AVAILABLE:
    from .pcb_graph_encoder import PCBGraphEncoder
    from .value_network import ValueNetwork, ValueNetworkTrainer
    from .policy_network import PolicyNetwork, PolicyNetworkTrainer, DRCContextEncoder
    from .dynamics_network import WorldModel, WorldModelTrainer

    class TrainingPipeline:
        """
        Unified training pipeline for all gaming AI networks.

        Trains:
        - PCB Graph Encoder (representation)
        - Value Network (state evaluation)
        - Policy Network (action selection)
        - World Model (dynamics prediction)
        """

        def __init__(
            self,
            hidden_dim: int = 256,
            device: str = 'cpu',
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            checkpoint_dir: Optional[Path] = None,
        ):
            """
            Initialize training pipeline.

            Args:
                hidden_dim: Hidden dimension for all networks
                device: Device for training ('cpu' or 'cuda')
                learning_rate: Learning rate
                batch_size: Training batch size
                checkpoint_dir: Directory for checkpoints
            """
            self.hidden_dim = hidden_dim
            self.device = device
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Initialize networks
            self.encoder = PCBGraphEncoder(hidden_dim=hidden_dim).to(device)
            self.value_network = ValueNetwork(input_dim=hidden_dim).to(device)
            self.policy_network = PolicyNetwork(input_dim=hidden_dim).to(device)
            self.world_model = WorldModel(observation_dim=hidden_dim, latent_dim=hidden_dim).to(device)

            # Initialize trainers
            self.value_trainer = ValueNetworkTrainer(
                self.value_network, learning_rate=learning_rate,
                batch_size=batch_size, device=device
            )
            self.policy_trainer = PolicyNetworkTrainer(
                self.policy_network, learning_rate=learning_rate,
                batch_size=batch_size, device=device
            )
            self.world_trainer = WorldModelTrainer(
                self.world_model, learning_rate=learning_rate,
                batch_size=batch_size, device=device
            )

            # Experience buffer
            self.buffer = ExperienceBuffer(
                capacity=100000,
                save_path=self.checkpoint_dir / 'experience_buffer.json'
            )

            # Training history
            self.training_history: List[Dict[str, Any]] = []
            self.epoch = 0

        def add_optimization_experience(
            self,
            states: List[np.ndarray],
            actions: List[Tuple[int, np.ndarray]],
            rewards: List[float],
            values: List[float],
            policies: List[np.ndarray],
            drc_contexts: List[np.ndarray],
            optimization_id: str,
        ) -> int:
            """
            Add experiences from an optimization run.

            Args:
                states: List of state embeddings
                actions: List of (category, params) tuples
                rewards: List of rewards
                values: List of value targets (from search)
                policies: List of policy targets (from MCTS)
                drc_contexts: List of DRC feature vectors
                optimization_id: Unique ID for this run

            Returns:
                Number of experiences added
            """
            experiences = []

            for i in range(len(states) - 1):
                exp = Experience(
                    state_embedding=states[i],
                    drc_context=drc_contexts[i],
                    action_category=actions[i][0],
                    action_params=actions[i][1],
                    reward=rewards[i],
                    next_state_embedding=states[i + 1],
                    done=(i == len(states) - 2),
                    value_target=values[i],
                    policy_target=policies[i],
                    optimization_id=optimization_id,
                    step=i,
                )
                experiences.append(exp)

            self.buffer.add_batch(experiences)
            return len(experiences)

        def train_epoch(self) -> Dict[str, float]:
            """
            Train all networks for one epoch.

            Returns:
                Dictionary of losses
            """
            if len(self.buffer) < self.batch_size:
                return {'error': 'Not enough experiences'}

            self.epoch += 1
            all_losses = {}

            # Sample experiences
            experiences, indices, weights = self.buffer.sample(self.batch_size)

            # Prepare tensors
            state_embeddings = torch.tensor(
                np.stack([exp.state_embedding for exp in experiences]),
                dtype=torch.float32, device=self.device
            )
            drc_contexts = torch.tensor(
                np.stack([exp.drc_context for exp in experiences]),
                dtype=torch.float32, device=self.device
            )
            value_targets = torch.tensor(
                [[exp.value_target] for exp in experiences],
                dtype=torch.float32, device=self.device
            )
            policy_targets = torch.tensor(
                np.stack([exp.policy_target for exp in experiences]),
                dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                [exp.action_category for exp in experiences],
                dtype=torch.long, device=self.device
            )
            rewards = torch.tensor(
                [exp.reward for exp in experiences],
                dtype=torch.float32, device=self.device
            )

            # Train value network
            from .value_network import ValueTrainingSample
            value_samples = [
                ValueTrainingSample(
                    state_embedding=exp.state_embedding,
                    target_value=exp.value_target,
                    violation_breakdown={},
                    optimization_id=exp.optimization_id,
                    iteration=exp.step,
                )
                for exp in experiences
            ]
            value_losses = self.value_trainer.train_epoch(value_samples)
            all_losses.update({f'value_{k}': v for k, v in value_losses.items()})

            # Train policy network
            from .policy_network import PolicyTrainingSample
            policy_samples = [
                PolicyTrainingSample(
                    state_embedding=exp.state_embedding,
                    drc_context=exp.drc_context,
                    target_probs=exp.policy_target,
                    action_taken=exp.action_category,
                    parameters_taken={},
                    reward=exp.reward,
                    optimization_id=exp.optimization_id,
                )
                for exp in experiences
            ]
            policy_losses = self.policy_trainer.train_epoch(policy_samples)
            all_losses.update({f'policy_{k}': v for k, v in policy_losses.items()})

            # Train world model
            from .dynamics_network import WorldModelTrainingSample
            world_samples = [
                WorldModelTrainingSample(
                    state_embedding=exp.state_embedding,
                    action_embedding=np.array([exp.action_category]),
                    next_state_embedding=exp.next_state_embedding,
                    reward=exp.reward,
                    done=exp.done,
                    optimization_id=exp.optimization_id,
                    step=exp.step,
                )
                for exp in experiences
            ]
            world_losses = self.world_trainer.train_epoch(world_samples)
            all_losses.update({f'world_{k}': v for k, v in world_losses.items()})

            # Update priorities based on TD error
            with torch.no_grad():
                predicted_values = self.value_network(state_embeddings)
                td_errors = (value_targets - predicted_values).abs().cpu().numpy().flatten()
                self.buffer.update_priorities(indices, td_errors)

            # Record history
            all_losses['epoch'] = self.epoch
            self.training_history.append(all_losses)

            return all_losses

        def train(
            self,
            num_epochs: int = 100,
            save_every: int = 10,
            log_every: int = 1,
        ) -> Dict[str, Any]:
            """
            Full training loop.

            Args:
                num_epochs: Number of epochs to train
                save_every: Save checkpoint every N epochs
                log_every: Log progress every N epochs

            Returns:
                Training summary
            """
            print(f"\nTraining Pipeline - {num_epochs} epochs")
            print("=" * 60)
            print(f"Buffer size: {len(self.buffer)}")
            print(f"Batch size: {self.batch_size}")
            print(f"Device: {self.device}")

            start_epoch = self.epoch
            best_loss = float('inf')

            for _ in range(num_epochs):
                losses = self.train_epoch()

                if 'error' in losses:
                    print(f"Skipping: {losses['error']}")
                    break

                if self.epoch % log_every == 0:
                    total_loss = sum(v for k, v in losses.items() if 'loss' in k)
                    print(f"Epoch {self.epoch}: total_loss={total_loss:.4f}")

                if self.epoch % save_every == 0:
                    self.save_checkpoint()

                # Track best
                total_loss = sum(v for k, v in losses.items() if 'loss' in k)
                if total_loss < best_loss:
                    best_loss = total_loss
                    self.save_checkpoint('best')

            print(f"\nTraining complete: {self.epoch - start_epoch} epochs")

            return {
                'epochs_trained': self.epoch - start_epoch,
                'final_losses': self.training_history[-1] if self.training_history else {},
                'best_loss': best_loss,
                'buffer_size': len(self.buffer),
            }

        def save_checkpoint(self, name: str = 'latest') -> None:
            """Save training checkpoint."""
            checkpoint_path = self.checkpoint_dir / f'checkpoint_{name}.pt'

            torch.save({
                'epoch': self.epoch,
                'encoder_state_dict': self.encoder.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'policy_network_state_dict': self.policy_network.state_dict(),
                'world_model_state_dict': self.world_model.state_dict(),
                'training_history': self.training_history[-100:],  # Last 100 epochs
            }, checkpoint_path)

            # Save buffer separately
            self.buffer.save()

        def load_checkpoint(self, name: str = 'latest') -> bool:
            """Load training checkpoint."""
            checkpoint_path = self.checkpoint_dir / f'checkpoint_{name}.pt'

            if not checkpoint_path.exists():
                return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.epoch = checkpoint['epoch']
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
            self.training_history = checkpoint.get('training_history', [])

            # Load buffer
            self.buffer.load()

            print(f"Loaded checkpoint from epoch {self.epoch}")
            return True

        def get_networks(self) -> Dict[str, Any]:
            """Get all trained networks."""
            return {
                'encoder': self.encoder,
                'value_network': self.value_network,
                'policy_network': self.policy_network,
                'world_model': self.world_model,
            }


else:
    # Fallback without PyTorch
    class TrainingPipeline:
        """Fallback training pipeline (no-op without PyTorch)."""

        def __init__(self, **kwargs):
            import warnings
            warnings.warn("PyTorch not available. Training disabled.")
            self.buffer = ExperienceBuffer()

        def train(self, *args, **kwargs):
            return {'error': 'PyTorch not available'}

        def save_checkpoint(self, *args):
            pass

        def load_checkpoint(self, *args):
            return False


if __name__ == '__main__':
    print("Training Pipeline Test")
    print("=" * 60)

    # Test experience buffer
    buffer = ExperienceBuffer(capacity=1000)

    # Add some experiences
    for i in range(100):
        exp = Experience(
            state_embedding=np.random.randn(256).astype(np.float32),
            drc_context=np.random.randn(12).astype(np.float32),
            action_category=np.random.randint(0, 9),
            action_params=np.random.randn(5).astype(np.float32),
            reward=np.random.randn(),
            next_state_embedding=np.random.randn(256).astype(np.float32),
            done=(i == 99),
            value_target=np.random.random(),
            policy_target=np.random.dirichlet(np.ones(9)).astype(np.float32),
            optimization_id=f"opt_{i // 20}",
            step=i % 20,
        )
        buffer.add(exp)

    print(f"\nBuffer statistics: {buffer.get_statistics()}")

    # Test sampling
    uniform_samples = buffer.sample_uniform(16)
    print(f"Uniform samples: {len(uniform_samples)}")

    stratified_samples = buffer.sample_stratified(16)
    print(f"Stratified samples: {len(stratified_samples)}")

    prioritized, indices, weights = buffer.sample(16)
    print(f"Prioritized samples: {len(prioritized)}")
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    if TORCH_AVAILABLE:
        print("\nTesting training pipeline...")
        pipeline = TrainingPipeline(hidden_dim=256, batch_size=16)

        # Add experiences from buffer
        pipeline.buffer = buffer

        # Train a few epochs
        result = pipeline.train(num_epochs=3, log_every=1)
        print(f"Training result: {result}")
    else:
        print("\nPyTorch not available, skipping pipeline test")
