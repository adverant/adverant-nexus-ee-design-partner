"""
Value Network - Learned PCB State Evaluation

This module implements an AlphaZero-style value network that predicts the
expected final quality score from a given PCB state. It replaces the
heuristic fitness function: 1.0 / (1.0 + violations/100)

Architecture:
- Input: PCB graph embedding from PCBGraphEncoder
- Hidden: Multi-layer MLP with residual connections
- Output: Scalar value in [0, 1] representing expected quality

Training:
- Supervised learning on (state, final_quality) pairs from optimization runs
- Self-play data from successful optimizations
- Temporal difference learning from value propagation

References:
- AlphaZero Value Network: https://arxiv.org/abs/1712.01815
- Neural Network Value Estimation: https://arxiv.org/abs/1911.08265
"""

import math
import json
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
class ValuePrediction:
    """Result of value network prediction."""
    value: float                          # Predicted quality score [0, 1]
    confidence: float                     # Prediction confidence [0, 1]
    uncertainty: float                    # Epistemic uncertainty estimate
    violation_estimates: Dict[str, float] # Predicted violation breakdown
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValueTrainingSample:
    """Training sample for value network."""
    state_embedding: np.ndarray   # PCB graph embedding
    target_value: float           # Actual final quality
    violation_breakdown: Dict[str, int]  # Actual violation counts
    optimization_id: str          # Which optimization run this came from
    iteration: int                # Iteration within the run
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_embedding': self.state_embedding.tolist(),
            'target_value': self.target_value,
            'violation_breakdown': self.violation_breakdown,
            'optimization_id': self.optimization_id,
            'iteration': self.iteration,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValueTrainingSample':
        return cls(
            state_embedding=np.array(data['state_embedding'], dtype=np.float32),
            target_value=data['target_value'],
            violation_breakdown=data['violation_breakdown'],
            optimization_id=data['optimization_id'],
            iteration=data['iteration'],
            timestamp=data.get('timestamp', datetime.now().isoformat()),
        )


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """Residual MLP block with pre-norm."""

        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp(self.norm(x))


    class ViolationPredictor(nn.Module):
        """
        Auxiliary head that predicts violation type breakdown.

        This multi-task learning helps the value network learn more
        informative representations.
        """

        # Common DRC violation types
        VIOLATION_TYPES = [
            'clearance',
            'track_width',
            'via_annular',
            'via_dangling',
            'track_dangling',
            'shorting_items',
            'silk_over_copper',
            'solder_mask_bridge',
            'courtyards_overlap',
            'unconnected',
        ]

        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.num_types = len(self.VIOLATION_TYPES)

            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_types),
                nn.Softplus(),  # Predict positive counts
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict violation counts for each type."""
            return self.predictor(x)

        def get_violation_dict(self, predictions: torch.Tensor) -> Dict[str, float]:
            """Convert prediction tensor to dictionary."""
            pred_np = predictions.detach().cpu().numpy().flatten()
            return {
                vtype: float(pred_np[i])
                for i, vtype in enumerate(self.VIOLATION_TYPES)
            }


    class UncertaintyEstimator(nn.Module):
        """
        Estimates epistemic uncertainty using MC Dropout.

        Higher uncertainty indicates the model is less confident,
        useful for guiding exploration in MCTS.
        """

        def __init__(self, input_dim: int, num_samples: int = 10, dropout: float = 0.2):
            super().__init__()
            self.num_samples = num_samples

            self.dropout_layer = nn.Dropout(dropout)
            self.variance_predictor = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1),
                nn.Softplus(),  # Positive variance
            )

        def forward(self, x: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Estimate uncertainty.

            Returns:
                mean_embedding: Averaged embedding
                uncertainty: Estimated uncertainty
            """
            if training:
                # During training, just return input and predicted variance
                return x, self.variance_predictor(x)

            # During inference, use MC Dropout
            self.train()  # Enable dropout
            samples = []
            for _ in range(self.num_samples):
                dropped = self.dropout_layer(x)
                samples.append(dropped)

            samples = torch.stack(samples)
            mean_embedding = samples.mean(dim=0)
            variance = samples.var(dim=0).mean(dim=-1, keepdim=True)

            self.eval()  # Restore eval mode
            return mean_embedding, variance + self.variance_predictor(x)


    class ValueNetwork(nn.Module):
        """
        Value Network for PCB State Evaluation.

        Predicts the expected final quality score from a PCB state embedding.
        Replaces heuristic: fitness = 1.0 / (1.0 + violations/100)

        Features:
        - Multi-task learning: value + violation breakdown
        - Uncertainty estimation for exploration
        - Residual connections for stable training
        - Confidence scoring based on input features
        """

        def __init__(
            self,
            input_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 4,
            dropout: float = 0.1,
            use_auxiliary_tasks: bool = True,
            use_uncertainty: bool = True,
        ):
            """
            Initialize Value Network.

            Args:
                input_dim: Dimension of input embedding (from PCBGraphEncoder)
                hidden_dim: Hidden layer dimension
                num_layers: Number of residual blocks
                dropout: Dropout rate
                use_auxiliary_tasks: Whether to predict violation breakdown
                use_uncertainty: Whether to estimate uncertainty
            """
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.use_auxiliary_tasks = use_auxiliary_tasks
            self.use_uncertainty = use_uncertainty

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

            # Residual blocks
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, dropout)
                for _ in range(num_layers)
            ])

            # Value head
            self.value_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )

            # Confidence head (estimates how confident the model is)
            self.confidence_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

            # Auxiliary task: violation breakdown
            if use_auxiliary_tasks:
                self.violation_predictor = ViolationPredictor(hidden_dim)

            # Uncertainty estimation
            if use_uncertainty:
                self.uncertainty_estimator = UncertaintyEstimator(hidden_dim)

        def forward(
            self,
            embedding: torch.Tensor,
            return_all: bool = False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
            """
            Forward pass.

            Args:
                embedding: PCB state embedding [batch_size, input_dim]
                return_all: Whether to return auxiliary outputs

            Returns:
                If return_all=False: value prediction [batch_size, 1]
                If return_all=True: (value, confidence, uncertainty, violations)
            """
            # Project input
            x = self.input_proj(embedding)

            # Residual blocks
            for block in self.blocks:
                x = block(x)

            # Value prediction
            value = self.value_head(x)

            if not return_all:
                return value

            # Confidence
            confidence = self.confidence_head(x)

            # Uncertainty
            if self.use_uncertainty:
                _, uncertainty = self.uncertainty_estimator(x, self.training)
            else:
                uncertainty = torch.zeros_like(value)

            # Violation breakdown
            if self.use_auxiliary_tasks:
                violations = self.violation_predictor(x)
            else:
                violations = torch.zeros(x.size(0), len(ViolationPredictor.VIOLATION_TYPES))

            return value, confidence, uncertainty, violations

        def predict(self, embedding: Union[np.ndarray, torch.Tensor]) -> ValuePrediction:
            """
            Make prediction with full output.

            Args:
                embedding: PCB state embedding

            Returns:
                ValuePrediction with all fields populated
            """
            self.eval()

            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)

            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            with torch.no_grad():
                value, confidence, uncertainty, violations = self(embedding, return_all=True)

            # Get violation dictionary
            if self.use_auxiliary_tasks:
                violation_dict = self.violation_predictor.get_violation_dict(violations[0])
            else:
                violation_dict = {}

            return ValuePrediction(
                value=value.item(),
                confidence=confidence.item(),
                uncertainty=uncertainty.item(),
                violation_estimates=violation_dict,
                metadata={
                    'embedding_norm': float(embedding.norm().item()),
                }
            )

        def compute_loss(
            self,
            embedding: torch.Tensor,
            target_value: torch.Tensor,
            target_violations: Optional[torch.Tensor] = None,
            value_weight: float = 1.0,
            violation_weight: float = 0.3,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute training loss.

            Args:
                embedding: Batch of embeddings [batch_size, input_dim]
                target_value: Target values [batch_size, 1]
                target_violations: Target violation counts [batch_size, num_types]
                value_weight: Weight for value loss
                violation_weight: Weight for violation loss

            Returns:
                total_loss: Combined loss
                loss_dict: Dictionary of individual losses
            """
            value, confidence, uncertainty, violations = self(embedding, return_all=True)

            # Value loss (MSE)
            value_loss = F.mse_loss(value, target_value)

            # Confidence calibration loss
            # High confidence when prediction is accurate
            prediction_error = (value - target_value).abs()
            target_confidence = 1.0 - prediction_error.clamp(0, 1)
            confidence_loss = F.mse_loss(confidence, target_confidence.detach())

            total_loss = value_weight * value_loss + 0.1 * confidence_loss

            loss_dict = {
                'value_loss': value_loss.item(),
                'confidence_loss': confidence_loss.item(),
            }

            # Violation prediction loss
            if self.use_auxiliary_tasks and target_violations is not None:
                violation_loss = F.mse_loss(violations, target_violations)
                total_loss += violation_weight * violation_loss
                loss_dict['violation_loss'] = violation_loss.item()

            loss_dict['total_loss'] = total_loss.item()

            return total_loss, loss_dict

        def save(self, path: Union[str, Path]) -> None:
            """Save model weights and config."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'state_dict': self.state_dict(),
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'num_layers': len(self.blocks),
                    'use_auxiliary_tasks': self.use_auxiliary_tasks,
                    'use_uncertainty': self.use_uncertainty,
                }
            }, path)

        @classmethod
        def load(cls, path: Union[str, Path]) -> 'ValueNetwork':
            """Load model from file."""
            path = Path(path)
            checkpoint = torch.load(path, map_location='cpu')

            config = checkpoint['config']
            model = cls(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                use_auxiliary_tasks=config['use_auxiliary_tasks'],
                use_uncertainty=config['use_uncertainty'],
            )
            model.load_state_dict(checkpoint['state_dict'])
            return model


    class ValueNetworkTrainer:
        """
        Trainer for the Value Network.

        Implements:
        - Mini-batch gradient descent
        - Learning rate scheduling
        - Early stopping
        - Validation monitoring
        """

        def __init__(
            self,
            model: ValueNetwork,
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

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
            )

            self.training_history: List[Dict[str, float]] = []
            self.best_val_loss = float('inf')

        def train_epoch(
            self,
            train_samples: List[ValueTrainingSample],
        ) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()

            # Shuffle samples
            np.random.shuffle(train_samples)

            total_losses = {'value_loss': 0.0, 'confidence_loss': 0.0, 'total_loss': 0.0}
            num_batches = 0

            for i in range(0, len(train_samples), self.batch_size):
                batch = train_samples[i:i + self.batch_size]

                # Prepare batch tensors
                embeddings = torch.tensor(
                    np.stack([s.state_embedding for s in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                targets = torch.tensor(
                    [[s.target_value] for s in batch],
                    dtype=torch.float32,
                    device=self.device,
                )

                # Prepare violation targets if available
                violation_targets = None
                if self.model.use_auxiliary_tasks:
                    violation_targets = torch.tensor(
                        [
                            [s.violation_breakdown.get(vtype, 0)
                             for vtype in ViolationPredictor.VIOLATION_TYPES]
                            for s in batch
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )

                # Forward pass
                self.optimizer.zero_grad()
                loss, loss_dict = self.model.compute_loss(
                    embeddings, targets, violation_targets
                )

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Accumulate losses
                for key, val in loss_dict.items():
                    total_losses[key] = total_losses.get(key, 0.0) + val
                num_batches += 1

            # Average losses
            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
            self.training_history.append(avg_losses)

            return avg_losses

        def validate(
            self,
            val_samples: List[ValueTrainingSample],
        ) -> Dict[str, float]:
            """Evaluate on validation set."""
            self.model.eval()

            total_losses = {'value_loss': 0.0, 'total_loss': 0.0}
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(val_samples), self.batch_size):
                    batch = val_samples[i:i + self.batch_size]

                    embeddings = torch.tensor(
                        np.stack([s.state_embedding for s in batch]),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    targets = torch.tensor(
                        [[s.target_value] for s in batch],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    _, loss_dict = self.model.compute_loss(embeddings, targets)

                    for key, val in loss_dict.items():
                        total_losses[key] = total_losses.get(key, 0.0) + val
                    num_batches += 1

            avg_losses = {k: v / max(1, num_batches) for k, v in total_losses.items()}

            # Update scheduler
            self.scheduler.step(avg_losses['total_loss'])

            # Track best
            if avg_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = avg_losses['total_loss']

            return avg_losses

        def train(
            self,
            train_samples: List[ValueTrainingSample],
            val_samples: Optional[List[ValueTrainingSample]] = None,
            num_epochs: int = 100,
            early_stopping_patience: int = 20,
            save_path: Optional[Path] = None,
        ) -> Dict[str, Any]:
            """
            Full training loop.

            Args:
                train_samples: Training data
                val_samples: Validation data (optional)
                num_epochs: Maximum epochs
                early_stopping_patience: Stop if no improvement
                save_path: Path to save best model

            Returns:
                Training results dictionary
            """
            best_epoch = 0
            patience_counter = 0

            for epoch in range(num_epochs):
                # Train
                train_losses = self.train_epoch(train_samples)

                # Validate
                if val_samples:
                    val_losses = self.validate(val_samples)
                    current_loss = val_losses['total_loss']
                else:
                    val_losses = {}
                    current_loss = train_losses['total_loss']

                # Print progress
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Train loss: {train_losses['total_loss']:.4f}")
                if val_losses:
                    print(f"  Val loss: {val_losses['total_loss']:.4f}")

                # Early stopping check
                if current_loss < self.best_val_loss:
                    self.best_val_loss = current_loss
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model
                    if save_path:
                        self.model.save(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            return {
                'best_epoch': best_epoch,
                'best_val_loss': self.best_val_loss,
                'training_history': self.training_history,
            }


else:
    # Fallback without PyTorch
    class ValueNetwork:
        """Fallback Value Network (requires PyTorch for full functionality)."""

        def __init__(self, input_dim: int = 256, **kwargs):
            self.input_dim = input_dim
            import warnings
            warnings.warn("PyTorch not available. ValueNetwork will use heuristic.")

        def predict(self, embedding: np.ndarray) -> ValuePrediction:
            """Fallback to heuristic value estimation."""
            # Use embedding statistics as proxy
            embedding_mean = np.mean(np.abs(embedding))
            value = 1.0 / (1.0 + embedding_mean)

            return ValuePrediction(
                value=float(value),
                confidence=0.5,
                uncertainty=0.3,
                violation_estimates={},
                metadata={'fallback': True}
            )


    class ValueNetworkTrainer:
        """Fallback trainer (no-op without PyTorch)."""

        def __init__(self, *args, **kwargs):
            import warnings
            warnings.warn("PyTorch not available. Training disabled.")

        def train(self, *args, **kwargs):
            return {'error': 'PyTorch not available'}


if __name__ == '__main__':
    print("Value Network Test")
    print("=" * 60)

    if TORCH_AVAILABLE:
        # Create model
        model = ValueNetwork(
            input_dim=256,
            hidden_dim=512,
            num_layers=4,
            use_auxiliary_tasks=True,
            use_uncertainty=True,
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Test forward pass
        batch_size = 8
        embedding = torch.randn(batch_size, 256)

        value, confidence, uncertainty, violations = model(embedding, return_all=True)
        print(f"\nForward pass:")
        print(f"  Value shape: {value.shape}")
        print(f"  Confidence shape: {confidence.shape}")
        print(f"  Uncertainty shape: {uncertainty.shape}")
        print(f"  Violations shape: {violations.shape}")

        # Test prediction
        single_embedding = np.random.randn(256).astype(np.float32)
        prediction = model.predict(single_embedding)
        print(f"\nPrediction:")
        print(f"  Value: {prediction.value:.4f}")
        print(f"  Confidence: {prediction.confidence:.4f}")
        print(f"  Uncertainty: {prediction.uncertainty:.4f}")
        print(f"  Violations: {prediction.violation_estimates}")

        # Test loss computation
        target_value = torch.rand(batch_size, 1)
        target_violations = torch.randint(0, 100, (batch_size, 10)).float()

        loss, loss_dict = model.compute_loss(embedding, target_value, target_violations)
        print(f"\nLoss computation:")
        print(f"  Total loss: {loss.item():.4f}")
        for key, val in loss_dict.items():
            print(f"  {key}: {val:.4f}")

        # Test training
        print("\nTesting trainer...")
        samples = [
            ValueTrainingSample(
                state_embedding=np.random.randn(256).astype(np.float32),
                target_value=np.random.random(),
                violation_breakdown={'clearance': np.random.randint(0, 50)},
                optimization_id='test',
                iteration=i,
            )
            for i in range(100)
        ]

        trainer = ValueNetworkTrainer(model, learning_rate=1e-3)
        losses = trainer.train_epoch(samples)
        print(f"  Epoch loss: {losses['total_loss']:.4f}")

    else:
        print("PyTorch not available, testing fallback")
        model = ValueNetwork()
        embedding = np.random.randn(256).astype(np.float32)
        prediction = model.predict(embedding)
        print(f"Prediction: {prediction}")
