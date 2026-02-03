"""
MAPOS-RQ Integration - Complete Gaming AI Optimizer

This module provides the unified MAPOSRQOptimizer that integrates all
gaming AI components with the existing MAPOS pipeline:

1. PCB Graph Encoder for state representation (or LLM-based encoding)
2. Value/Policy/Dynamics networks for learned optimization (or LLM backends)
3. MAP-Elites for quality-diversity
4. Red Queen evolution for adversarial improvement
5. Ralph Wiggum loop for persistent iteration

Usage:
    # LLM-first mode (default, no PyTorch needed)
    optimizer = MAPOSRQOptimizer(pcb_path, target_violations=50)
    result = await optimizer.optimize()

    # With optional GPU acceleration
    from .config import GamingAIConfig, InferenceProvider
    config = GamingAIConfig(use_neural_networks=True)
    optimizer = MAPOSRQOptimizer(pcb_path, config=config)

The optimizer uses LLM-based backends by default and can optionally
use neural networks (local PyTorch or third-party GPU providers).
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

import numpy as np

# Add parent directory for MAPOS imports
SCRIPT_DIR = Path(__file__).parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import violation fix map for REAL pcbnew operations
try:
    from violation_fix_map import (
        apply_suggestion_to_pcb,
        apply_fixes_for_violations,
        get_fixes_for_violation,
    )
    VIOLATION_FIX_MAP_AVAILABLE = True
except ImportError:
    VIOLATION_FIX_MAP_AVAILABLE = False
    logger.warning("violation_fix_map not available - modifications will use fallback")

# Local gaming AI imports - config first
from .config import (
    GamingAIConfig, OptimizationMode, InferenceProvider,
    LLMConfig, get_config
)
from .llm_backends import (
    LLMClient, LLMStateEncoder, LLMValueEstimator,
    LLMPolicyGenerator, LLMDynamicsSimulator, get_llm_backends
)
from .optional_gpu_backend import OptionalGPUInference, get_inference_backend
from .pcb_graph_encoder import PCBGraphEncoder, PCBGraph
from .map_elites import MAPElitesArchive, BehavioralDescriptor
from .red_queen_evolver import RedQueenEvolver, Champion, EvolutionRound
from .ralph_wiggum_optimizer import (
    RalphWiggumOptimizer, CompletionCriteria, OptimizationResult, OptimizationStatus
)
from .training import TrainingPipeline, ExperienceBuffer, Experience

# Configure module logger
logger = logging.getLogger(__name__)

# Optional PyTorch imports
try:
    import torch
    TORCH_AVAILABLE = True
    from .value_network import ValueNetwork
    from .policy_network import PolicyNetwork, DRCContextEncoder
    from .dynamics_network import WorldModel
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not available - using LLM-only mode")


@dataclass
class MAPOSRQConfig:
    """
    Configuration for MAPOS-RQ optimizer.

    This is a simplified config that wraps GamingAIConfig for backward compatibility.
    For full control, use GamingAIConfig directly.

    Default mode: LLM-first (no PyTorch required).
    """

    # Target settings
    target_violations: int = 50
    target_fitness: float = 0.9
    max_iterations: int = 100

    # Red Queen settings
    rq_rounds: int = 10
    rq_population_size: int = 50
    rq_iterations_per_round: int = 100
    rq_elite_count: int = 5

    # Ralph Wiggum settings
    max_stagnation: int = 15
    max_duration_hours: float = 24.0

    # Neural network settings (DISABLED by default - LLM-first)
    use_neural_networks: bool = False  # Changed default to False
    hidden_dim: int = 256
    checkpoint_path: Optional[str] = None

    # LLM settings (ENABLED by default)
    use_llm: bool = True
    llm_model: str = "anthropic/claude-opus-4.5"
    openrouter_api_key: Optional[str] = None

    # Optional GPU backend
    gpu_provider: Optional[str] = None  # "runpod", "modal", "replicate"
    gpu_endpoint: Optional[str] = None

    # Output settings
    output_dir: Optional[str] = None
    save_checkpoints: bool = True
    use_git: bool = False

    # Advanced settings
    mcts_simulations: int = 50
    mutation_rate: float = 0.8
    crossover_rate: float = 0.2
    temperature: float = 1.0

    def to_gaming_ai_config(self) -> GamingAIConfig:
        """Convert to full GamingAIConfig."""
        from .config import (
            GamingAIConfig, LLMConfig, NeuralNetworkConfig,
            EvolutionConfig, RalphWiggumConfig, InferenceProvider
        )

        # Determine GPU provider
        gpu_provider = InferenceProvider.NONE
        if self.gpu_provider:
            try:
                gpu_provider = InferenceProvider(self.gpu_provider)
            except ValueError:
                pass

        return GamingAIConfig(
            use_llm=self.use_llm,
            use_neural_networks=self.use_neural_networks,
            llm=LLMConfig(model=self.llm_model),
            neural_network=NeuralNetworkConfig(
                enabled=self.use_neural_networks,
                provider=gpu_provider,
                endpoint_id=self.gpu_endpoint,
            ),
            evolution=EvolutionConfig(
                num_rounds=self.rq_rounds,
                iterations_per_round=self.rq_iterations_per_round,
                population_size=self.rq_population_size,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
            ),
            ralph_wiggum=RalphWiggumConfig(
                target_violations=self.target_violations,
                target_fitness=self.target_fitness,
                max_iterations=self.max_iterations,
                max_stagnation_iterations=self.max_stagnation,
                max_duration_hours=self.max_duration_hours,
            ),
            output_dir=self.output_dir,
            save_checkpoints=self.save_checkpoints,
        )


@dataclass
class MAPOSRQResult:
    """Complete result from MAPOS-RQ optimization."""
    status: OptimizationStatus
    initial_violations: int
    final_violations: int
    improvement: int
    final_fitness: float
    final_generality: float
    total_iterations: int
    total_duration_seconds: float
    red_queen_rounds: int
    champions: List[Champion]
    best_solution_path: Optional[Path]
    training_experiences: int
    convergence_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MAPOSRQOptimizer:
    """
    Complete MAPOS-RQ Optimizer.

    Combines all gaming AI techniques into a unified optimizer:
    - AlphaZero-style: Neural value/policy networks
    - MuZero-style: Learned world model for planning
    - Digital Red Queen: Adversarial co-evolution
    - Ralph Wiggum: Persistent iteration until success
    """

    def __init__(
        self,
        pcb_path: Union[str, Path],
        config: Optional[MAPOSRQConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize MAPOS-RQ Optimizer.

        Args:
            pcb_path: Path to the PCB file to optimize
            config: Configuration (uses defaults if not provided)
            llm_client: LLM client for guided optimization
        """
        self.pcb_path = Path(pcb_path)
        self.config = config or MAPOSRQConfig()
        self.llm_client = llm_client

        # Setup output directory
        self.output_dir = Path(self.config.output_dir) if self.config.output_dir else \
            self.pcb_path.parent / 'mapos_rq_output'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_neural_networks()
        self._init_evolution_components()
        self._init_training_pipeline()

        # State tracking
        self.initial_violations: Optional[int] = None
        self.experiences_collected = 0

    def _init_neural_networks(self) -> None:
        """
        Initialize inference backends.

        Priority:
        1. LLM backends (default, no PyTorch needed)
        2. Optional GPU backends (RunPod, Modal, etc.)
        3. Local PyTorch (if configured and available)
        """
        # Initialize LLM backends (always available as fallback)
        self.llm_encoder: Optional[LLMStateEncoder] = None
        self.llm_value: Optional[LLMValueEstimator] = None
        self.llm_policy: Optional[LLMPolicyGenerator] = None
        self.llm_dynamics: Optional[LLMDynamicsSimulator] = None

        # Initialize PyTorch components (optional)
        self.encoder: Optional[Any] = None
        self.value_network: Optional[Any] = None
        self.policy_network: Optional[Any] = None
        self.world_model: Optional[Any] = None

        # Create LLM client if LLM is enabled
        if self.config.use_llm:
            try:
                llm_config = LLMConfig(model=self.config.llm_model)
                self.llm_client = LLMClient(llm_config)

                # Initialize LLM backends
                self.llm_encoder = LLMStateEncoder(llm_client=self.llm_client)
                self.llm_value = LLMValueEstimator(llm_client=self.llm_client)
                self.llm_policy = LLMPolicyGenerator(llm_client=self.llm_client)
                self.llm_dynamics = LLMDynamicsSimulator(llm_client=self.llm_client)

                logger.info("LLM backends initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM backends: {e}")

        # Initialize unified inference backend
        gaming_config = self.config.to_gaming_ai_config() if hasattr(self.config, 'to_gaming_ai_config') else None
        self.inference_backend = OptionalGPUInference(gaming_config)
        logger.info("Unified inference backend initialized")

        # Initialize local PyTorch if explicitly enabled and available
        if self.config.use_neural_networks and TORCH_AVAILABLE:
            try:
                self.encoder = PCBGraphEncoder(hidden_dim=self.config.hidden_dim)
                self.value_network = ValueNetwork(input_dim=self.config.hidden_dim)
                self.policy_network = PolicyNetwork(
                    input_dim=self.config.hidden_dim,
                    temperature=self.config.temperature,
                )
                self.world_model = WorldModel(
                    observation_dim=self.config.hidden_dim,
                    latent_dim=self.config.hidden_dim,
                )

                # Load checkpoint if available
                if self.config.checkpoint_path:
                    self._load_checkpoint(Path(self.config.checkpoint_path))

                logger.info(f"Local PyTorch networks initialized (hidden_dim={self.config.hidden_dim})")

            except Exception as e:
                logger.warning(f"Failed to initialize local PyTorch networks: {e}")
                self.encoder = None
        else:
            mode = "LLM-only" if self.config.use_llm else "heuristic-only"
            logger.info(f"Using {mode} mode (PyTorch disabled or not available)")

    def _init_evolution_components(self) -> None:
        """Initialize evolution components."""
        # Fitness function that integrates LLM value estimation
        def fitness_fn(solution: Any) -> float:
            """
            Compute fitness for a PCB solution.

            Uses LLM value estimation when available for more nuanced
            quality assessment, falling back to DRC-based heuristic.
            """
            # Try LLM value estimation first (provides better quality signal)
            if self.llm_value is not None:
                try:
                    # Use the synchronous fast method for evolution (async not suitable here)
                    value = self.llm_value.estimate_quality_fast(solution)
                    logger.debug(f"LLM fitness estimation: {value:.4f}")
                    return float(np.clip(value, 0.01, 0.99))
                except Exception as e:
                    logger.debug(f"LLM fitness failed, using heuristic: {e}")

            # Fallback: direct DRC-based fitness
            if hasattr(solution, 'run_drc'):
                try:
                    drc = solution.run_drc()
                    violations = getattr(drc, 'total_violations', 0)
                    fitness = 1.0 / (1.0 + violations / 100.0)
                    logger.debug(f"DRC fitness: {fitness:.4f} (violations={violations})")
                    return fitness
                except Exception as e:
                    logger.warning(f"DRC fitness failed: {e}")

            return 0.5  # Unknown quality

        # Red Queen Evolver
        self.red_queen = RedQueenEvolver(
            population_size=self.config.rq_population_size,
            iterations_per_round=self.config.rq_iterations_per_round,
            elite_count=self.config.rq_elite_count,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            llm_client=self.llm_client,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

        # Completion criteria
        self.criteria = CompletionCriteria(
            target_violations=self.config.target_violations,
            target_fitness=self.config.target_fitness,
            max_iterations=self.config.max_iterations,
            max_stagnation=self.config.max_stagnation,
            max_duration_hours=self.config.max_duration_hours,
        )

        print(f"Evolution components initialized")

    def _init_training_pipeline(self) -> None:
        """Initialize training pipeline."""
        if TORCH_AVAILABLE and self.config.use_neural_networks:
            self.training_pipeline = TrainingPipeline(
                hidden_dim=self.config.hidden_dim,
                checkpoint_dir=self.output_dir / 'training',
            )
        else:
            self.training_pipeline = None

        # Experience buffer (always available)
        self.experience_buffer = ExperienceBuffer(
            capacity=100000,
            save_path=self.output_dir / 'experiences.json',
        )

    def _load_checkpoint(self, path: Path) -> bool:
        """Load neural network checkpoint."""
        if not path.exists():
            return False

        try:
            checkpoint = torch.load(path, map_location='cpu')

            if 'encoder_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if 'value_network_state_dict' in checkpoint:
                self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            if 'policy_network_state_dict' in checkpoint:
                self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            if 'world_model_state_dict' in checkpoint:
                self.world_model.load_state_dict(checkpoint['world_model_state_dict'])

            print(f"Loaded checkpoint from {path}")
            return True

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    def _encode_state(self, pcb_state: Any) -> np.ndarray:
        """
        Encode PCB state to embedding.

        Uses the following priority:
        1. Local PyTorch encoder (if available and enabled)
        2. LLM-based encoder (default)
        3. Hash-based fallback
        """
        # Try local PyTorch encoder first if enabled
        if self.encoder is not None and TORCH_AVAILABLE:
            try:
                graph = PCBGraph.from_pcb_state(pcb_state)
                embedding = self.encoder.encode_graph(graph)
                return embedding.cpu().numpy().flatten()
            except Exception as e:
                logger.debug(f"PyTorch encoding failed: {e}")

        # Use LLM encoder if available
        if self.llm_encoder is not None:
            try:
                return self.llm_encoder.encode_state(pcb_state)
            except Exception as e:
                logger.debug(f"LLM encoding failed: {e}")

        # Fallback: deterministic embedding based on state hash
        hash_str = pcb_state.get_hash() if hasattr(pcb_state, 'get_hash') else str(id(pcb_state))
        hash_val = int(hash_str[:16], 16) if len(hash_str) >= 16 else hash(hash_str)
        rng = np.random.Generator(np.random.PCG64(hash_val % (2**31)))
        return rng.standard_normal(self.config.hidden_dim).astype(np.float32)

    def _get_value_estimate(
        self,
        embedding: np.ndarray,
        pcb_state: Optional[Any] = None
    ) -> float:
        """
        Get quality value estimate for a PCB state.

        Uses the following priority:
        1. Local PyTorch value network (if available and enabled)
        2. LLM-based value estimator with actual PCB state
        3. DRC-based heuristic fallback

        Args:
            embedding: State embedding vector (256D numpy array)
            pcb_state: PCB state object with run_drc() method. REQUIRED for
                       accurate LLM estimation. If None, falls back to heuristics.

        Returns:
            Quality score between 0.01 and 0.99

        Note:
            Previous implementation passed None to estimate_quality_fast(), which
            would cause AttributeError when trying to call pcb_state.run_drc().
            This fix ensures pcb_state is properly propagated through the call chain.
        """
        # Try local PyTorch network first (if explicitly enabled)
        if self.value_network is not None and TORCH_AVAILABLE:
            try:
                import torch
                emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    value = self.value_network(emb_tensor)
                logger.debug(f"PyTorch value estimation: {value.item():.4f}")
                return float(np.clip(value.item(), 0.01, 0.99))
            except Exception as e:
                logger.warning(f"PyTorch value estimation failed: {e}")

        # Use LLM value estimator if available (requires pcb_state)
        if self.llm_value is not None and pcb_state is not None:
            try:
                value = self.llm_value.estimate_quality_fast(pcb_state)
                logger.debug(f"LLM value estimation (fast): {value:.4f}")
                return float(np.clip(value, 0.01, 0.99))
            except AttributeError as e:
                logger.error(
                    f"LLM value estimation failed due to missing attribute: {e}. "
                    f"Ensure pcb_state has run_drc() method."
                )
            except Exception as e:
                logger.warning(f"LLM value estimation failed: {e}")

        # Fallback: direct DRC-based heuristic if pcb_state available
        if pcb_state is not None and hasattr(pcb_state, 'run_drc'):
            try:
                drc = pcb_state.run_drc()
                violations = getattr(drc, 'total_violations', 0)
                value = 1.0 / (1.0 + violations / 100.0)
                logger.debug(f"DRC heuristic value: {value:.4f} (violations={violations})")
                return float(np.clip(value, 0.01, 0.99))
            except Exception as e:
                logger.warning(f"DRC heuristic estimation failed: {e}")

        # Final fallback: embedding-based heuristic (least accurate)
        if pcb_state is None:
            logger.warning(
                "_get_value_estimate called without pcb_state. "
                "Using embedding-based heuristic which is less accurate."
            )
        value = float(np.clip(np.tanh(embedding.mean()) * 0.4 + 0.5, 0.01, 0.99))
        logger.debug(f"Embedding heuristic value: {value:.4f}")
        return value

    async def _get_policy_async(
        self,
        pcb_state: Any,
        embedding: np.ndarray,
        drc_features: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Get policy distribution using LLM for intelligent action selection.

        This async method uses LLMPolicyGenerator to provide intelligent
        modification suggestions instead of random uniform distribution.

        Args:
            pcb_state: PCB state object with DRC capabilities
            embedding: State embedding vector
            drc_features: DRC feature vector

        Returns:
            Tuple of (action_probabilities, modification_suggestions)
        """
        # Try LLM policy generator first
        if self.llm_policy is not None and pcb_state is not None:
            try:
                # Get DRC violations for context
                drc_violations = None
                if hasattr(pcb_state, 'run_drc'):
                    drc = pcb_state.run_drc()
                    drc_violations = getattr(drc, 'violations_by_type', {})

                # Get LLM suggestions
                suggestions = await self.llm_policy.suggest_modifications(
                    pcb_state=pcb_state,
                    drc_violations=drc_violations,
                    top_k=5
                )

                if suggestions:
                    # Convert suggestions to action probability distribution
                    probs = self.llm_policy.get_action_probabilities(pcb_state, suggestions)
                    logger.debug(f"LLM policy generated {len(suggestions)} suggestions")
                    return probs, suggestions

            except Exception as e:
                logger.warning(f"LLM policy generation failed: {e}")

        # Fallback to PyTorch policy network
        return self._get_policy(embedding, drc_features), []

    def _get_policy(self, embedding: np.ndarray, drc_features: np.ndarray) -> np.ndarray:
        """
        Get policy distribution (synchronous version).

        Uses the following priority:
        1. Local PyTorch policy network (if available and enabled)
        2. Uniform distribution fallback (for async LLM, use _get_policy_async)

        Args:
            embedding: State embedding vector (256D numpy array)
            drc_features: DRC context features (12D numpy array)

        Returns:
            Action probability distribution over 9 modification categories

        Note:
            For LLM-based policy, use _get_policy_async() instead.
            This synchronous method cannot call the async LLM client.
        """
        if self.policy_network is not None and TORCH_AVAILABLE:
            try:
                import torch
                emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                drc_tensor = torch.tensor(drc_features, dtype=torch.float32).unsqueeze(0)

                probs, _ = self.policy_network.get_category_distribution(emb_tensor, drc_tensor)
                result = probs.cpu().numpy().flatten()
                logger.debug(f"PyTorch policy distribution: max={result.max():.3f}")
                return result
            except Exception as e:
                logger.warning(f"PyTorch policy generation failed: {e}")

        # Fallback: uniform distribution
        # Note: For intelligent policy without PyTorch, use _get_policy_async()
        logger.debug("Using uniform policy distribution (no PyTorch available)")
        return np.ones(9) / 9

    async def _verify_prediction_with_drc(
        self,
        pcb_state: Any,
        modification: Any,
        prediction: Any,
        actual_new_state: Any
    ) -> Dict[str, Any]:
        """
        Verify LLM dynamics prediction against actual DRC results.

        This implements the DRC feedback loop to validate prediction accuracy
        and improve future predictions through experience collection.

        Args:
            pcb_state: Original PCB state before modification
            modification: The modification that was applied
            prediction: LLM prediction of what would happen
            actual_new_state: The actual PCB state after modification

        Returns:
            Dict with prediction accuracy metrics and feedback
        """
        feedback = {
            "prediction_accuracy": 0.0,
            "violations_fixed_predicted": [],
            "violations_fixed_actual": [],
            "violations_created_predicted": [],
            "violations_created_actual": [],
            "net_improvement_predicted": 0,
            "net_improvement_actual": 0,
            "feedback_for_llm": ""
        }

        try:
            # Get original DRC
            original_drc = None
            if hasattr(pcb_state, 'run_drc'):
                original_drc = pcb_state.run_drc()
            original_violations = {}
            if original_drc:
                original_violations = getattr(original_drc, 'violations_by_type', {})
                original_total = getattr(original_drc, 'total_violations', 0)
            else:
                original_total = 0

            # Get new DRC
            new_drc = None
            if hasattr(actual_new_state, 'run_drc'):
                new_drc = actual_new_state.run_drc()
            new_violations = {}
            if new_drc:
                new_violations = getattr(new_drc, 'violations_by_type', {})
                new_total = getattr(new_drc, 'total_violations', 0)
            else:
                new_total = 0

            # Compute actual changes
            violations_fixed_actual = []
            violations_created_actual = []

            for vtype in set(original_violations.keys()) | set(new_violations.keys()):
                orig_count = original_violations.get(vtype, 0)
                new_count = new_violations.get(vtype, 0)

                if new_count < orig_count:
                    violations_fixed_actual.append(vtype)
                elif new_count > orig_count:
                    violations_created_actual.append(vtype)

            actual_improvement = original_total - new_total

            # Get prediction values
            if prediction is not None:
                pred_fixed = getattr(prediction, 'violations_fixed', [])
                pred_created = getattr(prediction, 'violations_created', [])
                pred_improvement = getattr(prediction, 'net_improvement', 0)
            else:
                pred_fixed = []
                pred_created = []
                pred_improvement = 0

            # Compute prediction accuracy
            fixed_match = len(set(pred_fixed) & set(violations_fixed_actual))
            fixed_total = max(len(set(pred_fixed) | set(violations_fixed_actual)), 1)
            fixed_accuracy = fixed_match / fixed_total

            created_match = len(set(pred_created) & set(violations_created_actual))
            created_total = max(len(set(pred_created) | set(violations_created_actual)), 1)
            created_accuracy = created_match / created_total if violations_created_actual else 1.0

            improvement_accuracy = 1.0 - abs(pred_improvement - actual_improvement) / max(abs(actual_improvement), 10)
            improvement_accuracy = max(0.0, improvement_accuracy)

            overall_accuracy = (fixed_accuracy + created_accuracy + improvement_accuracy) / 3.0

            # Store results
            feedback["prediction_accuracy"] = overall_accuracy
            feedback["violations_fixed_predicted"] = pred_fixed
            feedback["violations_fixed_actual"] = violations_fixed_actual
            feedback["violations_created_predicted"] = pred_created
            feedback["violations_created_actual"] = violations_created_actual
            feedback["net_improvement_predicted"] = pred_improvement
            feedback["net_improvement_actual"] = actual_improvement

            # Generate feedback for improving future predictions
            if overall_accuracy < 0.7:
                feedback["feedback_for_llm"] = (
                    f"Prediction accuracy: {overall_accuracy:.1%}. "
                    f"Predicted {pred_improvement} improvement but actual was {actual_improvement}. "
                    f"Expected to fix {pred_fixed} but actually fixed {violations_fixed_actual}. "
                    f"Unexpected violations: {violations_created_actual}."
                )
            else:
                feedback["feedback_for_llm"] = f"Good prediction accuracy: {overall_accuracy:.1%}"

            logger.info(
                f"DRC feedback loop: accuracy={overall_accuracy:.1%}, "
                f"predicted={pred_improvement}, actual={actual_improvement}"
            )

        except Exception as e:
            logger.warning(f"DRC feedback verification failed: {e}")
            feedback["feedback_for_llm"] = f"Verification failed: {e}"

        return feedback

    async def _apply_llm_guided_modification(
        self,
        pcb_state: Any,
        max_attempts: int = 3
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
        """
        Apply LLM-guided modification with prediction verification.

        Uses LLM policy to suggest modifications, dynamics simulator to predict
        outcomes, and DRC feedback loop to verify and learn.

        Args:
            pcb_state: Current PCB state
            max_attempts: Maximum modification attempts

        Returns:
            Tuple of (new_state, modification, feedback) or (None, None, None) if failed
        """
        if self.llm_policy is None:
            return None, None, None

        for attempt in range(max_attempts):
            try:
                # Get state embedding
                embedding = self._encode_state(pcb_state)
                drc_features = np.zeros(12)  # DRC context

                # Get policy suggestions
                probs, suggestions = await self._get_policy_async(
                    pcb_state, embedding, drc_features
                )

                if not suggestions:
                    logger.debug("No LLM suggestions available")
                    continue

                # Pick top suggestion
                modification = suggestions[0]
                logger.info(
                    f"LLM suggested: {modification.mod_type} on {modification.target} "
                    f"(confidence={modification.confidence:.2f})"
                )

                # Predict outcome using dynamics simulator
                prediction = None
                if self.llm_dynamics is not None:
                    try:
                        prediction = await self.llm_dynamics.simulate_modification(
                            pcb_state, modification
                        )
                        logger.debug(
                            f"Dynamics prediction: net_improvement={prediction.net_improvement}, "
                            f"confidence={prediction.confidence:.2f}"
                        )
                    except Exception as e:
                        logger.warning(f"Dynamics prediction failed: {e}")

                # Apply modification using REAL pcbnew operations
                new_state = None

                # PRIMARY: Use violation_fix_map for real pcbnew operations
                if VIOLATION_FIX_MAP_AVAILABLE and hasattr(pcb_state, 'pcb_path'):
                    try:
                        pcb_path = getattr(pcb_state, 'pcb_path', None)
                        if pcb_path and Path(pcb_path).exists():
                            logger.info(f"Applying REAL fix via violation_fix_map: {modification.mod_type}")

                            # Execute real fix operation based on suggestion type
                            fix_result = apply_suggestion_to_pcb(
                                pcb_path=Path(pcb_path),
                                suggestion=modification,
                            )

                            if fix_result.get('success'):
                                # Reload PCB state to reflect actual file changes
                                from pcb_state import PCBState
                                new_state = PCBState.from_file(str(pcb_path))
                                if hasattr(pcb_state, 'state_id'):
                                    new_state.parent_id = pcb_state.state_id
                                if hasattr(pcb_state, 'generation'):
                                    new_state.generation = pcb_state.generation + 1
                                logger.info(f"REAL fix applied: {modification.mod_type} → {fix_result}")
                            else:
                                logger.warning(f"Fix returned failure: {fix_result.get('error', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"Real pcbnew fix failed: {e}")

                # FALLBACK 1: Try pcb_state.apply_modification (legacy path)
                if new_state is None and hasattr(pcb_state, 'apply_modification'):
                    try:
                        new_state = pcb_state.apply_modification(modification)
                        if new_state is not None:
                            logger.debug("Used legacy apply_modification fallback")
                    except Exception as e:
                        logger.warning(f"Legacy modification application failed: {e}")

                # FALLBACK 2: Use Red Queen random mutation
                if new_state is None:
                    new_state = self.red_queen._random_mutate(pcb_state)
                    if new_state is not None:
                        logger.debug("Used Red Queen random mutation fallback")

                if new_state is None:
                    continue

                # Verify prediction against actual DRC
                feedback = await self._verify_prediction_with_drc(
                    pcb_state, modification, prediction, new_state
                )

                return new_state, modification, feedback

            except Exception as e:
                logger.warning(f"LLM-guided modification attempt {attempt + 1} failed: {e}")
                continue

        return None, None, None

    def _collect_experience(
        self,
        pcb_state: Any,
        action: Tuple[int, np.ndarray],
        reward: float,
        next_state: Any,
        done: bool,
        value_target: float,
        policy_target: np.ndarray,
        optimization_id: str,
        step: int,
    ) -> None:
        """Collect experience for training."""
        state_emb = self._encode_state(pcb_state)
        next_emb = self._encode_state(next_state)

        # Get DRC context
        drc = pcb_state.run_drc() if hasattr(pcb_state, 'run_drc') else None
        drc_features = DRCContextEncoder.extract_features(drc) if TORCH_AVAILABLE else np.zeros(12)

        exp = Experience(
            state_embedding=state_emb,
            drc_context=drc_features,
            action_category=action[0],
            action_params=action[1],
            reward=reward,
            next_state_embedding=next_emb,
            done=done,
            value_target=value_target,
            policy_target=policy_target,
            optimization_id=optimization_id,
            step=step,
        )

        self.experience_buffer.add(exp)
        self.experiences_collected += 1

    async def optimize(self) -> MAPOSRQResult:
        """
        Run complete MAPOS-RQ optimization.

        Returns:
            MAPOSRQResult with full optimization results
        """
        start_time = datetime.now()

        print("\n" + "=" * 70)
        print("MAPOS-RQ: Gaming AI-Enhanced PCB Optimization")
        print("=" * 70)
        print(f"PCB: {self.pcb_path.name}")
        print(f"Target: <= {self.config.target_violations} violations")
        print(f"Neural networks: {'enabled' if self.encoder else 'disabled'}")
        print(f"LLM: {'enabled' if self.llm_client else 'disabled'}")
        print("=" * 70)

        # Load initial PCB state
        try:
            from pcb_state import PCBState
            initial_state = PCBState.from_file(str(self.pcb_path))
        except ImportError:
            print("ERROR: Could not import PCBState. Ensure MAPOS is properly installed.")
            return MAPOSRQResult(
                status=OptimizationStatus.FAILED,
                initial_violations=0,
                final_violations=0,
                improvement=0,
                final_fitness=0.0,
                final_generality=0.0,
                total_iterations=0,
                total_duration_seconds=0.0,
                red_queen_rounds=0,
                champions=[],
                best_solution_path=None,
                training_experiences=0,
                convergence_metrics={},
                metadata={'error': 'PCBState import failed'},
            )

        # Get initial metrics
        initial_drc = initial_state.run_drc()
        self.initial_violations = initial_drc.total_violations
        print(f"\nInitial violations: {self.initial_violations}")

        # Phase 1: Red Queen Evolution
        print("\n" + "-" * 60)
        print("PHASE 1: Red Queen Adversarial Evolution")
        print("-" * 60)

        current_state = initial_state
        best_violations = self.initial_violations
        best_state = initial_state

        optimization_id = f"mapos_rq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Track prediction feedback for learning
        prediction_feedback_history: List[Dict[str, Any]] = []

        for rq_round in range(self.config.rq_rounds):
            print(f"\n  Round {rq_round + 1}/{self.config.rq_rounds}")

            # Try LLM-guided modification first (more intelligent)
            llm_modified_state = None
            llm_modification = None
            llm_feedback = None

            if self.llm_policy is not None:
                llm_modified_state, llm_modification, llm_feedback = await self._apply_llm_guided_modification(
                    current_state
                )
                if llm_feedback:
                    prediction_feedback_history.append(llm_feedback)
                    logger.debug(f"LLM modification feedback: {llm_feedback.get('prediction_accuracy', 0):.1%} accuracy")

            # Create population from current state + LLM suggestion
            population = [current_state]

            # Add LLM-modified state to population if available
            if llm_modified_state is not None:
                population.append(llm_modified_state)

            # Fill rest with random mutations
            while len(population) < self.config.rq_population_size:
                mutated = self.red_queen._random_mutate(current_state)
                if mutated:
                    population.append(mutated)

            # Run evolution round
            round_result = await self.red_queen.run_round(population, rq_round)

            # Get best champion
            if round_result.champions:
                best_champion = max(round_result.champions, key=lambda c: c.fitness)
                current_state = best_champion.solution

                # Collect experience with proper DRC analysis
                drc = current_state.run_drc() if hasattr(current_state, 'run_drc') else None
                violations = drc.total_violations if drc else best_violations

                if violations < best_violations:
                    improvement = best_violations - violations
                    print(f"    Improvement: {best_violations} -> {violations} (-{improvement})")
                    best_violations = violations
                    best_state = current_state

                # Get actual policy distribution if LLM suggestions were used
                policy_target = np.ones(9) / 9  # Default uniform
                if llm_modification is not None and self.llm_policy is not None:
                    # Use the actual LLM-generated policy probabilities
                    try:
                        policy_target = self.llm_policy.get_action_probabilities(
                            current_state,
                            [llm_modification] if llm_modification else []
                        )
                    except Exception as e:
                        logger.debug(f"Failed to get policy target: {e}")

                # Get proper value estimate for the state
                embedding = self._encode_state(current_state)
                value_target = self._get_value_estimate(embedding, pcb_state=current_state)

                # Record experience with actual policy and value
                self._collect_experience(
                    pcb_state=initial_state,
                    action=(0, np.zeros(5)),
                    reward=self.initial_violations - violations,
                    next_state=current_state,
                    done=False,
                    value_target=value_target,
                    policy_target=policy_target,
                    optimization_id=optimization_id,
                    step=rq_round,
                )

            # Check early termination
            if best_violations <= self.config.target_violations:
                print(f"\n  Target reached in round {rq_round + 1}!")
                break

        # Log prediction accuracy summary
        if prediction_feedback_history:
            accuracies = [f.get('prediction_accuracy', 0) for f in prediction_feedback_history]
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"\n  LLM prediction accuracy: {avg_accuracy:.1%} (over {len(accuracies)} predictions)")

        # Phase 2: Ralph Wiggum Refinement (if not at target)
        if best_violations > self.config.target_violations:
            print("\n" + "-" * 60)
            print("PHASE 2: Ralph Wiggum Persistent Refinement")
            print("-" * 60)

            ralph_optimizer = RalphWiggumOptimizer(
                pcb_path=self.pcb_path,
                output_dir=self.output_dir / 'ralph',
                criteria=self.criteria,
                red_queen_evolver=self.red_queen,
                use_git=self.config.use_git,
                llm_client=self.llm_client,
            )

            # Start from best state
            ralph_optimizer.state.best_violations = best_violations

            ralph_result = await ralph_optimizer.run()

            best_violations = ralph_result.final_violations
            best_state = self.red_queen.get_all_champions()[-1].solution if self.red_queen.champions_history else best_state

        # Phase 3: Train neural networks (optional)
        if self.training_pipeline and self.experiences_collected > 100:
            print("\n" + "-" * 60)
            print("PHASE 3: Neural Network Training")
            print("-" * 60)

            self.training_pipeline.buffer = self.experience_buffer
            train_result = self.training_pipeline.train(num_epochs=10, log_every=2)
            print(f"Training complete: {train_result}")

            if self.config.save_checkpoints:
                self.training_pipeline.save_checkpoint()

        # Finalize
        duration = (datetime.now() - start_time).total_seconds()

        # Save best solution
        best_solution_path = self.output_dir / f'{self.pcb_path.stem}_optimized.kicad_pcb'
        if hasattr(best_state, 'save_to_file'):
            try:
                best_state.save_to_file(str(best_solution_path))
            except Exception:
                best_solution_path = None

        # Determine status
        if best_violations <= self.config.target_violations:
            status = OptimizationStatus.SUCCESS
        elif duration >= self.config.max_duration_hours * 3600:
            status = OptimizationStatus.MAX_ITERATIONS
        else:
            status = OptimizationStatus.STAGNATED

        # Get all champions
        all_champions = self.red_queen.get_all_champions()

        # Get convergence metrics
        convergence = self.red_queen._compute_convergence_metrics()

        # Build result
        result = MAPOSRQResult(
            status=status,
            initial_violations=self.initial_violations,
            final_violations=best_violations,
            improvement=self.initial_violations - best_violations,
            final_fitness=1.0 / (1.0 + best_violations / 100.0),
            final_generality=convergence.get('generality_progression', 0.0) + 0.5,
            total_iterations=len(self.red_queen.rounds_history),
            total_duration_seconds=duration,
            red_queen_rounds=len(self.red_queen.champions_history),
            champions=all_champions,
            best_solution_path=best_solution_path,
            training_experiences=self.experiences_collected,
            convergence_metrics=convergence,
            metadata={
                'pcb_path': str(self.pcb_path),
                'config': {
                    'target_violations': self.config.target_violations,
                    'rq_rounds': self.config.rq_rounds,
                    'use_neural_networks': self.config.use_neural_networks,
                    'use_llm': self.config.use_llm,
                },
            },
        )

        # Save result
        self._save_result(result)

        # Print summary
        print("\n" + "=" * 70)
        print("MAPOS-RQ OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Status: {status.name}")
        print(f"Initial violations: {self.initial_violations}")
        print(f"Final violations: {best_violations}")
        print(f"Improvement: {self.initial_violations - best_violations} ({100 * (1 - best_violations / max(1, self.initial_violations)):.1f}%)")
        print(f"Duration: {duration / 60:.1f} minutes")
        print(f"Red Queen rounds: {len(self.red_queen.champions_history)}")
        print(f"Total champions evolved: {len(all_champions)}")
        print(f"Training experiences: {self.experiences_collected}")
        if best_solution_path:
            print(f"Output: {best_solution_path}")
        print("=" * 70)

        return result

    def _save_result(self, result: MAPOSRQResult) -> None:
        """Save optimization result to disk."""
        result_path = self.output_dir / 'optimization_result.json'

        data = {
            'status': result.status.name,
            'initial_violations': result.initial_violations,
            'final_violations': result.final_violations,
            'improvement': result.improvement,
            'final_fitness': result.final_fitness,
            'total_iterations': result.total_iterations,
            'total_duration_seconds': result.total_duration_seconds,
            'red_queen_rounds': result.red_queen_rounds,
            'training_experiences': result.training_experiences,
            'convergence_metrics': result.convergence_metrics,
            'metadata': result.metadata,
            'timestamp': datetime.now().isoformat(),
        }

        with open(result_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save experience buffer
        self.experience_buffer.save()


async def optimize_pcb(
    pcb_path: Union[str, Path],
    target_violations: int = 50,
    max_iterations: int = 100,
    use_neural_networks: bool = True,
    use_llm: bool = True,
    output_dir: Optional[str] = None,
) -> MAPOSRQResult:
    """
    Convenience function to optimize a PCB file.

    Args:
        pcb_path: Path to PCB file
        target_violations: Target maximum violations
        max_iterations: Maximum iterations
        use_neural_networks: Whether to use neural networks
        use_llm: Whether to use LLM guidance
        output_dir: Output directory

    Returns:
        Optimization result
    """
    config = MAPOSRQConfig(
        target_violations=target_violations,
        max_iterations=max_iterations,
        use_neural_networks=use_neural_networks,
        use_llm=use_llm,
        output_dir=output_dir,
    )

    optimizer = MAPOSRQOptimizer(pcb_path, config)
    return await optimizer.optimize()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MAPOS-RQ PCB Optimizer')
    parser.add_argument('pcb_path', type=str, help='Path to PCB file')
    parser.add_argument('--target', type=int, default=50, help='Target violations')
    parser.add_argument('--max-iter', type=int, default=100, help='Max iterations')
    parser.add_argument('--no-neural', action='store_true', help='Disable neural networks')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    result = asyncio.run(optimize_pcb(
        pcb_path=args.pcb_path,
        target_violations=args.target,
        max_iterations=args.max_iter,
        use_neural_networks=not args.no_neural,
        use_llm=not args.no_llm,
        output_dir=args.output,
    ))

    print(f"\nFinal result: {result.status.name}")
    print(f"Violations: {result.initial_violations} -> {result.final_violations}")
