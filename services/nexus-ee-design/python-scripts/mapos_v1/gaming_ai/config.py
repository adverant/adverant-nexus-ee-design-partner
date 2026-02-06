#!/usr/bin/env python3
"""
Gaming AI Configuration - Centralized configuration for MAPOS Gaming AI.

This module extracts all hardcoded constants from the gaming AI components
and provides a unified configuration interface with LLM-first defaults.

Author: Adverant Inc.
License: MIT
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Configure module logger
logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Operating modes for gaming AI optimization."""
    STANDARD = "standard"      # Base MAPOS only (gaming AI disabled)
    GAMING_AI = "gaming_ai"    # Skip base MAPOS, use Gaming AI only
    HYBRID = "hybrid"          # Base MAPOS first, then Gaming AI refinement


class InferenceProvider(Enum):
    """Backend providers for neural network inference."""
    NONE = "none"              # No neural networks
    LOCAL = "local"            # Local PyTorch
    RUNPOD = "runpod"          # RunPod GPU instances
    MODAL = "modal"            # Modal serverless GPUs
    REPLICATE = "replicate"    # Replicate model serving
    TOGETHER = "together"      # Together AI inference


@dataclass
class LLMConfig:
    """Configuration for LLM provider (OpenRouter)."""

    provider: str = "openrouter"
    model: str = "anthropic/claude-3.5-haiku:beta"  # Fast, cost-effective for mutations
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY")
    )
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class DRCConfig:
    """Configuration for DRC analysis constants.

    These values are used for normalizing DRC violation counts
    in feature encoding. Previously hardcoded in policy_network.py.
    """

    # Maximum expected violations per type (for normalization)
    clearance_max: int = 100
    track_width_max: int = 100
    via_dangling_max: int = 50
    track_dangling_max: int = 50
    shorting_items_max: int = 20
    silk_over_copper_max: int = 200
    solder_mask_bridge_max: int = 100
    courtyard_overlap_max: int = 50

    # DRC execution settings
    timeout_seconds: int = 120
    retry_count: int = 2
    cache_results: bool = True
    cache_size: int = 1000


@dataclass
class EvolutionConfig:
    """Configuration for Red Queen evolution parameters.

    Previously hardcoded in red_queen_evolver.py.
    """

    # Round configuration
    num_rounds: int = 10
    iterations_per_round: int = 50
    population_size: int = 20

    # Mutation settings
    mutation_rate: float = 0.8
    crossover_rate: float = 0.2

    # Win/tie/loss margins (previously hardcoded as 0.01)
    win_margin: float = 0.01
    tie_margin: float = 0.01

    # Fitness weighting
    violation_weight: float = 0.5
    generality_weight: float = 0.5

    # MAP-Elites archive
    archive_dimensions: int = 10
    archive_bins_per_dimension: int = 10


@dataclass
class RalphWiggumConfig:
    """Configuration for Ralph Wiggum persistent optimizer.

    Previously hardcoded in ralph_wiggum_optimizer.py.
    """

    # Completion criteria
    target_violations: int = 100
    target_fitness: float = 0.9
    target_generality: float = 0.8
    max_iterations: int = 1000
    max_stagnation_iterations: int = 50
    max_duration_hours: float = 24.0

    # Stagnation detection (previously hardcoded as 5)
    stagnation_threshold: int = 5

    # Escalation settings
    escalation_strategies: List[str] = field(default_factory=lambda: [
        "increase_mutation",
        "reset_population",
        "switch_agents",
        "expand_search",
        "call_for_help"
    ])

    # Persistence
    checkpoint_interval: int = 10
    use_git: bool = False
    atomic_writes: bool = True
    file_locking: bool = True


@dataclass
class BoardConfig:
    """Configuration for PCB board defaults.

    Previously hardcoded in pcb_graph_encoder.py.
    """

    # Default board dimensions (mm)
    default_width: float = 250.0
    default_height: float = 85.0

    # Graph encoding settings
    node_embedding_dim: int = 128
    edge_embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 6

    # Normalization ranges
    position_max: float = 500.0
    via_diameter_max: float = 2.0
    trace_width_max: float = 5.0


@dataclass
class NeuralNetworkConfig:
    """Configuration for optional neural network backends."""

    # Enable/disable
    enabled: bool = False  # Disabled by default (LLM-first)

    # Provider settings
    provider: InferenceProvider = InferenceProvider.NONE
    endpoint_id: Optional[str] = None
    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GPU_PROVIDER_API_KEY")
    )

    # Fallback behavior
    fallback_to_llm: bool = True

    # Model paths (for local inference)
    encoder_checkpoint: Optional[str] = None
    value_checkpoint: Optional[str] = None
    policy_checkpoint: Optional[str] = None
    dynamics_checkpoint: Optional[str] = None

    # Inference settings
    batch_size: int = 1
    use_fp16: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging infrastructure."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    structured: bool = False
    include_metrics: bool = True


@dataclass
class GamingAIConfig:
    """
    Master configuration for Gaming AI optimization system.

    This is the main configuration class that aggregates all sub-configurations.
    Designed with LLM-first defaults - no PyTorch or GPU required by default.

    Example usage:
        config = GamingAIConfig(
            mode=OptimizationMode.HYBRID,
            llm=LLMConfig(model="anthropic/claude-opus-4.6"),
        )
    """

    # Operating mode
    mode: OptimizationMode = OptimizationMode.HYBRID

    # Primary intelligence (LLM-first)
    use_llm: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Optional neural networks (disabled by default)
    use_neural_networks: bool = False
    neural_network: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)

    # Sub-configurations
    drc: DRCConfig = field(default_factory=DRCConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    ralph_wiggum: RalphWiggumConfig = field(default_factory=RalphWiggumConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Output settings
    output_dir: Optional[str] = None
    save_checkpoints: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        self._setup_logging()

    def _validate(self):
        """Validate configuration values."""
        if self.use_llm and not self.llm.api_key:
            logger.warning(
                "LLM enabled but no API key provided. "
                "Set OPENROUTER_API_KEY environment variable."
            )

        if self.use_neural_networks and self.neural_network.provider == InferenceProvider.NONE:
            logger.warning(
                "Neural networks enabled but no provider configured. "
                "Falling back to LLM."
            )
            if self.neural_network.fallback_to_llm:
                self.use_neural_networks = False

        if self.mode == OptimizationMode.GAMING_AI and not (self.use_llm or self.use_neural_networks):
            raise ValueError(
                "Gaming AI mode requires either LLM or neural networks enabled."
            )

    def _setup_logging(self):
        """Configure logging based on settings."""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)

        # Configure root logger for gaming_ai package
        package_logger = logging.getLogger("mapos.gaming_ai")
        package_logger.setLevel(log_level)

        if not package_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(self.logging.format))
            package_logger.addHandler(handler)

        if self.logging.file_path:
            file_handler = logging.FileHandler(self.logging.file_path)
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            package_logger.addHandler(file_handler)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "use_llm": self.use_llm,
            "use_neural_networks": self.use_neural_networks,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
            },
            "neural_network": {
                "enabled": self.neural_network.enabled,
                "provider": self.neural_network.provider.value,
                "fallback_to_llm": self.neural_network.fallback_to_llm,
            },
            "drc": {
                "clearance_max": self.drc.clearance_max,
                "track_width_max": self.drc.track_width_max,
                "via_dangling_max": self.drc.via_dangling_max,
            },
            "evolution": {
                "num_rounds": self.evolution.num_rounds,
                "iterations_per_round": self.evolution.iterations_per_round,
                "mutation_rate": self.evolution.mutation_rate,
                "win_margin": self.evolution.win_margin,
            },
            "ralph_wiggum": {
                "target_violations": self.ralph_wiggum.target_violations,
                "max_iterations": self.ralph_wiggum.max_iterations,
                "stagnation_threshold": self.ralph_wiggum.stagnation_threshold,
            },
            "board": {
                "default_width": self.board.default_width,
                "default_height": self.board.default_height,
                "hidden_dim": self.board.hidden_dim,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GamingAIConfig":
        """Create configuration from dictionary."""
        mode = OptimizationMode(data.get("mode", "hybrid"))

        llm_data = data.get("llm", {})
        llm_config = LLMConfig(
            provider=llm_data.get("provider", "openrouter"),
            model=llm_data.get("model", "anthropic/claude-opus-4.6"),
            max_tokens=llm_data.get("max_tokens", 4096),
            temperature=llm_data.get("temperature", 0.7),
        )

        nn_data = data.get("neural_network", {})
        nn_config = NeuralNetworkConfig(
            enabled=nn_data.get("enabled", False),
            provider=InferenceProvider(nn_data.get("provider", "none")),
            fallback_to_llm=nn_data.get("fallback_to_llm", True),
        )

        drc_data = data.get("drc", {})
        drc_config = DRCConfig(
            clearance_max=drc_data.get("clearance_max", 100),
            track_width_max=drc_data.get("track_width_max", 100),
            via_dangling_max=drc_data.get("via_dangling_max", 50),
        )

        evolution_data = data.get("evolution", {})
        evolution_config = EvolutionConfig(
            num_rounds=evolution_data.get("num_rounds", 10),
            iterations_per_round=evolution_data.get("iterations_per_round", 50),
            mutation_rate=evolution_data.get("mutation_rate", 0.8),
            win_margin=evolution_data.get("win_margin", 0.01),
        )

        rw_data = data.get("ralph_wiggum", {})
        rw_config = RalphWiggumConfig(
            target_violations=rw_data.get("target_violations", 100),
            max_iterations=rw_data.get("max_iterations", 1000),
            stagnation_threshold=rw_data.get("stagnation_threshold", 5),
        )

        board_data = data.get("board", {})
        board_config = BoardConfig(
            default_width=board_data.get("default_width", 250.0),
            default_height=board_data.get("default_height", 85.0),
            hidden_dim=board_data.get("hidden_dim", 256),
        )

        return cls(
            mode=mode,
            use_llm=data.get("use_llm", True),
            use_neural_networks=data.get("use_neural_networks", False),
            llm=llm_config,
            neural_network=nn_config,
            drc=drc_config,
            evolution=evolution_config,
            ralph_wiggum=rw_config,
            board=board_config,
        )

    @classmethod
    def from_env(cls) -> "GamingAIConfig":
        """Create configuration from environment variables."""
        return cls(
            mode=OptimizationMode(os.environ.get("GAMING_AI_MODE", "hybrid")),
            use_llm=os.environ.get("GAMING_AI_USE_LLM", "true").lower() == "true",
            use_neural_networks=os.environ.get("GAMING_AI_USE_NN", "false").lower() == "true",
            llm=LLMConfig(
                model=os.environ.get("GAMING_AI_LLM_MODEL", "anthropic/claude-opus-4.6"),
            ),
            neural_network=NeuralNetworkConfig(
                provider=InferenceProvider(os.environ.get("GAMING_AI_GPU_PROVIDER", "none")),
                endpoint_id=os.environ.get("GAMING_AI_GPU_ENDPOINT"),
            ),
        )


# Default configuration instance
DEFAULT_CONFIG = GamingAIConfig()


def get_config() -> GamingAIConfig:
    """Get the current configuration (from environment or defaults)."""
    try:
        return GamingAIConfig.from_env()
    except Exception as e:
        logger.warning(f"Failed to load config from environment: {e}")
        return DEFAULT_CONFIG
