"""
Schematic Gaming AI Configuration

Centralized configuration for schematic optimization parameters,
extending the base gaming AI config with schematic-specific settings.
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class SchematicOptimizationMode(Enum):
    """Schematic optimization modes."""
    STANDARD = "standard"      # Basic generation, no optimization
    GAMING_AI = "gaming_ai"    # Full MAP-Elites + Red Queen + Ralph Wiggum
    HYBRID = "hybrid"          # LLM-guided with selective gaming AI
    FAST = "fast"              # Quick optimization, fewer iterations


@dataclass
class ERCWeights:
    """Weights for ERC violation scoring."""
    critical: float = 25.0     # Critical violations (multiple outputs, power issues)
    error: float = 10.0        # Error violations (floating inputs)
    warning: float = 3.0       # Warning violations (unconnected pins)


@dataclass
class BestPracticeWeights:
    """Weights for best practice violations."""
    critical: float = 20.0     # Critical BP violations
    error: float = 8.0         # Error BP violations
    warning: float = 2.0       # Warning BP violations


@dataclass
class FitnessConfig:
    """Configuration for fitness function calculation."""
    # Fitness component weights (must sum to 1.0)
    correctness_weight: float = 0.40   # ERC + BP compliance
    efficiency_weight: float = 0.30    # Cost efficiency
    reliability_weight: float = 0.20   # Thermal, decoupling, protection
    manufacturability_weight: float = 0.10  # Sourcing, availability

    # Scoring parameters
    erc_weights: ERCWeights = field(default_factory=ERCWeights)
    bp_weights: BestPracticeWeights = field(default_factory=BestPracticeWeights)

    # Target thresholds
    target_cost_per_function: float = 5.0   # USD per functional block
    min_decoupling_per_ic: float = 100e-9   # 100nF minimum per IC

    # Pass thresholds
    pass_fitness: float = 0.70      # Minimum fitness to pass
    excellent_fitness: float = 0.95  # Excellent fitness target


@dataclass
class MutationConfig:
    """Configuration for mutation operators."""
    # Strategy probabilities (must sum to 1.0)
    llm_guided_probability: float = 0.35
    topology_refinement_probability: float = 0.20
    component_optimization_probability: float = 0.25
    interface_hardening_probability: float = 0.10
    routing_optimization_probability: float = 0.10

    # LLM mutation settings
    max_llm_retries: int = 3
    llm_temperature: float = 0.7

    # Component mutation settings
    allow_package_change: bool = True
    allow_manufacturer_change: bool = True
    prefer_automotive_grade: bool = False

    # Topology mutation settings
    allow_regulator_type_change: bool = True  # LDO <-> Switching
    allow_power_domain_split: bool = True

    # Interface mutation settings
    add_esd_protection: bool = True
    add_termination_resistors: bool = True


@dataclass
class EvolutionConfig:
    """Configuration for Red Queen evolution."""
    # Population settings
    population_size: int = 20
    elite_count: int = 5

    # Competition settings
    rounds_per_generation: int = 10
    win_margin: float = 0.02      # 2% margin for win
    tie_margin: float = 0.01      # 1% margin for tie

    # Adversarial settings
    adversarial_mutation_rate: float = 0.3
    crossover_rate: float = 0.2


@dataclass
class RalphWiggumConfig:
    """Configuration for persistent optimization loop."""
    # Iteration limits
    max_iterations: int = 500
    target_fitness: float = 0.95

    # Stagnation detection
    stagnation_threshold: int = 50       # Iterations without improvement
    escalation_multiplier: float = 1.5   # Mutation rate increase on stagnation

    # Persistence settings
    checkpoint_interval: int = 10        # Save state every N iterations
    git_commit_on_improvement: bool = True

    # Timeout settings
    iteration_timeout: float = 30.0      # Seconds per iteration
    total_timeout: float = 3600.0        # Total optimization timeout (1 hour)


@dataclass
class ArchiveConfig:
    """Configuration for MAP-Elites archive."""
    # Grid resolution (bins per dimension)
    complexity_bins: int = 5         # component_count, net_count, sheet_count
    strategy_bins: int = 4           # power_distribution, interface_isolation
    quality_bins: int = 5            # erc_count, bp_adherence, cost_efficiency
    manufacturing_bins: int = 3      # footprint_variety, sourcing_difficulty

    # Archive settings
    max_archive_size: int = 1000
    elite_replacement_threshold: float = 0.01  # 1% improvement needed to replace


@dataclass
class BehaviorDescriptorConfig:
    """Configuration for behavioral descriptor computation."""
    # Normalization ranges
    max_component_count: int = 200
    max_net_count: int = 100
    max_sheet_count: int = 10
    max_footprint_variety: int = 30
    max_erc_violations: int = 50

    # Feature extraction settings
    include_thermal_features: bool = True
    include_cost_features: bool = True
    include_sourcing_features: bool = True


@dataclass
class SchematicGamingAIConfig:
    """Main configuration for schematic gaming AI optimization."""

    # Mode selection
    mode: SchematicOptimizationMode = SchematicOptimizationMode.HYBRID

    # LLM settings (inherited from base config, can override)
    llm_model: str = "anthropic/claude-opus-4.6"  # Cost-effective for mutations
    llm_api_key: Optional[str] = None  # From env: OPENROUTER_API_KEY

    # Component configs
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    ralph_wiggum: RalphWiggumConfig = field(default_factory=RalphWiggumConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    behavior: BehaviorDescriptorConfig = field(default_factory=BehaviorDescriptorConfig)

    # Debug settings
    verbose: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Get API key from environment if not provided
        if self.llm_api_key is None:
            self.llm_api_key = os.environ.get("OPENROUTER_API_KEY")
            if not self.llm_api_key:
                logger.warning(
                    "LLM enabled but no API key provided. "
                    "Set OPENROUTER_API_KEY environment variable."
                )

        # Validate fitness weights sum to 1.0
        total_weight = (
            self.fitness.correctness_weight +
            self.fitness.efficiency_weight +
            self.fitness.reliability_weight +
            self.fitness.manufacturability_weight
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Fitness weights must sum to 1.0, got {total_weight}")

        # Validate mutation probabilities sum to 1.0
        total_prob = (
            self.mutation.llm_guided_probability +
            self.mutation.topology_refinement_probability +
            self.mutation.component_optimization_probability +
            self.mutation.interface_hardening_probability +
            self.mutation.routing_optimization_probability
        )
        if abs(total_prob - 1.0) > 0.001:
            raise ValueError(f"Mutation probabilities must sum to 1.0, got {total_prob}")


# Global config instance
_config: Optional[SchematicGamingAIConfig] = None


def get_schematic_config() -> SchematicGamingAIConfig:
    """Get or create the global schematic gaming AI configuration."""
    global _config
    if _config is None:
        _config = SchematicGamingAIConfig()
    return _config


def set_schematic_config(config: SchematicGamingAIConfig) -> None:
    """Set the global schematic gaming AI configuration."""
    global _config
    _config = config


def reset_schematic_config() -> None:
    """Reset to default configuration."""
    global _config
    _config = None
