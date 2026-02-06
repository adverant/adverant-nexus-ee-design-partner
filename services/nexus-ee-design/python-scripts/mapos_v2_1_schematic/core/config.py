"""
MAPO v2.1 Schematic Configuration

Centralized configuration for the LLM-orchestrated Gaming AI
schematic generation pipeline.

Author: Nexus EE Design Team
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SchematicMAPOConfig:
    """
    Configuration for MAPO v2.1 Schematic Pipeline.
    
    Controls all aspects of the schematic generation:
    - LLM orchestration (OpenRouter)
    - Nexus-memory integration
    - Gaming AI optimization
    - Smoke test validation
    - Fitness function weights
    
    Philosophy: "Opus 4.6 Thinks, Gaming AI Explores, Algorithms Execute, Memory Learns"
    """
    
    # ===== LLM Configuration =====
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = "anthropic/claude-opus-4.6"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_temperature: float = 0.3  # Lower for more deterministic responses
    llm_max_tokens: int = 4096
    llm_timeout: float = 120.0  # seconds
    
    # ===== Nexus-Memory Configuration =====
    nexus_api_key: str = field(
        default_factory=lambda: os.environ.get("NEXUS_API_KEY", "")
    )
    nexus_api_url: str = field(
        default_factory=lambda: os.environ.get("NEXUS_API_URL", "https://api.adverant.ai")
    )
    enable_memory: bool = True
    memory_recall_limit: int = 10  # Max memories to recall per query
    memory_store_successful: bool = True  # Auto-store successful resolutions
    
    # ===== Symbol Resolution =====
    symbol_cache_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "symbol_cache"
    )
    use_graphrag: bool = False  # Neo4j-based symbol search
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # Symbol source priority (higher = tried first)
    symbol_source_priority: tuple = (
        "nexus_memory",      # Learned resolutions first
        "kicad_worker",      # Official KiCad libraries
        "snapeda",           # SnapEDA API
        "ultralibrarian",    # UltraLibrarian API
        "local_cache",       # Previously fetched symbols
        "llm_generated",     # LLM-generated as last resort
    )
    
    # ===== Gaming AI Configuration =====
    map_elites_enabled: bool = True
    red_queen_enabled: bool = True
    ralph_wiggum_max_iterations: int = 100
    
    # MAP-Elites settings
    map_elites_archive_dims: tuple = (5, 5, 5, 5, 5)  # 5x5x5x5x5 = 3125 cells
    map_elites_behavior_dims: int = 10  # 10D behavioral descriptor
    
    # Red Queen settings
    red_queen_adversary_strength: float = 0.8
    red_queen_evolution_rate: float = 0.1
    
    # Mutation settings
    mutation_rate: float = 0.2
    mutation_operators: tuple = (
        ("llm_guided", 0.35),        # LLM recommends mutation
        ("topology_refinement", 0.20),
        ("component_optimization", 0.25),
        ("interface_hardening", 0.10),
        ("routing_optimization", 0.10),
    )
    
    # ===== Smoke Test Configuration =====
    smoke_test_enabled: bool = True
    require_smoke_test_pass: bool = True
    spice_simulator: str = "ngspice"  # ngspice, xyce, ltspice
    smoke_test_timeout: float = 60.0  # seconds
    
    # Smoke test thresholds
    max_dc_current_ma: float = 1000.0  # Max expected DC current
    max_dc_voltage_v: float = 100.0     # Max expected DC voltage
    convergence_tolerance: float = 1e-9
    
    # ===== Fitness Function Configuration =====
    # Weights for multi-objective optimization
    correctness_weight: float = 0.40  # ERC + connection completeness
    wiring_weight: float = 0.30       # Wire routing quality
    simulation_weight: float = 0.20   # Smoke test results
    cost_weight: float = 0.10         # BOM cost efficiency
    
    # Convergence targets
    target_fitness: float = 0.95
    min_acceptable_fitness: float = 0.70
    
    # ===== Validation Configuration =====
    validation_threshold: float = 0.85
    max_validation_iterations: int = 5
    
    # ERC settings
    erc_check_power_pins: bool = True
    erc_check_unconnected: bool = True
    erc_check_duplicate_refs: bool = True
    
    # ===== Output Configuration =====
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "output" / "mapo_v2_1"
    )
    
    # Export settings
    auto_export: bool = True
    export_pdf: bool = True
    export_svg: bool = True
    export_png: bool = True
    
    # NFS sync for artifact storage
    nfs_base_path: str = field(
        default_factory=lambda: os.environ.get("ARTIFACT_STORAGE_PATH", "/data/artifacts")
    )
    project_id: Optional[str] = None
    
    # ===== Logging =====
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: Optional[Path] = None
    
    # ===== Performance Tuning =====
    max_concurrent_symbol_fetches: int = 5
    batch_size: int = 10  # Components per batch for LLM calls
    cache_llm_responses: bool = True
    
    def __post_init__(self):
        """Initialize paths and validate configuration."""
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol_cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set log file path if not specified
        if self.log_to_file and not self.log_file_path:
            self.log_file_path = self.output_dir / "mapo_v2_1.log"
    
    @classmethod
    def from_env(cls) -> "SchematicMAPOConfig":
        """Create config from environment variables."""
        return cls(
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            nexus_api_key=os.environ.get("NEXUS_API_KEY", ""),
            nexus_api_url=os.environ.get("NEXUS_API_URL", "https://api.adverant.ai"),
            smoke_test_enabled=os.environ.get("SMOKE_TEST_ENABLED", "true").lower() == "true",
            map_elites_enabled=os.environ.get("MAP_ELITES_ENABLED", "true").lower() == "true",
            red_queen_enabled=os.environ.get("RED_QUEEN_ENABLED", "true").lower() == "true",
        )
    
    def validate(self) -> list:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check API keys
        if not self.openrouter_api_key:
            issues.append("OPENROUTER_API_KEY not set - LLM features disabled")
        
        if self.enable_memory and not self.nexus_api_key:
            issues.append("NEXUS_API_KEY not set - memory features disabled")
        
        # Check fitness weights sum to 1.0
        total_weight = (
            self.correctness_weight +
            self.wiring_weight +
            self.simulation_weight +
            self.cost_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Fitness weights sum to {total_weight}, should be 1.0")
        
        # Check mutation operator weights sum to 1.0
        mutation_weight_sum = sum(w for _, w in self.mutation_operators)
        if abs(mutation_weight_sum - 1.0) > 0.01:
            issues.append(f"Mutation weights sum to {mutation_weight_sum}, should be 1.0")
        
        return issues
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "openrouter_model": self.openrouter_model,
            "enable_memory": self.enable_memory,
            "map_elites_enabled": self.map_elites_enabled,
            "red_queen_enabled": self.red_queen_enabled,
            "smoke_test_enabled": self.smoke_test_enabled,
            "target_fitness": self.target_fitness,
            "fitness_weights": {
                "correctness": self.correctness_weight,
                "wiring": self.wiring_weight,
                "simulation": self.simulation_weight,
                "cost": self.cost_weight,
            },
            "output_dir": str(self.output_dir),
        }


# Singleton config instance
_config_instance: Optional[SchematicMAPOConfig] = None


def get_config() -> SchematicMAPOConfig:
    """Get or create the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = SchematicMAPOConfig.from_env()
    return _config_instance


def set_config(config: SchematicMAPOConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config
