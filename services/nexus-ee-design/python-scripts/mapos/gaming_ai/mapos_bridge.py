"""
MAPOS Bridge - Integration with Existing MAPOS Pipeline

This module provides seamless integration between the Gaming AI optimization
system and the existing MAPOS multi-agent pipeline. It extends the base
MultiAgentOptimizer with Gaming AI capabilities.

Key Features:
1. Uses existing MAPOS pre-DRC fixes before Gaming AI optimization
2. Integrates with existing PCBState representation
3. Preserves all existing MAPOS phases while adding Gaming AI phases
4. Provides backward compatibility with existing MAPOS CLI

Usage:
    # As drop-in replacement for MultiAgentOptimizer
    optimizer = GamingAIMultiAgentOptimizer(pcb_path, config)
    result = await optimizer.optimize()

    # Or use the CLI
    python -m mapos.gaming_ai.mapos_bridge board.kicad_pcb --gaming-ai
"""

import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import shutil

# Add parent directory for MAPOS imports
SCRIPT_DIR = Path(__file__).parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import existing MAPOS components
try:
    from multi_agent_optimizer import (
        MultiAgentOptimizer, OptimizationConfig, OptimizationResult as MAPOSResult
    )
    from pcb_state import PCBState, DRCResult, PCBModification, ModificationType
    from generator_agents import AgentPool, create_default_agent_pool
    MAPOS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MAPOS modules not available: {e}")
    MAPOS_AVAILABLE = False

# Import gaming AI components
from .pcb_graph_encoder import PCBGraph, PCBGraphEncoder
from .map_elites import MAPElitesArchive, BehavioralDescriptor
from .red_queen_evolver import RedQueenEvolver, Champion, EvolutionRound
from .ralph_wiggum_optimizer import (
    RalphWiggumOptimizer, CompletionCriteria, OptimizationResult as RWResult, OptimizationStatus
)
from .training import ExperienceBuffer, Experience

try:
    import torch
    TORCH_AVAILABLE = True
    from .value_network import ValueNetwork, ValuePrediction
    from .policy_network import PolicyNetwork, ModificationCategory
    from .dynamics_network import WorldModel
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


@dataclass
class GamingAIConfig:
    """
    Extended configuration for Gaming AI integration.

    Default mode: LLM-first (no PyTorch required).
    """

    # Base MAPOS settings (passed to parent)
    target_violations: int = 100
    max_time_minutes: int = 30
    mcts_iterations: int = 50
    evolution_generations: int = 30
    refinement_cycles: int = 10

    # Gaming AI settings
    enable_gaming_ai: bool = True
    rq_rounds: int = 10
    rq_population_size: int = 50
    rq_iterations_per_round: int = 100

    # Mode selection: "standard", "gaming_ai", or "hybrid"
    mode: str = "hybrid"

    # Neural network settings (DISABLED by default - LLM-first)
    use_neural_networks: bool = False  # Changed default to False
    hidden_dim: int = 256
    checkpoint_path: Optional[str] = None

    # LLM settings (ENABLED by default)
    use_llm: bool = True
    llm_model: str = "anthropic/claude-opus-4.6"

    # Optional GPU backend
    gpu_provider: Optional[str] = None  # "runpod", "modal", "replicate"
    gpu_endpoint: Optional[str] = None

    # Ralph Wiggum settings
    max_stagnation: int = 15
    max_duration_hours: float = 24.0

    # Output settings
    save_gaming_ai_checkpoints: bool = True

    def to_mapos_config(self) -> 'OptimizationConfig':
        """Convert to base MAPOS OptimizationConfig."""
        if not MAPOS_AVAILABLE:
            raise RuntimeError("MAPOS modules not available")

        return OptimizationConfig(
            target_violations=self.target_violations,
            max_time_minutes=self.max_time_minutes,
            mcts_iterations=self.mcts_iterations,
            evolution_generations=self.evolution_generations,
            refinement_cycles=self.refinement_cycles,
        )


@dataclass
class GamingAIResult:
    """Extended result including Gaming AI metrics."""

    # Base MAPOS result fields
    initial_violations: int
    final_violations: int
    total_improvement: int
    improvement_percent: float
    target_reached: bool
    phases_completed: List[str]
    total_time_seconds: float
    best_state: Optional[Any]
    phase_results: Dict[str, Any]

    # Gaming AI specific fields
    gaming_ai_enabled: bool
    red_queen_rounds: int
    total_champions: int
    archive_coverage: float
    convergence_metrics: Dict[str, float]
    neural_network_used: bool
    training_experiences: int

    @classmethod
    def from_mapos_result(
        cls,
        mapos_result: 'MAPOSResult',
        gaming_ai_phases: Dict[str, Any],
    ) -> 'GamingAIResult':
        """Create from base MAPOS result with Gaming AI additions."""
        return cls(
            initial_violations=mapos_result.initial_violations,
            final_violations=mapos_result.final_violations,
            total_improvement=mapos_result.total_improvement,
            improvement_percent=mapos_result.improvement_percent,
            target_reached=mapos_result.target_reached,
            phases_completed=mapos_result.phases_completed,
            total_time_seconds=mapos_result.total_time_seconds,
            best_state=mapos_result.best_state,
            phase_results={**mapos_result.phase_results, **gaming_ai_phases},
            gaming_ai_enabled=gaming_ai_phases.get('gaming_ai_enabled', False),
            red_queen_rounds=gaming_ai_phases.get('red_queen_rounds', 0),
            total_champions=gaming_ai_phases.get('total_champions', 0),
            archive_coverage=gaming_ai_phases.get('archive_coverage', 0.0),
            convergence_metrics=gaming_ai_phases.get('convergence_metrics', {}),
            neural_network_used=gaming_ai_phases.get('neural_network_used', False),
            training_experiences=gaming_ai_phases.get('training_experiences', 0),
        )


class GamingAIMultiAgentOptimizer:
    """
    Extended Multi-Agent Optimizer with Gaming AI capabilities.

    This class wraps the existing MultiAgentOptimizer and adds Gaming AI
    optimization phases:

    1. Base MAPOS phases (pre-DRC, MCTS, Evolution, Tournament, Refinement)
    2. Red Queen adversarial evolution
    3. Ralph Wiggum persistent refinement
    4. Neural network training

    The optimizer can run in three modes:
    - Standard: Just base MAPOS (enable_gaming_ai=False)
    - Gaming AI only: Skip base MAPOS, use Gaming AI
    - Hybrid: Run base MAPOS first, then Gaming AI for refinement
    """

    def __init__(
        self,
        pcb_path: str,
        config: Optional[GamingAIConfig] = None,
        output_dir: Optional[str] = None,
        llm_client: Optional[Any] = None,
        mode: str = "hybrid",
    ):
        """
        Initialize extended optimizer.

        Args:
            pcb_path: Path to PCB file
            config: Gaming AI configuration
            output_dir: Output directory
            llm_client: LLM client for mutations
            mode: "standard", "gaming_ai", or "hybrid"
        """
        self.pcb_path = Path(pcb_path)
        self.config = config or GamingAIConfig()
        self.output_dir = Path(output_dir) if output_dir else self.pcb_path.parent / "gaming_ai_output"
        self.llm_client = llm_client
        self.mode = mode

        # Validate PCB file
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base MAPOS optimizer (if available)
        self.base_optimizer: Optional[MultiAgentOptimizer] = None
        if MAPOS_AVAILABLE and mode in ("standard", "hybrid"):
            try:
                self.base_optimizer = MultiAgentOptimizer(
                    pcb_path=str(self.pcb_path),
                    config=self.config.to_mapos_config(),
                    output_dir=str(self.output_dir / 'mapos'),
                )
            except Exception as e:
                print(f"Warning: Could not initialize base MAPOS optimizer: {e}")

        # Initialize Gaming AI components
        self._init_gaming_ai_components()

        # State tracking
        self.current_state: Optional[PCBState] = None
        self.best_state: Optional[PCBState] = None
        self.best_violations = float('inf')
        self.experiences_collected = 0

    def _init_gaming_ai_components(self) -> None:
        """Initialize Gaming AI components."""
        # Neural networks
        self.encoder: Optional[PCBGraphEncoder] = None
        self.value_network: Optional[ValueNetwork] = None
        self.policy_network: Optional[PolicyNetwork] = None
        self.world_model: Optional[WorldModel] = None

        if self.config.use_neural_networks and TORCH_AVAILABLE:
            try:
                self.encoder = PCBGraphEncoder(hidden_dim=self.config.hidden_dim)
                self.value_network = ValueNetwork(input_dim=self.config.hidden_dim)
                self.policy_network = PolicyNetwork(input_dim=self.config.hidden_dim)
                self.world_model = WorldModel(
                    observation_dim=self.config.hidden_dim,
                    latent_dim=self.config.hidden_dim,
                )

                # Load checkpoint if provided
                if self.config.checkpoint_path:
                    self._load_checkpoint(Path(self.config.checkpoint_path))

            except Exception as e:
                print(f"Warning: Could not initialize neural networks: {e}")

        # Fitness function for Red Queen
        def fitness_fn(solution: Any) -> float:
            if hasattr(solution, 'run_drc'):
                try:
                    drc = solution.run_drc()
                    return 1.0 / (1.0 + drc.total_violations / 100.0)
                except Exception:
                    return 0.0
            return 0.5

        # Red Queen Evolver
        self.red_queen = RedQueenEvolver(
            population_size=self.config.rq_population_size,
            iterations_per_round=self.config.rq_iterations_per_round,
            elite_count=5,
            mutation_rate=0.8,
            crossover_rate=0.2,
            llm_client=self.llm_client,
            fitness_fn=fitness_fn,
            descriptor_fn=BehavioralDescriptor.from_pcb_state,
        )

        # MAP-Elites Archive
        self.archive = MAPElitesArchive(
            dimensions=10,
            bins_per_dimension=10,
        )

        # Experience buffer
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

            if 'encoder_state_dict' in checkpoint and self.encoder:
                self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if 'value_network_state_dict' in checkpoint and self.value_network:
                self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            if 'policy_network_state_dict' in checkpoint and self.policy_network:
                self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            if 'world_model_state_dict' in checkpoint and self.world_model:
                self.world_model.load_state_dict(checkpoint['world_model_state_dict'])

            print(f"Loaded checkpoint from {path}")
            return True

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    def _encode_state(self, state: PCBState) -> np.ndarray:
        """Encode PCB state to neural embedding."""
        if self.encoder is not None and TORCH_AVAILABLE:
            try:
                graph = PCBGraph.from_pcb_state(state)
                embedding = self.encoder.encode_graph(graph)
                return embedding.detach().cpu().numpy().flatten()
            except Exception:
                pass

        # Fallback: hash-based pseudo-embedding
        hash_val = int(state.get_hash(), 16) if hasattr(state, 'get_hash') else 0
        np.random.seed(hash_val % (2**31))
        return np.random.randn(self.config.hidden_dim).astype(np.float32)

    def _predict_value(self, state: PCBState) -> float:
        """Predict value using neural network."""
        if self.value_network is not None and TORCH_AVAILABLE:
            try:
                embedding = self._encode_state(state)
                emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    value = self.value_network(emb_tensor)
                return value.item()
            except Exception:
                pass

        # Fallback: heuristic
        if hasattr(state, '_drc_result') and state._drc_result:
            return state._drc_result.fitness_score
        return 0.5

    def _sample_action(self, state: PCBState) -> Tuple[int, np.ndarray]:
        """Sample action from policy network."""
        if self.policy_network is not None and TORCH_AVAILABLE:
            try:
                embedding = self._encode_state(state)
                emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                drc_context = torch.zeros(1, 12)  # Placeholder DRC context

                with torch.no_grad():
                    category, params = self.policy_network.sample_action(emb_tensor, drc_context)

                return category.item(), params.cpu().numpy().flatten()
            except Exception:
                pass

        # Fallback: random action
        category = np.random.randint(0, 9)
        params = np.random.randn(5)
        return category, params

    def _apply_action_to_state(
        self,
        state: PCBState,
        action_category: int,
        action_params: np.ndarray,
    ) -> PCBState:
        """Apply sampled action to create new state."""
        # Map category to modification type
        category_to_mod = {
            0: ModificationType.MOVE_COMPONENT,
            1: ModificationType.ADJUST_TRACE_WIDTH,
            2: ModificationType.ADJUST_VIA_SIZE,
            3: ModificationType.ADJUST_ZONE,
            4: ModificationType.ADJUST_CLEARANCE,
            5: ModificationType.MOVE_VIA,
            6: ModificationType.DELETE_VIA,
            7: ModificationType.ADJUST_SOLDER_MASK,
            8: ModificationType.MOVE_SILKSCREEN,
        }

        mod_type = category_to_mod.get(action_category, ModificationType.ADJUST_CLEARANCE)

        # Build parameters from action
        if mod_type == ModificationType.MOVE_COMPONENT:
            # Select random component if available
            if state.components:
                ref = list(state.components.keys())[int(abs(action_params[0]) * len(state.components)) % len(state.components)]
                comp = state.components[ref]
                params = {
                    'x': comp.x + action_params[1] * 5.0,
                    'y': comp.y + action_params[2] * 5.0,
                    'rotation': (comp.rotation + action_params[3] * 45) % 360,
                }
                target = ref
            else:
                return state.copy()

        elif mod_type == ModificationType.ADJUST_CLEARANCE:
            param_names = ['signal_clearance', 'power_clearance', 'zone_clearance']
            param_name = param_names[int(abs(action_params[0]) * 3) % 3]
            params = {
                'param_name': param_name,
                'value': 0.1 + abs(action_params[1]) * 0.4,
            }
            target = param_name

        elif mod_type == ModificationType.ADJUST_VIA_SIZE:
            if state.nets:
                net = list(state.nets.keys())[int(abs(action_params[0]) * len(state.nets)) % len(state.nets)]
                params = {
                    'diameter': 0.6 + abs(action_params[1]) * 0.6,
                    'drill': 0.3 + abs(action_params[2]) * 0.3,
                }
                target = net
            else:
                return state.copy()

        elif mod_type == ModificationType.ADJUST_ZONE:
            if state.zones:
                zone_key = list(state.zones.keys())[int(abs(action_params[0]) * len(state.zones)) % len(state.zones)]
                params = {
                    'clearance': 0.2 + abs(action_params[1]) * 0.3,
                    'thermal_gap': 0.3 + abs(action_params[2]) * 0.5,
                }
                target = zone_key
            else:
                return state.copy()

        else:
            # Default: clearance adjustment
            params = {
                'param_name': 'signal_clearance',
                'value': 0.15 + abs(action_params[0]) * 0.1,
            }
            target = 'signal_clearance'
            mod_type = ModificationType.ADJUST_CLEARANCE

        # Create and apply modification
        mod = PCBModification(
            mod_type=mod_type,
            target=target,
            parameters=params,
            description=f"Gaming AI action (category={action_category})",
            source_agent="gaming_ai",
            confidence=0.8,
        )

        return state.apply_modification(mod)

    async def _run_gaming_ai_phase(self) -> Dict[str, Any]:
        """Run Gaming AI optimization phase."""
        print("\n[GAMING AI] Red Queen Adversarial Evolution")
        print("-" * 40)

        if self.current_state is None:
            if MAPOS_AVAILABLE:
                self.current_state = PCBState.from_file(str(self.pcb_path))
            else:
                print("ERROR: Cannot load PCB state - MAPOS not available")
                return {'error': 'MAPOS not available'}

        initial_violations = self.current_state.run_drc().total_violations
        best_violations = initial_violations

        # Build initial population
        population = [self.current_state]
        for _ in range(self.config.rq_population_size - 1):
            mutant = self.red_queen._random_mutate(self.current_state)
            if mutant:
                population.append(mutant)
            else:
                # Random modification
                action_cat, action_params = self._sample_action(self.current_state)
                mutant = self._apply_action_to_state(self.current_state, action_cat, action_params)
                population.append(mutant)

        # Run Red Queen rounds
        total_champions = 0
        for rq_round in range(self.config.rq_rounds):
            round_result = await self.red_queen.run_round(population, rq_round)

            if round_result.champions:
                total_champions += len(round_result.champions)

                # Get best champion
                best_champion = max(round_result.champions, key=lambda c: c.fitness)

                # Evaluate
                if hasattr(best_champion.solution, 'run_drc'):
                    try:
                        drc = best_champion.solution.run_drc()
                        if drc.total_violations < best_violations:
                            best_violations = drc.total_violations
                            self.best_state = best_champion.solution
                            self.best_violations = best_violations
                            print(f"  Round {rq_round + 1}: New best = {best_violations} violations")
                    except Exception:
                        pass

                # Update population for next round
                population = [c.solution for c in round_result.champions[:5]]
                while len(population) < self.config.rq_population_size:
                    base = population[np.random.randint(len(population))]
                    action_cat, action_params = self._sample_action(base)
                    mutant = self._apply_action_to_state(base, action_cat, action_params)
                    population.append(mutant)

            # Check early termination
            if best_violations <= self.config.target_violations:
                print(f"\nTarget reached in round {rq_round + 1}!")
                break

        # Update archive with final population
        for solution in population:
            if hasattr(solution, 'run_drc'):
                try:
                    drc = solution.run_drc()
                    fitness = 1.0 / (1.0 + drc.total_violations / 100.0)
                    descriptor = BehavioralDescriptor.from_pcb_state(solution)
                    self.archive.add(solution, fitness, descriptor)
                except Exception:
                    pass

        # Get archive stats
        archive_stats = self.archive.get_statistics()

        return {
            'gaming_ai_enabled': True,
            'initial_violations': initial_violations,
            'final_violations': best_violations,
            'improvement': initial_violations - best_violations,
            'red_queen_rounds': len(self.red_queen.rounds_history),
            'total_champions': total_champions,
            'archive_coverage': archive_stats.coverage,
            'convergence_metrics': self.red_queen._compute_convergence_metrics(),
            'neural_network_used': self.encoder is not None,
            'training_experiences': self.experiences_collected,
        }

    async def _run_ralph_wiggum_phase(self) -> Dict[str, Any]:
        """Run Ralph Wiggum persistent refinement."""
        print("\n[GAMING AI] Ralph Wiggum Persistent Refinement")
        print("-" * 40)

        criteria = CompletionCriteria(
            target_violations=self.config.target_violations,
            target_fitness=0.9,
            max_iterations=100,
            max_stagnation=self.config.max_stagnation,
            max_duration_hours=self.config.max_duration_hours,
        )

        ralph_optimizer = RalphWiggumOptimizer(
            pcb_path=self.pcb_path,
            output_dir=self.output_dir / 'ralph_wiggum',
            criteria=criteria,
            red_queen_evolver=self.red_queen,
            use_git=False,
            llm_client=self.llm_client,
        )

        # Start from current best
        if self.best_violations < float('inf'):
            ralph_optimizer.state.best_violations = int(self.best_violations)

        result = await ralph_optimizer.run()

        if result.final_violations < self.best_violations:
            self.best_violations = result.final_violations
            # Note: Would need to track best state from Ralph Wiggum

        return {
            'status': result.status.name,
            'final_violations': result.final_violations,
            'iterations': result.iterations,
            'duration_seconds': result.duration_seconds,
        }

    async def optimize(self) -> GamingAIResult:
        """
        Run complete optimization pipeline.

        Returns:
            GamingAIResult with all metrics
        """
        import time
        start_time = time.time()

        print("\n" + "=" * 70)
        print("GAMING AI MULTI-AGENT PCB OPTIMIZATION")
        print("=" * 70)
        print(f"PCB: {self.pcb_path.name}")
        print(f"Mode: {self.mode}")
        print(f"Target: {self.config.target_violations} violations")
        print(f"Neural networks: {'enabled' if self.encoder else 'disabled'}")
        print("=" * 70)

        phases_completed = []
        gaming_ai_results: Dict[str, Any] = {
            'gaming_ai_enabled': self.config.enable_gaming_ai,
            'red_queen_rounds': 0,
            'total_champions': 0,
            'archive_coverage': 0.0,
            'convergence_metrics': {},
            'neural_network_used': self.encoder is not None,
            'training_experiences': 0,
        }

        # Phase 1: Base MAPOS (if hybrid mode)
        base_result: Optional[MAPOSResult] = None
        if self.mode in ("standard", "hybrid") and self.base_optimizer:
            print("\n[PHASE 1] Running Base MAPOS Pipeline...")
            try:
                base_result = await self.base_optimizer.optimize()
                phases_completed.extend(base_result.phases_completed)

                # Update best state
                if base_result.best_state:
                    self.best_state = base_result.best_state
                    self.best_violations = base_result.final_violations
                    self.current_state = base_result.best_state

            except Exception as e:
                print(f"  Base MAPOS failed: {e}")

        # Phase 2: Gaming AI (if enabled)
        if self.config.enable_gaming_ai and self.mode in ("gaming_ai", "hybrid"):
            # Skip if target already reached
            if self.best_violations <= self.config.target_violations:
                print("\nTarget already reached - skipping Gaming AI phase")
            else:
                # Load initial state if not set
                if self.current_state is None and MAPOS_AVAILABLE:
                    self.current_state = PCBState.from_file(str(self.pcb_path))

                # Run Red Queen phase
                rq_results = await self._run_gaming_ai_phase()
                gaming_ai_results.update(rq_results)
                phases_completed.append("red_queen")

                # Run Ralph Wiggum if still above target
                if self.best_violations > self.config.target_violations:
                    rw_results = await self._run_ralph_wiggum_phase()
                    gaming_ai_results['ralph_wiggum'] = rw_results
                    phases_completed.append("ralph_wiggum")

        # Calculate final metrics
        total_time = time.time() - start_time

        # Get initial violations
        if base_result:
            initial_violations = base_result.initial_violations
        elif self.current_state:
            initial_violations = self.current_state.run_drc().total_violations
        else:
            initial_violations = gaming_ai_results.get('initial_violations', 0)

        improvement = initial_violations - int(self.best_violations)
        improvement_pct = 100 * improvement / max(1, initial_violations)

        # Build final result
        if base_result:
            result = GamingAIResult.from_mapos_result(base_result, gaming_ai_results)
            result.final_violations = int(self.best_violations)
            result.total_improvement = improvement
            result.improvement_percent = improvement_pct
            result.phases_completed = phases_completed
            result.total_time_seconds = total_time
            result.target_reached = self.best_violations <= self.config.target_violations
        else:
            result = GamingAIResult(
                initial_violations=initial_violations,
                final_violations=int(self.best_violations),
                total_improvement=improvement,
                improvement_percent=improvement_pct,
                target_reached=self.best_violations <= self.config.target_violations,
                phases_completed=phases_completed,
                total_time_seconds=total_time,
                best_state=self.best_state,
                phase_results=gaming_ai_results,
                gaming_ai_enabled=self.config.enable_gaming_ai,
                red_queen_rounds=gaming_ai_results.get('red_queen_rounds', 0),
                total_champions=gaming_ai_results.get('total_champions', 0),
                archive_coverage=gaming_ai_results.get('archive_coverage', 0.0),
                convergence_metrics=gaming_ai_results.get('convergence_metrics', {}),
                neural_network_used=self.encoder is not None,
                training_experiences=self.experiences_collected,
            )

        # Save results
        self._save_results(result)

        # Print summary
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Initial violations: {initial_violations}")
        print(f"Final violations: {result.final_violations}")
        print(f"Improvement: {improvement} ({improvement_pct:.1f}%)")
        print(f"Target reached: {result.target_reached}")
        print(f"Phases: {', '.join(phases_completed)}")
        print(f"Duration: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Gaming AI rounds: {result.red_queen_rounds}")
        print(f"Champions evolved: {result.total_champions}")
        print("=" * 70)

        return result

    def _save_results(self, result: GamingAIResult) -> None:
        """Save optimization results."""
        result_path = self.output_dir / 'gaming_ai_result.json'

        data = {
            'timestamp': datetime.now().isoformat(),
            'pcb_path': str(self.pcb_path),
            'mode': self.mode,
            'initial_violations': result.initial_violations,
            'final_violations': result.final_violations,
            'improvement': result.total_improvement,
            'improvement_percent': result.improvement_percent,
            'target': self.config.target_violations,
            'target_reached': result.target_reached,
            'phases_completed': result.phases_completed,
            'total_time_seconds': result.total_time_seconds,
            'gaming_ai': {
                'enabled': result.gaming_ai_enabled,
                'red_queen_rounds': result.red_queen_rounds,
                'total_champions': result.total_champions,
                'archive_coverage': result.archive_coverage,
                'neural_network_used': result.neural_network_used,
                'training_experiences': result.training_experiences,
            },
            'convergence_metrics': result.convergence_metrics,
        }

        with open(result_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save experience buffer
        if self.experience_buffer and self.experiences_collected > 0:
            self.experience_buffer.save()

        # Save archive
        archive_path = self.output_dir / 'map_elites_archive.json'
        self.archive.save(archive_path)


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Gaming AI Multi-Agent PCB Optimizer"
    )
    parser.add_argument(
        "pcb_path",
        help="Path to KiCad PCB file"
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=100,
        help="Target violation count (default: 100)"
    )
    parser.add_argument(
        "--max-time", "-m",
        type=int,
        default=30,
        help="Maximum time in minutes (default: 30)"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "gaming_ai", "hybrid"],
        default="hybrid",
        help="Optimization mode (default: hybrid)"
    )
    parser.add_argument(
        "--rq-rounds",
        type=int,
        default=10,
        help="Red Queen rounds (default: 10)"
    )
    parser.add_argument(
        "--no-neural",
        action="store_true",
        help="Disable neural networks"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Neural network checkpoint path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory"
    )

    args = parser.parse_args()

    config = GamingAIConfig(
        target_violations=args.target,
        max_time_minutes=args.max_time,
        enable_gaming_ai=args.mode != "standard",
        rq_rounds=args.rq_rounds,
        use_neural_networks=not args.no_neural,
        checkpoint_path=args.checkpoint,
    )

    optimizer = GamingAIMultiAgentOptimizer(
        pcb_path=args.pcb_path,
        config=config,
        output_dir=args.output,
        mode=args.mode,
    )

    result = await optimizer.optimize()

    sys.exit(0 if result.target_reached else 1)


if __name__ == '__main__':
    asyncio.run(main())
