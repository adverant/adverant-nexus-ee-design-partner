"""
Shared pytest fixtures for Gaming AI tests.

Provides mock PCB states, neural network instances, and other shared resources.
"""

import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch
import tempfile
import json
import sys

# Add parent directories to path
TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
MAPOS_DIR = GAMING_AI_DIR.parent
if str(MAPOS_DIR) not in sys.path:
    sys.path.insert(0, str(MAPOS_DIR))
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))


# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Mock PCB State for testing without actual KiCad files
# ============================================================================

@dataclass
class MockComponentPosition:
    """Mock component position."""
    reference: str
    x: float
    y: float
    rotation: float
    layer: str
    footprint: str


@dataclass
class MockTraceSegment:
    """Mock trace segment."""
    net_name: str
    layer: str
    width: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass
class MockVia:
    """Mock via."""
    x: float
    y: float
    diameter: float
    drill: float
    net_name: str
    layers: tuple


@dataclass
class MockZone:
    """Mock zone."""
    name: str
    net_name: str
    layer: str
    clearance: float
    min_thickness: float
    thermal_gap: float
    bounds: tuple


@dataclass
class MockDRCResult:
    """Mock DRC result."""
    total_violations: int
    errors: int
    warnings: int
    unconnected: int
    violations_by_type: Dict[str, int]
    top_violations: List[Dict]

    @property
    def fitness_score(self) -> float:
        """Calculate fitness score."""
        base_score = 1.0 / (1.0 + self.total_violations / 100)
        error_bonus = 0.1 if self.errors < 50 else 0
        connection_bonus = 0.1 if self.unconnected < 5 else 0
        return min(1.0, base_score + error_bonus + connection_bonus)


class MockPCBState:
    """Mock PCB state for testing."""

    def __init__(
        self,
        num_components: int = 10,
        num_traces: int = 50,
        num_vias: int = 20,
        num_zones: int = 4,
        violations: int = 100,
    ):
        self.state_id = f"mock_{np.random.randint(10000)}"
        self.parent_id = None
        self.generation = 0

        # Generate mock components
        self.components: Dict[str, MockComponentPosition] = {}
        for i in range(num_components):
            ref = f"U{i+1}"
            self.components[ref] = MockComponentPosition(
                reference=ref,
                x=10 + i * 20,
                y=10 + (i % 3) * 20,
                rotation=(i * 45) % 360,
                layer="F.Cu" if i % 2 == 0 else "B.Cu",
                footprint=f"Package_SO:SOIC-{8 + (i % 3) * 8}",
            )

        # Generate mock traces
        self.traces: List[MockTraceSegment] = []
        for i in range(num_traces):
            self.traces.append(MockTraceSegment(
                net_name=f"Net{i % 10}",
                layer=f"In{(i % 4) + 1}.Cu",
                width=0.2 + (i % 5) * 0.1,
                start_x=float(i * 5),
                start_y=float((i * 3) % 100),
                end_x=float(i * 5 + 10),
                end_y=float((i * 3 + 10) % 100),
            ))

        # Generate mock vias
        self.vias: List[MockVia] = []
        for i in range(num_vias):
            self.vias.append(MockVia(
                x=float(i * 10),
                y=float((i * 7) % 80),
                diameter=0.8,
                drill=0.4,
                net_name=f"Net{i % 10}",
                layers=("F.Cu", "B.Cu"),
            ))

        # Generate mock zones
        self.zones: Dict[str, MockZone] = {}
        zone_names = ["GND_F.Cu", "GND_B.Cu", "+3V3_F.Cu", "+3V3_B.Cu"]
        for i, name in enumerate(zone_names[:num_zones]):
            self.zones[name] = MockZone(
                name=name.split("_")[0],
                net_name=name.split("_")[0],
                layer=name.split("_")[1] if "_" in name else "F.Cu",
                clearance=0.25,
                min_thickness=0.25,
                thermal_gap=0.5,
                bounds=(0, 0, 250, 85),
            )

        # Mock nets
        self.nets = {f"Net{i}": i for i in range(10)}
        self.nets["GND"] = 0
        self.nets["+3V3"] = 1

        # Parameters
        self.parameters = {
            'signal_trace_width': 0.25,
            'power_trace_width': 1.0,
            'hv_trace_width': 2.0,
            'via_diameter': 0.8,
            'via_drill': 0.4,
            'zone_clearance': 0.25,
        }

        # Cached DRC result
        self._violations = violations
        self._drc_result: Optional[MockDRCResult] = None
        self._fitness: Optional[float] = None

        # Modifications
        self.modifications: List[Any] = []

    def run_drc(self) -> MockDRCResult:
        """Return mock DRC result."""
        if self._drc_result is None:
            self._drc_result = MockDRCResult(
                total_violations=self._violations,
                errors=self._violations // 2,
                warnings=self._violations // 4,
                unconnected=self._violations // 10,
                violations_by_type={
                    'clearance': self._violations // 3,
                    'track_width': self._violations // 4,
                    'via_hole': self._violations // 5,
                    'silk_over_copper': self._violations // 6,
                },
                top_violations=[],
            )
        return self._drc_result

    @property
    def fitness(self) -> float:
        """Get fitness score."""
        if self._fitness is None:
            self._fitness = self.run_drc().fitness_score
        return self._fitness

    def copy(self) -> 'MockPCBState':
        """Create a copy of this state."""
        new_state = MockPCBState(
            num_components=len(self.components),
            num_traces=len(self.traces),
            num_vias=len(self.vias),
            num_zones=len(self.zones),
            violations=self._violations,
        )
        new_state.parent_id = self.state_id
        new_state.generation = self.generation + 1
        new_state.components = dict(self.components)
        new_state.traces = list(self.traces)
        new_state.vias = list(self.vias)
        new_state.zones = dict(self.zones)
        new_state.parameters = dict(self.parameters)
        new_state.modifications = list(self.modifications)
        return new_state

    def apply_modification(self, mod: Any) -> 'MockPCBState':
        """Apply modification and return new state."""
        new_state = self.copy()
        new_state.modifications.append(mod)
        # Randomly improve violations slightly
        if np.random.random() < 0.3:
            new_state._violations = max(0, new_state._violations - np.random.randint(1, 5))
        return new_state

    def get_hash(self) -> str:
        """Get state hash."""
        import hashlib
        state_str = f"{self.state_id}_{len(self.modifications)}_{self._violations}"
        return hashlib.md5(state_str.encode()).hexdigest()[:12]

    def save_to_file(self, path: str) -> Path:
        """Save mock state to file."""
        data = {
            'state_id': self.state_id,
            'components': len(self.components),
            'violations': self._violations,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        return Path(path)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'state_id': self.state_id,
            'parent_id': self.parent_id,
            'generation': self.generation,
            'component_count': len(self.components),
            'via_count': len(self.vias),
            'trace_count': len(self.traces),
            'zone_count': len(self.zones),
            'violations': self._violations,
        }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_pcb_state():
    """Create a mock PCB state."""
    return MockPCBState(
        num_components=20,
        num_traces=100,
        num_vias=40,
        num_zones=6,
        violations=500,
    )


@pytest.fixture
def mock_pcb_state_low_violations():
    """Create a mock PCB state with low violations."""
    return MockPCBState(
        num_components=20,
        num_traces=100,
        num_vias=40,
        num_zones=6,
        violations=25,
    )


@pytest.fixture
def mock_pcb_state_high_violations():
    """Create a mock PCB state with high violations."""
    return MockPCBState(
        num_components=50,
        num_traces=200,
        num_vias=80,
        num_zones=8,
        violations=2000,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pcb_file(temp_dir):
    """Create a mock PCB file."""
    pcb_content = '''(kicad_pcb (version 20230620) (generator pcbnew)
  (general
    (thickness 1.6)
  )
  (paper "A4")
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
  (net 0 "")
  (net 1 "GND")
  (net 2 "+3V3")
  (footprint "Package_SO:SOIC-8"
    (layer "F.Cu")
    (at 100 50)
    (property "Reference" "U1")
  )
  (segment (start 100 50) (end 110 60) (width 0.25) (layer "F.Cu") (net 1))
  (via (at 105 55) (size 0.8) (drill 0.4) (layers "F.Cu" "B.Cu") (net 1))
)'''
    pcb_path = temp_dir / "test_board.kicad_pcb"
    with open(pcb_path, 'w') as f:
        f.write(pcb_content)
    return pcb_path


@pytest.fixture
def random_embedding():
    """Generate a random embedding vector."""
    return np.random.randn(256).astype(np.float32)


@pytest.fixture
def random_drc_context():
    """Generate random DRC context features."""
    return np.random.randn(12).astype(np.float32)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()

    async def mock_generate(prompt, **kwargs):
        return MagicMock(text="Move component U1 by 0.5mm to the right to reduce clearance violation.")

    client.generate = mock_generate
    return client


# ============================================================================
# PyTorch-specific fixtures
# ============================================================================

@pytest.fixture
def torch_device():
    """Get appropriate PyTorch device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.device("cpu")


@pytest.fixture
def pcb_graph_encoder(torch_device):
    """Create PCB graph encoder."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from ..pcb_graph_encoder import PCBGraphEncoder
    return PCBGraphEncoder(hidden_dim=64, num_layers=2).to(torch_device)


@pytest.fixture
def value_network(torch_device):
    """Create value network."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from ..value_network import ValueNetwork
    return ValueNetwork(input_dim=64, hidden_dim=128, num_layers=2).to(torch_device)


@pytest.fixture
def policy_network(torch_device):
    """Create policy network."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from ..policy_network import PolicyNetwork
    return PolicyNetwork(input_dim=64, hidden_dim=128).to(torch_device)


@pytest.fixture
def world_model(torch_device):
    """Create world model."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from ..dynamics_network import WorldModel
    return WorldModel(observation_dim=64, latent_dim=64).to(torch_device)


@pytest.fixture
def map_elites_archive():
    """Create MAP-Elites archive."""
    from ..map_elites import MAPElitesArchive
    return MAPElitesArchive(dimensions=10, bins_per_dimension=5)


@pytest.fixture
def experience_buffer(temp_dir):
    """Create experience buffer."""
    from ..training import ExperienceBuffer
    return ExperienceBuffer(
        capacity=1000,
        save_path=temp_dir / "experiences.json",
    )


# ============================================================================
# Helper functions
# ============================================================================

def create_mock_population(size: int = 10, base_violations: int = 500) -> List[MockPCBState]:
    """Create a population of mock PCB states."""
    population = []
    for i in range(size):
        state = MockPCBState(
            num_components=10 + i,
            num_traces=50 + i * 5,
            num_vias=20 + i * 2,
            violations=base_violations - i * 10,
        )
        population.append(state)
    return population


def create_random_experiences(count: int = 100, dim: int = 256) -> List[dict]:
    """Create random experience data for testing."""
    experiences = []
    for i in range(count):
        exp = {
            'state_embedding': np.random.randn(dim).astype(np.float32),
            'drc_context': np.random.randn(12).astype(np.float32),
            'action_category': np.random.randint(0, 9),
            'action_params': np.random.randn(5).astype(np.float32),
            'reward': np.random.randn() * 10,
            'next_state_embedding': np.random.randn(dim).astype(np.float32),
            'done': np.random.random() < 0.1,
            'value_target': np.random.random(),
            'policy_target': np.random.randn(9).astype(np.float32),
        }
        experiences.append(exp)
    return experiences
