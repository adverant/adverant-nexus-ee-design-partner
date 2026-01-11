"""
MAP-Elites Archive - Quality-Diversity for PCB Optimization

This module implements the MAP-Elites algorithm, which maintains an archive
of diverse, high-quality solutions organized by behavioral characteristics.

Key concepts:
- Behavioral Descriptor: How a solution "behaves" (e.g., routing density, via count)
- Archive Cell: Grid position determined by behavioral descriptor
- Elite: Best solution found for a specific behavioral cell

The archive enables:
1. Diverse exploration: Solutions cover different strategies
2. Stepping stones: Sub-optimal solutions can lead to better ones
3. Quality-diversity: Maintain both performance and variety

References:
- MAP-Elites: https://arxiv.org/abs/1504.04909
- Digital Red Queen: https://arxiv.org/abs/2601.03335
- Quality-Diversity: https://quality-diversity.github.io/
"""

import math
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Callable, Union
from datetime import datetime
from collections import defaultdict

import numpy as np


@dataclass
class BehavioralDescriptor:
    """
    Behavioral characteristics of a PCB design.

    These descriptors capture HOW a design achieves its goals,
    not just how well. Different descriptors can have similar
    fitness but represent fundamentally different strategies.
    """

    # Core routing characteristics
    routing_density: float          # Total trace length / board area
    via_count: int                  # Number of vias
    layer_utilization: float        # Fraction of layers used
    zone_coverage: float            # Power/ground zone coverage

    # Design strategy indicators
    thermal_spread: float           # Heat distribution variance
    signal_length_variance: float   # Variance in critical signal lengths
    component_clustering: float     # How clustered components are
    power_path_directness: float    # Efficiency of power distribution

    # DRC-related
    min_clearance_ratio: float      # Actual / required clearance
    silk_density: float             # Silkscreen area density

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for computation."""
        return np.array([
            self.routing_density,
            self.via_count / 500.0,  # Normalize
            self.layer_utilization,
            self.zone_coverage,
            self.thermal_spread,
            self.signal_length_variance,
            self.component_clustering,
            self.power_path_directness,
            self.min_clearance_ratio,
            self.silk_density,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'routing_density': self.routing_density,
            'via_count': self.via_count,
            'layer_utilization': self.layer_utilization,
            'zone_coverage': self.zone_coverage,
            'thermal_spread': self.thermal_spread,
            'signal_length_variance': self.signal_length_variance,
            'component_clustering': self.component_clustering,
            'power_path_directness': self.power_path_directness,
            'min_clearance_ratio': self.min_clearance_ratio,
            'silk_density': self.silk_density,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BehavioralDescriptor':
        """Create from dictionary."""
        return cls(
            routing_density=data.get('routing_density', 0.0),
            via_count=int(data.get('via_count', 0)),
            layer_utilization=data.get('layer_utilization', 0.0),
            zone_coverage=data.get('zone_coverage', 0.0),
            thermal_spread=data.get('thermal_spread', 0.0),
            signal_length_variance=data.get('signal_length_variance', 0.0),
            component_clustering=data.get('component_clustering', 0.0),
            power_path_directness=data.get('power_path_directness', 0.0),
            min_clearance_ratio=data.get('min_clearance_ratio', 0.0),
            silk_density=data.get('silk_density', 0.0),
        )

    @classmethod
    def from_pcb_state(cls, pcb_state: Any) -> 'BehavioralDescriptor':
        """
        Extract behavioral descriptor from a PCBState.

        Args:
            pcb_state: PCBState object from pcb_state.py

        Returns:
            BehavioralDescriptor characterizing the design
        """
        # Board dimensions (default to reference PCB)
        board_width = 250.0
        board_height = 85.0
        board_area = board_width * board_height

        # Routing density: total trace length / board area
        total_trace_length = sum(
            math.sqrt((t.end_x - t.start_x)**2 + (t.end_y - t.start_y)**2)
            for t in pcb_state.traces
        ) if hasattr(pcb_state, 'traces') else 0.0
        routing_density = total_trace_length / board_area

        # Via count
        via_count = len(pcb_state.vias) if hasattr(pcb_state, 'vias') else 0

        # Layer utilization: unique layers used / total layers
        layers_used = set()
        for trace in getattr(pcb_state, 'traces', []):
            layers_used.add(trace.layer)
        for via in getattr(pcb_state, 'vias', []):
            if hasattr(via, 'layers'):
                layers_used.update(via.layers)
        layer_utilization = len(layers_used) / 10.0  # Assuming 10-layer board

        # Zone coverage: total zone area / board area
        zone_area = 0.0
        for zone in getattr(pcb_state, 'zones', {}).values():
            if hasattr(zone, 'bounds'):
                w = zone.bounds[2] - zone.bounds[0]
                h = zone.bounds[3] - zone.bounds[1]
                zone_area += w * h
        zone_coverage = min(1.0, zone_area / board_area)

        # Thermal spread: variance in component positions (proxy for heat distribution)
        comp_positions = []
        for comp in getattr(pcb_state, 'components', {}).values():
            comp_positions.append((comp.x, comp.y))

        if len(comp_positions) > 1:
            positions = np.array(comp_positions)
            thermal_spread = np.std(positions[:, 0]) * np.std(positions[:, 1]) / 1000.0
        else:
            thermal_spread = 0.0

        # Signal length variance: variance in trace lengths per net
        net_lengths: Dict[str, List[float]] = defaultdict(list)
        for trace in getattr(pcb_state, 'traces', []):
            length = math.sqrt((trace.end_x - trace.start_x)**2 +
                               (trace.end_y - trace.start_y)**2)
            net_lengths[trace.net_name].append(length)

        if net_lengths:
            total_lengths = [sum(lengths) for lengths in net_lengths.values()]
            signal_length_variance = np.std(total_lengths) / 100.0 if total_lengths else 0.0
        else:
            signal_length_variance = 0.0

        # Component clustering: inverse of average distance between components
        if len(comp_positions) > 1:
            distances = []
            for i, (x1, y1) in enumerate(comp_positions):
                for j, (x2, y2) in enumerate(comp_positions[i+1:], i+1):
                    distances.append(math.sqrt((x2-x1)**2 + (y2-y1)**2))
            avg_distance = np.mean(distances) if distances else 100.0
            component_clustering = 1.0 / (1.0 + avg_distance / 50.0)
        else:
            component_clustering = 0.5

        # Power path directness: ratio of shortest path to actual path for power nets
        # Simplified: use average trace width as proxy (wider = more direct power)
        power_traces = [t for t in getattr(pcb_state, 'traces', [])
                        if 'V' in t.net_name.upper() or 'GND' in t.net_name.upper()]
        if power_traces:
            avg_power_width = np.mean([t.width for t in power_traces])
            power_path_directness = min(1.0, avg_power_width / 2.0)
        else:
            power_path_directness = 0.5

        # Clearance ratio: actual parameters vs required
        params = getattr(pcb_state, 'parameters', {})
        actual_clearance = params.get('signal_clearance', 0.15)
        min_clearance_ratio = actual_clearance / 0.1  # 0.1mm is typical minimum

        # Silk density: estimate from component count and footprints
        silk_density = min(1.0, len(getattr(pcb_state, 'components', {})) / 200.0)

        return cls(
            routing_density=float(routing_density),
            via_count=via_count,
            layer_utilization=float(layer_utilization),
            zone_coverage=float(zone_coverage),
            thermal_spread=float(thermal_spread),
            signal_length_variance=float(signal_length_variance),
            component_clustering=float(component_clustering),
            power_path_directness=float(power_path_directness),
            min_clearance_ratio=float(min_clearance_ratio),
            silk_density=float(silk_density),
        )


@dataclass
class ArchiveCell:
    """
    A cell in the MAP-Elites archive.

    Each cell stores the best solution found for a particular
    behavioral characteristic combination.
    """
    indices: Tuple[int, ...]         # Grid indices for this cell
    elite: Optional[Any] = None      # Best solution (PCBState or similar)
    fitness: float = float('-inf')   # Fitness of the elite
    descriptor: Optional[BehavioralDescriptor] = None
    visits: int = 0                  # Number of times this cell was visited
    improvements: int = 0            # Number of times elite was replaced
    last_updated: str = ""           # Timestamp of last update
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self, solution: Any, fitness: float, descriptor: BehavioralDescriptor) -> bool:
        """
        Try to update this cell with a new solution.

        Returns True if the solution was accepted (better than current elite).
        """
        self.visits += 1

        if fitness > self.fitness:
            self.elite = solution
            self.fitness = fitness
            self.descriptor = descriptor
            self.improvements += 1
            self.last_updated = datetime.now().isoformat()
            return True

        return False

    def is_empty(self) -> bool:
        """Check if cell has no elite."""
        return self.elite is None


@dataclass
class ArchiveStatistics:
    """Statistics about the MAP-Elites archive."""
    total_cells: int                  # Total number of cells in grid
    filled_cells: int                 # Number of cells with elites
    coverage: float                   # Fraction of cells filled
    avg_fitness: float                # Average fitness of elites
    max_fitness: float                # Best fitness in archive
    min_fitness: float                # Worst fitness in archive
    total_visits: int                 # Total visits across all cells
    total_improvements: int           # Total improvements across all cells
    diversity_score: float            # Measure of behavioral diversity


class MAPElitesArchive:
    """
    MAP-Elites archive for quality-diversity optimization.

    The archive is organized as a multi-dimensional grid where each
    dimension corresponds to a behavioral characteristic. Each cell
    stores the best solution found with those characteristics.

    Key features:
    - Automatically discretizes continuous behavioral descriptors
    - Maintains diversity through behavioral niching
    - Supports custom fitness functions and descriptors
    """

    # Default behavioral dimensions to use for archive
    DEFAULT_DIMENSIONS = [
        ('routing_density', 0.0, 2.0, 10),      # (name, min, max, bins)
        ('via_count', 0, 500, 10),
        ('zone_coverage', 0.0, 1.0, 5),
        ('component_clustering', 0.0, 1.0, 5),
    ]

    def __init__(
        self,
        dimensions: Optional[List[Tuple[str, float, float, int]]] = None,
        fitness_fn: Optional[Callable[[Any], float]] = None,
        descriptor_fn: Optional[Callable[[Any], BehavioralDescriptor]] = None,
    ):
        """
        Initialize MAP-Elites archive.

        Args:
            dimensions: List of (name, min, max, bins) for each behavioral dimension
            fitness_fn: Function to compute fitness from solution
            descriptor_fn: Function to compute behavioral descriptor from solution
        """
        self.dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self.dimension_names = [d[0] for d in self.dimensions]
        self.dimension_mins = np.array([d[1] for d in self.dimensions])
        self.dimension_maxs = np.array([d[2] for d in self.dimensions])
        self.dimension_bins = [d[3] for d in self.dimensions]

        # Create grid shape
        self.grid_shape = tuple(self.dimension_bins)
        self.total_cells = np.prod(self.grid_shape)

        # Initialize empty archive
        self.archive: Dict[Tuple[int, ...], ArchiveCell] = {}

        # Fitness and descriptor functions
        self.fitness_fn = fitness_fn or self._default_fitness
        self.descriptor_fn = descriptor_fn or BehavioralDescriptor.from_pcb_state

        # History tracking
        self.insertion_history: List[Dict[str, Any]] = []

    def _default_fitness(self, solution: Any) -> float:
        """Default fitness function using DRC results."""
        if hasattr(solution, 'run_drc'):
            drc = solution.run_drc()
            return 1.0 / (1.0 + drc.total_violations / 100.0)
        elif hasattr(solution, 'fitness'):
            return solution.fitness
        else:
            return 0.0

    def _discretize(self, descriptor: BehavioralDescriptor) -> Tuple[int, ...]:
        """
        Convert behavioral descriptor to grid indices.

        Returns:
            Tuple of indices for each dimension
        """
        indices = []
        descriptor_dict = descriptor.to_dict()

        for name, min_val, max_val, bins in self.dimensions:
            value = descriptor_dict.get(name, 0.0)

            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val + 1e-8)
            normalized = np.clip(normalized, 0.0, 1.0)

            # Discretize to bin index
            idx = int(normalized * (bins - 1))
            idx = np.clip(idx, 0, bins - 1)
            indices.append(idx)

        return tuple(indices)

    def add(
        self,
        solution: Any,
        fitness: Optional[float] = None,
        descriptor: Optional[BehavioralDescriptor] = None,
    ) -> bool:
        """
        Try to add a solution to the archive.

        Args:
            solution: The solution to add
            fitness: Pre-computed fitness (computed if not provided)
            descriptor: Pre-computed descriptor (computed if not provided)

        Returns:
            True if solution was added (improved a cell)
        """
        # Compute fitness if not provided
        if fitness is None:
            fitness = self.fitness_fn(solution)

        # Compute descriptor if not provided
        if descriptor is None:
            descriptor = self.descriptor_fn(solution)

        # Get cell indices
        indices = self._discretize(descriptor)

        # Get or create cell
        if indices not in self.archive:
            self.archive[indices] = ArchiveCell(indices=indices)

        cell = self.archive[indices]

        # Try to update
        accepted = cell.update(solution, fitness, descriptor)

        # Record history
        self.insertion_history.append({
            'timestamp': datetime.now().isoformat(),
            'indices': indices,
            'fitness': fitness,
            'accepted': accepted,
        })

        return accepted

    def get_elite(self, indices: Tuple[int, ...]) -> Optional[Any]:
        """Get elite solution at specific indices."""
        if indices in self.archive:
            return self.archive[indices].elite
        return None

    def get_all_elites(self) -> List[Any]:
        """Get all elite solutions in the archive."""
        return [cell.elite for cell in self.archive.values() if not cell.is_empty()]

    def get_best_elite(self) -> Optional[Any]:
        """Get the elite with highest fitness."""
        if not self.archive:
            return None

        best_cell = max(self.archive.values(), key=lambda c: c.fitness)
        return best_cell.elite

    def get_pareto_front(self, objectives: List[str] = None) -> List[Any]:
        """
        Get Pareto-optimal solutions.

        For each pair of solutions, check if one dominates the other
        across multiple objectives.
        """
        if objectives is None:
            objectives = ['fitness', 'routing_density', 'zone_coverage']

        elites = []
        for cell in self.archive.values():
            if cell.is_empty():
                continue

            obj_values = {'fitness': cell.fitness}
            if cell.descriptor:
                obj_values.update(cell.descriptor.to_dict())

            elites.append((cell.elite, obj_values))

        # Find non-dominated solutions
        pareto_front = []
        for i, (elite_i, obj_i) in enumerate(elites):
            dominated = False
            for j, (elite_j, obj_j) in enumerate(elites):
                if i == j:
                    continue

                # Check if j dominates i
                better_in_all = all(
                    obj_j.get(o, 0) >= obj_i.get(o, 0) for o in objectives
                )
                strictly_better = any(
                    obj_j.get(o, 0) > obj_i.get(o, 0) for o in objectives
                )

                if better_in_all and strictly_better:
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(elite_i)

        return pareto_front

    def sample(self, strategy: str = 'uniform') -> Optional[Any]:
        """
        Sample a solution from the archive.

        Args:
            strategy: Sampling strategy
                - 'uniform': Uniform random from filled cells
                - 'fitness_weighted': Probability proportional to fitness
                - 'curiosity': Favor less-visited cells

        Returns:
            Sampled solution or None if archive is empty
        """
        filled_cells = [c for c in self.archive.values() if not c.is_empty()]

        if not filled_cells:
            return None

        if strategy == 'uniform':
            cell = np.random.choice(filled_cells)

        elif strategy == 'fitness_weighted':
            fitnesses = np.array([c.fitness for c in filled_cells])
            fitnesses = fitnesses - fitnesses.min() + 1e-8  # Shift to positive
            probs = fitnesses / fitnesses.sum()
            idx = np.random.choice(len(filled_cells), p=probs)
            cell = filled_cells[idx]

        elif strategy == 'curiosity':
            # Favor cells with fewer visits
            visits = np.array([c.visits for c in filled_cells])
            curiosity = 1.0 / (visits + 1)
            probs = curiosity / curiosity.sum()
            idx = np.random.choice(len(filled_cells), p=probs)
            cell = filled_cells[idx]

        else:
            cell = np.random.choice(filled_cells)

        return cell.elite

    def get_unexplored_regions(self) -> List[Tuple[int, ...]]:
        """
        Find grid regions with no elites.

        Useful for guiding exploration toward novel behaviors.
        """
        unexplored = []

        # Generate all possible indices
        from itertools import product
        for indices in product(*[range(b) for b in self.dimension_bins]):
            if indices not in self.archive or self.archive[indices].is_empty():
                unexplored.append(indices)

        return unexplored

    def get_statistics(self) -> ArchiveStatistics:
        """Compute archive statistics."""
        filled_cells = [c for c in self.archive.values() if not c.is_empty()]

        if not filled_cells:
            return ArchiveStatistics(
                total_cells=self.total_cells,
                filled_cells=0,
                coverage=0.0,
                avg_fitness=0.0,
                max_fitness=0.0,
                min_fitness=0.0,
                total_visits=0,
                total_improvements=0,
                diversity_score=0.0,
            )

        fitnesses = [c.fitness for c in filled_cells]
        visits = [c.visits for c in filled_cells]
        improvements = [c.improvements for c in filled_cells]

        # Diversity score: average pairwise distance in behavior space
        if len(filled_cells) > 1:
            descriptors = [c.descriptor.to_vector() for c in filled_cells if c.descriptor]
            if len(descriptors) > 1:
                desc_array = np.stack(descriptors)
                distances = []
                for i in range(len(desc_array)):
                    for j in range(i + 1, len(desc_array)):
                        dist = np.linalg.norm(desc_array[i] - desc_array[j])
                        distances.append(dist)
                diversity_score = np.mean(distances) if distances else 0.0
            else:
                diversity_score = 0.0
        else:
            diversity_score = 0.0

        return ArchiveStatistics(
            total_cells=self.total_cells,
            filled_cells=len(filled_cells),
            coverage=len(filled_cells) / self.total_cells,
            avg_fitness=float(np.mean(fitnesses)),
            max_fitness=float(np.max(fitnesses)),
            min_fitness=float(np.min(fitnesses)),
            total_visits=sum(visits),
            total_improvements=sum(improvements),
            diversity_score=float(diversity_score),
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save archive to disk.

        Note: Only saves metadata and descriptors, not actual solutions
        (which may be large PCBState objects).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'dimensions': self.dimensions,
            'grid_shape': self.grid_shape,
            'cells': [],
        }

        for indices, cell in self.archive.items():
            cell_data = {
                'indices': list(indices),
                'fitness': cell.fitness,
                'descriptor': cell.descriptor.to_dict() if cell.descriptor else None,
                'visits': cell.visits,
                'improvements': cell.improvements,
                'last_updated': cell.last_updated,
                'metadata': cell.metadata,
            }
            data['cells'].append(cell_data)

        data['statistics'] = {
            'filled_cells': len([c for c in self.archive.values() if not c.is_empty()]),
            'total_insertions': len(self.insertion_history),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MAPElitesArchive':
        """Load archive from disk (metadata only)."""
        with open(path) as f:
            data = json.load(f)

        archive = cls(dimensions=data['dimensions'])

        for cell_data in data['cells']:
            indices = tuple(cell_data['indices'])
            cell = ArchiveCell(
                indices=indices,
                fitness=cell_data['fitness'],
                descriptor=BehavioralDescriptor.from_dict(cell_data['descriptor'])
                    if cell_data['descriptor'] else None,
                visits=cell_data['visits'],
                improvements=cell_data['improvements'],
                last_updated=cell_data['last_updated'],
                metadata=cell_data.get('metadata', {}),
            )
            archive.archive[indices] = cell

        return archive

    def visualize(self, dim1: int = 0, dim2: int = 1) -> str:
        """
        Create ASCII visualization of 2D slice of archive.

        Args:
            dim1: First dimension to show (x-axis)
            dim2: Second dimension to show (y-axis)

        Returns:
            ASCII art representation
        """
        bins1 = self.dimension_bins[dim1]
        bins2 = self.dimension_bins[dim2]

        grid = [['.' for _ in range(bins1)] for _ in range(bins2)]

        for indices, cell in self.archive.items():
            if cell.is_empty():
                continue

            x = indices[dim1]
            y = indices[dim2]

            # Fitness-based character
            if cell.fitness >= 0.8:
                char = '#'
            elif cell.fitness >= 0.6:
                char = '*'
            elif cell.fitness >= 0.4:
                char = '+'
            else:
                char = 'o'

            grid[bins2 - 1 - y][x] = char  # Flip y for display

        # Build output
        name1 = self.dimension_names[dim1]
        name2 = self.dimension_names[dim2]

        lines = [f"MAP-Elites Archive ({name1} x {name2})"]
        lines.append("-" * (bins1 + 4))

        for row in grid:
            lines.append(f"| {''.join(row)} |")

        lines.append("-" * (bins1 + 4))
        lines.append(f"Legend: #=0.8+, *=0.6+, +=0.4+, o=<0.4, .=empty")

        return "\n".join(lines)


if __name__ == '__main__':
    print("MAP-Elites Archive Test")
    print("=" * 60)

    # Create archive with custom dimensions
    archive = MAPElitesArchive(
        dimensions=[
            ('routing_density', 0.0, 2.0, 10),
            ('via_count', 0, 500, 10),
            ('zone_coverage', 0.0, 1.0, 5),
        ]
    )

    print(f"Grid shape: {archive.grid_shape}")
    print(f"Total cells: {archive.total_cells}")

    # Add some test solutions
    class MockSolution:
        def __init__(self, fitness, density, vias, coverage):
            self._fitness = fitness
            self.routing_density = density
            self.via_count = vias
            self.zone_coverage = coverage

        @property
        def fitness(self):
            return self._fitness

    # Add solutions with different behaviors
    for i in range(50):
        fitness = np.random.random()
        density = np.random.uniform(0, 2)
        vias = np.random.randint(0, 500)
        coverage = np.random.random()

        solution = MockSolution(fitness, density, vias, coverage)

        descriptor = BehavioralDescriptor(
            routing_density=density,
            via_count=vias,
            layer_utilization=0.5,
            zone_coverage=coverage,
            thermal_spread=0.3,
            signal_length_variance=0.2,
            component_clustering=0.5,
            power_path_directness=0.6,
            min_clearance_ratio=1.2,
            silk_density=0.3,
        )

        accepted = archive.add(solution, fitness, descriptor)
        if accepted:
            print(f"  Added solution {i}: fitness={fitness:.3f}, "
                  f"density={density:.2f}, vias={vias}")

    # Get statistics
    stats = archive.get_statistics()
    print(f"\nArchive Statistics:")
    print(f"  Filled cells: {stats.filled_cells}/{stats.total_cells} ({stats.coverage:.1%})")
    print(f"  Fitness: avg={stats.avg_fitness:.3f}, max={stats.max_fitness:.3f}")
    print(f"  Total visits: {stats.total_visits}")
    print(f"  Diversity score: {stats.diversity_score:.3f}")

    # Sample solutions
    print(f"\nSampling:")
    for strategy in ['uniform', 'fitness_weighted', 'curiosity']:
        sample = archive.sample(strategy)
        if sample:
            print(f"  {strategy}: fitness={sample.fitness:.3f}")

    # Visualize
    print(f"\n{archive.visualize(0, 1)}")

    # Get Pareto front
    pareto = archive.get_pareto_front()
    print(f"\nPareto front size: {len(pareto)}")

    # Unexplored regions
    unexplored = archive.get_unexplored_regions()
    print(f"Unexplored regions: {len(unexplored)}")
