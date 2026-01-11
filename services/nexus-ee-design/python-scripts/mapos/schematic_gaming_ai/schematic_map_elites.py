"""
Schematic MAP-Elites Archive

Quality-diversity archive for maintaining diverse, high-quality
schematic solutions organized by behavioral characteristics.
"""

import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from copy import deepcopy

import numpy as np

from .config import get_schematic_config, ArchiveConfig
from .behavior_descriptor import SchematicBehaviorDescriptor, compute_schematic_descriptor
from .fitness_function import SchematicFitness, compute_schematic_fitness

logger = logging.getLogger(__name__)


@dataclass
class SchematicArchiveCell:
    """A cell in the MAP-Elites archive containing an elite schematic."""
    schematic: Dict[str, Any]
    fitness: SchematicFitness
    descriptor: SchematicBehaviorDescriptor
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schematic": self.schematic,
            "fitness": self.fitness.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchematicArchiveCell":
        """Create from dictionary."""
        return cls(
            schematic=data["schematic"],
            fitness=SchematicFitness.from_dict(data["fitness"]),
            descriptor=SchematicBehaviorDescriptor.from_dict(data["descriptor"]),
            generation=data.get("generation", 0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            update_count=data.get("update_count", 0),
        )


@dataclass
class SchematicArchiveStatistics:
    """Statistics about the MAP-Elites archive."""
    total_cells: int = 0
    occupied_cells: int = 0
    coverage: float = 0.0
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    median_fitness: float = 0.0
    total_evaluations: int = 0
    total_replacements: int = 0
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cells": self.total_cells,
            "occupied_cells": self.occupied_cells,
            "coverage": self.coverage,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "median_fitness": self.median_fitness,
            "total_evaluations": self.total_evaluations,
            "total_replacements": self.total_replacements,
            "generation": self.generation,
        }


class SchematicMAPElitesArchive:
    """
    MAP-Elites archive for schematic quality-diversity optimization.

    Maintains a grid of elite schematics where each cell represents
    a different behavioral niche (design strategy).
    """

    def __init__(self, config: Optional[ArchiveConfig] = None):
        """
        Initialize archive.

        Args:
            config: Archive configuration
        """
        self.config = config or get_schematic_config().archive

        # Grid dimensions based on behavioral descriptor
        # 10D descriptor discretized into bins
        self.grid_dims = self._compute_grid_dims()
        self.total_cells = int(np.prod(self.grid_dims))

        # Archive storage: grid coordinates -> cell
        self.archive: Dict[Tuple[int, ...], SchematicArchiveCell] = {}

        # Statistics
        self.generation = 0
        self.total_evaluations = 0
        self.total_replacements = 0

        logger.info(f"Initialized MAP-Elites archive with {self.total_cells} cells")

    def _compute_grid_dims(self) -> Tuple[int, ...]:
        """Compute grid dimensions based on config."""
        return (
            self.config.complexity_bins,      # component_count
            self.config.complexity_bins,      # net_count
            self.config.complexity_bins,      # sheet_count
            self.config.strategy_bins,        # power_distribution
            self.config.strategy_bins,        # interface_isolation
            self.config.quality_bins,         # erc_violations (inverted)
            self.config.quality_bins,         # bp_adherence
            self.config.quality_bins,         # cost_efficiency
            self.config.manufacturing_bins,   # footprint_variety
            self.config.manufacturing_bins,   # sourcing_difficulty
        )

    def _descriptor_to_cell(self, descriptor: SchematicBehaviorDescriptor) -> Tuple[int, ...]:
        """
        Convert behavioral descriptor to grid cell coordinates.

        Args:
            descriptor: Behavioral descriptor

        Returns:
            Tuple of grid coordinates
        """
        vector = descriptor.to_vector()

        # Discretize each dimension
        coords = []
        for i, (val, bins) in enumerate(zip(vector, self.grid_dims)):
            # Clip to [0, 1] and discretize
            val = np.clip(val, 0.0, 0.9999)
            coord = int(val * bins)
            coords.append(coord)

        return tuple(coords)

    def add(
        self,
        schematic: Dict[str, Any],
        fitness: Optional[SchematicFitness] = None,
        descriptor: Optional[SchematicBehaviorDescriptor] = None
    ) -> Tuple[bool, Optional[SchematicArchiveCell]]:
        """
        Try to add schematic to archive.

        Args:
            schematic: Schematic dictionary
            fitness: Optional pre-computed fitness
            descriptor: Optional pre-computed descriptor

        Returns:
            (was_added, replaced_cell_or_None)
        """
        self.total_evaluations += 1

        # Compute fitness and descriptor if not provided
        if fitness is None:
            fitness = compute_schematic_fitness(schematic)

        if descriptor is None:
            descriptor = compute_schematic_descriptor(schematic)

        # Get cell coordinates
        cell_coords = self._descriptor_to_cell(descriptor)

        # Check if cell is empty or new solution is better
        existing = self.archive.get(cell_coords)

        if existing is None:
            # Empty cell - add solution
            new_cell = SchematicArchiveCell(
                schematic=deepcopy(schematic),
                fitness=fitness,
                descriptor=descriptor,
                generation=self.generation,
            )
            self.archive[cell_coords] = new_cell
            logger.debug(f"Added to empty cell {cell_coords}, fitness={fitness.total:.3f}")
            return True, None

        # Cell occupied - check if new is better
        improvement = fitness.total - existing.fitness.total
        threshold = self.config.elite_replacement_threshold

        if improvement > threshold:
            # Replace existing elite
            old_cell = existing
            new_cell = SchematicArchiveCell(
                schematic=deepcopy(schematic),
                fitness=fitness,
                descriptor=descriptor,
                generation=self.generation,
                update_count=existing.update_count + 1,
            )
            self.archive[cell_coords] = new_cell
            self.total_replacements += 1
            logger.debug(
                f"Replaced cell {cell_coords}, "
                f"fitness {existing.fitness.total:.3f} -> {fitness.total:.3f}"
            )
            return True, old_cell

        return False, None

    def get_best(self) -> Optional[SchematicArchiveCell]:
        """Get the cell with highest fitness."""
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda c: c.fitness.total)

    def get_random_elite(self) -> Optional[SchematicArchiveCell]:
        """Get a random elite from the archive."""
        if not self.archive:
            return None
        coords = list(self.archive.keys())
        return self.archive[coords[np.random.randint(len(coords))]]

    def get_diverse_sample(self, n: int) -> List[SchematicArchiveCell]:
        """
        Get n diverse elites from archive.

        Uses farthest point sampling in behavioral space.
        """
        if not self.archive:
            return []

        cells = list(self.archive.values())
        if len(cells) <= n:
            return cells

        # Start with best elite
        sample = [self.get_best()]
        sample_vectors = [sample[0].descriptor.to_vector()]

        while len(sample) < n:
            # Find cell farthest from current sample
            max_min_dist = -1
            farthest = None

            for cell in cells:
                if cell in sample:
                    continue

                vec = cell.descriptor.to_vector()
                min_dist = min(
                    np.linalg.norm(vec - sv) for sv in sample_vectors
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest = cell

            if farthest:
                sample.append(farthest)
                sample_vectors.append(farthest.descriptor.to_vector())
            else:
                break

        return sample

    def get_statistics(self) -> SchematicArchiveStatistics:
        """Compute archive statistics."""
        occupied = len(self.archive)

        if occupied == 0:
            return SchematicArchiveStatistics(
                total_cells=self.total_cells,
                generation=self.generation,
                total_evaluations=self.total_evaluations,
            )

        fitnesses = [c.fitness.total for c in self.archive.values()]

        return SchematicArchiveStatistics(
            total_cells=self.total_cells,
            occupied_cells=occupied,
            coverage=occupied / self.total_cells,
            best_fitness=max(fitnesses),
            average_fitness=np.mean(fitnesses),
            median_fitness=np.median(fitnesses),
            total_evaluations=self.total_evaluations,
            total_replacements=self.total_replacements,
            generation=self.generation,
        )

    def increment_generation(self) -> None:
        """Increment generation counter."""
        self.generation += 1

    def save(self, path: Union[str, Path]) -> None:
        """Save archive to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0.0",
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "total_replacements": self.total_replacements,
            "grid_dims": self.grid_dims,
            "cells": {
                str(k): v.to_dict() for k, v in self.archive.items()
            }
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved archive to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load archive from JSON file."""
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        self.generation = data.get("generation", 0)
        self.total_evaluations = data.get("total_evaluations", 0)
        self.total_replacements = data.get("total_replacements", 0)

        self.archive = {}
        for k, v in data.get("cells", {}).items():
            # Parse string key back to tuple
            coords = tuple(map(int, k.strip("()").split(",")))
            self.archive[coords] = SchematicArchiveCell.from_dict(v)

        logger.info(f"Loaded archive from {path}: {len(self.archive)} cells")

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire archive to dictionary."""
        return {
            "statistics": self.get_statistics().to_dict(),
            "cells": [c.to_dict() for c in self.archive.values()],
        }
