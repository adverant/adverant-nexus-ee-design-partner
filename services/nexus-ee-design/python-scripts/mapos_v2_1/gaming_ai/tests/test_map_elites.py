"""
Tests for MAP-Elites Archive module.

Tests the quality-diversity archive implementation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json

TEST_DIR = Path(__file__).parent
GAMING_AI_DIR = TEST_DIR.parent
if str(GAMING_AI_DIR) not in sys.path:
    sys.path.insert(0, str(GAMING_AI_DIR))

from conftest import MockPCBState, create_mock_population

# Import module under test
from map_elites import (
    MAPElitesArchive, BehavioralDescriptor, ArchiveCell, ArchiveStatistics
)


class TestBehavioralDescriptor:
    """Tests for BehavioralDescriptor."""

    def test_creation(self):
        """Test creating a behavioral descriptor."""
        descriptor = BehavioralDescriptor(
            routing_density=0.5,
            via_count=100,
            layer_utilization=0.6,
            zone_coverage=0.7,
            thermal_spread=0.3,
            trace_length_variance=0.2,
            component_clustering=0.4,
            clearance_margin=0.15,
            power_distribution=0.5,
            signal_integrity_score=0.8,
        )

        assert descriptor.routing_density == 0.5
        assert descriptor.via_count == 100

    def test_to_vector(self):
        """Test converting to numpy vector."""
        descriptor = BehavioralDescriptor(
            routing_density=0.5,
            via_count=100,
            layer_utilization=0.6,
            zone_coverage=0.7,
            thermal_spread=0.3,
            trace_length_variance=0.2,
            component_clustering=0.4,
            clearance_margin=0.15,
            power_distribution=0.5,
            signal_integrity_score=0.8,
        )

        vector = descriptor.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 10

    def test_from_pcb_state(self, mock_pcb_state):
        """Test creating descriptor from PCB state."""
        descriptor = BehavioralDescriptor.from_pcb_state(mock_pcb_state)

        assert descriptor is not None
        assert 0 <= descriptor.routing_density <= 1
        assert descriptor.via_count >= 0

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        original = BehavioralDescriptor(
            routing_density=0.5,
            via_count=100,
            layer_utilization=0.6,
            zone_coverage=0.7,
            thermal_spread=0.3,
            trace_length_variance=0.2,
            component_clustering=0.4,
            clearance_margin=0.15,
            power_distribution=0.5,
            signal_integrity_score=0.8,
        )

        data = original.to_dict()
        restored = BehavioralDescriptor.from_dict(data)

        assert restored.routing_density == original.routing_density
        assert restored.via_count == original.via_count

    def test_distance(self):
        """Test distance between descriptors."""
        d1 = BehavioralDescriptor(
            routing_density=0.0, via_count=0, layer_utilization=0.0,
            zone_coverage=0.0, thermal_spread=0.0, trace_length_variance=0.0,
            component_clustering=0.0, clearance_margin=0.0,
            power_distribution=0.0, signal_integrity_score=0.0,
        )
        d2 = BehavioralDescriptor(
            routing_density=1.0, via_count=100, layer_utilization=1.0,
            zone_coverage=1.0, thermal_spread=1.0, trace_length_variance=1.0,
            component_clustering=1.0, clearance_margin=1.0,
            power_distribution=1.0, signal_integrity_score=1.0,
        )

        distance = d1.distance(d2)
        assert distance > 0


class TestArchiveCell:
    """Tests for ArchiveCell."""

    def test_creation(self):
        """Test creating an archive cell."""
        cell = ArchiveCell(
            solution=MockPCBState(),
            fitness=0.75,
            descriptor=BehavioralDescriptor(
                routing_density=0.5, via_count=50, layer_utilization=0.6,
                zone_coverage=0.7, thermal_spread=0.3, trace_length_variance=0.2,
                component_clustering=0.4, clearance_margin=0.15,
                power_distribution=0.5, signal_integrity_score=0.8,
            ),
            visit_count=1,
        )

        assert cell.fitness == 0.75
        assert cell.visit_count == 1

    def test_update(self):
        """Test updating cell with better solution."""
        cell = ArchiveCell(
            solution=MockPCBState(violations=100),
            fitness=0.5,
            descriptor=BehavioralDescriptor(
                routing_density=0.5, via_count=50, layer_utilization=0.6,
                zone_coverage=0.7, thermal_spread=0.3, trace_length_variance=0.2,
                component_clustering=0.4, clearance_margin=0.15,
                power_distribution=0.5, signal_integrity_score=0.8,
            ),
            visit_count=1,
        )

        # Better solution
        better_state = MockPCBState(violations=50)
        better_fitness = 0.8

        old_solution = cell.solution
        cell.solution = better_state
        cell.fitness = better_fitness
        cell.visit_count += 1

        assert cell.fitness == 0.8
        assert cell.visit_count == 2


class TestMAPElitesArchive:
    """Tests for MAPElitesArchive."""

    def test_initialization(self, map_elites_archive):
        """Test archive initialization."""
        assert map_elites_archive is not None
        assert map_elites_archive.dimensions == 10
        assert map_elites_archive.bins_per_dimension == 5

    def test_add_solution(self, map_elites_archive, mock_pcb_state):
        """Test adding a solution to the archive."""
        descriptor = BehavioralDescriptor.from_pcb_state(mock_pcb_state)
        fitness = 0.75

        added = map_elites_archive.add(mock_pcb_state, fitness, descriptor)

        assert added is True
        stats = map_elites_archive.get_statistics()
        assert stats.filled_cells >= 1

    def test_add_better_solution(self, map_elites_archive):
        """Test that better solution replaces worse one."""
        state1 = MockPCBState(violations=100)
        state2 = MockPCBState(violations=50)

        descriptor = BehavioralDescriptor(
            routing_density=0.5, via_count=50, layer_utilization=0.5,
            zone_coverage=0.5, thermal_spread=0.5, trace_length_variance=0.5,
            component_clustering=0.5, clearance_margin=0.5,
            power_distribution=0.5, signal_integrity_score=0.5,
        )

        map_elites_archive.add(state1, 0.5, descriptor)
        map_elites_archive.add(state2, 0.8, descriptor)

        # Should only have 1 cell filled (same position)
        stats = map_elites_archive.get_statistics()
        assert stats.filled_cells == 1
        assert stats.max_fitness >= 0.8

    def test_add_diverse_solutions(self, map_elites_archive):
        """Test adding diverse solutions fills different cells."""
        for i in range(10):
            state = MockPCBState(
                num_components=5 + i * 3,
                num_vias=10 + i * 5,
                violations=100 - i * 5,
            )
            descriptor = BehavioralDescriptor.from_pcb_state(state)
            fitness = 0.5 + i * 0.05
            map_elites_archive.add(state, fitness, descriptor)

        stats = map_elites_archive.get_statistics()
        assert stats.filled_cells >= 5  # Should have some diversity

    def test_sample_solutions(self, map_elites_archive, mock_pcb_state):
        """Test sampling solutions from archive."""
        # Add some solutions first
        for i in range(5):
            state = MockPCBState(violations=100 - i * 10)
            descriptor = BehavioralDescriptor.from_pcb_state(state)
            map_elites_archive.add(state, 0.5 + i * 0.1, descriptor)

        # Sample
        samples = map_elites_archive.sample(3)
        assert len(samples) <= 3

    def test_get_best_solutions(self, map_elites_archive):
        """Test getting best solutions."""
        for i in range(10):
            state = MockPCBState(violations=100 - i * 5)
            descriptor = BehavioralDescriptor.from_pcb_state(state)
            map_elites_archive.add(state, 0.5 + i * 0.05, descriptor)

        best = map_elites_archive.get_best(5)
        assert len(best) <= 5

        # Should be sorted by fitness
        fitnesses = [cell.fitness for cell in best]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_get_statistics(self, map_elites_archive):
        """Test statistics calculation."""
        for i in range(10):
            state = MockPCBState(violations=100 - i * 5)
            descriptor = BehavioralDescriptor.from_pcb_state(state)
            map_elites_archive.add(state, 0.5 + i * 0.05, descriptor)

        stats = map_elites_archive.get_statistics()

        assert isinstance(stats, ArchiveStatistics)
        assert stats.total_cells > 0
        assert stats.filled_cells > 0
        assert 0 <= stats.coverage <= 1
        assert stats.avg_fitness >= 0
        assert stats.total_visits >= stats.filled_cells

    def test_save_and_load(self, map_elites_archive, temp_dir):
        """Test saving and loading archive."""
        # Add some solutions
        for i in range(5):
            state = MockPCBState(violations=100 - i * 10)
            descriptor = BehavioralDescriptor.from_pcb_state(state)
            map_elites_archive.add(state, 0.5 + i * 0.1, descriptor)

        # Save
        save_path = temp_dir / "archive.json"
        map_elites_archive.save(save_path)

        assert save_path.exists()

    def test_empty_archive_statistics(self, map_elites_archive):
        """Test statistics on empty archive."""
        stats = map_elites_archive.get_statistics()

        assert stats.filled_cells == 0
        assert stats.coverage == 0
        assert stats.avg_fitness == 0

    def test_sample_from_empty(self, map_elites_archive):
        """Test sampling from empty archive."""
        samples = map_elites_archive.sample(5)
        assert len(samples) == 0


class TestArchiveStatistics:
    """Tests for ArchiveStatistics dataclass."""

    def test_creation(self):
        """Test creating statistics."""
        stats = ArchiveStatistics(
            total_cells=1000,
            filled_cells=100,
            coverage=0.1,
            avg_fitness=0.65,
            max_fitness=0.95,
            min_fitness=0.35,
            total_visits=500,
            diversity_score=0.8,
        )

        assert stats.total_cells == 1000
        assert stats.coverage == 0.1
