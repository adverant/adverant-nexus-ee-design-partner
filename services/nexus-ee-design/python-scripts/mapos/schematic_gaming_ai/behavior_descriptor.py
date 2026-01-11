"""
Schematic Behavioral Descriptors

Defines a 10-dimensional feature space characterizing how a schematic
achieves its goals - the "behavior" or "strategy" of a design.

Different descriptors with similar fitness represent fundamentally
different design philosophies, enabling quality-diversity optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

import numpy as np

from .config import get_schematic_config, BehaviorDescriptorConfig

logger = logging.getLogger(__name__)


class PowerDistributionStrategy(Enum):
    """Power distribution topology strategies."""
    STAR = "star"           # Single point distribution
    RADIAL = "radial"       # Hub-and-spoke from regulators
    TREE = "tree"           # Hierarchical distribution
    SPLIT = "split"         # Separate analog/digital/RF domains
    UNKNOWN = "unknown"


@dataclass
class SchematicBehaviorDescriptor:
    """
    Behavioral characteristics of a schematic design.

    These descriptors capture HOW a design achieves its goals,
    not just how well. Different descriptors can have similar
    fitness but represent fundamentally different strategies.

    The 10-dimensional feature space covers:
    - Complexity (3D): component_count, net_count, sheet_count
    - Strategy (2D): power_distribution, interface_isolation
    - Quality (3D): erc_violations, bp_adherence, cost_efficiency
    - Manufacturing (2D): footprint_variety, sourcing_difficulty
    """

    # Complexity metrics (3 dimensions)
    component_count: int = 0             # Total unique components
    net_count: int = 0                   # Total electrical nets
    sheet_count: int = 1                 # Hierarchical sheets (1 = flat)

    # Design strategy indicators (2 dimensions)
    power_distribution_strategy: PowerDistributionStrategy = PowerDistributionStrategy.UNKNOWN
    interface_isolation: float = 0.0     # Isolation degree (0-1)

    # Quality metrics (3 dimensions)
    erc_violation_count: int = 0         # ERC errors
    best_practice_adherence: float = 0.0  # Fraction of BP rules passed (0-1)
    cost_efficiency: float = 0.0         # Cost per functional block (normalized)

    # Manufacturing considerations (2 dimensions)
    footprint_variety: int = 0           # Different package types
    sourcing_difficulty: float = 0.0     # Avg component availability (0=easy, 1=hard)

    # Additional metadata (not part of vector)
    decoupling_efficiency: float = 0.0   # Actual / required decoupling
    thermal_spreading: float = 0.0       # Heat path diversity
    protection_coverage: float = 0.0     # Protected interfaces / total interfaces

    def to_vector(self, config: Optional[BehaviorDescriptorConfig] = None) -> np.ndarray:
        """
        Convert to 10-dimensional normalized behavioral vector.

        Args:
            config: Configuration for normalization ranges

        Returns:
            10D numpy array with values in [0, 1]
        """
        if config is None:
            config = get_schematic_config().behavior

        # Normalize to [0, 1] range
        vector = np.array([
            # Complexity (3D)
            min(self.component_count / config.max_component_count, 1.0),
            min(self.net_count / config.max_net_count, 1.0),
            min(self.sheet_count / config.max_sheet_count, 1.0),

            # Strategy (2D)
            self._power_strategy_to_float(),
            np.clip(self.interface_isolation, 0.0, 1.0),

            # Quality (3D) - note: erc_violations inverted (lower is better)
            1.0 - min(self.erc_violation_count / config.max_erc_violations, 1.0),
            np.clip(self.best_practice_adherence, 0.0, 1.0),
            np.clip(self.cost_efficiency, 0.0, 1.0),

            # Manufacturing (2D)
            min(self.footprint_variety / config.max_footprint_variety, 1.0),
            np.clip(self.sourcing_difficulty, 0.0, 1.0),
        ], dtype=np.float32)

        return vector

    def _power_strategy_to_float(self) -> float:
        """Convert power distribution strategy to float [0, 1]."""
        strategy_map = {
            PowerDistributionStrategy.STAR: 0.0,
            PowerDistributionStrategy.RADIAL: 0.25,
            PowerDistributionStrategy.TREE: 0.5,
            PowerDistributionStrategy.SPLIT: 0.75,
            PowerDistributionStrategy.UNKNOWN: 1.0,
        }
        return strategy_map.get(self.power_distribution_strategy, 0.5)

    @classmethod
    def from_vector(cls, vector: np.ndarray, config: Optional[BehaviorDescriptorConfig] = None) -> "SchematicBehaviorDescriptor":
        """
        Create descriptor from 10D vector (inverse of to_vector).

        Args:
            vector: 10D numpy array with values in [0, 1]
            config: Configuration for denormalization

        Returns:
            SchematicBehaviorDescriptor instance
        """
        if config is None:
            config = get_schematic_config().behavior

        if len(vector) != 10:
            raise ValueError(f"Expected 10D vector, got {len(vector)}D")

        return cls(
            component_count=int(vector[0] * config.max_component_count),
            net_count=int(vector[1] * config.max_net_count),
            sheet_count=max(1, int(vector[2] * config.max_sheet_count)),
            power_distribution_strategy=cls._float_to_power_strategy(vector[3]),
            interface_isolation=float(vector[4]),
            erc_violation_count=int((1.0 - vector[5]) * config.max_erc_violations),
            best_practice_adherence=float(vector[6]),
            cost_efficiency=float(vector[7]),
            footprint_variety=int(vector[8] * config.max_footprint_variety),
            sourcing_difficulty=float(vector[9]),
        )

    @staticmethod
    def _float_to_power_strategy(value: float) -> PowerDistributionStrategy:
        """Convert float [0, 1] back to power distribution strategy."""
        if value < 0.125:
            return PowerDistributionStrategy.STAR
        elif value < 0.375:
            return PowerDistributionStrategy.RADIAL
        elif value < 0.625:
            return PowerDistributionStrategy.TREE
        elif value < 0.875:
            return PowerDistributionStrategy.SPLIT
        else:
            return PowerDistributionStrategy.UNKNOWN

    def distance(self, other: "SchematicBehaviorDescriptor") -> float:
        """
        Compute Euclidean distance to another descriptor.

        Args:
            other: Another SchematicBehaviorDescriptor

        Returns:
            Euclidean distance in [0, sqrt(10)] range
        """
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))

    def similarity(self, other: "SchematicBehaviorDescriptor") -> float:
        """
        Compute similarity score (inverse of distance).

        Args:
            other: Another SchematicBehaviorDescriptor

        Returns:
            Similarity in [0, 1] range (1 = identical)
        """
        max_distance = np.sqrt(10)  # Maximum possible distance
        return 1.0 - (self.distance(other) / max_distance)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_count": self.component_count,
            "net_count": self.net_count,
            "sheet_count": self.sheet_count,
            "power_distribution_strategy": self.power_distribution_strategy.value,
            "interface_isolation": self.interface_isolation,
            "erc_violation_count": self.erc_violation_count,
            "best_practice_adherence": self.best_practice_adherence,
            "cost_efficiency": self.cost_efficiency,
            "footprint_variety": self.footprint_variety,
            "sourcing_difficulty": self.sourcing_difficulty,
            "decoupling_efficiency": self.decoupling_efficiency,
            "thermal_spreading": self.thermal_spreading,
            "protection_coverage": self.protection_coverage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchematicBehaviorDescriptor":
        """Create from dictionary."""
        return cls(
            component_count=data.get("component_count", 0),
            net_count=data.get("net_count", 0),
            sheet_count=data.get("sheet_count", 1),
            power_distribution_strategy=PowerDistributionStrategy(
                data.get("power_distribution_strategy", "unknown")
            ),
            interface_isolation=data.get("interface_isolation", 0.0),
            erc_violation_count=data.get("erc_violation_count", 0),
            best_practice_adherence=data.get("best_practice_adherence", 0.0),
            cost_efficiency=data.get("cost_efficiency", 0.0),
            footprint_variety=data.get("footprint_variety", 0),
            sourcing_difficulty=data.get("sourcing_difficulty", 0.0),
            decoupling_efficiency=data.get("decoupling_efficiency", 0.0),
            thermal_spreading=data.get("thermal_spreading", 0.0),
            protection_coverage=data.get("protection_coverage", 0.0),
        )


def compute_schematic_descriptor(schematic: Dict[str, Any]) -> SchematicBehaviorDescriptor:
    """
    Compute behavioral descriptor from schematic data.

    Args:
        schematic: Schematic dictionary with components, nets, sheets, etc.

    Returns:
        SchematicBehaviorDescriptor capturing design characteristics
    """
    # Extract basic counts
    components = schematic.get("components", [])
    nets = schematic.get("nets", [])
    sheets = schematic.get("sheets", [{"name": "main"}])

    component_count = len(components)
    net_count = len(nets)
    sheet_count = len(sheets)

    # Analyze power distribution strategy
    power_strategy = _analyze_power_distribution(schematic)

    # Compute interface isolation
    interface_isolation = _compute_interface_isolation(schematic)

    # Extract quality metrics
    validation = schematic.get("validation_results", {})
    erc_violations = validation.get("erc_violations", 0) + validation.get("erc_warnings", 0)

    # Compute best practice adherence
    bp_violations = validation.get("bp_violations", 0)
    total_bp_rules = 5  # ERC001-005
    bp_adherence = max(0.0, 1.0 - (bp_violations / total_bp_rules))

    # Compute cost efficiency
    cost_efficiency = _compute_cost_efficiency(schematic)

    # Analyze manufacturing characteristics
    footprint_variety = _count_unique_footprints(components)
    sourcing_difficulty = _compute_sourcing_difficulty(components)

    # Compute additional metrics
    decoupling_efficiency = _compute_decoupling_efficiency(schematic)
    thermal_spreading = _compute_thermal_spreading(schematic)
    protection_coverage = _compute_protection_coverage(schematic)

    return SchematicBehaviorDescriptor(
        component_count=component_count,
        net_count=net_count,
        sheet_count=sheet_count,
        power_distribution_strategy=power_strategy,
        interface_isolation=interface_isolation,
        erc_violation_count=erc_violations,
        best_practice_adherence=bp_adherence,
        cost_efficiency=cost_efficiency,
        footprint_variety=footprint_variety,
        sourcing_difficulty=sourcing_difficulty,
        decoupling_efficiency=decoupling_efficiency,
        thermal_spreading=thermal_spreading,
        protection_coverage=protection_coverage,
    )


def _analyze_power_distribution(schematic: Dict[str, Any]) -> PowerDistributionStrategy:
    """Analyze power distribution topology from schematic."""
    components = schematic.get("components", [])
    nets = schematic.get("nets", [])

    # Count power-related components
    regulators = [c for c in components if "regulator" in c.get("type", "").lower()]
    power_nets = [n for n in nets if any(p in n.get("name", "").upper()
                                          for p in ["VDD", "VCC", "3V3", "5V", "12V", "GND"])]

    if len(regulators) == 0:
        return PowerDistributionStrategy.UNKNOWN

    # Check for domain separation (AGND, DGND)
    gnd_nets = [n for n in nets if "GND" in n.get("name", "").upper()]
    if len(gnd_nets) > 1:
        return PowerDistributionStrategy.SPLIT

    # Check for hierarchical distribution
    if len(regulators) > 2:
        return PowerDistributionStrategy.TREE

    # Check for radial distribution (multiple outputs from single regulator)
    if len(power_nets) > 3:
        return PowerDistributionStrategy.RADIAL

    return PowerDistributionStrategy.STAR


def _compute_interface_isolation(schematic: Dict[str, Any]) -> float:
    """Compute degree of interface isolation (0-1)."""
    components = schematic.get("components", [])

    # Count isolation components (buffers, optocouplers, isolators)
    isolation_keywords = ["buffer", "isolator", "optocoupler", "iso", "galvanic"]
    isolation_components = [c for c in components
                           if any(kw in c.get("type", "").lower() for kw in isolation_keywords)]

    # Count interfaces that need isolation
    interface_keywords = ["uart", "spi", "i2c", "can", "usb", "ethernet", "rs485", "rs232"]
    interface_components = [c for c in components
                           if any(kw in c.get("type", "").lower() for kw in interface_keywords)]

    if len(interface_components) == 0:
        return 1.0  # No interfaces = fully isolated

    return min(1.0, len(isolation_components) / len(interface_components))


def _compute_cost_efficiency(schematic: Dict[str, Any]) -> float:
    """Compute cost efficiency (normalized 0-1)."""
    components = schematic.get("components", [])

    # Estimate total cost
    total_cost = 0.0
    for component in components:
        unit_cost = component.get("unit_cost", 0.0)
        if unit_cost > 0:
            total_cost += unit_cost
        else:
            # Estimate based on type
            ctype = component.get("type", "").lower()
            if "mcu" in ctype or "microcontroller" in ctype:
                total_cost += 5.0
            elif "regulator" in ctype:
                total_cost += 1.0
            elif "capacitor" in ctype or "resistor" in ctype:
                total_cost += 0.02
            else:
                total_cost += 0.50

    # Count functional blocks
    sheets = schematic.get("sheets", [{"name": "main"}])
    block_count = max(1, len(sheets))

    # Target: $5 per functional block
    target_cost = block_count * 5.0
    efficiency = min(1.0, target_cost / max(total_cost, 0.01))

    return efficiency


def _count_unique_footprints(components: List[Dict[str, Any]]) -> int:
    """Count unique footprint types."""
    footprints = set()
    for component in components:
        footprint = component.get("footprint", "")
        if footprint:
            footprints.add(footprint)
    return len(footprints)


def _compute_sourcing_difficulty(components: List[Dict[str, Any]]) -> float:
    """Compute average sourcing difficulty (0=easy, 1=hard)."""
    if not components:
        return 0.0

    total_difficulty = 0.0
    for component in components:
        # Check for sourcing indicators
        availability = component.get("availability", "in_stock")
        lead_time = component.get("lead_time_days", 0)

        if availability == "obsolete":
            total_difficulty += 1.0
        elif availability == "limited":
            total_difficulty += 0.7
        elif lead_time > 30:
            total_difficulty += 0.5
        elif lead_time > 7:
            total_difficulty += 0.2
        else:
            total_difficulty += 0.0

    return total_difficulty / len(components)


def _compute_decoupling_efficiency(schematic: Dict[str, Any]) -> float:
    """Compute ratio of actual to required decoupling."""
    components = schematic.get("components", [])

    # Count ICs
    ics = [c for c in components if c.get("type", "").lower() in
           ["mcu", "microcontroller", "ic", "chip", "fpga", "cpld"]]

    # Count decoupling capacitors (100nF or similar)
    decoupling = [c for c in components
                  if "capacitor" in c.get("type", "").lower()
                  and c.get("value", "").lower() in ["100nf", "0.1uf", "100n"]]

    required = max(1, len(ics))
    actual = len(decoupling)

    return min(1.0, actual / required)


def _compute_thermal_spreading(schematic: Dict[str, Any]) -> float:
    """Compute thermal path diversity (0-1)."""
    components = schematic.get("components", [])

    # Count heat-generating components
    heat_sources = [c for c in components
                    if c.get("type", "").lower() in
                    ["regulator", "mosfet", "transistor", "led_driver", "motor_driver"]]

    # Count thermal management components
    thermal_mgmt = [c for c in components
                    if any(kw in c.get("type", "").lower()
                           for kw in ["heatsink", "thermal", "pad", "via"])]

    if len(heat_sources) == 0:
        return 1.0  # No heat sources = no thermal issues

    return min(1.0, len(thermal_mgmt) / len(heat_sources))


def _compute_protection_coverage(schematic: Dict[str, Any]) -> float:
    """Compute fraction of interfaces with protection."""
    components = schematic.get("components", [])

    # Count external interfaces
    interface_keywords = ["connector", "jack", "plug", "header", "terminal"]
    interfaces = [c for c in components
                  if any(kw in c.get("type", "").lower() for kw in interface_keywords)]

    # Count protection components
    protection_keywords = ["tvs", "esd", "fuse", "ptc", "varistor", "clamp", "transil"]
    protection = [c for c in components
                  if any(kw in c.get("type", "").lower() for kw in protection_keywords)]

    if len(interfaces) == 0:
        return 1.0  # No interfaces = fully protected

    return min(1.0, len(protection) / len(interfaces))
