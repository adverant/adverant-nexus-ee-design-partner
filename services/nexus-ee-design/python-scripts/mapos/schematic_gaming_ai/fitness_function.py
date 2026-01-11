"""
Schematic Fitness Functions

Multi-objective fitness evaluation for schematics combining:
1. Correctness - ERC + Best Practice compliance
2. Efficiency - Cost per functional block
3. Reliability - Thermal, decoupling, protection
4. Manufacturability - Sourcing, availability
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from .config import get_schematic_config, FitnessConfig

logger = logging.getLogger(__name__)


@dataclass
class FitnessWeights:
    """Configurable weights for fitness components."""
    correctness: float = 0.40
    efficiency: float = 0.30
    reliability: float = 0.20
    manufacturability: float = 0.10

    def validate(self) -> bool:
        """Check weights sum to 1.0."""
        total = self.correctness + self.efficiency + self.reliability + self.manufacturability
        return abs(total - 1.0) < 0.001

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for vectorized operations."""
        return np.array([
            self.correctness,
            self.efficiency,
            self.reliability,
            self.manufacturability
        ])


@dataclass
class SchematicFitness:
    """
    Multi-objective fitness result for a schematic.

    The total fitness is a weighted combination of four objectives:
    - correctness: ERC compliance + best practices
    - efficiency: Cost per functional block
    - reliability: Thermal management + decoupling + protection
    - manufacturability: Component availability + sourcing
    """
    total: float = 0.0
    correctness: float = 0.0
    efficiency: float = 0.0
    reliability: float = 0.0
    manufacturability: float = 0.0

    # Detailed breakdown
    erc_score: float = 0.0
    bp_score: float = 0.0
    cost_score: float = 0.0
    thermal_score: float = 0.0
    decoupling_score: float = 0.0
    protection_score: float = 0.0
    sourcing_score: float = 0.0
    availability_score: float = 0.0

    # Violation counts
    critical_violations: int = 0
    error_violations: int = 0
    warning_violations: int = 0

    def is_passing(self, config: Optional[FitnessConfig] = None) -> bool:
        """Check if schematic passes minimum fitness threshold."""
        if config is None:
            config = get_schematic_config().fitness
        return self.total >= config.pass_fitness and self.critical_violations == 0

    def is_excellent(self, config: Optional[FitnessConfig] = None) -> bool:
        """Check if schematic achieves excellent fitness."""
        if config is None:
            config = get_schematic_config().fitness
        return self.total >= config.excellent_fitness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "correctness": self.correctness,
            "efficiency": self.efficiency,
            "reliability": self.reliability,
            "manufacturability": self.manufacturability,
            "breakdown": {
                "erc_score": self.erc_score,
                "bp_score": self.bp_score,
                "cost_score": self.cost_score,
                "thermal_score": self.thermal_score,
                "decoupling_score": self.decoupling_score,
                "protection_score": self.protection_score,
                "sourcing_score": self.sourcing_score,
                "availability_score": self.availability_score,
            },
            "violations": {
                "critical": self.critical_violations,
                "error": self.error_violations,
                "warning": self.warning_violations,
            },
            "passing": self.is_passing(),
            "excellent": self.is_excellent(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchematicFitness":
        """Create from dictionary."""
        breakdown = data.get("breakdown", {})
        violations = data.get("violations", {})
        return cls(
            total=data.get("total", 0.0),
            correctness=data.get("correctness", 0.0),
            efficiency=data.get("efficiency", 0.0),
            reliability=data.get("reliability", 0.0),
            manufacturability=data.get("manufacturability", 0.0),
            erc_score=breakdown.get("erc_score", 0.0),
            bp_score=breakdown.get("bp_score", 0.0),
            cost_score=breakdown.get("cost_score", 0.0),
            thermal_score=breakdown.get("thermal_score", 0.0),
            decoupling_score=breakdown.get("decoupling_score", 0.0),
            protection_score=breakdown.get("protection_score", 0.0),
            sourcing_score=breakdown.get("sourcing_score", 0.0),
            availability_score=breakdown.get("availability_score", 0.0),
            critical_violations=violations.get("critical", 0),
            error_violations=violations.get("error", 0),
            warning_violations=violations.get("warning", 0),
        )

    def __gt__(self, other: "SchematicFitness") -> bool:
        """Compare by total fitness."""
        return self.total > other.total

    def __lt__(self, other: "SchematicFitness") -> bool:
        """Compare by total fitness."""
        return self.total < other.total

    def __eq__(self, other: object) -> bool:
        """Compare by total fitness."""
        if not isinstance(other, SchematicFitness):
            return NotImplemented
        return abs(self.total - other.total) < 0.001


def compute_schematic_fitness(
    schematic: Dict[str, Any],
    weights: Optional[FitnessWeights] = None,
    config: Optional[FitnessConfig] = None
) -> SchematicFitness:
    """
    Compute multi-objective fitness for a schematic.

    Args:
        schematic: Schematic dictionary with components, validation results, etc.
        weights: Optional custom weights (defaults to config)
        config: Optional fitness configuration

    Returns:
        SchematicFitness with total score and component breakdowns
    """
    if config is None:
        config = get_schematic_config().fitness

    if weights is None:
        weights = FitnessWeights(
            correctness=config.correctness_weight,
            efficiency=config.efficiency_weight,
            reliability=config.reliability_weight,
            manufacturability=config.manufacturability_weight,
        )

    # 1. CORRECTNESS (ERC + Best Practices)
    erc_score, critical, errors, warnings = _compute_erc_score(schematic, config)
    bp_score, bp_violations = _compute_bp_score(schematic, config)
    correctness = erc_score * 0.7 + bp_score * 0.3

    # 2. EFFICIENCY (Cost per functional block)
    cost_score = _compute_cost_score(schematic, config)
    efficiency = cost_score

    # 3. RELIABILITY (Thermal + Decoupling + Protection)
    thermal_score = _compute_thermal_score(schematic)
    decoupling_score = _compute_decoupling_score(schematic, config)
    protection_score = _compute_protection_score(schematic)
    reliability = (thermal_score + decoupling_score + protection_score) / 3.0

    # 4. MANUFACTURABILITY (Sourcing + Availability)
    sourcing_score, availability_score = _compute_manufacturing_scores(schematic)
    manufacturability = (sourcing_score + availability_score) / 2.0

    # Compute weighted total
    total = (
        correctness * weights.correctness +
        efficiency * weights.efficiency +
        reliability * weights.reliability +
        manufacturability * weights.manufacturability
    )

    # Penalty for critical violations
    if critical > 0:
        total *= 0.5  # 50% penalty for critical violations

    return SchematicFitness(
        total=total,
        correctness=correctness,
        efficiency=efficiency,
        reliability=reliability,
        manufacturability=manufacturability,
        erc_score=erc_score,
        bp_score=bp_score,
        cost_score=cost_score,
        thermal_score=thermal_score,
        decoupling_score=decoupling_score,
        protection_score=protection_score,
        sourcing_score=sourcing_score,
        availability_score=availability_score,
        critical_violations=critical,
        error_violations=errors,
        warning_violations=warnings,
    )


def _compute_erc_score(
    schematic: Dict[str, Any],
    config: FitnessConfig
) -> Tuple[float, int, int, int]:
    """
    Compute ERC compliance score.

    Returns:
        (score, critical_count, error_count, warning_count)
    """
    validation = schematic.get("validation_results", {})

    # Count violations by severity
    critical = validation.get("critical_violations", 0)
    errors = validation.get("erc_violations", 0)
    warnings = validation.get("erc_warnings", 0)

    # Calculate penalty
    penalty = (
        critical * config.erc_weights.critical +
        errors * config.erc_weights.error +
        warnings * config.erc_weights.warning
    )

    # Score starts at 100, subtract penalties
    score = max(0.0, (100.0 - penalty) / 100.0)

    return score, critical, errors, warnings


def _compute_bp_score(
    schematic: Dict[str, Any],
    config: FitnessConfig
) -> Tuple[float, int]:
    """
    Compute best practices compliance score.

    Returns:
        (score, violation_count)
    """
    validation = schematic.get("validation_results", {})
    bp_violations = validation.get("bp_violations", 0)
    bp_warnings = validation.get("bp_warnings", 0)

    # Calculate penalty
    penalty = (
        bp_violations * config.bp_weights.error +
        bp_warnings * config.bp_weights.warning
    )

    # Score starts at 100, subtract penalties
    score = max(0.0, (100.0 - penalty) / 100.0)

    return score, bp_violations


def _compute_cost_score(schematic: Dict[str, Any], config: FitnessConfig) -> float:
    """Compute cost efficiency score (0-1)."""
    components = schematic.get("components", [])
    sheets = schematic.get("sheets", [{"name": "main"}])

    # Estimate total cost
    total_cost = 0.0
    for component in components:
        unit_cost = component.get("unit_cost", 0.0)
        quantity = component.get("quantity", 1)
        if unit_cost > 0:
            total_cost += unit_cost * quantity
        else:
            # Estimate based on type
            total_cost += _estimate_component_cost(component)

    # Target cost based on functional blocks
    block_count = max(1, len(sheets))
    target_cost = block_count * config.target_cost_per_function

    # Score: higher is better (under target is good)
    if total_cost <= target_cost:
        return 1.0
    else:
        # Linear decrease above target
        overage = (total_cost - target_cost) / target_cost
        return max(0.0, 1.0 - overage * 0.5)


def _estimate_component_cost(component: Dict[str, Any]) -> float:
    """Estimate component cost based on type."""
    ctype = component.get("type", "").lower()
    value = component.get("value", "")

    # Cost estimates by type
    if "mcu" in ctype or "microcontroller" in ctype:
        return 5.0
    elif "fpga" in ctype or "cpld" in ctype:
        return 15.0
    elif "regulator" in ctype:
        if "switching" in ctype:
            return 2.0
        return 0.50
    elif "mosfet" in ctype or "transistor" in ctype:
        return 0.30
    elif "capacitor" in ctype:
        # Larger caps cost more
        if "uf" in value.lower() or "µf" in value.lower():
            return 0.10
        return 0.02
    elif "resistor" in ctype:
        return 0.01
    elif "inductor" in ctype:
        return 0.30
    elif "connector" in ctype:
        return 0.50
    elif "crystal" in ctype:
        return 0.25
    elif "diode" in ctype or "led" in ctype:
        return 0.05
    else:
        return 0.20


def _compute_thermal_score(schematic: Dict[str, Any]) -> float:
    """Compute thermal management score (0-1)."""
    components = schematic.get("components", [])

    # Identify heat-generating components
    heat_sources = []
    thermal_mgmt = []

    for component in components:
        ctype = component.get("type", "").lower()

        if any(kw in ctype for kw in ["regulator", "mosfet", "driver", "power"]):
            heat_sources.append(component)

        if any(kw in ctype for kw in ["heatsink", "thermal", "fan"]):
            thermal_mgmt.append(component)

    if len(heat_sources) == 0:
        return 1.0  # No heat sources

    # Check for thermal management coverage
    coverage = min(1.0, len(thermal_mgmt) / len(heat_sources))

    # Bonus for using components with thermal pads
    thermal_pads = sum(1 for c in heat_sources if "pad" in c.get("footprint", "").lower())
    pad_bonus = min(0.2, thermal_pads / len(heat_sources) * 0.2)

    return min(1.0, coverage + pad_bonus)


def _compute_decoupling_score(schematic: Dict[str, Any], config: FitnessConfig) -> float:
    """Compute decoupling capacitor adequacy score (0-1)."""
    components = schematic.get("components", [])

    # Count ICs that need decoupling
    ics = [c for c in components
           if c.get("type", "").lower() in
           ["mcu", "microcontroller", "ic", "chip", "fpga", "cpld", "regulator"]]

    # Count decoupling capacitors (100nF typical)
    decoupling = [c for c in components
                  if "capacitor" in c.get("type", "").lower()
                  and _is_decoupling_cap(c)]

    # Also count bulk capacitors
    bulk = [c for c in components
            if "capacitor" in c.get("type", "").lower()
            and _is_bulk_cap(c)]

    if len(ics) == 0:
        return 1.0

    # Target: 1 decoupling cap per IC + 1 bulk per power domain
    required_decoupling = len(ics)
    power_domains = _count_power_domains(schematic)
    required_bulk = max(1, power_domains)

    decoupling_ratio = min(1.0, len(decoupling) / max(1, required_decoupling))
    bulk_ratio = min(1.0, len(bulk) / max(1, required_bulk))

    return decoupling_ratio * 0.7 + bulk_ratio * 0.3


def _is_decoupling_cap(component: Dict[str, Any]) -> bool:
    """Check if component is a decoupling capacitor."""
    value = component.get("value", "").lower()
    decoupling_values = ["100nf", "0.1uf", "100n", "0.1µf", "0.1 uf", "100 nf"]
    return any(v in value for v in decoupling_values)


def _is_bulk_cap(component: Dict[str, Any]) -> bool:
    """Check if component is a bulk capacitor."""
    value = component.get("value", "").lower()
    # Bulk caps typically 10uF or larger
    bulk_indicators = ["10uf", "10µf", "22uf", "47uf", "100uf", "10 uf"]
    return any(v in value for v in bulk_indicators)


def _count_power_domains(schematic: Dict[str, Any]) -> int:
    """Count distinct power domains in schematic."""
    nets = schematic.get("nets", [])
    power_nets = set()

    power_prefixes = ["vdd", "vcc", "3v3", "5v", "12v", "1v8", "2v5"]
    for net in nets:
        name = net.get("name", "").lower()
        for prefix in power_prefixes:
            if prefix in name:
                power_nets.add(prefix)

    return max(1, len(power_nets))


def _compute_protection_score(schematic: Dict[str, Any]) -> float:
    """Compute interface protection score (0-1)."""
    components = schematic.get("components", [])

    # Count external interfaces
    interfaces = [c for c in components
                  if c.get("type", "").lower() in
                  ["connector", "jack", "header", "terminal", "usb"]]

    # Count protection components
    protection = [c for c in components
                  if any(kw in c.get("type", "").lower()
                         for kw in ["tvs", "esd", "fuse", "ptc", "varistor", "clamp"])]

    if len(interfaces) == 0:
        return 1.0

    # Ideal: 1 protection per interface
    return min(1.0, len(protection) / len(interfaces))


def _compute_manufacturing_scores(schematic: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute manufacturing-related scores.

    Returns:
        (sourcing_score, availability_score)
    """
    components = schematic.get("components", [])

    if not components:
        return 1.0, 1.0

    # Sourcing score based on supplier count
    total_suppliers = 0
    total_leadtime = 0

    for component in components:
        suppliers = component.get("suppliers", [])
        total_suppliers += len(suppliers) if suppliers else 1

        leadtime = component.get("lead_time_days", 7)
        total_leadtime += leadtime

    avg_suppliers = total_suppliers / len(components)
    avg_leadtime = total_leadtime / len(components)

    # Sourcing: more suppliers is better (target: 3+)
    sourcing_score = min(1.0, avg_suppliers / 3.0)

    # Availability: shorter leadtime is better (target: 7 days)
    if avg_leadtime <= 7:
        availability_score = 1.0
    elif avg_leadtime <= 30:
        availability_score = 0.8 - (avg_leadtime - 7) * 0.02
    else:
        availability_score = max(0.3, 0.5 - (avg_leadtime - 30) * 0.01)

    return sourcing_score, availability_score
