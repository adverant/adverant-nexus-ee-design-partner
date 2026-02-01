"""
Layout Optimizer Agent - Optimizes schematic component placement.

Implements IPC-2221 and IEEE 315 standards for:
1. Signal flow organization (left-to-right, top-to-bottom)
2. Zone-based placement (power, input, processing, output, passive)
3. Component spacing optimization
4. Bypass capacitor placement near ICs
5. Grid alignment verification

Author: Nexus EE Design Team
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PlacementZone(Enum):
    """Schematic placement zones following signal flow convention."""
    POWER = "power"           # Top: regulators, power ICs
    INPUT = "input"           # Left: input connectors, sensors
    PROCESSING = "processing" # Center: MCU, logic ICs
    OUTPUT = "output"         # Right: drivers, output connectors
    PASSIVE = "passive"       # Near parent: bypass caps, pull-ups
    SUPPORT = "support"       # Bottom: crystals, test points


@dataclass
class LayoutConstraint:
    """Constraint for component placement."""
    component_ref: str
    zone: PlacementZone
    near_ref: Optional[str] = None      # For bypass caps: place near this IC
    min_spacing: float = 7.62           # mm (300 mil default)
    preferred_rotation: int = 0         # 0, 90, 180, 270
    group_with: Optional[str] = None    # Group with another component
    priority: int = 0                   # Higher = place first


@dataclass
class OptimizationResult:
    """Result from layout optimization."""
    success: bool
    original_positions: Dict[str, Tuple[float, float]]
    optimized_positions: Dict[str, Tuple[float, float]]
    violations: List[str]
    improvements: List[str]
    grid_corrections: int
    spacing_corrections: int


class LayoutOptimizerAgent:
    """
    Optimizes schematic layout following professional standards.

    Placement Strategy:
    1. Analyze component connectivity graph
    2. Assign components to zones based on function
    3. Place power components at top
    4. Place processing chain left-to-right
    5. Place bypass caps adjacent to ICs
    6. Verify grid alignment and spacing
    """

    # Standard grid unit (100 mil = 2.54mm)
    GRID_UNIT = 2.54

    # Zone positions (mm from origin)
    ZONE_POSITIONS = {
        PlacementZone.POWER: {"y": 30.0, "x_start": 50.0},
        PlacementZone.INPUT: {"y": 80.0, "x_start": 20.0},
        PlacementZone.PROCESSING: {"y": 80.0, "x_start": 80.0},
        PlacementZone.OUTPUT: {"y": 80.0, "x_start": 180.0},
        PlacementZone.PASSIVE: {"y": 130.0, "x_start": 30.0},
        PlacementZone.SUPPORT: {"y": 160.0, "x_start": 50.0},
    }

    # Minimum spacing rules (mm)
    SPACING_RULES = {
        "ic_to_ic": 40.0,           # 40mm between IC centers
        "ic_to_passive": 15.0,       # 15mm IC to passive
        "passive_to_passive": 10.0,  # 10mm between passives
        "bypass_to_ic": 5.08,        # 200 mil bypass to IC
        "connector_edge": 10.0,      # 10mm from edge for connectors
    }

    # Component category to zone mapping
    CATEGORY_ZONES = {
        # Power zone
        "Power": PlacementZone.POWER,
        "Regulator": PlacementZone.POWER,
        "LDO": PlacementZone.POWER,

        # Input zone
        "Connector": PlacementZone.INPUT,
        "Sensor": PlacementZone.INPUT,
        "CAN_Transceiver": PlacementZone.INPUT,
        "Thermistor": PlacementZone.INPUT,

        # Processing zone
        "MCU": PlacementZone.PROCESSING,
        "IC": PlacementZone.PROCESSING,
        "Amplifier": PlacementZone.PROCESSING,

        # Output zone
        "Gate_Driver": PlacementZone.OUTPUT,
        "MOSFET": PlacementZone.OUTPUT,
        "BJT": PlacementZone.OUTPUT,
        "Transistor": PlacementZone.OUTPUT,
        "LED": PlacementZone.OUTPUT,

        # Passive zone
        "Capacitor": PlacementZone.PASSIVE,
        "Resistor": PlacementZone.PASSIVE,
        "Inductor": PlacementZone.PASSIVE,

        # Support zone
        "Crystal": PlacementZone.SUPPORT,
        "Diode": PlacementZone.SUPPORT,
        "TVS": PlacementZone.SUPPORT,
        "Fuse": PlacementZone.SUPPORT,
    }

    def __init__(self):
        """Initialize the layout optimizer."""
        self.constraints: List[LayoutConstraint] = []
        self.component_graph: Dict[str, Set[str]] = {}

    def optimize_layout(
        self,
        symbols: List[Any],  # List[SymbolInstance]
        connections: List[Any],  # List[Connection]
        bom: Optional[List[Dict]] = None,
    ) -> OptimizationResult:
        """
        Optimize layout for a schematic sheet.

        Args:
            symbols: List of SymbolInstance objects
            connections: List of Connection objects
            bom: Optional BOM with component categories

        Returns:
            OptimizationResult with before/after positions
        """
        logger.info(f"Optimizing layout for {len(symbols)} components")

        # Store original positions
        original_positions = {s.reference: s.position for s in symbols}

        # Build connectivity graph
        self._build_connectivity_graph(symbols, connections)

        # Generate constraints from component analysis
        constraints = self._generate_constraints(symbols, bom)

        # Assign zones to components
        zone_assignments = self._assign_zones(symbols, constraints)

        # Calculate optimized positions
        optimized_positions = self._calculate_positions(
            symbols, zone_assignments, constraints
        )

        # Apply positions to symbols
        violations = []
        improvements = []
        grid_corrections = 0
        spacing_corrections = 0

        for symbol in symbols:
            new_pos = optimized_positions.get(symbol.reference)
            if new_pos:
                old_pos = symbol.position

                # Grid snap
                snapped_pos = self._snap_to_grid(new_pos)
                if snapped_pos != new_pos:
                    grid_corrections += 1
                    new_pos = snapped_pos

                # Apply position
                symbol.position = new_pos

                # Track improvement
                if old_pos != new_pos:
                    dx = abs(new_pos[0] - old_pos[0])
                    dy = abs(new_pos[1] - old_pos[1])
                    if dx > 1 or dy > 1:
                        improvements.append(
                            f"{symbol.reference}: moved ({dx:.1f}, {dy:.1f})mm"
                        )

        # Verify spacing constraints
        spacing_violations = self._verify_spacing(symbols)
        violations.extend(spacing_violations)

        # Fix spacing violations
        for violation in spacing_violations:
            if self._fix_spacing_violation(symbols, violation):
                spacing_corrections += 1

        logger.info(
            f"Layout optimization complete: {len(improvements)} improvements, "
            f"{len(violations)} violations, {grid_corrections} grid corrections"
        )

        return OptimizationResult(
            success=len(violations) == 0,
            original_positions=original_positions,
            optimized_positions={s.reference: s.position for s in symbols},
            violations=violations,
            improvements=improvements,
            grid_corrections=grid_corrections,
            spacing_corrections=spacing_corrections,
        )

    def _build_connectivity_graph(
        self,
        symbols: List[Any],
        connections: List[Any]
    ):
        """Build graph of component connections."""
        self.component_graph = {s.reference: set() for s in symbols}

        for conn in connections:
            from_ref = conn.from_ref
            to_ref = conn.to_ref

            if from_ref in self.component_graph and to_ref in self.component_graph:
                self.component_graph[from_ref].add(to_ref)
                self.component_graph[to_ref].add(from_ref)

    def _generate_constraints(
        self,
        symbols: List[Any],
        bom: Optional[List[Dict]]
    ) -> List[LayoutConstraint]:
        """Generate placement constraints for all components."""
        constraints = []

        # Build reference to category lookup from BOM
        ref_categories = {}
        if bom:
            for item in bom:
                ref = item.get("reference")
                if ref:
                    ref_categories[ref] = item.get("category", "Other")

        # Identify ICs and their bypass capacitors
        ics = []
        bypass_caps = []

        for symbol in symbols:
            category = ref_categories.get(symbol.reference, "Other")

            # Identify ICs
            if category in ["MCU", "IC", "Gate_Driver", "Amplifier", "Power", "Regulator"]:
                ics.append(symbol)

            # Identify potential bypass caps
            if category == "Capacitor":
                # Check if connected to an IC (likely bypass cap)
                connected = self.component_graph.get(symbol.reference, set())
                for ic in ics:
                    if ic.reference in connected:
                        bypass_caps.append((symbol, ic))
                        break

        # Generate IC constraints
        for ic in ics:
            category = ref_categories.get(ic.reference, "IC")
            zone = self.CATEGORY_ZONES.get(category, PlacementZone.PROCESSING)
            constraints.append(LayoutConstraint(
                component_ref=ic.reference,
                zone=zone,
                min_spacing=self.SPACING_RULES["ic_to_ic"],
                priority=10,
            ))

        # Generate bypass cap constraints (place near their IC)
        for cap, ic in bypass_caps:
            constraints.append(LayoutConstraint(
                component_ref=cap.reference,
                zone=PlacementZone.PASSIVE,
                near_ref=ic.reference,
                min_spacing=self.SPACING_RULES["bypass_to_ic"],
                priority=8,
            ))

        # Generate constraints for remaining components
        for symbol in symbols:
            # Skip if already has constraint
            if any(c.component_ref == symbol.reference for c in constraints):
                continue

            category = ref_categories.get(symbol.reference, "Other")
            zone = self.CATEGORY_ZONES.get(category, PlacementZone.SUPPORT)

            constraints.append(LayoutConstraint(
                component_ref=symbol.reference,
                zone=zone,
                min_spacing=self.SPACING_RULES.get(
                    "passive_to_passive" if zone == PlacementZone.PASSIVE else "ic_to_passive",
                    10.0
                ),
                priority=5 if zone in [PlacementZone.PROCESSING, PlacementZone.OUTPUT] else 3,
            ))

        return constraints

    def _assign_zones(
        self,
        symbols: List[Any],
        constraints: List[LayoutConstraint]
    ) -> Dict[str, PlacementZone]:
        """Assign each component to a placement zone."""
        assignments = {}

        for constraint in constraints:
            assignments[constraint.component_ref] = constraint.zone

        # Assign any missing components to SUPPORT zone
        for symbol in symbols:
            if symbol.reference not in assignments:
                assignments[symbol.reference] = PlacementZone.SUPPORT

        return assignments

    def _calculate_positions(
        self,
        symbols: List[Any],
        zone_assignments: Dict[str, PlacementZone],
        constraints: List[LayoutConstraint]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate optimized positions for all components."""
        positions = {}

        # Group symbols by zone
        zone_groups: Dict[PlacementZone, List[Any]] = {zone: [] for zone in PlacementZone}
        for symbol in symbols:
            zone = zone_assignments.get(symbol.reference, PlacementZone.SUPPORT)
            zone_groups[zone].append(symbol)

        # Build constraint lookup
        constraint_map = {c.component_ref: c for c in constraints}

        # Place each zone
        for zone, zone_symbols in zone_groups.items():
            zone_pos = self.ZONE_POSITIONS.get(zone, {"y": 100.0, "x_start": 50.0})
            base_y = zone_pos["y"]
            base_x = zone_pos["x_start"]

            # Sort by priority
            zone_symbols.sort(
                key=lambda s: constraint_map.get(s.reference, LayoutConstraint(s.reference, zone)).priority,
                reverse=True
            )

            # Calculate positions
            if zone == PlacementZone.POWER:
                # Power ICs in horizontal row at top
                spacing = self.SPACING_RULES["ic_to_ic"]
                for i, symbol in enumerate(zone_symbols):
                    positions[symbol.reference] = (base_x + i * spacing, base_y)

            elif zone == PlacementZone.PROCESSING:
                # MCU and main ICs in horizontal flow
                spacing = self.SPACING_RULES["ic_to_ic"]
                for i, symbol in enumerate(zone_symbols):
                    positions[symbol.reference] = (base_x + i * spacing, base_y)

            elif zone == PlacementZone.OUTPUT:
                # Drivers and output components
                spacing = self.SPACING_RULES["ic_to_ic"]
                for i, symbol in enumerate(zone_symbols):
                    positions[symbol.reference] = (base_x + i * spacing, base_y)

            elif zone == PlacementZone.INPUT:
                # Input connectors on left edge, vertical arrangement
                spacing = 20.0  # Vertical spacing for connectors
                for i, symbol in enumerate(zone_symbols):
                    positions[symbol.reference] = (base_x, base_y + i * spacing)

            elif zone == PlacementZone.PASSIVE:
                # Passives in grid, but check for bypass cap constraints
                row_size = 8
                x_spacing = self.SPACING_RULES["passive_to_passive"]
                y_spacing = 15.0

                non_bypass = []
                for symbol in zone_symbols:
                    constraint = constraint_map.get(symbol.reference)
                    if constraint and constraint.near_ref:
                        # This is a bypass cap - place near its IC
                        ic_pos = positions.get(constraint.near_ref)
                        if ic_pos:
                            # Place below and slightly right of IC
                            positions[symbol.reference] = (
                                ic_pos[0] + 5.08,  # 200 mil right
                                ic_pos[1] + 10.16  # 400 mil below
                            )
                        else:
                            non_bypass.append(symbol)
                    else:
                        non_bypass.append(symbol)

                # Place remaining passives in grid
                for i, symbol in enumerate(non_bypass):
                    row = i // row_size
                    col = i % row_size
                    positions[symbol.reference] = (
                        base_x + col * x_spacing,
                        base_y + row * y_spacing
                    )

            else:
                # Support components in grid
                row_size = 6
                x_spacing = 15.0
                y_spacing = 15.0
                for i, symbol in enumerate(zone_symbols):
                    row = i // row_size
                    col = i % row_size
                    positions[symbol.reference] = (
                        base_x + col * x_spacing,
                        base_y + row * y_spacing
                    )

        return positions

    def _snap_to_grid(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Snap position to 100 mil grid."""
        x = round(position[0] / self.GRID_UNIT) * self.GRID_UNIT
        y = round(position[1] / self.GRID_UNIT) * self.GRID_UNIT
        return (x, y)

    def _verify_spacing(self, symbols: List[Any]) -> List[str]:
        """Verify spacing constraints are met."""
        violations = []

        # Check pairwise spacing
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                dx = abs(s1.position[0] - s2.position[0])
                dy = abs(s1.position[1] - s2.position[1])
                distance = (dx ** 2 + dy ** 2) ** 0.5

                # Get minimum required spacing
                min_spacing = self.SPACING_RULES["passive_to_passive"]

                # ICs need more spacing
                if s1.reference.startswith("U") and s2.reference.startswith("U"):
                    min_spacing = self.SPACING_RULES["ic_to_ic"]
                elif s1.reference.startswith("U") or s2.reference.startswith("U"):
                    min_spacing = self.SPACING_RULES["ic_to_passive"]

                if distance < min_spacing and distance > 0:
                    violations.append(
                        f"Spacing violation: {s1.reference} and {s2.reference} "
                        f"are {distance:.1f}mm apart (min: {min_spacing}mm)"
                    )

        return violations

    def _fix_spacing_violation(self, symbols: List[Any], violation: str) -> bool:
        """Attempt to fix a spacing violation."""
        # Parse violation to get component references
        match = re.search(r"(\w+) and (\w+)", violation)
        if not match:
            return False

        ref1, ref2 = match.groups()

        # Find the symbols
        s1 = next((s for s in symbols if s.reference == ref1), None)
        s2 = next((s for s in symbols if s.reference == ref2), None)

        if not s1 or not s2:
            return False

        # Move the second symbol away
        dx = s2.position[0] - s1.position[0]
        dy = s2.position[1] - s1.position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        if distance == 0:
            # Overlapping - move horizontally
            s2.position = (s2.position[0] + self.SPACING_RULES["ic_to_passive"], s2.position[1])
            return True

        # Calculate direction and required distance
        min_spacing = self.SPACING_RULES["passive_to_passive"]
        if s1.reference.startswith("U") or s2.reference.startswith("U"):
            min_spacing = self.SPACING_RULES["ic_to_passive"]

        # Scale factor to achieve minimum spacing
        scale = min_spacing / distance
        if scale > 1:
            # Move s2 away from s1
            new_dx = dx * scale
            new_dy = dy * scale
            s2.position = self._snap_to_grid((
                s1.position[0] + new_dx,
                s1.position[1] + new_dy
            ))
            return True

        return False


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Layout Optimizer Agent test")
    print("Run with actual SymbolInstance objects from schematic assembler")
