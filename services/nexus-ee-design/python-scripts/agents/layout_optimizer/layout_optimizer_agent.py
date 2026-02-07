"""
Layout Optimizer Agent - Professional signal flow-based component placement.

MAPO v3.1 - Replaces simplistic zone-based placement with graph-theoretic
signal flow analysis to produce industry-standard schematic layouts.

Key improvements over v3.0:
1. Signal flow graph analysis (not hardcoded zones)
2. Topological sort for component layering
3. Functional subsystem grouping
4. Critical path identification
5. Proximity constraint enforcement
6. Ideation context integration

Implements IPC-2221 and IEEE 315 standards for professional schematics.

Author: Nexus EE Design Team
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .signal_flow_analyzer import (
    SignalFlowAnalyzer,
    SignalFlowAnalysis,
    SignalPath,
    ComponentLayer,
    FunctionalGroup,
)

# Import ideation context types for placement hints
try:
    from ideation_context import PlacementContext, SubsystemBlock
except ImportError:
    PlacementContext = None
    SubsystemBlock = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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
    analysis: Optional[SignalFlowAnalysis] = None
    metrics: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Layout Optimizer Agent
# ---------------------------------------------------------------------------


class LayoutOptimizerAgent:
    """
    Optimizes schematic layout using professional signal flow analysis.

    Placement Strategy (MAPO v3.1):
    1. Build connectivity graph from netlist
    2. Analyze signal flow paths (sources → sinks)
    3. Determine component layers via topological sort
    4. Group components by functional subsystem
    5. Apply signal flow left-to-right, power top-to-bottom
    6. Enforce proximity constraints (bypass caps near ICs)
    7. Apply separation zones (analog/digital, power/signal)
    8. Optimize for wire length and crossing minimization

    This produces layouts that match professional schematic standards,
    not amateur zone-based arrangements.
    """

    # Standard grid unit (100 mil = 2.54mm)
    GRID_UNIT = 2.54

    # Canvas dimensions (mm)
    CANVAS_WIDTH = 254.0   # 10 inches
    CANVAS_HEIGHT = 190.5  # 7.5 inches

    # Spacing rules (mm)
    SPACING_RULES = {
        "ic_to_ic": 40.0,           # 40mm between IC centers
        "ic_to_passive": 15.0,       # 15mm IC to passive
        "passive_to_passive": 10.0,  # 10mm between passives
        "bypass_to_ic": 5.08,        # 200 mil bypass to IC
        "connector_edge": 10.0,      # 10mm from edge for connectors
        "layer_spacing": 60.0,       # Horizontal spacing between layers
        "vertical_spacing": 20.0,    # Vertical spacing within layer
    }

    def __init__(self):
        """Initialize the layout optimizer."""
        self.analyzer = SignalFlowAnalyzer()

    def optimize_layout(
        self,
        symbols: List[Any],  # List[SymbolInstance]
        connections: List[Any],  # List[Connection]
        bom: Optional[List[Dict]] = None,
        placement_hints: Optional[Any] = None,
    ) -> OptimizationResult:
        """
        Optimize layout for a schematic sheet using signal flow analysis.

        Args:
            symbols: List of SymbolInstance objects
            connections: List of Connection objects
            bom: Optional BOM with component categories
            placement_hints: Optional PlacementContext from ideation

        Returns:
            OptimizationResult with before/after positions and metrics
        """
        logger.info(f"Optimizing layout for {len(symbols)} components (MAPO v3.1 signal flow)")

        # Store original positions
        original_positions = {s.reference: s.position for s in symbols}

        # Build netlist from connections
        netlist = self._build_netlist(connections, symbols)

        # Build BOM if not provided
        if not bom:
            bom = self._build_bom_from_symbols(symbols)

        # Step 1: Analyze signal flow
        analysis = self.analyzer.analyze(
            netlist=netlist,
            bom=bom,
            ideation_context=placement_hints
        )

        logger.info(
            f"Signal flow analysis: {len(analysis.signal_paths)} paths, "
            f"{len(analysis.component_layers)} layers, "
            f"{len(analysis.functional_groups)} groups"
        )

        # Step 2: Calculate positions based on signal flow
        component_positions = self._calculate_signal_flow_positions(
            symbols=symbols,
            analysis=analysis,
            placement_hints=placement_hints
        )

        # Step 3: Apply positions to symbols
        improvements = []
        grid_corrections = 0

        for symbol in symbols:
            new_pos = component_positions.get(symbol.reference)
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

        # Step 4: Verify spacing constraints
        violations = self._verify_spacing(symbols)

        # Step 5: Fix spacing violations
        spacing_corrections = 0
        for violation in violations:
            if self._fix_spacing_violation(symbols, violation):
                spacing_corrections += 1

        # Step 6: Calculate quality metrics
        metrics = self._calculate_quality_metrics(
            symbols=symbols,
            connections=connections,
            analysis=analysis,
            original_positions=original_positions
        )

        logger.info(
            f"Layout optimization complete: {len(improvements)} improvements, "
            f"{len(violations)} violations, {grid_corrections} grid corrections, "
            f"{spacing_corrections} spacing corrections"
        )
        logger.info(
            f"Quality metrics: wire_length={metrics.get('total_wire_length', 0):.1f}mm, "
            f"crossings={metrics.get('wire_crossings', 0)}, "
            f"signal_flow_score={metrics.get('signal_flow_score', 0):.2f}"
        )

        return OptimizationResult(
            success=len(violations) == 0,
            original_positions=original_positions,
            optimized_positions={s.reference: s.position for s in symbols},
            violations=violations,
            improvements=improvements,
            grid_corrections=grid_corrections,
            spacing_corrections=spacing_corrections,
            analysis=analysis,
            metrics=metrics
        )

    # -------------------------------------------------------------------------
    # Signal flow positioning
    # -------------------------------------------------------------------------

    def _calculate_signal_flow_positions(
        self,
        symbols: List[Any],
        analysis: SignalFlowAnalysis,
        placement_hints: Optional[Any] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate component positions based on signal flow analysis.

        Strategy:
        1. Place components by layer (left to right)
        2. Within each layer, arrange by functional group
        3. Apply proximity constraints
        4. Apply separation constraints
        5. Power components at top, signals in middle, passives below
        """
        positions = {}

        # Step 1: Place by layer (horizontal signal flow)
        layer_positions = self._place_by_layer(analysis.component_layers)
        positions.update(layer_positions)

        # Step 2: Adjust for functional grouping
        if analysis.functional_groups:
            self._adjust_for_functional_groups(
                positions,
                analysis.functional_groups,
                placement_hints
            )

        # Step 3: Apply proximity constraints (bypass caps near ICs, etc.)
        self._apply_proximity_constraints(
            positions,
            analysis.critical_proximity_pairs
        )

        # Step 4: Apply separation constraints (analog away from digital)
        self._apply_separation_constraints(
            positions,
            analysis.separation_zones
        )

        # Step 5: Ensure all symbols have positions
        for symbol in symbols:
            if symbol.reference not in positions:
                # Fallback position
                positions[symbol.reference] = (
                    self.CANVAS_WIDTH / 2,
                    self.CANVAS_HEIGHT / 2
                )

        return positions

    def _place_by_layer(
        self,
        component_layers: List[ComponentLayer]
    ) -> Dict[str, Tuple[float, float]]:
        """Place components by signal flow layer (left to right)."""
        positions = {}

        if not component_layers:
            return positions

        # Calculate layer x-positions (left to right)
        num_layers = len(component_layers)
        layer_spacing = min(
            self.SPACING_RULES["layer_spacing"],
            (self.CANVAS_WIDTH - 40) / max(num_layers, 1)
        )

        for layer in component_layers:
            # Base x-position for this layer
            layer_x = 20.0 + layer.x_position_hint * (self.CANVAS_WIDTH - 40.0)

            # Arrange components vertically within layer
            y_pos = 30.0
            for comp_ref in layer.components:
                positions[comp_ref] = (layer_x, y_pos)
                y_pos += self.SPACING_RULES["vertical_spacing"]

        return positions

    def _adjust_for_functional_groups(
        self,
        positions: Dict[str, Tuple[float, float]],
        functional_groups: List[FunctionalGroup],
        placement_hints: Optional[Any]
    ):
        """Adjust positions to group functional subsystems."""
        for group in functional_groups:
            if not group.components:
                continue

            # If ideation provides position hint, use it
            if group.position_hint:
                x_hint, y_hint = group.position_hint
                for comp_ref in group.components:
                    if comp_ref in positions:
                        positions[comp_ref] = (x_hint, y_hint)
                continue

            # Otherwise, calculate centroid and cluster components
            if len(group.components) == 1:
                continue

            # Get existing positions
            group_positions = [
                positions[ref] for ref in group.components
                if ref in positions
            ]

            if not group_positions:
                continue

            # Calculate centroid
            avg_x = sum(p[0] for p in group_positions) / len(group_positions)
            avg_y = sum(p[1] for p in group_positions) / len(group_positions)

            # Move components closer to centroid (clustering)
            for i, comp_ref in enumerate(group.components):
                if comp_ref in positions:
                    old_x, old_y = positions[comp_ref]
                    # Move 50% toward centroid
                    new_x = old_x + 0.5 * (avg_x - old_x)
                    new_y = old_y + 0.5 * (avg_y - old_y)
                    positions[comp_ref] = (new_x, new_y)

    def _apply_proximity_constraints(
        self,
        positions: Dict[str, Tuple[float, float]],
        proximity_pairs: List[Tuple[str, str]]
    ):
        """Move components that must be close together."""
        for comp1, comp2 in proximity_pairs:
            if comp1 in positions and comp2 in positions:
                pos1 = positions[comp1]
                # Place comp2 near comp1 (below and slightly right)
                positions[comp2] = (
                    pos1[0] + self.SPACING_RULES["bypass_to_ic"],
                    pos1[1] + 10.0  # 10mm below
                )

    def _apply_separation_constraints(
        self,
        positions: Dict[str, Tuple[float, float]],
        separation_zones: Dict[str, List[str]]
    ):
        """Ensure zones are spatially separated."""
        # Move analog components to left side
        analog_comps = separation_zones.get('analog', [])
        for comp_ref in analog_comps:
            if comp_ref in positions:
                x, y = positions[comp_ref]
                if x > self.CANVAS_WIDTH / 2:
                    # Move to left side
                    positions[comp_ref] = (x - 60.0, y)

        # Move digital components to right side
        digital_comps = separation_zones.get('digital', [])
        for comp_ref in digital_comps:
            if comp_ref in positions:
                x, y = positions[comp_ref]
                if x < self.CANVAS_WIDTH / 2:
                    # Move to right side
                    positions[comp_ref] = (x + 60.0, y)

        # Move power components to top
        power_comps = separation_zones.get('power', [])
        for comp_ref in power_comps:
            if comp_ref in positions:
                x, y = positions[comp_ref]
                if y > 50.0:
                    # Move to top
                    positions[comp_ref] = (x, 30.0)

    # -------------------------------------------------------------------------
    # Quality metrics
    # -------------------------------------------------------------------------

    def _calculate_quality_metrics(
        self,
        symbols: List[Any],
        connections: List[Any],
        analysis: SignalFlowAnalysis,
        original_positions: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Calculate layout quality metrics."""
        metrics = {}

        # Total wire length
        metrics['total_wire_length'] = self._calculate_wire_length(
            {s.reference: s.position for s in symbols},
            connections
        )

        # Wire crossings
        metrics['wire_crossings'] = self._count_crossings(
            {s.reference: s.position for s in symbols},
            connections
        )

        # Signal flow score (0.0-1.0)
        metrics['signal_flow_score'] = self._calculate_signal_flow_score(
            {s.reference: s.position for s in symbols},
            analysis
        )

        # Improvement percentage
        if original_positions:
            original_wire_length = self._calculate_wire_length(
                original_positions,
                connections
            )
            if original_wire_length > 0:
                improvement = (
                    (original_wire_length - metrics['total_wire_length'])
                    / original_wire_length
                    * 100
                )
                metrics['wire_length_improvement_pct'] = improvement

        return metrics

    def _calculate_wire_length(
        self,
        positions: Dict[str, Tuple[float, float]],
        connections: List[Any]
    ) -> float:
        """Calculate total wire length (Manhattan distance)."""
        total = 0.0
        for conn in connections:
            from_ref = getattr(conn, 'from_ref', None)
            to_ref = getattr(conn, 'to_ref', None)

            if from_ref in positions and to_ref in positions:
                pos1 = positions[from_ref]
                pos2 = positions[to_ref]
                # Manhattan distance
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                total += distance

        return total

    def _count_crossings(
        self,
        positions: Dict[str, Tuple[float, float]],
        connections: List[Any]
    ) -> int:
        """Count number of wire crossings (heuristic)."""
        # Simplified crossing count: check all pairs of connections
        crossings = 0
        conn_list = []

        for conn in connections:
            from_ref = getattr(conn, 'from_ref', None)
            to_ref = getattr(conn, 'to_ref', None)
            if from_ref in positions and to_ref in positions:
                conn_list.append((positions[from_ref], positions[to_ref]))

        # Check each pair of connections for intersection
        for i, (a1, a2) in enumerate(conn_list):
            for b1, b2 in conn_list[i+1:]:
                if self._lines_intersect(a1, a2, b1, b2):
                    crossings += 1

        return crossings

    def _lines_intersect(
        self,
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        b1: Tuple[float, float],
        b2: Tuple[float, float]
    ) -> bool:
        """Check if two line segments intersect."""
        def ccw(a, b, c):
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

    def _calculate_signal_flow_score(
        self,
        positions: Dict[str, Tuple[float, float]],
        analysis: SignalFlowAnalysis
    ) -> float:
        """
        Calculate signal flow clarity score (0.0-1.0).

        Higher score = better left-to-right signal flow.
        """
        if not analysis.signal_paths:
            return 0.5

        total_score = 0.0
        num_paths = len(analysis.signal_paths)

        for path in analysis.signal_paths:
            # Check if components are arranged left-to-right
            source_ref = path.source_component
            sink_refs = path.sink_components

            if source_ref not in positions:
                continue

            source_x = positions[source_ref][0]
            sink_xs = [
                positions[ref][0] for ref in sink_refs
                if ref in positions
            ]

            if not sink_xs:
                continue

            # Check if sinks are to the right of source
            avg_sink_x = sum(sink_xs) / len(sink_xs)
            if avg_sink_x > source_x:
                # Good flow
                path_score = 1.0
            else:
                # Backwards flow
                path_score = 0.0

            # Weight by criticality
            total_score += path_score * path.criticality

        return total_score / max(num_paths, 1)

    # -------------------------------------------------------------------------
    # Netlist and BOM construction
    # -------------------------------------------------------------------------

    def _build_netlist(
        self,
        connections: List[Any],
        symbols: List[Any]
    ) -> List[Dict]:
        """Build netlist from Connection objects."""
        # Group connections by net name
        nets_dict = {}

        for conn in connections:
            net_name = getattr(conn, 'net_name', 'NET')
            from_ref = getattr(conn, 'from_ref', None)
            from_pin = getattr(conn, 'from_pin', '')
            to_ref = getattr(conn, 'to_ref', None)
            to_pin = getattr(conn, 'to_pin', '')

            if not net_name or not from_ref or not to_ref:
                continue

            if net_name not in nets_dict:
                nets_dict[net_name] = []

            # Add both pins to this net
            nets_dict[net_name].append({
                'component': from_ref,
                'pin': from_pin
            })
            nets_dict[net_name].append({
                'component': to_ref,
                'pin': to_pin
            })

        # Convert to list format
        netlist = []
        for net_name, pins in nets_dict.items():
            # Deduplicate pins
            unique_pins = []
            seen = set()
            for pin in pins:
                key = (pin['component'], pin['pin'])
                if key not in seen:
                    seen.add(key)
                    unique_pins.append(pin)

            netlist.append({
                'net_name': net_name,
                'pins': unique_pins
            })

        return netlist

    def _build_bom_from_symbols(self, symbols: List[Any]) -> List[Dict]:
        """Build minimal BOM from symbol instances."""
        bom = []
        for symbol in symbols:
            # Try to infer category from reference designator
            ref = symbol.reference
            category = "Other"

            if ref.startswith('U'):
                category = "IC"
            elif ref.startswith('R'):
                category = "Resistor"
            elif ref.startswith('C'):
                category = "Capacitor"
            elif ref.startswith('L'):
                category = "Inductor"
            elif ref.startswith('J'):
                category = "Connector"
            elif ref.startswith('D'):
                category = "Diode"
            elif ref.startswith('Q'):
                category = "Transistor"

            bom.append({
                'reference': ref,
                'category': category,
                'value': getattr(symbol, 'value', ''),
                'part_number': ''
            })

        return bom

    # -------------------------------------------------------------------------
    # Grid and spacing
    # -------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Layout Optimizer Agent (MAPO v3.1) - Signal Flow Analysis")
    print("Run with actual SymbolInstance objects from schematic assembler")
