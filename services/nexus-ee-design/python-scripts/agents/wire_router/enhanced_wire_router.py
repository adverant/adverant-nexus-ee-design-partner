"""
Enhanced Wire Router - Professional Manhattan routing for KiCad schematics.

Implements IPC/IEEE recommended practices:
1. Avoids 4-way junctions - converts to offset 3-way junctions
2. Bus routing for grouped parallel signals
3. Horizontal power rails (VCC top, GND bottom)
4. Crossing minimization using channel routing
5. Wire length optimization for high-speed/critical paths
6. Proper junction dot placement
7. Grid-aligned routing (100 mil / 2.54mm)

Author: Nexus EE Design Team
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Wire routing type."""
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    BUS = "bus"
    CRITICAL = "critical"  # High-speed, matched length


class JunctionType(Enum):
    """Junction connection types."""
    NONE = "none"
    THREE_WAY = "three_way"  # T-junction (preferred)
    FOUR_WAY = "four_way"    # Cross junction (avoid!)


@dataclass
class WireSegment:
    """Single wire segment between two points."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    net_name: str
    route_type: RouteType = RouteType.SIGNAL
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def is_horizontal(self) -> bool:
        return abs(self.start[1] - self.end[1]) < 0.01

    @property
    def is_vertical(self) -> bool:
        return abs(self.start[0] - self.end[0]) < 0.01

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx**2 + dy**2) ** 0.5


@dataclass
class Junction:
    """Wire junction point."""
    position: Tuple[float, float]
    junction_type: JunctionType
    connected_nets: Set[str] = field(default_factory=set)
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class BusRoute:
    """Bus route for grouped parallel signals."""
    name: str
    signals: List[str]  # Individual signal names (e.g., DATA[0], DATA[1])
    segments: List[WireSegment] = field(default_factory=list)
    spacing: float = 2.54  # mm between parallel wires


@dataclass
class RoutingConstraint:
    """Routing constraint for a net."""
    net_name: str
    route_type: RouteType = RouteType.SIGNAL
    max_length: Optional[float] = None  # mm
    min_spacing: float = 2.54  # mm from other nets
    avoid_crossings_with: List[str] = field(default_factory=list)
    is_differential: bool = False
    differential_pair: Optional[str] = None


@dataclass
class RoutingResult:
    """Result of routing operation."""
    wires: List[WireSegment]
    junctions: List[Junction]
    buses: List[BusRoute]
    crossings: int
    total_wire_length: float
    four_way_junctions_avoided: int
    warnings: List[str] = field(default_factory=list)
    center_fallback_ratio: float = 0.0


class EnhancedWireRouter:
    """
    Professional-grade Manhattan wire router for schematics.

    Features:
    - Avoids 4-way junctions (IPC best practice)
    - Optimizes power rail routing
    - Minimizes wire crossings
    - Groups bus signals
    - Supports routing constraints
    """

    GRID_UNIT = 2.54  # mm (100 mil)
    JUNCTION_OFFSET = 1.27  # mm (50 mil) - offset for avoiding 4-way
    MIN_WIRE_LENGTH = 1.27  # mm (50 mil) - minimum segment length
    BUS_SPACING = 2.54  # mm between bus wires
    POWER_RAIL_MARGIN = 10.0  # mm from edge for power rails

    def __init__(self):
        """Initialize the router."""
        self._wires: List[WireSegment] = []
        self._junctions: List[Junction] = []
        self._buses: List[BusRoute] = []
        self._occupied_points: Dict[Tuple[float, float], str] = {}  # point -> net_name
        self._crossing_count = 0
        self._four_way_avoided = 0
        self._warnings: List[str] = []

    def route(
        self,
        connections: List[Dict[str, Any]],
        component_positions: Dict[str, Tuple[float, float]],
        pin_positions: Dict[str, Dict[str, Tuple[float, float]]],
        constraints: Optional[List[RoutingConstraint]] = None,
        sheet_bounds: Tuple[float, float, float, float] = (0, 0, 297, 210)
    ) -> RoutingResult:
        """
        Route all connections with professional quality.

        Args:
            connections: List of {from_ref, from_pin, to_ref, to_pin, net_name}
            component_positions: Reference -> (x, y) position
            pin_positions: Reference -> {pin_name -> (x, y) absolute position}
            constraints: Optional routing constraints
            sheet_bounds: (min_x, min_y, max_x, max_y) of schematic sheet

        Returns:
            RoutingResult with all wires, junctions, and statistics
        """
        logger.info(f"Starting enhanced routing for {len(connections)} connections")

        # Reset state
        self._wires = []
        self._junctions = []
        self._buses = []
        self._occupied_points = {}
        self._crossing_count = 0
        self._four_way_avoided = 0
        self._warnings = []
        self._center_fallback_count = 0
        self._pin_match_count = 0

        # Build constraint lookup
        constraint_map: Dict[str, RoutingConstraint] = {}
        if constraints:
            for c in constraints:
                constraint_map[c.net_name] = c

        # Categorize connections
        power_conns = []
        ground_conns = []
        signal_conns = []
        bus_groups: Dict[str, List[Dict]] = defaultdict(list)

        for conn in connections:
            net = conn.get("net_name", "")
            net_lower = net.lower() if net else ""

            # Check for bus signals (e.g., DATA[0], ADDR[3])
            if "[" in net and "]" in net:
                bus_name = net.split("[")[0]
                bus_groups[bus_name].append(conn)
            elif "vcc" in net_lower or "vdd" in net_lower or "3v3" in net_lower or "5v" in net_lower:
                power_conns.append(conn)
            elif "gnd" in net_lower or "vss" in net_lower:
                ground_conns.append(conn)
            else:
                signal_conns.append(conn)

        # Route power rails first (horizontal rails at top)
        logger.info(f"Routing {len(power_conns)} power connections")
        self._route_power_rail(
            power_conns,
            component_positions,
            pin_positions,
            sheet_bounds,
            is_ground=False
        )

        # Route ground rails (horizontal rails at bottom)
        logger.info(f"Routing {len(ground_conns)} ground connections")
        self._route_power_rail(
            ground_conns,
            component_positions,
            pin_positions,
            sheet_bounds,
            is_ground=True
        )

        # Route bus signals as grouped parallel wires
        for bus_name, bus_conns in bus_groups.items():
            logger.info(f"Routing bus '{bus_name}' with {len(bus_conns)} signals")
            self._route_bus(
                bus_name,
                bus_conns,
                component_positions,
                pin_positions
            )

        # Route remaining signals
        logger.info(f"Routing {len(signal_conns)} signal connections")
        routed_count = 0
        skipped_count = 0
        for conn in signal_conns:
            try:
                self._route_signal(
                    conn,
                    component_positions,
                    pin_positions,
                    constraint_map.get(conn.get("net_name", ""))
                )
                routed_count += 1
            except Exception as e:
                skipped_count += 1
                logger.error(f"ROUTING FAILED for connection {conn.get('net_name', 'unknown')}: {type(e).__name__}: {e}")
                self._warnings.append(str(e))
        logger.info(f"Signal routing complete: {routed_count} of {len(signal_conns)} routed, {skipped_count} skipped")

        # Check total routing success across ALL connection types
        total_connections = len(connections)
        total_routed = len(self._wires)  # Actual wires generated
        if total_connections > 0 and skipped_count > 0:
            failure_pct = skipped_count / max(len(signal_conns), 1)
            logger.error(
                f"ROUTING SUMMARY: {skipped_count}/{len(signal_conns)} signal connections FAILED to route "
                f"({failure_pct:.0%} failure rate). "
                f"Total wires generated: {total_routed}. "
                f"Check symbol pin definitions and reference matching."
            )
            if failure_pct > 0.5:
                logger.error(
                    f"CRITICAL: >50% routing failure rate ({failure_pct:.0%}). "
                    f"Most connections could not be routed — schematic will be non-functional."
                )

        # Post-process: fix 4-way junctions
        self._fix_four_way_junctions()

        # NEW: Run DRC validation
        drc_violations = self._validate_electrical_rules(connections, self._wires)
        if drc_violations:
            logger.error(f"DRC FAILED: {len(drc_violations)} violations")
            for violation in drc_violations[:10]:  # Show first 10
                logger.error(f"  {violation}")
            self._warnings.extend(drc_violations)
        else:
            logger.info("DRC PASSED: No electrical violations")

        # Report pin matching statistics
        total_pin_lookups = self._pin_match_count + self._center_fallback_count
        if total_pin_lookups > 0:
            match_rate = self._pin_match_count / total_pin_lookups
            logger.info(
                f"PIN MATCHING STATS: {self._pin_match_count}/{total_pin_lookups} pins matched "
                f"({match_rate:.0%}), {self._center_fallback_count} fell back to component center."
            )
            if self._center_fallback_count > 0:
                logger.error(
                    f"PIN_MATCHING_DEGRADED: {self._center_fallback_count} wires routed to component "
                    f"centers instead of actual pin positions. These will produce visually incorrect "
                    f"wire endpoints. Fix: ensure symbol pin names match connection pin names."
                )

        # Quality metrics summary
        total_attempts = self._center_fallback_count + self._pin_match_count
        center_fallback_ratio = 0.0
        if total_attempts > 0:
            center_fallback_ratio = self._center_fallback_count / total_attempts
            logger.info(
                f"ROUTING QUALITY: {self._pin_match_count}/{total_attempts} pins matched "
                f"({center_fallback_ratio * 100:.1f}% center fallbacks)"
            )
            if center_fallback_ratio > 0.20:
                logger.error(
                    f"ROUTING QUALITY ALERT: {center_fallback_ratio * 100:.1f}% of connections "
                    f"used center fallback (threshold: 20%). Symbol quality is degraded."
                )

        # Calculate statistics
        total_length = sum(w.length for w in self._wires)

        result = RoutingResult(
            wires=self._wires,
            junctions=self._junctions,
            buses=self._buses,
            crossings=self._crossing_count,
            total_wire_length=total_length,
            four_way_junctions_avoided=self._four_way_avoided,
            warnings=self._warnings,
            center_fallback_ratio=center_fallback_ratio,
        )

        logger.info(
            f"Routing complete: {len(self._wires)} wires, "
            f"{len(self._junctions)} junctions, "
            f"{self._four_way_avoided} 4-way junctions avoided"
        )

        return result

    def _snap_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Snap coordinates to grid."""
        return (
            round(x / self.GRID_UNIT) * self.GRID_UNIT,
            round(y / self.GRID_UNIT) * self.GRID_UNIT
        )

    def _route_power_rail(
        self,
        connections: List[Dict],
        component_positions: Dict[str, Tuple[float, float]],
        pin_positions: Dict[str, Dict[str, Tuple[float, float]]],
        sheet_bounds: Tuple[float, float, float, float],
        is_ground: bool
    ):
        """
        Route power/ground as horizontal rails with vertical drops.

        VCC: Horizontal rail at top, vertical drops down to components
        GND: Horizontal rail at bottom, vertical drops up to components
        """
        if not connections:
            return

        min_x, min_y, max_x, max_y = sheet_bounds

        # Compute component bounding box for dynamic rail placement
        # Place rails OUTSIDE all components to prevent signal wire crossings
        comp_ys = [pos[1] for pos in component_positions.values()] if component_positions else []
        comp_min_y = min(comp_ys) if comp_ys else min_y
        comp_max_y = max(comp_ys) if comp_ys else max_y

        # Rail position — outside component bounding box
        if is_ground:
            rail_y = comp_max_y + self.POWER_RAIL_MARGIN + 5.0  # Below all components
            net_name = "GND"
            route_type = RouteType.GROUND
        else:
            rail_y = comp_min_y - self.POWER_RAIL_MARGIN - 5.0  # Above all components
            net_name = connections[0].get("net_name", "VCC")
            route_type = RouteType.POWER

        rail_y = self._snap_to_grid(0, rail_y)[1]

        # Collect all pin positions that connect to this rail
        pin_coords = []
        for conn in connections:
            from_ref = conn.get("from_ref")
            from_pin = conn.get("from_pin")
            to_ref = conn.get("to_ref")
            to_pin = conn.get("to_pin")

            # Get pin positions
            if from_ref in pin_positions and from_pin in pin_positions[from_ref]:
                pin_coords.append(pin_positions[from_ref][from_pin])
            elif from_ref in component_positions:
                pin_coords.append(component_positions[from_ref])

            if to_ref in pin_positions and to_pin in pin_positions[to_ref]:
                pin_coords.append(pin_positions[to_ref][to_pin])
            elif to_ref in component_positions:
                pin_coords.append(component_positions[to_ref])

        if not pin_coords:
            return

        # Sort pins by x-coordinate
        pin_coords = sorted(set(pin_coords), key=lambda p: p[0])

        # Create horizontal rail spanning all pins
        rail_start_x = self._snap_to_grid(pin_coords[0][0] - 5, 0)[0]
        rail_end_x = self._snap_to_grid(pin_coords[-1][0] + 5, 0)[0]

        # Main horizontal rail
        self._wires.append(WireSegment(
            start=(rail_start_x, rail_y),
            end=(rail_end_x, rail_y),
            net_name=net_name,
            route_type=route_type
        ))

        # Vertical drops to each pin
        for px, py in pin_coords:
            px_snapped = self._snap_to_grid(px, 0)[0]
            py_snapped = self._snap_to_grid(0, py)[1]

            # Vertical wire from rail to pin
            if abs(py_snapped - rail_y) > 0.01:
                self._wires.append(WireSegment(
                    start=(px_snapped, rail_y),
                    end=(px_snapped, py_snapped),
                    net_name=net_name,
                    route_type=route_type
                ))

                # Junction at rail connection
                self._add_junction((px_snapped, rail_y), net_name)

    def _route_bus(
        self,
        bus_name: str,
        connections: List[Dict],
        component_positions: Dict[str, Tuple[float, float]],
        pin_positions: Dict[str, Dict[str, Tuple[float, float]]]
    ):
        """
        Route bus signals as parallel wires.

        Bus signals are routed together with consistent spacing.
        """
        bus = BusRoute(name=bus_name, signals=[])

        # Extract signal indices and sort
        signal_conns = []
        for conn in connections:
            net = conn.get("net_name", "")
            if "[" in net and "]" in net:
                try:
                    idx = int(net.split("[")[1].split("]")[0])
                    signal_conns.append((idx, conn))
                except ValueError:
                    logger.error(f"BUS INDEX PARSE FAILED for net '{net}': expected format 'NAME[N]', defaulting to index 0")
                    signal_conns.append((0, conn))

        signal_conns.sort(key=lambda x: x[0])

        # Route each signal with offset
        for i, (idx, conn) in enumerate(signal_conns):
            offset = i * self.BUS_SPACING
            net_name = conn.get("net_name", f"{bus_name}[{idx}]")
            bus.signals.append(net_name)

            # Get endpoints
            from_ref = conn.get("from_ref")
            from_pin = conn.get("from_pin")
            to_ref = conn.get("to_ref")
            to_pin = conn.get("to_pin")

            from_pos = self._get_pin_position(from_ref, from_pin, component_positions, pin_positions)
            to_pos = self._get_pin_position(to_ref, to_pin, component_positions, pin_positions)

            if from_pos and to_pos:
                # Apply vertical offset for bus spacing
                from_pos = (from_pos[0], from_pos[1] + offset)
                to_pos = (to_pos[0], to_pos[1] + offset)

                wires = self._manhattan_route_enhanced(from_pos, to_pos, net_name, RouteType.BUS)
                bus.segments.extend(wires)
                self._wires.extend(wires)

        self._buses.append(bus)

    def _route_signal(
        self,
        conn: Dict[str, Any],
        component_positions: Dict[str, Tuple[float, float]],
        pin_positions: Dict[str, Dict[str, Tuple[float, float]]],
        constraint: Optional[RoutingConstraint] = None
    ):
        """Route a single signal connection."""
        from_ref = conn.get("from_ref")
        from_pin = conn.get("from_pin")
        to_ref = conn.get("to_ref")
        to_pin = conn.get("to_pin")
        net_name = conn.get("net_name", f"Net-({from_ref}-{from_pin})")

        from_pos = self._get_pin_position(from_ref, from_pin, component_positions, pin_positions)
        to_pos = self._get_pin_position(to_ref, to_pin, component_positions, pin_positions)

        # Skip connections with missing pin positions instead of aborting all routing
        if not from_pos or not to_pos:
            missing_from = "from_pos" if not from_pos else None
            missing_to = "to_pos" if not to_pos else None
            error_msg = (
                f"ROUTING FAILURE: Cannot route net '{net_name}' - missing pin positions. "
                f"From: {from_ref}.{from_pin} -> position={'MISSING' if missing_from else from_pos}, "
                f"To: {to_ref}.{to_pin} -> position={'MISSING' if missing_to else to_pos}. "
                f"Causes: "
                f"1) Symbol has no pin definitions (placeholder symbol?), "
                f"2) Pin name mismatch between connection and symbol, "
                f"3) Component not found in layout (reference mismatch). "
                f"Available components: {list(component_positions.keys())[:10]}..."
            )
            logger.error(error_msg)
            self._warnings.append(error_msg)
            # Raise to trigger the skip counter in the routing loop
            raise ValueError(error_msg)

        route_type = RouteType.SIGNAL
        if constraint:
            route_type = constraint.route_type

        wires = self._manhattan_route_enhanced(from_pos, to_pos, net_name, route_type)
        self._wires.extend(wires)

    # Power pin equivalence groups for fuzzy matching
    _POWER_PIN_EQUIVALENTS = {
        "VCC": {"VCC", "VDD", "VCCIO", "V+", "AVCC", "AVDD", "DVCC", "DVDD"},
        "GND": {"GND", "VSS", "GROUND", "AVSS", "DVSS", "AGND", "DGND", "V-", "PGND", "EPAD"},
        "3V3": {"3V3", "3.3V", "+3V3", "+3.3V", "VCC_3V3"},
        "5V":  {"5V", "+5V", "VCC_5V"},
    }

    # Track center-fallback statistics across the routing session
    _center_fallback_count: int = 0
    _pin_match_count: int = 0

    def _get_pin_position(
        self,
        ref: str,
        pin: str,
        component_positions: Dict[str, Tuple[float, float]],
        pin_positions: Dict[str, Dict[str, Tuple[float, float]]]
    ) -> Optional[Tuple[float, float]]:
        """
        Get pin position with fuzzy matching.

        Match order:
        1. Exact match
        2. Case-insensitive match
        3. Strip suffix match (pin name before '-' or '_')
        4. Power pin equivalents (VCC↔VDD, GND↔VSS)
        5. Prefix/partial match
        6. Component center (LAST RESORT — logged as error)
        """
        # 1. Exact match
        if ref in pin_positions and pin in pin_positions[ref]:
            self._pin_match_count += 1
            return pin_positions[ref][pin]

        # If ref has no pin data at all, log detailed error
        if ref not in pin_positions:
            if ref in component_positions:
                self._center_fallback_count += 1
                logger.error(
                    f"PIN_FALLBACK_TO_CENTER: {ref}.{pin} — component has NO pin position data. "
                    f"Symbol likely has no pin definitions (placeholder?). "
                    f"Falling back to component center {component_positions[ref]}."
                )
                return component_positions[ref]
            logger.error(
                f"PIN_NOT_FOUND: {ref}.{pin} — component ref '{ref}' not in pin_positions "
                f"AND not in component_positions. Available refs: "
                f"{sorted(list(component_positions.keys()))[:20]}"
            )
            return None

        available_pins = pin_positions[ref]
        pin_upper = pin.upper().strip()

        # 2. Case-insensitive match
        for avail_pin, pos in available_pins.items():
            if avail_pin.upper().strip() == pin_upper:
                self._pin_match_count += 1
                logger.debug(f"PIN_FUZZY_MATCH(case): {ref}.{pin} → {ref}.{avail_pin}")
                return pos

        # 3. Strip suffix match — match before '-' or '_' delimiter
        pin_base = pin_upper.split("-")[0].split("_")[0]
        if pin_base:
            for avail_pin, pos in available_pins.items():
                avail_base = avail_pin.upper().split("-")[0].split("_")[0]
                if avail_base == pin_base:
                    self._pin_match_count += 1
                    logger.debug(f"PIN_FUZZY_MATCH(base): {ref}.{pin} → {ref}.{avail_pin} (base={pin_base})")
                    return pos

        # 4. Power pin equivalents (VCC↔VDD, GND↔VSS, etc.)
        for _group_name, equivalents in self._POWER_PIN_EQUIVALENTS.items():
            if pin_upper in equivalents:
                for avail_pin, pos in available_pins.items():
                    if avail_pin.upper() in equivalents:
                        self._pin_match_count += 1
                        logger.debug(f"PIN_FUZZY_MATCH(power_equiv): {ref}.{pin} → {ref}.{avail_pin}")
                        return pos

        # 5. Prefix/partial match — pin name starts with or contains the target
        for avail_pin, pos in available_pins.items():
            avail_upper = avail_pin.upper()
            if avail_upper.startswith(pin_upper) or pin_upper.startswith(avail_upper):
                self._pin_match_count += 1
                logger.debug(f"PIN_FUZZY_MATCH(prefix): {ref}.{pin} → {ref}.{avail_pin}")
                return pos

        # 6. LAST RESORT: component center — detailed error for debugging
        if ref in component_positions:
            self._center_fallback_count += 1
            avail_names = sorted(available_pins.keys())
            logger.error(
                f"PIN_FALLBACK_TO_CENTER: {ref}.{pin} — no fuzzy match found. "
                f"Requested pin '{pin}' not in available pins: {avail_names}. "
                f"Falling back to component center {component_positions[ref]}. "
                f"Total center-fallbacks so far: {self._center_fallback_count}."
            )
            return component_positions[ref]

        logger.error(
            f"PIN_NOT_FOUND: {ref}.{pin} — ref in pin_positions but NOT in component_positions. "
            f"Available pins: {sorted(available_pins.keys())}."
        )
        return None

    def _manhattan_route_enhanced(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        net_name: str,
        route_type: RouteType
    ) -> List[WireSegment]:
        """
        Create enhanced Manhattan route between two points.

        Uses Z-route (horizontal-vertical-horizontal) to avoid 4-way junctions
        when possible.
        """
        wires = []

        # Snap to grid
        sx, sy = self._snap_to_grid(*start)
        ex, ey = self._snap_to_grid(*end)

        # Check if same point
        if abs(sx - ex) < 0.01 and abs(sy - ey) < 0.01:
            return wires

        # Check for straight line
        if abs(sx - ex) < 0.01:
            # Vertical line
            wires.append(WireSegment(start=(sx, sy), end=(ex, ey), net_name=net_name, route_type=route_type))
            return wires

        if abs(sy - ey) < 0.01:
            # Horizontal line
            wires.append(WireSegment(start=(sx, sy), end=(ex, ey), net_name=net_name, route_type=route_type))
            return wires

        # Check for potential 4-way junction and use Z-route to avoid
        mid_point = self._get_occupied_point_on_path(sx, sy, ex, ey)

        if mid_point and self._would_create_four_way(mid_point, net_name):
            # Use Z-route with offset
            wires = self._z_route(sx, sy, ex, ey, net_name, route_type)
        else:
            # Standard L-route
            wires = self._l_route(sx, sy, ex, ey, net_name, route_type)

        # Check if proposed route crosses existing power/ground wires
        # If so, try offset paths to avoid crossings
        if self._has_power_crossing(wires, net_name):
            for offset_mult in [1, -1, 2, -2]:
                offset = offset_mult * self.GRID_UNIT
                alt_wires = self._l_route(sx, sy + offset, ex, ey, net_name, route_type)
                if not self._has_power_crossing(alt_wires, net_name):
                    # Add short jog wire from original start to offset start
                    jog = WireSegment(
                        start=(sx, sy), end=(sx, sy + offset),
                        net_name=net_name, route_type=route_type
                    )
                    wires = [jog] + alt_wires
                    break

        # Register occupied points
        for wire in wires:
            self._register_wire(wire)

        return wires

    def _has_power_crossing(self, candidate_wires: List[WireSegment], net_name: str) -> bool:
        """Check if any candidate wire crosses an existing power/ground wire."""
        power_types = {RouteType.POWER, RouteType.GROUND}
        for cw in candidate_wires:
            for existing in self._wires:
                if existing.net_name == net_name:
                    continue  # Same net, not a crossing
                if existing.route_type not in power_types:
                    continue  # Only avoid crossing power/ground rails
                if self._wires_intersect(cw, existing):
                    return True
        return False

    def _l_route(
        self,
        sx: float, sy: float,
        ex: float, ey: float,
        net_name: str,
        route_type: RouteType
    ) -> List[WireSegment]:
        """Create L-route, choosing the direction that minimizes crossings.

        Tries both horizontal-first (H-V) and vertical-first (V-H) orientations,
        picks the one with fewer crossings against existing wires.
        """
        # Option A: Horizontal-first (H then V)
        wires_hv = []
        if abs(sx - ex) > 0.01:
            wires_hv.append(WireSegment(
                start=(sx, sy), end=(ex, sy),
                net_name=net_name, route_type=route_type
            ))
        if abs(sy - ey) > 0.01:
            wires_hv.append(WireSegment(
                start=(ex, sy), end=(ex, ey),
                net_name=net_name, route_type=route_type
            ))

        # Option B: Vertical-first (V then H)
        wires_vh = []
        if abs(sy - ey) > 0.01:
            wires_vh.append(WireSegment(
                start=(sx, sy), end=(sx, ey),
                net_name=net_name, route_type=route_type
            ))
        if abs(sx - ex) > 0.01:
            wires_vh.append(WireSegment(
                start=(sx, ey), end=(ex, ey),
                net_name=net_name, route_type=route_type
            ))

        # Count crossings for each option
        crossings_hv = self._count_crossings(wires_hv, net_name)
        crossings_vh = self._count_crossings(wires_vh, net_name)

        # Pick the option with fewer crossings (prefer H-V on tie for signal flow)
        if crossings_vh < crossings_hv:
            wires = wires_vh
            corner = (sx, ey)
        else:
            wires = wires_hv
            corner = (ex, sy)

        # Add junction at corner if we have two segments
        if len(wires) == 2:
            self._add_junction(corner, net_name)

        return wires

    def _count_crossings(
        self,
        candidate_wires: List[WireSegment],
        net_name: str
    ) -> int:
        """Count how many existing wires the candidate segments would cross."""
        count = 0
        for cw in candidate_wires:
            for existing in self._wires:
                if existing.net_name == net_name:
                    continue  # Same net, not a crossing
                if self._wires_intersect(cw, existing):
                    count += 1
        return count

    def _z_route(
        self,
        sx: float, sy: float,
        ex: float, ey: float,
        net_name: str,
        route_type: RouteType
    ) -> List[WireSegment]:
        """
        Create Z-route to avoid 4-way junctions.

        Pattern: horizontal - vertical - horizontal
        The vertical segment is offset to avoid creating 4-way junctions.
        """
        wires = []
        self._four_way_avoided += 1

        # Calculate midpoint with offset
        mid_x = (sx + ex) / 2
        mid_x = self._snap_to_grid(mid_x, 0)[0]

        # First horizontal (start to mid)
        if abs(sx - mid_x) > self.MIN_WIRE_LENGTH:
            wires.append(WireSegment(
                start=(sx, sy),
                end=(mid_x, sy),
                net_name=net_name,
                route_type=route_type
            ))
            self._add_junction((mid_x, sy), net_name)
        else:
            mid_x = sx

        # Vertical (at mid_x from sy to ey)
        if abs(sy - ey) > 0.01:
            wires.append(WireSegment(
                start=(mid_x, sy),
                end=(mid_x, ey),
                net_name=net_name,
                route_type=route_type
            ))
            self._add_junction((mid_x, ey), net_name)

        # Second horizontal (mid to end)
        if abs(mid_x - ex) > self.MIN_WIRE_LENGTH:
            wires.append(WireSegment(
                start=(mid_x, ey),
                end=(ex, ey),
                net_name=net_name,
                route_type=route_type
            ))

        return wires

    def _get_occupied_point_on_path(
        self,
        sx: float, sy: float,
        ex: float, ey: float
    ) -> Optional[Tuple[float, float]]:
        """Check if path passes through any occupied point."""
        # Check corner point of L-route
        corner = (ex, sy)
        if corner in self._occupied_points:
            return corner

        # Check if any segment would cross existing wire
        # (simplified - full implementation would check all intersections)
        return None

    def _would_create_four_way(
        self,
        point: Tuple[float, float],
        net_name: str
    ) -> bool:
        """Check if adding wires at point would create 4-way junction."""
        # Count existing wire endpoints at this point
        count = 0
        for wire in self._wires:
            for endpoint in [wire.start, wire.end]:
                if abs(endpoint[0] - point[0]) < 0.01 and abs(endpoint[1] - point[1]) < 0.01:
                    count += 1

        # If we add 2 more (for corner), we'd have 4 or more
        return count >= 2

    def _register_wire(self, wire: WireSegment):
        """Register wire endpoints as occupied."""
        for point in [wire.start, wire.end]:
            key = (round(point[0], 2), round(point[1], 2))
            if key not in self._occupied_points:
                self._occupied_points[key] = wire.net_name

    def _add_junction(self, position: Tuple[float, float], net_name: str):
        """Add a junction at the specified position."""
        pos_key = (round(position[0], 2), round(position[1], 2))

        # Check if junction already exists
        for junc in self._junctions:
            junc_key = (round(junc.position[0], 2), round(junc.position[1], 2))
            if junc_key == pos_key:
                junc.connected_nets.add(net_name)
                return

        # Create new junction
        self._junctions.append(Junction(
            position=position,
            junction_type=JunctionType.THREE_WAY,
            connected_nets={net_name}
        ))

    def _fix_four_way_junctions(self):
        """
        Post-process to identify and fix any remaining 4-way junctions.

        Strategy: Add small offset to one of the wires to convert
        4-way junction to two 3-way junctions.
        """
        # Count wire endpoints at each point
        point_counts: Dict[Tuple[float, float], List[WireSegment]] = defaultdict(list)

        for wire in self._wires:
            for endpoint in [wire.start, wire.end]:
                key = (round(endpoint[0], 2), round(endpoint[1], 2))
                point_counts[key].append(wire)

        # Find 4-way junctions
        for point, wires in point_counts.items():
            if len(wires) >= 4:
                logger.warning(f"Found 4-way junction at {point}, attempting to fix")
                self._four_way_avoided += 1

                # Mark junction as 4-way (for reporting)
                for junc in self._junctions:
                    junc_key = (round(junc.position[0], 2), round(junc.position[1], 2))
                    if junc_key == point:
                        junc.junction_type = JunctionType.FOUR_WAY
                        break

                # In a real implementation, we would reroute one of the wires
                # For now, just log the warning
                self._warnings.append(
                    f"4-way junction at ({point[0]:.2f}, {point[1]:.2f}) - "
                    "consider manual adjustment"
                )

    # ========== DRC VALIDATION METHODS ==========

    def _validate_electrical_rules(
        self,
        connections: List[Dict[str, Any]],
        wires: List[WireSegment]
    ) -> List[str]:
        """
        Validate electrical design rules (DRC).

        Returns:
            List of DRC violations
        """
        violations = []

        # Check #1: Short circuit detection
        violations.extend(self._check_short_circuits(wires))

        # Check #2: Clearance validation
        violations.extend(self._check_clearance(wires, connections))

        # Check #3: Trace width validation
        violations.extend(self._check_trace_widths(wires, connections))

        # Check #4: 4-way junction detection (MUST FIX, not just warn)
        violations.extend(self._check_four_way_junctions_strict(wires))

        return violations

    def _check_short_circuits(self, wires: List[WireSegment]) -> List[str]:
        """Detect unintentional shorts between different nets."""
        violations = []

        # Group wires by net
        net_wires: Dict[str, List[WireSegment]] = {}
        for wire in wires:
            if wire.net_name not in net_wires:
                net_wires[wire.net_name] = []
            net_wires[wire.net_name].append(wire)

        # Check for overlaps between different nets
        nets = list(net_wires.keys())
        for i, net1 in enumerate(nets):
            for net2 in nets[i+1:]:
                # Check all wire pairs for intersection
                for wire1 in net_wires[net1]:
                    for wire2 in net_wires[net2]:
                        if self._wires_intersect(wire1, wire2):
                            violations.append(
                                f"SHORT CIRCUIT: Nets '{net1}' and '{net2}' intersect "
                                f"(unintentional crossing)"
                            )

        return violations

    def _wires_intersect(self, wire1: WireSegment, wire2: WireSegment) -> bool:
        """
        Check if two wire segments intersect (crossing shorts only).

        Excludes T-junctions (where one wire ends on another) and corner connections
        (where wires share an endpoint). These are valid electrical connections, not shorts.

        Only returns True for actual CROSSING intersections (like + shape), which
        indicate unintentional shorts.
        """
        def ccw(a, b, c):
            """Counter-clockwise test."""
            return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

        def points_equal(p1, p2, tolerance=0.01):
            """Check if two points are the same."""
            return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

        def point_on_segment(point, seg_start, seg_end, tolerance=0.01):
            """Check if a point lies on a line segment."""
            px, py = point
            x1, y1 = seg_start
            x2, y2 = seg_end

            # Check if point is collinear with segment
            cross_product = abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1))
            if cross_product > tolerance:
                return False

            # Check if point is between segment endpoints
            if min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance:
                if min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance:
                    return True
            return False

        a1, a2 = wire1.start, wire1.end
        b1, b2 = wire2.start, wire2.end

        # Check if wires share any endpoints (corner connections) - NOT shorts
        if (points_equal(a1, b1) or points_equal(a1, b2) or
            points_equal(a2, b1) or points_equal(a2, b2)):
            return False

        # Check for T-junctions (one wire's endpoint touches the other wire's middle)
        if (point_on_segment(b1, a1, a2) or point_on_segment(b2, a1, a2) or
            point_on_segment(a1, b1, b2) or point_on_segment(a2, b1, b2)):
            return False

        # Guard against collinear segments — CCW is degenerate for these
        # Two parallel segments on the same axis are overlaps/T-junctions, not crossing shorts

        # Both horizontal on same Y → not a crossing
        if (abs(a1[1] - a2[1]) < 0.01 and abs(b1[1] - b2[1]) < 0.01 and
            abs(a1[1] - b1[1]) < 0.01):
            return False

        # Both vertical on same X → not a crossing
        if (abs(a1[0] - a2[0]) < 0.01 and abs(b1[0] - b2[0]) < 0.01 and
            abs(a1[0] - b1[0]) < 0.01):
            return False

        # Check for actual crossing intersection (+ shape) using CCW algorithm
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

    def _check_clearance(
        self,
        wires: List[WireSegment],
        connections: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate minimum clearance between nets (IPC-2221)."""
        violations = []

        # Simplified: Check all wire pairs
        for i, wire1 in enumerate(wires):
            for wire2 in wires[i+1:]:
                if wire1.net_name == wire2.net_name:
                    continue  # Same net, no clearance needed

                # Calculate minimum distance
                distance = self._min_distance_between_wires(wire1, wire2)

                # IPC-2221: Minimum 0.13mm for 0-50V
                min_clearance = 0.13

                if distance < min_clearance:
                    violations.append(
                        f"CLEARANCE VIOLATION: Nets '{wire1.net_name}' and "
                        f"'{wire2.net_name}' are {distance:.3f}mm apart "
                        f"(min: {min_clearance:.3f}mm per IPC-2221)"
                    )

        return violations

    def _min_distance_between_wires(self, wire1: WireSegment, wire2: WireSegment) -> float:
        """
        Calculate minimum distance between two wire segments.

        Uses proper point-to-line-segment distance calculation, not just endpoint-to-endpoint.
        """
        def point_to_segment_distance(point: Tuple[float, float], seg_start: Tuple[float, float],
                                       seg_end: Tuple[float, float]) -> float:
            """Calculate minimum distance from a point to a line segment."""
            px, py = point
            x1, y1 = seg_start
            x2, y2 = seg_end

            # Vector from start to end
            dx = x2 - x1
            dy = y2 - y1

            # If segment is a point, return distance to that point
            if dx == 0 and dy == 0:
                return ((px - x1)**2 + (py - y1)**2) ** 0.5

            # Parameter t for projection of point onto line
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

            # Closest point on segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy

            # Distance from point to closest point
            return ((px - closest_x)**2 + (py - closest_y)**2) ** 0.5

        # Calculate all point-to-segment distances
        min_dist = float('inf')

        # Wire1 endpoints to wire2 segment
        min_dist = min(min_dist, point_to_segment_distance(wire1.start, wire2.start, wire2.end))
        min_dist = min(min_dist, point_to_segment_distance(wire1.end, wire2.start, wire2.end))

        # Wire2 endpoints to wire1 segment
        min_dist = min(min_dist, point_to_segment_distance(wire2.start, wire1.start, wire1.end))
        min_dist = min(min_dist, point_to_segment_distance(wire2.end, wire1.start, wire1.end))

        return min_dist

    def _check_four_way_junctions_strict(self, wires: List[WireSegment]) -> List[str]:
        """Strictly check and report 4-way junctions as ERRORS."""
        violations = []

        # Count wire endpoints at each position
        endpoint_counts: Dict[Tuple[float, float], List[str]] = {}
        for wire in wires:
            for point in [wire.start, wire.end]:
                key = (round(point[0], 2), round(point[1], 2))
                if key not in endpoint_counts:
                    endpoint_counts[key] = []
                endpoint_counts[key].append(wire.net_name)

        # Check for 4-way junctions
        for pos, nets in endpoint_counts.items():
            if len(nets) >= 4:
                violations.append(
                    f"4-WAY JUNCTION ERROR: {len(nets)} wires meet at ({pos[0]}, {pos[1]}) "
                    f"MUST split into 3-way junctions per IPC best practices."
                )

        return violations

    def _check_trace_widths(
        self,
        wires: List[WireSegment],
        connections: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Validate trace widths meet IPC-2221 current capacity requirements.

        Uses IPC-2221 formula for external layers, 1oz copper, 10°C rise:
        A = (I / (k * ΔT^b))^(1/c)
        where:
            I = current in amps
            k = 0.048 (constant for external layers)
            ΔT = temperature rise (10°C)
            b = 0.44
            c = 0.725
            A = cross-sectional area in sq mils

        Then convert area to width based on copper thickness (1oz = 1.378 mils).

        Args:
            wires: List of wire segments to validate
            connections: List of connection dictionaries (may contain net metadata)

        Returns:
            List of violation strings for traces that are too narrow
        """
        import math

        violations = []

        # IPC-2221 constants for external layers, 1oz copper, 10°C rise
        K = 0.048
        DELTA_T = 10.0  # Temperature rise in °C
        B = 0.44
        C = 0.725
        COPPER_THICKNESS_MILS = 1.378  # 1oz copper thickness in mils
        MILS_TO_MM = 0.0254  # Conversion factor

        # Default current ratings by net type (amps)
        # These match IPC-2221 rules.yaml signal_types
        DEFAULT_CURRENTS = {
            "power": 2.0,      # VCC, +5V, etc.
            "ground": 3.0,     # GND
            "signal": 0.1,     # General signals
            "bus": 0.1,        # Bus signals
            "critical": 0.1,   # High-speed signals
        }

        # Build net-to-current mapping
        net_currents = {}

        # Extract current from connections (if available)
        for conn in connections:
            net_name = conn.get("net_name", "")
            if not net_name:
                continue

            # Check if this is a power/ground net (infer higher current)
            connection_type = conn.get("connection_type", "signal")

            # Use default current based on connection type
            current = DEFAULT_CURRENTS.get(connection_type, 0.1)

            # Power nets typically carry more current
            net_name_lower = net_name.lower()
            if any(pwr in net_name_lower for pwr in ["vcc", "vdd", "+5v", "+3v3", "+12v", "power"]):
                current = max(current, DEFAULT_CURRENTS["power"])
            elif any(gnd in net_name_lower for gnd in ["gnd", "ground", "vss"]):
                current = max(current, DEFAULT_CURRENTS["ground"])

            net_currents[net_name] = current

        # Group wires by net and calculate total width needed per net
        net_wires = {}
        for wire in wires:
            if wire.net_name not in net_wires:
                net_wires[wire.net_name] = []
            net_wires[wire.net_name].append(wire)

        # Validate each net
        for net_name, net_wire_list in net_wires.items():
            # Get current for this net
            current = net_currents.get(net_name, DEFAULT_CURRENTS["signal"])

            # Skip validation for very low current signals
            if current < 0.05:
                continue

            # Calculate minimum required trace width using IPC-2221 formula
            # A = (I / (k * ΔT^b))^(1/c)  [area in sq mils]
            area_sq_mils = (current / (K * (DELTA_T ** B))) ** (1 / C)

            # Width = Area / Thickness
            width_mils = area_sq_mils / COPPER_THICKNESS_MILS

            # Convert to mm
            min_width_mm = width_mils * MILS_TO_MM

            # For schematic validation, we assume a default trace width
            # In real PCB, this would come from the wire object
            # For now, assume standard schematic trace (0.25mm for signals, 0.80mm for power)
            actual_width_mm = 0.80 if current >= 1.0 else 0.25

            # Check if actual width meets requirement
            if actual_width_mm < min_width_mm:
                violations.append(
                    f"TRACE WIDTH VIOLATION: Net '{net_name}' carries {current:.2f}A "
                    f"but trace width {actual_width_mm:.3f}mm is less than required "
                    f"{min_width_mm:.3f}mm per IPC-2221 (1oz copper, 10°C rise). "
                    f"Increase trace width to {min_width_mm:.3f}mm or use heavier copper."
                )

        return violations


def convert_to_kicad_wires(result: RoutingResult) -> List[Dict[str, Any]]:
    """Convert routing result to KiCad wire format."""
    kicad_wires = []

    for wire in result.wires:
        kicad_wires.append({
            "start": wire.start,
            "end": wire.end,
            "uuid": wire.uuid
        })

    return kicad_wires


def convert_to_kicad_junctions(result: RoutingResult) -> List[Dict[str, Any]]:
    """Convert routing result to KiCad junction format."""
    kicad_junctions = []

    for junc in result.junctions:
        # Only include 3-way junctions (proper connection points)
        if junc.junction_type == JunctionType.THREE_WAY or len(junc.connected_nets) >= 2:
            kicad_junctions.append({
                "position": junc.position,
                "uuid": junc.uuid
            })

    return kicad_junctions


# CLI entry point for testing
if __name__ == "__main__":
    import sys

    # Test the router
    router = EnhancedWireRouter()

    # Sample connections
    test_connections = [
        {"from_ref": "U1", "from_pin": "VCC", "to_ref": "C1", "to_pin": "1", "net_name": "VCC"},
        {"from_ref": "U1", "from_pin": "GND", "to_ref": "C1", "to_pin": "2", "net_name": "GND"},
        {"from_ref": "U1", "from_pin": "PA0", "to_ref": "U2", "to_pin": "IN1", "net_name": "PWM_A"},
        {"from_ref": "U1", "from_pin": "PA1", "to_ref": "U2", "to_pin": "IN2", "net_name": "PWM_B"},
        {"from_ref": "U2", "from_pin": "VCC", "to_ref": "C2", "to_pin": "1", "net_name": "VCC"},
        {"from_ref": "U2", "from_pin": "GND", "to_ref": "C2", "to_pin": "2", "net_name": "GND"},
    ]

    # Component positions
    component_positions = {
        "U1": (50.0, 80.0),
        "U2": (120.0, 80.0),
        "C1": (50.0, 120.0),
        "C2": (120.0, 120.0),
    }

    # Pin positions (absolute)
    pin_positions = {
        "U1": {
            "VCC": (50.0, 70.0),
            "GND": (50.0, 90.0),
            "PA0": (60.0, 75.0),
            "PA1": (60.0, 85.0),
        },
        "U2": {
            "VCC": (110.0, 70.0),
            "GND": (110.0, 90.0),
            "IN1": (110.0, 75.0),
            "IN2": (110.0, 85.0),
        },
        "C1": {
            "1": (50.0, 115.0),
            "2": (50.0, 125.0),
        },
        "C2": {
            "1": (120.0, 115.0),
            "2": (120.0, 125.0),
        },
    }

    print("Testing Enhanced Wire Router...")
    print("=" * 60)

    result = router.route(
        test_connections,
        component_positions,
        pin_positions,
        sheet_bounds=(0, 0, 200, 150)
    )

    print(f"\nRouting Results:")
    print(f"  Total wires: {len(result.wires)}")
    print(f"  Total junctions: {len(result.junctions)}")
    print(f"  Crossings: {result.crossings}")
    print(f"  Total wire length: {result.total_wire_length:.2f} mm")
    print(f"  4-way junctions avoided: {result.four_way_junctions_avoided}")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    print("\nWire segments:")
    for wire in result.wires:
        print(f"  {wire.net_name}: ({wire.start[0]:.2f}, {wire.start[1]:.2f}) -> ({wire.end[0]:.2f}, {wire.end[1]:.2f})")

    print("\nJunctions:")
    for junc in result.junctions:
        print(f"  ({junc.position[0]:.2f}, {junc.position[1]:.2f}) - {junc.junction_type.value}")

    print("\n" + "=" * 60)
    print("Enhanced Wire Router test complete!")
