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
    remaining_overlaps: int = 0


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

    # Canvas dimensions (mm) — A2 sheet provides ample room for complex schematics.
    # A4 (254×190.5) was too small for 20+ components: passives overlapped IC bodies.
    # A2 (420×594mm landscape) gives ~3.5× the area, eliminating crowding.
    CANVAS_WIDTH = 594.0   # A2 landscape width
    CANVAS_HEIGHT = 420.0  # A2 landscape height

    # Spacing rules (mm)
    SPACING_RULES = {
        "ic_to_ic": 60.0,           # 60mm between IC centers (was 40mm — too tight with large ICs)
        "ic_to_passive": 25.0,       # 25mm IC to passive (was 15mm — passives were on top of ICs)
        "passive_to_passive": 15.0,  # 15mm between passives (was 10mm)
        "bypass_to_ic": 15.0,        # 15mm bypass to IC (was 5mm — caused overlap with IC body)
        "connector_edge": 10.0,      # 10mm from edge for connectors
        "layer_spacing": 80.0,       # Horizontal spacing between layers
        "vertical_spacing": 40.0,    # Vertical spacing within layer
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
            metrics=metrics,
            remaining_overlaps=getattr(self, '_last_remaining_overlaps', 0) or 0
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
        Calculate component positions using subsystem-based placement.

        Strategy (MAPO v3.2 — rewritten for professional layout):
        1. Build subsystem regions from ideation or inferred from BOM/groups
        2. Place ICs as anchors within their subsystem region
        3. Place passives relative to their connected IC (bypass caps close)
        4. Place connectors at canvas edges (inputs left, outputs right)
        5. Collision resolution pass
        """
        positions: Dict[str, Tuple[float, float]] = {}

        # Build symbol lookup: ref -> symbol
        sym_lookup = {s.reference: s for s in symbols}

        # Build connectivity map: ref -> set of connected refs
        connectivity: Dict[str, Set[str]] = {}
        for path in analysis.signal_paths:
            refs = [path.source_component] + path.sink_components
            for r in refs:
                if r not in connectivity:
                    connectivity[r] = set()
            for r in refs[1:]:
                connectivity[path.source_component].add(r)
                if r not in connectivity:
                    connectivity[r] = set()
                connectivity[r].add(path.source_component)

        # Classify every component
        ic_refs = []
        passive_refs = []  # R, C, L
        connector_refs = []  # J
        power_refs = []  # Voltage regulators, power-related ICs
        other_refs = []
        for s in symbols:
            ref = s.reference
            if ref.startswith('J'):
                connector_refs.append(ref)
            elif ref.startswith('U'):
                ic_refs.append(ref)
            elif ref.startswith(('R', 'C', 'L')):
                passive_refs.append(ref)
            elif ref.startswith(('D', 'Q')):
                other_refs.append(ref)
            else:
                other_refs.append(ref)

        # Check for power-related ICs in separation_zones
        power_zone_refs = set(analysis.separation_zones.get('power', []))
        power_ics = [r for r in ic_refs if r in power_zone_refs]
        signal_ics = [r for r in ic_refs if r not in power_zone_refs]

        logger.info(
            f"Component classification: {len(signal_ics)} signal ICs, {len(power_ics)} power ICs, "
            f"{len(passive_refs)} passives, {len(connector_refs)} connectors, {len(other_refs)} other"
        )

        # --- Step 1: Build subsystem regions ---
        # Use functional groups from analysis (enriched by ideation if available)
        subsystems = self._build_subsystem_regions(
            analysis.functional_groups, symbols, signal_ics, power_ics,
            connector_refs, passive_refs, other_refs, connectivity
        )

        logger.info(f"Built {len(subsystems)} subsystem regions for placement")

        # --- Step 2: Allocate non-overlapping rectangular regions on canvas ---
        region_allocations = self._allocate_regions(subsystems)

        # --- Step 3: Place components within their subsystem region ---
        for subsys_name, region in region_allocations.items():
            subsys = subsystems[subsys_name]
            rx, ry, rw, rh = region  # x, y, width, height

            # Separate ICs and passives within this subsystem
            sub_ics = [r for r in subsys['components'] if r.startswith('U')]
            sub_passives = [r for r in subsys['components'] if r.startswith(('R', 'C', 'L'))]
            sub_connectors = [r for r in subsys['components'] if r.startswith('J')]
            sub_other = [r for r in subsys['components']
                         if not r.startswith(('U', 'R', 'C', 'L', 'J'))]

            # Sort ICs by connectivity centrality: most connected IC goes to center
            if sub_ics:
                ic_degree = {
                    ref: len(connectivity.get(ref, set()))
                    for ref in sub_ics
                }
                sub_ics_sorted = sorted(sub_ics, key=lambda r: ic_degree.get(r, 0), reverse=True)

                # Place most connected IC in center, others radiate outward
                ic_y = ry + rh * 0.30
                # Use the updated ic_to_ic spacing rule (60mm), not a hardcoded 40mm cap.
                # The larger canvas provides room; don't shrink spacing below the rule minimum.
                ic_spacing = max(self.SPACING_RULES["ic_to_ic"], rw / max(len(sub_ics), 1))
                center_x = rx + rw / 2

                if len(sub_ics_sorted) == 1:
                    positions[sub_ics_sorted[0]] = (center_x, ic_y)
                else:
                    # Center placement: [3rd, 1st(center), 2nd, 4th, ...]
                    center_order = []
                    left_idx = 0
                    right_idx = 0
                    for rank, ref in enumerate(sub_ics_sorted):
                        if rank == 0:
                            center_order.append((0, ref))  # Most connected → center
                        elif rank % 2 == 1:
                            right_idx += 1
                            center_order.append((right_idx, ref))
                        else:
                            left_idx -= 1
                            center_order.append((left_idx, ref))

                    for offset_idx, ref in center_order:
                        positions[ref] = (center_x + offset_idx * ic_spacing, ic_y)

                if len(sub_ics) > 1:
                    logger.debug(
                        f"  {subsys_name}: IC centrality order: "
                        + ", ".join(f"{r}({ic_degree.get(r,0)})" for r in sub_ics_sorted)
                    )

            # Place connectors at region edges
            for i, conn_ref in enumerate(sub_connectors):
                if subsys.get('edge') == 'left':
                    positions[conn_ref] = (rx + 5.0, ry + 20.0 + i * 15.0)
                elif subsys.get('edge') == 'right':
                    positions[conn_ref] = (rx + rw - 5.0, ry + 20.0 + i * 15.0)
                else:
                    positions[conn_ref] = (rx + 5.0, ry + 20.0 + i * 15.0)

            # Place passives relative to their connected IC.
            # Group passives by their primary IC, then lay them out in a horizontal
            # row BELOW the IC.  This eliminates the column-stacking that caused
            # passives to overlap IC bodies in the old algorithm.
            placed_passives = set()
            ic_passive_groups: Dict[str, List[str]] = {}
            passive_no_ic: List[str] = []

            for p_ref in sub_passives:
                connected_ics = [r for r in connectivity.get(p_ref, set())
                                 if r in positions and r.startswith('U')]
                if connected_ics:
                    primary_ic = connected_ics[0]
                    ic_passive_groups.setdefault(primary_ic, []).append(p_ref)
                else:
                    passive_no_ic.append(p_ref)

            # For each IC, place its passives in a row below
            passive_row_spacing = self.SPACING_RULES["passive_to_passive"]
            passive_row_offset_y = self.SPACING_RULES["bypass_to_ic"] + 10.0  # below IC center
            for ic_ref, passives_for_ic in ic_passive_groups.items():
                ic_pos = positions[ic_ref]
                n = len(passives_for_ic)
                # Center the row of passives below the IC
                row_total_w = (n - 1) * passive_row_spacing
                start_x = ic_pos[0] - row_total_w / 2.0
                for idx, p_ref in enumerate(passives_for_ic):
                    positions[p_ref] = (
                        start_x + idx * passive_row_spacing,
                        ic_pos[1] + passive_row_offset_y
                    )
                    placed_passives.add(p_ref)

            # Place remaining passives (not connected to any IC) in a grid below ICs
            unplaced_passives = [p for p in sub_passives if p not in placed_passives]
            unplaced_passives.extend(passive_no_ic)
            if unplaced_passives:
                passive_y = ry + rh * 0.75
                cols = max(int(len(unplaced_passives) ** 0.5) + 1, 3)
                p_spacing_x = max(passive_row_spacing, rw / (cols + 1))
                for idx, p_ref in enumerate(unplaced_passives):
                    col = idx % cols
                    row = idx // cols
                    positions[p_ref] = (
                        rx + 15.0 + col * p_spacing_x,
                        passive_y + row * passive_row_spacing
                    )

            # Place other components (diodes, transistors) near related ICs
            for i, o_ref in enumerate(sub_other):
                connected = [r for r in connectivity.get(o_ref, set())
                             if r in positions]
                if connected:
                    anchor = positions[connected[0]]
                    positions[o_ref] = (anchor[0] + 15.0, anchor[1] + 15.0 + i * 10.0)
                else:
                    positions[o_ref] = (rx + 20.0 + i * 12.0, ry + rh * 0.85)

        # --- Step 4: Ensure all symbols have positions ---
        unplaced = [s for s in symbols if s.reference not in positions]
        if unplaced:
            logger.error(
                f"LAYOUT ERROR: {len(unplaced)} components still unplaced after subsystem placement: "
                f"{[s.reference for s in unplaced][:20]}"
            )
            # Place in a grid at bottom of canvas
            for idx, symbol in enumerate(unplaced):
                col = idx % 8
                row = idx // 8
                positions[symbol.reference] = (
                    30.0 + col * 25.0,
                    self.CANVAS_HEIGHT - 40.0 + row * 15.0
                )

        # --- Step 5: Collision resolution ---
        self._last_remaining_overlaps = self._resolve_collisions(positions, symbols=symbols)

        # --- Step 6: Apply proximity constraints (bypass caps near ICs) ---
        self._apply_proximity_constraints(
            positions,
            analysis.critical_proximity_pairs
        )

        return positions

    def _build_subsystem_regions(
        self,
        functional_groups: List[FunctionalGroup],
        symbols: List[Any],
        signal_ics: List[str],
        power_ics: List[str],
        connector_refs: List[str],
        passive_refs: List[str],
        other_refs: List[str],
        connectivity: Dict[str, Set[str]],
    ) -> Dict[str, Dict]:
        """
        Build subsystem definitions for placement.

        Uses functional groups from signal flow analysis (enriched by ideation).
        Components not in any group are assigned to the closest IC's group.
        """
        subsystems: Dict[str, Dict] = {}
        assigned = set()

        # Process functional groups from analysis
        for group in functional_groups:
            if not group.components:
                continue
            # Infer category from group name
            gname = group.group_name
            gname_lower = gname.lower()
            if 'power' in gname_lower or 'supply' in gname_lower:
                cat = 'power'
            elif 'connector' in gname_lower or 'input' in gname_lower or 'output' in gname_lower:
                cat = 'connector'
            else:
                cat = 'signal'
            subsystems[gname] = {
                'components': list(group.components),
                'category': cat,
                'edge': None,
            }
            assigned.update(group.components)

        # Create power subsystem for power ICs not in any group
        unassigned_power = [r for r in power_ics if r not in assigned]
        if unassigned_power:
            subsystems['Power Supply'] = {
                'components': list(unassigned_power),
                'category': 'power',
                'edge': None,
            }
            assigned.update(unassigned_power)

        # Create connector subsystem (split input/output by analysis)
        input_conns = []
        output_conns = []
        for conn_ref in connector_refs:
            if conn_ref not in assigned:
                # Heuristic: connectors connected to sources are inputs, to sinks are outputs
                connected = connectivity.get(conn_ref, set())
                if any(r in signal_ics[:len(signal_ics)//2] for r in connected):
                    input_conns.append(conn_ref)
                else:
                    output_conns.append(conn_ref)

        if input_conns:
            subsystems['Input Connectors'] = {
                'components': input_conns,
                'category': 'connector',
                'edge': 'left',
            }
            assigned.update(input_conns)
        if output_conns:
            subsystems['Output Connectors'] = {
                'components': output_conns,
                'category': 'connector',
                'edge': 'right',
            }
            assigned.update(output_conns)
        # Any remaining connectors
        remaining_conns = [r for r in connector_refs if r not in assigned]
        if remaining_conns:
            subsystems.setdefault('Input Connectors', {
                'components': [],
                'category': 'connector',
                'edge': 'left',
            })
            subsystems['Input Connectors']['components'].extend(remaining_conns)
            assigned.update(remaining_conns)

        # Assign unassigned ICs to their own subsystem
        for ic_ref in signal_ics:
            if ic_ref not in assigned:
                subsystems[f'IC_{ic_ref}'] = {
                    'components': [ic_ref],
                    'category': 'signal',
                    'edge': None,
                }
                assigned.add(ic_ref)

        # Assign unassigned passives to the subsystem of their nearest connected IC
        for p_ref in passive_refs + other_refs:
            if p_ref in assigned:
                continue
            connected_ics = [r for r in connectivity.get(p_ref, set())
                            if r in assigned]
            if connected_ics:
                # Find which subsystem contains the connected IC
                for ss_name, ss_data in subsystems.items():
                    if connected_ics[0] in ss_data['components']:
                        ss_data['components'].append(p_ref)
                        assigned.add(p_ref)
                        break
            if p_ref not in assigned:
                # Last resort: put in "Misc" subsystem
                subsystems.setdefault('Miscellaneous', {
                    'components': [],
                    'category': 'other',
                    'edge': None,
                })
                subsystems['Miscellaneous']['components'].append(p_ref)
                assigned.add(p_ref)

        for ss_name, ss_data in subsystems.items():
            logger.info(f"  Subsystem '{ss_name}': {len(ss_data['components'])} components ({ss_data['category']})")

        return subsystems

    def _allocate_regions(
        self,
        subsystems: Dict[str, Dict],
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Allocate non-overlapping rectangular regions on the canvas.

        Returns: {subsystem_name: (x, y, width, height)}

        Layout strategy:
        - Input connectors: left edge
        - Power supply: top strip
        - Signal subsystems: middle area, arranged in rows
        - Output connectors: right edge
        """
        regions: Dict[str, Tuple[float, float, float, float]] = {}

        margin = 15.0
        usable_w = self.CANVAS_WIDTH - 2 * margin
        usable_h = self.CANVAS_HEIGHT - 2 * margin

        # Categorize subsystems
        power_ss = {k: v for k, v in subsystems.items() if v['category'] == 'power'}
        left_conn_ss = {k: v for k, v in subsystems.items()
                        if v['category'] == 'connector' and v.get('edge') == 'left'}
        right_conn_ss = {k: v for k, v in subsystems.items()
                         if v['category'] == 'connector' and v.get('edge') == 'right'}
        signal_ss = {k: v for k, v in subsystems.items()
                     if k not in power_ss and k not in left_conn_ss and k not in right_conn_ss}

        # Allocate left connector strip (10% width)
        left_w = usable_w * 0.10 if left_conn_ss else 0
        if left_conn_ss:
            y_offset = margin
            for name, ss in left_conn_ss.items():
                h = max(usable_h * 0.5, len(ss['components']) * 15.0)
                regions[name] = (margin, y_offset, left_w, min(h, usable_h))
                y_offset += h + 10.0

        # Allocate right connector strip (10% width)
        right_w = usable_w * 0.10 if right_conn_ss else 0
        if right_conn_ss:
            y_offset = margin
            for name, ss in right_conn_ss.items():
                h = max(usable_h * 0.5, len(ss['components']) * 15.0)
                regions[name] = (margin + usable_w - right_w, y_offset, right_w, min(h, usable_h))
                y_offset += h + 10.0

        # Remaining area for power + signal
        mid_x = margin + left_w + 5.0
        mid_w = usable_w - left_w - right_w - 10.0

        # Allocate power strip at top (20% height)
        power_h = usable_h * 0.20 if power_ss else 0
        if power_ss:
            x_offset = mid_x
            per_w = mid_w / max(len(power_ss), 1)
            for name, ss in power_ss.items():
                regions[name] = (x_offset, margin, per_w, power_h)
                x_offset += per_w

        # Allocate signal subsystems in the remaining area
        signal_y_start = margin + power_h + 5.0
        signal_h = usable_h - power_h - 10.0

        if signal_ss:
            # Arrange signal subsystems in a grid, weighted by component count
            num_ss = len(signal_ss)
            cols = max(int(num_ss ** 0.5 + 0.5), 1)
            rows = (num_ss + cols - 1) // cols

            # Build row lists for weight computation
            row_lists: List[List[Tuple[str, Dict]]] = [[] for _ in range(rows)]
            for idx, item in enumerate(signal_ss.items()):
                row_lists[idx // cols].append(item)

            # Compute per-row height weight (proportional to max component count in row)
            row_weights = []
            for row_items in row_lists:
                if row_items:
                    row_weights.append(max(len(ss['components']) for _, ss in row_items))
                else:
                    row_weights.append(1)
            total_row_weight = sum(row_weights) or 1

            y_cursor = signal_y_start
            for r_idx, row_items in enumerate(row_lists):
                if not row_items:
                    continue
                row_h = signal_h * (row_weights[r_idx] / total_row_weight)

                # Compute per-column width weight within this row
                col_weights = [max(len(ss['components']), 1) for _, ss in row_items]
                total_col_weight = sum(col_weights) or 1

                x_cursor = mid_x
                for c_idx, (name, ss) in enumerate(row_items):
                    cell_w = mid_w * (col_weights[c_idx] / total_col_weight)
                    regions[name] = (
                        x_cursor,
                        y_cursor,
                        cell_w - 5.0,
                        row_h - 5.0,
                    )
                    x_cursor += cell_w
                y_cursor += row_h

        return regions

    def _resolve_collisions(
        self,
        positions: Dict[str, Tuple[float, float]],
        symbols: Optional[List] = None,
    ):
        """Push overlapping components apart using AABB collision detection.

        Uses actual component dimensions (computed from pin bounding boxes)
        instead of treating components as points.
        """
        # Build component size dict from symbols or defaults
        component_sizes: Dict[str, Tuple[float, float]] = {}
        symbol_lookup = {}
        if symbols:
            for sym in symbols:
                symbol_lookup[sym.reference] = sym

        for ref in positions:
            sym = symbol_lookup.get(ref)
            if sym and hasattr(sym, 'pins') and sym.pins:
                xs = [p.position[0] for p in sym.pins]
                ys = [p.position[1] for p in sym.pins]
                if xs and ys:
                    w = max(xs) - min(xs) + 12.0  # Add padding for body + labels
                    h = max(ys) - min(ys) + 10.0
                    component_sizes[ref] = (max(w, 10.0), max(h, 8.0))
                    continue
            # Default sizes by reference prefix
            if ref.startswith('U'):
                component_sizes[ref] = (30.0, 35.0)  # ICs are large
            elif ref.startswith(('R', 'C', 'L')):
                component_sizes[ref] = (10.0, 6.0)   # Passives are small
            elif ref.startswith('Q'):
                component_sizes[ref] = (12.0, 10.0)  # Transistors
            elif ref.startswith('J'):
                component_sizes[ref] = (15.0, 12.0)  # Connectors
            elif ref.startswith('D'):
                component_sizes[ref] = (8.0, 5.0)    # Diodes
            else:
                component_sizes[ref] = (12.0, 10.0)  # Default

        max_iterations = 100
        margin = 2.54  # Grid margin between components
        refs = list(positions.keys())

        for iteration in range(max_iterations):
            moved = False
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    r1, r2 = refs[i], refs[j]
                    p1, p2 = positions[r1], positions[r2]
                    s1, s2 = component_sizes.get(r1, (12.0, 10.0)), component_sizes.get(r2, (12.0, 10.0))

                    # AABB overlap check
                    hw1, hh1 = s1[0] / 2 + margin, s1[1] / 2 + margin
                    hw2, hh2 = s2[0] / 2 + margin, s2[1] / 2 + margin

                    overlap_x = (hw1 + hw2) - abs(p1[0] - p2[0])
                    overlap_y = (hh1 + hh2) - abs(p1[1] - p2[1])

                    if overlap_x > 0 and overlap_y > 0:
                        # Push apart along axis of least overlap
                        if overlap_x < overlap_y:
                            # Push horizontally
                            push = overlap_x / 2 + 0.5
                            if p1[0] <= p2[0]:
                                positions[r1] = (p1[0] - push, p1[1])
                                positions[r2] = (p2[0] + push, p2[1])
                            else:
                                positions[r1] = (p1[0] + push, p1[1])
                                positions[r2] = (p2[0] - push, p2[1])
                        else:
                            # Push vertically
                            push = overlap_y / 2 + 0.5
                            if p1[1] <= p2[1]:
                                positions[r1] = (p1[0], p1[1] - push)
                                positions[r2] = (p2[0], p2[1] + push)
                            else:
                                positions[r1] = (p1[0], p1[1] + push)
                                positions[r2] = (p2[0], p2[1] - push)
                        moved = True

            if not moved:
                logger.info(f"Collision resolution converged after {iteration + 1} iterations")
                break
        else:
            logger.warning(
                f"Collision resolution did NOT converge after {max_iterations} iterations. "
                f"Some components may still overlap."
            )

        # Validation pass: count remaining overlaps
        remaining_overlaps = 0
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                r1, r2 = refs[i], refs[j]
                p1, p2 = positions[r1], positions[r2]
                s1, s2 = component_sizes.get(r1, (12.0, 10.0)), component_sizes.get(r2, (12.0, 10.0))
                hw1, hh1 = s1[0] / 2, s1[1] / 2
                hw2, hh2 = s2[0] / 2, s2[1] / 2
                if (hw1 + hw2) - abs(p1[0] - p2[0]) > 0 and (hh1 + hh2) - abs(p1[1] - p2[1]) > 0:
                    remaining_overlaps += 1

        if remaining_overlaps > 0:
            logger.error(f"LAYOUT QUALITY: {remaining_overlaps} overlapping component pairs remain after collision resolution")
        else:
            logger.info("LAYOUT QUALITY: 0 overlapping component pairs - all collisions resolved")

        return remaining_overlaps

    def _apply_proximity_constraints(
        self,
        positions: Dict[str, Tuple[float, float]],
        proximity_pairs: List[Tuple[str, str]]
    ):
        """Move components that must be close together (e.g., bypass caps near ICs)."""
        for comp1, comp2 in proximity_pairs:
            if comp1 in positions and comp2 in positions:
                pos1 = positions[comp1]
                pos2 = positions[comp2]
                # Check current distance
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dist = (dx**2 + dy**2) ** 0.5
                target_dist = self.SPACING_RULES["bypass_to_ic"]

                if dist > target_dist * 3:
                    # Too far — move comp2 close to comp1
                    # Place slightly below-right of comp1
                    positions[comp2] = (
                        pos1[0] + target_dist,
                        pos1[1] + 8.0
                    )

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
