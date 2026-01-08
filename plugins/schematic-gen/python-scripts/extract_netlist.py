#!/usr/bin/env python3
"""
Netlist Extraction from KiCad Schematics

Parses .kicad_sch S-expression files and extracts:
1. Components (symbols) with their pins
2. Wires and their endpoints
3. Labels (net names)
4. Connectivity graph: which pins connect to which nets

This is the CRITICAL MISSING LINK between schematic and PCB generation.

Usage:
    python extract_netlist.py --path /path/to/schematic.kicad_sch --json

Output:
    JSON with components, nets, and connectivity
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict


@dataclass
class Pin:
    """A component pin."""
    number: str
    name: str
    uuid: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class Component:
    """A schematic component."""
    reference: str
    value: str
    lib_id: str
    footprint: str
    x: float
    y: float
    rotation: float
    uuid: str
    pins: List[Pin] = field(default_factory=list)


@dataclass
class Wire:
    """A schematic wire."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    uuid: str


@dataclass
class Label:
    """A net label."""
    name: str
    x: float
    y: float
    uuid: str


@dataclass
class GlobalLabel:
    """A global label for hierarchical schematics."""
    name: str
    x: float
    y: float
    uuid: str


@dataclass
class PowerSymbol:
    """A power symbol (VCC, GND, etc)."""
    name: str  # e.g., "+3.3V", "GND"
    x: float
    y: float
    uuid: str


class NetlistExtractor:
    """
    Extracts netlist from KiCad schematic files.

    The extraction process:
    1. Parse all symbols (components) and their pins
    2. Parse all wires
    3. Parse all labels (local and global)
    4. Parse all power symbols
    5. Build connectivity graph by matching wire endpoints to pin locations
    6. Assign net names from labels and power symbols
    """

    def __init__(self, schematic_path: str):
        self.path = schematic_path
        self.content = ""
        self.components: List[Component] = []
        self.wires: List[Wire] = []
        self.labels: List[Label] = []
        self.global_labels: List[GlobalLabel] = []
        self.power_symbols: List[PowerSymbol] = []

        # Connectivity tolerance (points within this distance are connected)
        self.tolerance = 0.1

    def load(self):
        """Load schematic file."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Schematic not found: {self.path}")

        with open(self.path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    def extract(self) -> Dict[str, Any]:
        """
        Extract netlist from schematic.

        Returns:
            Dictionary with components, nets, and connectivity
        """
        self.load()

        # Parse schematic elements
        self._parse_symbols()
        self._parse_wires()
        self._parse_labels()
        self._parse_power_symbols()

        # Build connectivity
        connectivity = self._build_connectivity()

        # Build netlist output
        return self._build_netlist(connectivity)

    def _parse_symbols(self):
        """Parse component symbols."""
        # Match symbol blocks
        symbol_pattern = r'\(symbol\s+\(lib_id\s+"([^"]+)"\)\s+\(at\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\)'

        # More comprehensive pattern to get full symbol block
        full_symbol_pattern = (
            r'\(symbol\s*\n?\s*'
            r'\(lib_id\s+"([^"]+)"\)\s*\n?\s*'
            r'\(at\s+([\d.-]+)\s+([\d.-]+)\s*([\d.-]*)\)'
            r'.*?'
            r'\(property\s+"Reference"\s+"([^"]+)"'
            r'.*?'
            r'\(property\s+"Value"\s+"([^"]+)"'
            r'.*?'
            r'(?:\(property\s+"Footprint"\s+"([^"]*)"|)'
            r'.*?'
            r'\(uuid\s+"([^"]+)"\)'
        )

        for match in re.finditer(full_symbol_pattern, self.content, re.DOTALL):
            lib_id = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            rotation = float(match.group(4)) if match.group(4) else 0.0
            reference = match.group(5)
            value = match.group(6)
            footprint = match.group(7) if match.group(7) else ""
            uuid = match.group(8)

            # Skip power symbols (handled separately)
            if lib_id.startswith("power:"):
                continue

            # Skip hidden references (like #PWR)
            if reference.startswith("#"):
                continue

            component = Component(
                reference=reference,
                value=value,
                lib_id=lib_id,
                footprint=footprint,
                x=x,
                y=y,
                rotation=rotation,
                uuid=uuid
            )

            # Extract pins from the symbol block
            symbol_block_start = match.start()
            # Find the end of this symbol block
            depth = 0
            end_pos = symbol_block_start
            for i, char in enumerate(self.content[symbol_block_start:]):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        end_pos = symbol_block_start + i + 1
                        break

            symbol_block = self.content[symbol_block_start:end_pos]

            # Parse pins in this symbol
            pin_pattern = r'\(pin\s+"(\d+)".*?\(uuid\s+"([^"]+)"\)'
            for pin_match in re.finditer(pin_pattern, symbol_block, re.DOTALL):
                pin_number = pin_match.group(1)
                pin_uuid = pin_match.group(2)

                # Calculate pin position based on component rotation
                # This is simplified - real implementation would use library pin offsets
                component.pins.append(Pin(
                    number=pin_number,
                    name="",
                    uuid=pin_uuid,
                    x=x,  # Simplified - actual pin position depends on symbol
                    y=y
                ))

            self.components.append(component)

    def _parse_wires(self):
        """Parse wire connections."""
        # Match wire elements
        wire_pattern = r'\(wire\s*\n?\s*\(pts\s+\(xy\s+([\d.-]+)\s+([\d.-]+)\)\s+\(xy\s+([\d.-]+)\s+([\d.-]+)\)\).*?\(uuid\s+"([^"]+)"\)'

        for match in re.finditer(wire_pattern, self.content, re.DOTALL):
            x1 = float(match.group(1))
            y1 = float(match.group(2))
            x2 = float(match.group(3))
            y2 = float(match.group(4))
            uuid = match.group(5)

            self.wires.append(Wire(
                start=(x1, y1),
                end=(x2, y2),
                uuid=uuid
            ))

    def _parse_labels(self):
        """Parse net labels."""
        # Local labels
        label_pattern = r'\(label\s+"([^"]+)"\s+\(at\s+([\d.-]+)\s+([\d.-]+).*?\(uuid\s+"([^"]+)"\)'

        for match in re.finditer(label_pattern, self.content, re.DOTALL):
            name = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            uuid = match.group(4)

            self.labels.append(Label(name=name, x=x, y=y, uuid=uuid))

        # Global labels
        global_label_pattern = r'\(global_label\s+"([^"]+)"\s+\(.*?\(at\s+([\d.-]+)\s+([\d.-]+).*?\(uuid\s+"([^"]+)"\)'

        for match in re.finditer(global_label_pattern, self.content, re.DOTALL):
            name = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            uuid = match.group(4)

            self.global_labels.append(GlobalLabel(name=name, x=x, y=y, uuid=uuid))

    def _parse_power_symbols(self):
        """Parse power symbols (VCC, GND, etc)."""
        # Power symbols have lib_id starting with "power:"
        power_pattern = (
            r'\(symbol\s*\n?\s*'
            r'\(lib_id\s+"power:([^"]+)"\)\s*\n?\s*'
            r'\(at\s+([\d.-]+)\s+([\d.-]+)'
            r'.*?'
            r'\(uuid\s+"([^"]+)"\)'
        )

        for match in re.finditer(power_pattern, self.content, re.DOTALL):
            name = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            uuid = match.group(4)

            self.power_symbols.append(PowerSymbol(name=name, x=x, y=y, uuid=uuid))

    def _points_close(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if two points are within tolerance."""
        return abs(p1[0] - p2[0]) < self.tolerance and abs(p1[1] - p2[1]) < self.tolerance

    def _build_connectivity(self) -> Dict[str, Set[Tuple[str, str]]]:
        """
        Build connectivity graph using union-find algorithm.

        Returns:
            Dictionary mapping net names to sets of (component_ref, pin_number)
        """
        # Collect all connection points
        points: List[Tuple[float, float, str, str]] = []  # (x, y, type, identifier)

        # Add wire endpoints
        for wire in self.wires:
            points.append((wire.start[0], wire.start[1], 'wire', wire.uuid))
            points.append((wire.end[0], wire.end[1], 'wire', wire.uuid))

        # Add label positions
        for label in self.labels:
            points.append((label.x, label.y, 'label', label.name))

        # Add global label positions
        for glabel in self.global_labels:
            points.append((glabel.x, glabel.y, 'global_label', glabel.name))

        # Add power symbol positions
        for power in self.power_symbols:
            points.append((power.x, power.y, 'power', power.name))

        # Add component pin positions (simplified - using component center)
        for comp in self.components:
            for pin in comp.pins:
                points.append((pin.x, pin.y, 'pin', f"{comp.reference}.{pin.number}"))

        # Build union-find structure
        parent = {i: i for i in range(len(points))}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union points that are close together
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if self._points_close((points[i][0], points[i][1]),
                                       (points[j][0], points[j][1])):
                    union(i, j)

        # Group points by their root
        groups = defaultdict(list)
        for i, point in enumerate(points):
            groups[find(i)].append(point)

        # Build net connectivity
        connectivity: Dict[str, Set[Tuple[str, str]]] = {}

        for group_id, group_points in groups.items():
            # Find net name (from labels or power symbols)
            net_name = None
            pins_in_net = set()

            for x, y, ptype, identifier in group_points:
                if ptype == 'label':
                    net_name = identifier
                elif ptype == 'global_label':
                    net_name = identifier
                elif ptype == 'power':
                    net_name = identifier
                elif ptype == 'pin':
                    ref, pin_num = identifier.split('.')
                    pins_in_net.add((ref, pin_num))

            # If no name found, generate one
            if net_name is None and pins_in_net:
                net_name = f"Net_{group_id}"

            if net_name and pins_in_net:
                if net_name in connectivity:
                    connectivity[net_name].update(pins_in_net)
                else:
                    connectivity[net_name] = pins_in_net

        return connectivity

    def _build_netlist(self, connectivity: Dict[str, Set[Tuple[str, str]]]) -> Dict[str, Any]:
        """Build final netlist structure."""
        # Components
        components = []
        for comp in self.components:
            components.append({
                'reference': comp.reference,
                'value': comp.value,
                'lib_id': comp.lib_id,
                'footprint': comp.footprint,
                'position': {'x': comp.x, 'y': comp.y, 'rotation': comp.rotation},
                'pins': [
                    {'number': p.number, 'uuid': p.uuid}
                    for p in comp.pins
                ]
            })

        # Nets
        nets = []
        net_id = 1
        for net_name, connections in connectivity.items():
            nets.append({
                'id': net_id,
                'name': net_name,
                'connections': list(connections)
            })
            net_id += 1

        # Summary
        return {
            'source_file': self.path,
            'components': components,
            'nets': nets,
            'summary': {
                'component_count': len(components),
                'net_count': len(nets),
                'wire_count': len(self.wires),
                'label_count': len(self.labels) + len(self.global_labels),
                'power_symbol_count': len(self.power_symbols)
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract netlist from KiCad schematic'
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='Path to .kicad_sch file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: stdout)'
    )

    args = parser.parse_args()

    try:
        extractor = NetlistExtractor(args.path)
        netlist = extractor.extract()

        if args.json:
            output = json.dumps(netlist, indent=2)
        else:
            # Human-readable format
            output = f"Schematic: {netlist['source_file']}\n"
            output += f"\nComponents ({netlist['summary']['component_count']}):\n"
            for comp in netlist['components']:
                output += f"  {comp['reference']}: {comp['value']} ({comp['lib_id']})\n"
                output += f"    Footprint: {comp['footprint']}\n"

            output += f"\nNets ({netlist['summary']['net_count']}):\n"
            for net in netlist['nets']:
                connections = ', '.join(f"{c[0]}.{c[1]}" for c in net['connections'])
                output += f"  {net['name']} (ID {net['id']}): {connections}\n"

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Netlist written to: {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
