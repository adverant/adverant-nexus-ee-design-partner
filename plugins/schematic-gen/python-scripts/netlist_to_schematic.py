#!/usr/bin/env python3
"""
Netlist to KiCad Schematic Converter

Converts SKiDL-generated netlists to KiCad schematic format (.kicad_sch).
Supports automatic symbol placement and wire routing.

This module handles conversion for any circuit type:
- Digital circuits
- Analog circuits
- RF/microwave circuits
- Mixed-signal designs
- Power electronics
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import math
import re

try:
    import sexpdata
except ImportError:
    print("ERROR: sexpdata not installed. Run: pip install sexpdata", file=sys.stderr)
    sys.exit(1)


@dataclass
class Position:
    """2D position in schematic coordinates (mils)."""
    x: float
    y: float

    def to_mm(self) -> Tuple[float, float]:
        """Convert to millimeters (KiCad native unit)."""
        return (self.x * 0.0254, self.y * 0.0254)

    def offset(self, dx: float, dy: float) -> 'Position':
        """Create offset position."""
        return Position(self.x + dx, self.y + dy)


@dataclass
class PinConnection:
    """Represents a pin connection in the netlist."""
    component_ref: str
    pin_number: str
    pin_name: str


@dataclass
class NetlistNet:
    """Represents a net in the netlist."""
    name: str
    code: int
    connections: List[PinConnection] = field(default_factory=list)


@dataclass
class NetlistComponent:
    """Represents a component in the netlist."""
    reference: str
    value: str
    footprint: str
    library: str = 'Device'
    symbol: str = ''
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class SchematicSymbol:
    """Represents a placed symbol in the schematic."""
    component: NetlistComponent
    position: Position
    rotation: int = 0
    mirror: str = ''
    unit: int = 1


@dataclass
class SchematicWire:
    """Represents a wire segment in the schematic."""
    start: Position
    end: Position
    net_name: str = ''


@dataclass
class SchematicLabel:
    """Represents a net label in the schematic."""
    text: str
    position: Position
    rotation: int = 0
    label_type: str = 'local'  # local, global, hierarchical, power


class NetlistParser:
    """
    Parser for various netlist formats.

    Supports:
    - KiCad netlist format
    - SKiDL-generated netlists
    - SPICE netlist format
    """

    def __init__(self, netlist_path: str):
        self.path = netlist_path
        self.components: List[NetlistComponent] = []
        self.nets: List[NetlistNet] = []

    def parse(self) -> Tuple[List[NetlistComponent], List[NetlistNet]]:
        """Parse the netlist file."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Netlist file not found: {self.path}")

        with open(self.path, 'r') as f:
            content = f.read()

        # Detect format and parse
        if content.strip().startswith('(export'):
            self._parse_kicad_netlist(content)
        elif content.strip().startswith('*') or '.subckt' in content.lower():
            self._parse_spice_netlist(content)
        else:
            self._parse_skidl_netlist(content)

        return self.components, self.nets

    def _parse_kicad_netlist(self, content: str):
        """Parse KiCad format netlist."""
        try:
            sexp = sexpdata.loads(content)
            self._extract_kicad_components(sexp)
            self._extract_kicad_nets(sexp)
        except Exception as e:
            raise ValueError(f"Failed to parse KiCad netlist: {e}")

    def _extract_kicad_components(self, sexp: Any):
        """Extract components from KiCad S-expression."""
        def find_node(node, name):
            if isinstance(node, list):
                for item in node:
                    if isinstance(item, list) and len(item) > 0:
                        if hasattr(item[0], 'value') and item[0].value() == name:
                            return item
                        result = find_node(item, name)
                        if result:
                            return result
            return None

        components_node = find_node(sexp, 'components')
        if not components_node:
            return

        for item in components_node[1:]:
            if isinstance(item, list) and len(item) > 0:
                if hasattr(item[0], 'value') and item[0].value() == 'comp':
                    comp = self._parse_component(item)
                    if comp:
                        self.components.append(comp)

    def _parse_component(self, comp_sexp: List) -> Optional[NetlistComponent]:
        """Parse a single component from S-expression."""
        ref = ''
        value = ''
        footprint = ''
        library = 'Device'

        for item in comp_sexp[1:]:
            if not isinstance(item, list) or len(item) < 2:
                continue

            key = item[0].value() if hasattr(item[0], 'value') else str(item[0])

            if key == 'ref':
                ref = str(item[1])
            elif key == 'value':
                value = str(item[1])
            elif key == 'footprint':
                footprint = str(item[1])
            elif key == 'libsource':
                for sub in item[1:]:
                    if isinstance(sub, list) and len(sub) >= 2:
                        if hasattr(sub[0], 'value') and sub[0].value() == 'lib':
                            library = str(sub[1])

        if ref:
            return NetlistComponent(
                reference=ref,
                value=value,
                footprint=footprint,
                library=library
            )
        return None

    def _extract_kicad_nets(self, sexp: Any):
        """Extract nets from KiCad S-expression."""
        def find_nodes(node, name):
            results = []
            if isinstance(node, list):
                for item in node:
                    if isinstance(item, list) and len(item) > 0:
                        if hasattr(item[0], 'value') and item[0].value() == name:
                            results.append(item)
                        results.extend(find_nodes(item, name))
            return results

        net_nodes = find_nodes(sexp, 'net')
        for net_node in net_nodes:
            net = self._parse_net(net_node)
            if net:
                self.nets.append(net)

    def _parse_net(self, net_sexp: List) -> Optional[NetlistNet]:
        """Parse a single net from S-expression."""
        name = ''
        code = 0
        connections = []

        for item in net_sexp[1:]:
            if not isinstance(item, list):
                continue

            key = item[0].value() if hasattr(item[0], 'value') else str(item[0])

            if key == 'name':
                name = str(item[1])
            elif key == 'code':
                code = int(item[1])
            elif key == 'node':
                ref = ''
                pin = ''
                for sub in item[1:]:
                    if isinstance(sub, list) and len(sub) >= 2:
                        subkey = sub[0].value() if hasattr(sub[0], 'value') else str(sub[0])
                        if subkey == 'ref':
                            ref = str(sub[1])
                        elif subkey == 'pin':
                            pin = str(sub[1])
                if ref and pin:
                    connections.append(PinConnection(
                        component_ref=ref,
                        pin_number=pin,
                        pin_name=pin
                    ))

        if name:
            return NetlistNet(name=name, code=code, connections=connections)
        return None

    def _parse_spice_netlist(self, content: str):
        """Parse SPICE format netlist."""
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            ref = parts[0]
            # SPICE components: R, C, L, M, Q, D, etc.
            if ref[0] in 'RCLMQDVIXE':
                self.components.append(NetlistComponent(
                    reference=ref,
                    value=parts[-1] if len(parts) > 3 else parts[2],
                    footprint='',
                    library='Device'
                ))

    def _parse_skidl_netlist(self, content: str):
        """Parse SKiDL-generated netlist."""
        # SKiDL generates KiCad-compatible netlists, so use KiCad parser
        self._parse_kicad_netlist(content)


class SchematicGenerator:
    """
    Generates KiCad schematic files from netlists.

    Handles:
    - Component placement
    - Wire routing
    - Label generation
    - Multi-sheet schematics
    """

    GRID_SIZE = 100  # mils
    COMPONENT_SPACING_X = 400  # mils
    COMPONENT_SPACING_Y = 300  # mils
    PAGE_WIDTH = 11000  # mils (A3)
    PAGE_HEIGHT = 8500  # mils (A3)

    def __init__(self):
        self.symbols: List[SchematicSymbol] = []
        self.wires: List[SchematicWire] = []
        self.labels: List[SchematicLabel] = []
        self.component_positions: Dict[str, Position] = {}

    def generate(
        self,
        components: List[NetlistComponent],
        nets: List[NetlistNet],
        output_path: str
    ):
        """
        Generate schematic from components and nets.

        Args:
            components: List of components
            nets: List of nets
            output_path: Output .kicad_sch file path
        """
        # Place components
        self._place_components(components)

        # Generate wires for nets
        self._generate_wires(nets)

        # Generate labels
        self._generate_labels(nets)

        # Write schematic file
        self._write_schematic(output_path)

    def _place_components(self, components: List[NetlistComponent]):
        """Place components on the schematic."""
        # Group components by type
        groups: Dict[str, List[NetlistComponent]] = {}
        for comp in components:
            prefix = re.match(r'^([A-Z]+)', comp.reference)
            group = prefix.group(1) if prefix else 'OTHER'
            if group not in groups:
                groups[group] = []
            groups[group].append(comp)

        # Sort each group by reference number
        for group in groups.values():
            group.sort(key=lambda c: int(re.search(r'\d+', c.reference).group() or 0))

        # Place by group
        y_offset = 500
        for group_name, group_comps in groups.items():
            x_offset = 500

            for i, comp in enumerate(group_comps):
                pos = Position(
                    x_offset + (i % 10) * self.COMPONENT_SPACING_X,
                    y_offset + (i // 10) * self.COMPONENT_SPACING_Y
                )

                self.symbols.append(SchematicSymbol(
                    component=comp,
                    position=pos,
                    rotation=0
                ))

                self.component_positions[comp.reference] = pos

            # Move to next row for next group
            rows = (len(group_comps) + 9) // 10
            y_offset += rows * self.COMPONENT_SPACING_Y + 200

    def _generate_wires(self, nets: List[NetlistNet]):
        """Generate wires for net connections."""
        for net in nets:
            if len(net.connections) < 2:
                continue

            # Get positions of connected components
            positions = []
            for conn in net.connections:
                pos = self.component_positions.get(conn.component_ref)
                if pos:
                    # Offset to approximate pin position
                    pin_pos = pos.offset(100, 0)
                    positions.append(pin_pos)

            # Create wire segments connecting all positions
            if len(positions) >= 2:
                # Simple star topology - connect all to first
                base_pos = positions[0]
                for pos in positions[1:]:
                    self.wires.append(SchematicWire(
                        start=base_pos,
                        end=pos,
                        net_name=net.name
                    ))

    def _generate_labels(self, nets: List[NetlistNet]):
        """Generate net labels."""
        power_nets = {'VCC', 'VDD', 'GND', 'VSS', 'VIN', 'VBUS', 'AGND', 'PGND'}

        for net in nets:
            if not net.connections:
                continue

            # Get position from first connection
            first_conn = net.connections[0]
            pos = self.component_positions.get(first_conn.component_ref)
            if not pos:
                continue

            label_pos = pos.offset(150, -50)

            # Determine label type
            label_type = 'local'
            if net.name.upper() in power_nets or net.name.startswith('V'):
                label_type = 'power'
            elif net.name.startswith('/') or '/' in net.name:
                label_type = 'hierarchical'
            elif len(net.connections) > 5:
                label_type = 'global'

            self.labels.append(SchematicLabel(
                text=net.name,
                position=label_pos,
                label_type=label_type
            ))

    def _write_schematic(self, output_path: str):
        """Write the KiCad schematic file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate S-expression content
        content = self._generate_sexp()

        with open(output_path, 'w') as f:
            f.write(content)

    def _generate_sexp(self) -> str:
        """Generate KiCad schematic S-expression."""
        lines = []

        # Header
        lines.append('(kicad_sch (version 20230121) (generator "skidl_schematic_gen")')
        lines.append('')
        lines.append('  (uuid "00000000-0000-0000-0000-000000000001")')
        lines.append('')

        # Paper size
        lines.append('  (paper "A3")')
        lines.append('')

        # Title block
        lines.append('  (title_block')
        lines.append(f'    (title "Generated Schematic")')
        lines.append(f'    (date "{datetime.now().strftime("%Y-%m-%d")}")')
        lines.append(f'    (rev "1.0")')
        lines.append(f'    (company "Adverant")')
        lines.append('  )')
        lines.append('')

        # Library symbols
        lines.append('  (lib_symbols')
        added_symbols: Set[str] = set()
        for symbol in self.symbols:
            symbol_key = f"{symbol.component.library}:{symbol.component.reference[0]}"
            if symbol_key not in added_symbols:
                lines.append(self._generate_lib_symbol(symbol.component))
                added_symbols.add(symbol_key)
        lines.append('  )')
        lines.append('')

        # Symbols (component instances)
        for symbol in self.symbols:
            lines.append(self._generate_symbol_instance(symbol))
            lines.append('')

        # Wires
        for wire in self.wires:
            lines.append(self._generate_wire(wire))

        # Labels
        for label in self.labels:
            lines.append(self._generate_label(label))

        lines.append(')')

        return '\n'.join(lines)

    def _generate_lib_symbol(self, comp: NetlistComponent) -> str:
        """Generate library symbol definition."""
        ref_prefix = re.match(r'^([A-Z]+)', comp.reference)
        prefix = ref_prefix.group(1) if ref_prefix else 'U'

        # Basic symbol template based on component type
        symbol_name = f"{comp.library}:{prefix}"

        return f'''    (symbol "{symbol_name}" (pin_numbers hide) (pin_names hide) (in_bom yes) (on_board yes)
      (property "Reference" "{prefix}" (at 0 1.27 0) (effects (font (size 1.27 1.27))))
      (property "Value" "{prefix}" (at 0 -1.27 0) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "{symbol_name}_1_1"
        (rectangle (start -2.54 2.54) (end 2.54 -2.54)
          (stroke (width 0) (type default))
          (fill (type background))
        )
        (pin passive line (at -5.08 0 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 5.08 0 180) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      )
    )'''

    def _generate_symbol_instance(self, symbol: SchematicSymbol) -> str:
        """Generate symbol instance in schematic."""
        x_mm, y_mm = symbol.position.to_mm()
        ref_prefix = re.match(r'^([A-Z]+)', symbol.component.reference)
        prefix = ref_prefix.group(1) if ref_prefix else 'U'

        return f'''  (symbol (lib_id "{symbol.component.library}:{prefix}") (at {x_mm:.2f} {y_mm:.2f} 0) (unit 1)
    (uuid "{symbol.component.reference}")
    (property "Reference" "{symbol.component.reference}" (at {x_mm:.2f} {y_mm + 2:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Value" "{symbol.component.value}" (at {x_mm:.2f} {y_mm - 2:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Footprint" "{symbol.component.footprint}" (at {x_mm:.2f} {y_mm:.2f} 0)
      (effects (font (size 1.27 1.27)) hide)
    )
  )'''

    def _generate_wire(self, wire: SchematicWire) -> str:
        """Generate wire segment."""
        start_mm = wire.start.to_mm()
        end_mm = wire.end.to_mm()

        return f'''  (wire (pts (xy {start_mm[0]:.2f} {start_mm[1]:.2f}) (xy {end_mm[0]:.2f} {end_mm[1]:.2f}))
    (stroke (width 0) (type default))
  )'''

    def _generate_label(self, label: SchematicLabel) -> str:
        """Generate net label."""
        x_mm, y_mm = label.position.to_mm()

        if label.label_type == 'power':
            return f'''  (power_label "{label.text}" (at {x_mm:.2f} {y_mm:.2f} 0)
    (effects (font (size 1.27 1.27)))
  )'''
        elif label.label_type == 'global':
            return f'''  (global_label "{label.text}" (shape passive) (at {x_mm:.2f} {y_mm:.2f} 0)
    (effects (font (size 1.27 1.27)))
  )'''
        else:
            return f'''  (label "{label.text}" (at {x_mm:.2f} {y_mm:.2f} 0)
    (effects (font (size 1.27 1.27)))
  )'''


def main():
    parser = argparse.ArgumentParser(
        description='Convert netlist to KiCad schematic'
    )
    parser.add_argument(
        '--netlists', '-n',
        type=str,
        required=True,
        help='Comma-separated list of netlist files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output schematic file path'
    )

    args = parser.parse_args()

    # Parse all netlists
    all_components: List[NetlistComponent] = []
    all_nets: List[NetlistNet] = []

    for netlist_path in args.netlists.split(','):
        netlist_path = netlist_path.strip()
        if not netlist_path:
            continue

        try:
            parser_obj = NetlistParser(netlist_path)
            components, nets = parser_obj.parse()
            all_components.extend(components)
            all_nets.extend(nets)
            print(f"Parsed {netlist_path}: {len(components)} components, {len(nets)} nets")
        except Exception as e:
            print(f"ERROR: Failed to parse {netlist_path}: {e}", file=sys.stderr)
            sys.exit(1)

    # Generate schematic
    generator = SchematicGenerator()
    try:
        generator.generate(all_components, all_nets, args.output)
        print(f"Generated schematic: {args.output}")
        print(f"  Total symbols: {len(generator.symbols)}")
        print(f"  Total wires: {len(generator.wires)}")
        print(f"  Total labels: {len(generator.labels)}")
    except Exception as e:
        print(f"ERROR: Failed to generate schematic: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
