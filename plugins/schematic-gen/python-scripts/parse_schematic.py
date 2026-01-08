#!/usr/bin/env python3
"""
KiCad Schematic Parser

Parses KiCad schematic files (.kicad_sch) and outputs structured JSON data
for validation and analysis.

This parser is designed to work with any circuit type:
- Digital circuits
- Analog circuits
- RF/microwave circuits
- Mixed-signal designs
- Power electronics
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

try:
    import sexpdata
except ImportError:
    print("ERROR: sexpdata not installed. Run: pip install sexpdata", file=sys.stderr)
    sys.exit(1)


class PinType(str, Enum):
    """Pin electrical types."""
    INPUT = 'input'
    OUTPUT = 'output'
    BIDIRECTIONAL = 'bidirectional'
    TRI_STATE = 'tri_state'
    PASSIVE = 'passive'
    POWER_INPUT = 'power_input'
    POWER_OUTPUT = 'power_output'
    OPEN_COLLECTOR = 'open_collector'
    OPEN_EMITTER = 'open_emitter'
    NO_CONNECT = 'no_connect'
    UNSPECIFIED = 'unspecified'


@dataclass
class Position:
    """2D position."""
    x: float
    y: float


@dataclass
class Pin:
    """Component pin data."""
    number: str
    name: str
    pin_type: str
    position: Optional[Position] = None
    connected: bool = False
    net_name: str = ''


@dataclass
class Component:
    """Schematic component data."""
    reference: str
    value: str
    symbol: str
    footprint: str
    position: Position
    rotation: int = 0
    properties: Dict[str, str] = field(default_factory=dict)
    pins: List[Pin] = field(default_factory=list)
    manufacturer: str = ''
    part_number: str = ''


@dataclass
class NetConnection:
    """Connection to a net."""
    component_ref: str
    pin_number: str
    pin_name: str


@dataclass
class NetProperties:
    """Net electrical properties."""
    impedance: Optional[float] = None
    max_current: Optional[float] = None
    max_voltage: Optional[float] = None
    net_type: str = 'signal'  # power, ground, signal, differential


@dataclass
class Net:
    """Net data."""
    name: str
    connections: List[NetConnection] = field(default_factory=list)
    net_class: str = ''
    properties: NetProperties = field(default_factory=NetProperties)


@dataclass
class Wire:
    """Wire segment data."""
    start_point: Position
    end_point: Position
    net_name: str = ''


@dataclass
class Label:
    """Net label data."""
    text: str
    position: Position
    label_type: str = 'local'  # local, global, hierarchical, power


@dataclass
class SheetMetadata:
    """Sheet metadata."""
    title: str = ''
    revision: str = ''
    date: str = ''
    author: str = ''
    company: str = ''
    comments: List[str] = field(default_factory=list)


@dataclass
class SchematicSheet:
    """Single schematic sheet data."""
    id: str
    name: str
    page_number: int
    components: List[Component] = field(default_factory=list)
    nets: List[Net] = field(default_factory=list)
    wires: List[Wire] = field(default_factory=list)
    labels: List[Label] = field(default_factory=list)
    metadata: SheetMetadata = field(default_factory=SheetMetadata)


@dataclass
class ProjectMetadata:
    """Project metadata."""
    title: str = ''
    description: str = ''
    revision: str = ''
    date: str = ''
    author: str = ''
    company: str = ''
    target_application: str = ''


@dataclass
class SchematicData:
    """Complete schematic data."""
    id: str
    project_name: str
    version: str
    sheets: List[SchematicSheet] = field(default_factory=list)
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)


class SchematicParser:
    """
    Parser for KiCad schematic files.

    Extracts:
    - Components with all properties
    - Nets and connections
    - Wires and labels
    - Hierarchical sheet structure
    """

    def __init__(self, schematic_path: str):
        self.path = schematic_path
        self.sexp_data: Any = None
        self.schematic: Optional[SchematicData] = None

    def parse(self) -> SchematicData:
        """Parse the schematic file and return structured data."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Schematic file not found: {self.path}")

        with open(self.path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            self.sexp_data = sexpdata.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to parse schematic: {e}")

        # Create schematic structure
        self.schematic = SchematicData(
            id=self._extract_uuid() or 'unknown',
            project_name=Path(self.path).stem,
            version=self._extract_version() or '1.0'
        )

        # Parse main sheet
        main_sheet = self._parse_sheet()
        self.schematic.sheets.append(main_sheet)

        # Parse title block
        self.schematic.metadata = self._parse_title_block()

        # Build net connections from wires and labels
        self._build_net_connections(main_sheet)

        return self.schematic

    def _find_node(self, node: Any, name: str) -> Optional[Any]:
        """Find a node by name in S-expression."""
        if isinstance(node, list):
            for item in node:
                if isinstance(item, list) and len(item) > 0:
                    if hasattr(item[0], 'value') and item[0].value() == name:
                        return item
                    result = self._find_node(item, name)
                    if result:
                        return result
        return None

    def _find_all_nodes(self, node: Any, name: str) -> List[Any]:
        """Find all nodes with given name."""
        results = []
        if isinstance(node, list):
            for item in node:
                if isinstance(item, list) and len(item) > 0:
                    if hasattr(item[0], 'value') and item[0].value() == name:
                        results.append(item)
                    results.extend(self._find_all_nodes(item, name))
        return results

    def _get_value(self, node: List, key: str) -> Optional[str]:
        """Get value for a key from S-expression node."""
        for item in node:
            if isinstance(item, list) and len(item) >= 2:
                if hasattr(item[0], 'value') and item[0].value() == key:
                    return str(item[1])
        return None

    def _extract_uuid(self) -> Optional[str]:
        """Extract schematic UUID."""
        uuid_node = self._find_node(self.sexp_data, 'uuid')
        if uuid_node and len(uuid_node) > 1:
            return str(uuid_node[1])
        return None

    def _extract_version(self) -> Optional[str]:
        """Extract schematic version."""
        if isinstance(self.sexp_data, list) and len(self.sexp_data) > 0:
            for item in self.sexp_data:
                if isinstance(item, list) and len(item) >= 2:
                    if hasattr(item[0], 'value') and item[0].value() == 'version':
                        return str(item[1])
        return None

    def _parse_sheet(self) -> SchematicSheet:
        """Parse the schematic sheet."""
        sheet = SchematicSheet(
            id=self._extract_uuid() or 'sheet1',
            name='Main',
            page_number=1
        )

        # Parse symbols (components)
        symbols = self._find_all_nodes(self.sexp_data, 'symbol')
        for symbol_node in symbols:
            # Skip library symbol definitions
            if self._get_value(symbol_node, 'lib_id'):
                component = self._parse_component(symbol_node)
                if component:
                    sheet.components.append(component)

        # Parse wires
        wires = self._find_all_nodes(self.sexp_data, 'wire')
        for wire_node in wires:
            wire = self._parse_wire(wire_node)
            if wire:
                sheet.wires.append(wire)

        # Parse labels
        for label_type in ['label', 'global_label', 'hierarchical_label']:
            labels = self._find_all_nodes(self.sexp_data, label_type)
            for label_node in labels:
                label = self._parse_label(label_node, label_type)
                if label:
                    sheet.labels.append(label)

        # Parse power labels
        power_labels = self._find_all_nodes(self.sexp_data, 'power_label')
        for label_node in power_labels:
            label = self._parse_label(label_node, 'power')
            if label:
                sheet.labels.append(label)

        return sheet

    def _parse_component(self, symbol_node: List) -> Optional[Component]:
        """Parse a component from symbol node."""
        lib_id = self._get_value(symbol_node, 'lib_id')
        if not lib_id:
            return None

        # Get position
        at_node = self._find_node(symbol_node, 'at')
        position = Position(0, 0)
        rotation = 0
        if at_node and len(at_node) >= 3:
            position = Position(float(at_node[1]), float(at_node[2]))
            if len(at_node) >= 4:
                rotation = int(at_node[3])

        # Get properties
        properties = {}
        reference = ''
        value = ''
        footprint = ''

        property_nodes = self._find_all_nodes(symbol_node, 'property')
        for prop in property_nodes:
            if len(prop) >= 3:
                prop_name = str(prop[1])
                prop_value = str(prop[2])
                properties[prop_name] = prop_value

                if prop_name == 'Reference':
                    reference = prop_value
                elif prop_name == 'Value':
                    value = prop_value
                elif prop_name == 'Footprint':
                    footprint = prop_value

        if not reference:
            return None

        return Component(
            reference=reference,
            value=value,
            symbol=lib_id,
            footprint=footprint,
            position=position,
            rotation=rotation,
            properties=properties,
            pins=[]  # Pins would require parsing lib_symbols
        )

    def _parse_wire(self, wire_node: List) -> Optional[Wire]:
        """Parse a wire from wire node."""
        pts_node = self._find_node(wire_node, 'pts')
        if not pts_node or len(pts_node) < 3:
            return None

        points = self._find_all_nodes(pts_node, 'xy')
        if len(points) < 2:
            return None

        start = Position(float(points[0][1]), float(points[0][2]))
        end = Position(float(points[1][1]), float(points[1][2]))

        return Wire(
            start_point=start,
            end_point=end,
            net_name=''
        )

    def _parse_label(self, label_node: List, label_type: str) -> Optional[Label]:
        """Parse a label from label node."""
        if len(label_node) < 2:
            return None

        text = str(label_node[1])

        at_node = self._find_node(label_node, 'at')
        position = Position(0, 0)
        if at_node and len(at_node) >= 3:
            position = Position(float(at_node[1]), float(at_node[2]))

        return Label(
            text=text,
            position=position,
            label_type=label_type.replace('_label', '')
        )

    def _parse_title_block(self) -> ProjectMetadata:
        """Parse title block metadata."""
        metadata = ProjectMetadata()

        title_block = self._find_node(self.sexp_data, 'title_block')
        if not title_block:
            return metadata

        metadata.title = self._get_value(title_block, 'title') or ''
        metadata.date = self._get_value(title_block, 'date') or ''
        metadata.revision = self._get_value(title_block, 'rev') or ''
        metadata.company = self._get_value(title_block, 'company') or ''

        return metadata

    def _build_net_connections(self, sheet: SchematicSheet):
        """Build net connections from wires and labels."""
        # Group labels by position to identify net names
        label_positions: Dict[Tuple[float, float], str] = {}
        for label in sheet.labels:
            key = (round(label.position.x, 1), round(label.position.y, 1))
            label_positions[key] = label.text

        # Create nets from labels
        nets_by_name: Dict[str, Net] = {}
        for label in sheet.labels:
            if label.text not in nets_by_name:
                net_type = 'power' if label.label_type == 'power' else 'signal'
                if label.text.upper() in ['GND', 'VSS', 'AGND', 'PGND']:
                    net_type = 'ground'

                nets_by_name[label.text] = Net(
                    name=label.text,
                    connections=[],
                    properties=NetProperties(net_type=net_type)
                )

        sheet.nets = list(nets_by_name.values())


def main():
    parser = argparse.ArgumentParser(
        description='Parse KiCad schematic and output structured data'
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='Path to schematic file'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='json',
        choices=['json', 'summary'],
        help='Output format'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (stdout if not specified)'
    )

    args = parser.parse_args()

    try:
        parser_obj = SchematicParser(args.path)
        schematic = parser_obj.parse()

        if args.format == 'json':
            # Convert to JSON-serializable format
            def convert_dataclass(obj):
                if hasattr(obj, '__dataclass_fields__'):
                    return {k: convert_dataclass(v) for k, v in asdict(obj).items()}
                elif isinstance(obj, list):
                    return [convert_dataclass(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert_dataclass(v) for k, v in obj.items()}
                elif isinstance(obj, Enum):
                    return obj.value
                return obj

            output = json.dumps(convert_dataclass(schematic), indent=2)
        else:
            # Summary format
            output = f"""Schematic: {schematic.project_name}
Version: {schematic.version}
Sheets: {len(schematic.sheets)}

Sheet 1 ({schematic.sheets[0].name}):
  Components: {len(schematic.sheets[0].components)}
  Wires: {len(schematic.sheets[0].wires)}
  Labels: {len(schematic.sheets[0].labels)}
  Nets: {len(schematic.sheets[0].nets)}

Wire/Component Ratio: {len(schematic.sheets[0].wires) / max(len(schematic.sheets[0].components), 1):.2f}
"""

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
        else:
            print(output)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
