#!/usr/bin/env python3
"""
KiCad Schematic Parser

Parses KiCad schematic files (.kicad_sch) and extracts component information,
net connections, and hierarchical structure.
"""

import json
import sys
from pathlib import Path
from typing import Any

import sexpdata


def parse_sexp_file(filepath: str) -> Any:
    """Parse a KiCad S-expression file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return sexpdata.loads(content)


def extract_symbol(symbol_sexp: list) -> dict:
    """Extract component information from a symbol S-expression."""
    component = {
        'reference': '',
        'value': '',
        'footprint': '',
        'position': {'x': 0, 'y': 0},
        'rotation': 0,
        'properties': {},
        'pins': []
    }

    for item in symbol_sexp:
        if isinstance(item, list) and len(item) > 0:
            key = str(item[0])

            if key == 'lib_id':
                component['lib_id'] = str(item[1])

            elif key == 'at':
                component['position']['x'] = float(item[1])
                component['position']['y'] = float(item[2])
                if len(item) > 3:
                    component['rotation'] = float(item[3])

            elif key == 'property':
                prop_name = str(item[1])
                prop_value = str(item[2]) if len(item) > 2 else ''

                if prop_name == 'Reference':
                    component['reference'] = prop_value
                elif prop_name == 'Value':
                    component['value'] = prop_value
                elif prop_name == 'Footprint':
                    component['footprint'] = prop_value
                else:
                    component['properties'][prop_name] = prop_value

            elif key == 'pin':
                pin = {
                    'name': str(item[1]) if len(item) > 1 else '',
                    'uuid': ''
                }
                for subitem in item:
                    if isinstance(subitem, list) and str(subitem[0]) == 'uuid':
                        pin['uuid'] = str(subitem[1])
                component['pins'].append(pin)

    return component


def extract_wire(wire_sexp: list) -> dict:
    """Extract wire information from a wire S-expression."""
    wire = {
        'points': [],
        'uuid': ''
    }

    for item in wire_sexp:
        if isinstance(item, list) and len(item) > 0:
            key = str(item[0])

            if key == 'pts':
                for pt in item[1:]:
                    if isinstance(pt, list) and str(pt[0]) == 'xy':
                        wire['points'].append({
                            'x': float(pt[1]),
                            'y': float(pt[2])
                        })

            elif key == 'uuid':
                wire['uuid'] = str(item[1])

    return wire


def extract_label(label_sexp: list) -> dict:
    """Extract label/net name from a label S-expression."""
    label = {
        'name': '',
        'type': 'local',
        'position': {'x': 0, 'y': 0},
        'rotation': 0,
        'uuid': ''
    }

    if len(label_sexp) > 1:
        label['name'] = str(label_sexp[1])

    for item in label_sexp:
        if isinstance(item, list) and len(item) > 0:
            key = str(item[0])

            if key == 'at':
                label['position']['x'] = float(item[1])
                label['position']['y'] = float(item[2])
                if len(item) > 3:
                    label['rotation'] = float(item[3])

            elif key == 'uuid':
                label['uuid'] = str(item[1])

    return label


def parse_schematic(filepath: str) -> dict:
    """Parse a KiCad schematic file and extract all information."""
    sexp = parse_sexp_file(filepath)

    schematic = {
        'version': '',
        'generator': '',
        'uuid': '',
        'paper': 'A4',
        'components': [],
        'wires': [],
        'labels': [],
        'global_labels': [],
        'hierarchical_labels': [],
        'sheets': [],
        'junctions': [],
        'no_connects': []
    }

    for item in sexp:
        if isinstance(item, list) and len(item) > 0:
            key = str(item[0])

            if key == 'version':
                schematic['version'] = str(item[1])

            elif key == 'generator':
                schematic['generator'] = str(item[1])

            elif key == 'uuid':
                schematic['uuid'] = str(item[1])

            elif key == 'paper':
                schematic['paper'] = str(item[1])

            elif key == 'symbol':
                component = extract_symbol(item)
                schematic['components'].append(component)

            elif key == 'wire':
                wire = extract_wire(item)
                schematic['wires'].append(wire)

            elif key == 'label':
                label = extract_label(item)
                label['type'] = 'local'
                schematic['labels'].append(label)

            elif key == 'global_label':
                label = extract_label(item)
                label['type'] = 'global'
                schematic['global_labels'].append(label)

            elif key == 'hierarchical_label':
                label = extract_label(item)
                label['type'] = 'hierarchical'
                schematic['hierarchical_labels'].append(label)

            elif key == 'sheet':
                sheet = {
                    'name': '',
                    'filename': '',
                    'uuid': '',
                    'position': {'x': 0, 'y': 0},
                    'size': {'w': 0, 'h': 0}
                }
                for subitem in item:
                    if isinstance(subitem, list) and len(subitem) > 0:
                        subkey = str(subitem[0])
                        if subkey == 'at':
                            sheet['position']['x'] = float(subitem[1])
                            sheet['position']['y'] = float(subitem[2])
                        elif subkey == 'size':
                            sheet['size']['w'] = float(subitem[1])
                            sheet['size']['h'] = float(subitem[2])
                        elif subkey == 'uuid':
                            sheet['uuid'] = str(subitem[1])
                        elif subkey == 'property':
                            if str(subitem[1]) == 'Sheetname':
                                sheet['name'] = str(subitem[2])
                            elif str(subitem[1]) == 'Sheetfile':
                                sheet['filename'] = str(subitem[2])
                schematic['sheets'].append(sheet)

            elif key == 'junction':
                junction = {'position': {'x': 0, 'y': 0}, 'uuid': ''}
                for subitem in item:
                    if isinstance(subitem, list) and len(subitem) > 0:
                        subkey = str(subitem[0])
                        if subkey == 'at':
                            junction['position']['x'] = float(subitem[1])
                            junction['position']['y'] = float(subitem[2])
                        elif subkey == 'uuid':
                            junction['uuid'] = str(subitem[1])
                schematic['junctions'].append(junction)

            elif key == 'no_connect':
                nc = {'position': {'x': 0, 'y': 0}, 'uuid': ''}
                for subitem in item:
                    if isinstance(subitem, list) and len(subitem) > 0:
                        subkey = str(subitem[0])
                        if subkey == 'at':
                            nc['position']['x'] = float(subitem[1])
                            nc['position']['y'] = float(subitem[2])
                        elif subkey == 'uuid':
                            nc['uuid'] = str(subitem[1])
                schematic['no_connects'].append(nc)

    return schematic


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: parse_schematic.py <schematic_file.kicad_sch>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        schematic = parse_schematic(filepath)
        print(json.dumps(schematic, indent=2))
    except Exception as e:
        print(f"Error parsing schematic: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()