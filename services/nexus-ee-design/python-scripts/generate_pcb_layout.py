#!/usr/bin/env python3
"""
PCB Layout Generator - Converts schematics to initial PCB layouts.

This script is called by the TypeScript layout-worker to generate
initial PCB layouts from KiCad schematics.

Input: JSON file with:
  - layoutId: string
  - schematicId: string
  - schematicPath: string (path to .kicad_sch file)
  - outputPath: string (where to save .kicad_pcb)
  - workDir: string

Output (JSON to stdout):
  - success: boolean
  - score: number (0-100 initial quality score)
  - filePath: string (path to generated .kicad_pcb)
  - drcViolations: number
  - components: number (count)
  - message: string

Author: Nexus EE Design Team
"""

import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Component:
    """A component to place on the PCB."""
    reference: str  # U1, R1, C1, etc.
    value: str
    footprint: str
    position: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class BoardOutline:
    """PCB board outline."""
    width: float  # mm
    height: float  # mm
    origin: Tuple[float, float] = (100.0, 100.0)


class PCBLayoutGenerator:
    """Generates initial PCB layouts from KiCad schematics."""

    # Default footprints for common categories
    DEFAULT_FOOTPRINTS = {
        'Capacitor': 'Capacitor_SMD:C_0805_2012Metric',
        'Resistor': 'Resistor_SMD:R_0805_2012Metric',
        'Inductor': 'Inductor_SMD:L_1210_3225Metric',
        'MCU': 'Package_QFP:LQFP-48_7x7mm_P0.5mm',
        'MOSFET': 'Package_TO_SOT_SMD:TO-263-3_TabPin2',
        'Gate_Driver': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
        'Amplifier': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
        'CAN_Transceiver': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
        'Connector': 'Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical',
        'LED': 'LED_SMD:LED_0805_2012Metric',
        'Diode': 'Diode_SMD:D_SOD-123',
        'Crystal': 'Crystal:Crystal_SMD_3215-2Pin_3.2x1.5mm',
        'Default': 'Resistor_SMD:R_0805_2012Metric',
    }

    # Footprint sizes for placement (width, height in mm)
    FOOTPRINT_SIZES = {
        'LQFP-48': (9.0, 9.0),
        'LQFP-64': (12.0, 12.0),
        'LQFP-100': (16.0, 16.0),
        'SOIC-8': (6.0, 5.0),
        'SOIC-16': (10.0, 4.0),
        'TO-263': (10.0, 15.0),
        'C_0805': (2.0, 1.25),
        'R_0805': (2.0, 1.25),
        'L_1210': (3.2, 2.5),
        'Default': (4.0, 4.0),
    }

    def __init__(self, schematic_path: Path, output_path: Path):
        self.schematic_path = schematic_path
        self.output_path = output_path
        self.components: List[Component] = []
        self.board_outline = BoardOutline(width=100.0, height=80.0)

    def parse_schematic(self) -> int:
        """Parse schematic and extract components with footprints."""
        if not self.schematic_path.exists():
            raise FileNotFoundError(f"Schematic not found: {self.schematic_path}")

        content = self.schematic_path.read_text()

        # Extract symbol instances with footprints
        # Pattern: (symbol (lib_id "xxx") ... (property "Footprint" "yyy") ... (property "Reference" "zzz"))
        symbol_pattern = re.compile(
            r'\(symbol\s+\(lib_id\s+"([^"]+)"\).*?'
            r'\(property\s+"Reference"\s+"([^"]+)".*?'
            r'\(property\s+"Value"\s+"([^"]*)".*?'
            r'\(property\s+"Footprint"\s+"([^"]*)"',
            re.DOTALL
        )

        # Also try alternative ordering
        alt_pattern = re.compile(
            r'\(symbol\s+\(lib_id\s+"([^"]+)"\).*?'
            r'(?:\(property\s+"Reference"\s+"([^"]+)".*?)?'
            r'(?:\(property\s+"Value"\s+"([^"]*)".*?)?'
            r'(?:\(property\s+"Footprint"\s+"([^"]*)")?',
            re.DOTALL
        )

        matches = symbol_pattern.findall(content)

        for lib_id, reference, value, footprint in matches:
            # Skip power symbols
            if reference.startswith('#') or lib_id.startswith('power:'):
                continue

            # Determine footprint
            if not footprint or footprint == '""':
                footprint = self._get_default_footprint(lib_id, reference)

            comp = Component(
                reference=reference,
                value=value,
                footprint=footprint
            )
            self.components.append(comp)

        return len(self.components)

    def _get_default_footprint(self, lib_id: str, reference: str) -> str:
        """Get default footprint based on component type."""
        # Check by reference prefix
        prefix = re.match(r'^([A-Z]+)', reference)
        if prefix:
            prefix = prefix.group(1)
            if prefix in ['U']:
                if 'STM32' in lib_id or 'MCU' in lib_id:
                    return self.DEFAULT_FOOTPRINTS['MCU']
                return self.DEFAULT_FOOTPRINTS['Gate_Driver']
            elif prefix in ['R']:
                return self.DEFAULT_FOOTPRINTS['Resistor']
            elif prefix in ['C']:
                return self.DEFAULT_FOOTPRINTS['Capacitor']
            elif prefix in ['L']:
                return self.DEFAULT_FOOTPRINTS['Inductor']
            elif prefix in ['Q']:
                return self.DEFAULT_FOOTPRINTS['MOSFET']
            elif prefix in ['D']:
                return self.DEFAULT_FOOTPRINTS['Diode']
            elif prefix in ['J']:
                return self.DEFAULT_FOOTPRINTS['Connector']
            elif prefix in ['Y']:
                return self.DEFAULT_FOOTPRINTS['Crystal']

        return self.DEFAULT_FOOTPRINTS['Default']

    def _get_footprint_size(self, footprint: str) -> Tuple[float, float]:
        """Get footprint dimensions for placement."""
        for key, size in self.FOOTPRINT_SIZES.items():
            if key in footprint:
                return size
        return self.FOOTPRINT_SIZES['Default']

    def calculate_board_size(self) -> BoardOutline:
        """Calculate board size based on components."""
        if not self.components:
            return BoardOutline(width=50.0, height=40.0)

        # Calculate total area needed
        total_area = 0.0
        for comp in self.components:
            w, h = self._get_footprint_size(comp.footprint)
            total_area += w * h * 2.5  # 2.5x overhead for routing

        # Assume roughly square aspect ratio
        import math
        side = math.sqrt(total_area)

        # Add margins
        margin = 10.0
        width = max(50.0, side + 2 * margin)
        height = max(40.0, side * 0.8 + 2 * margin)

        return BoardOutline(width=width, height=height)

    def place_components(self):
        """Place components using grid-based algorithm."""
        if not self.components:
            return

        self.board_outline = self.calculate_board_size()

        # Simple grid placement
        x = self.board_outline.origin[0] + 10.0
        y = self.board_outline.origin[1] + 10.0
        max_x = self.board_outline.origin[0] + self.board_outline.width - 10.0
        row_height = 0.0

        for comp in self.components:
            w, h = self._get_footprint_size(comp.footprint)

            # Check if we need new row
            if x + w > max_x:
                x = self.board_outline.origin[0] + 10.0
                y += row_height + 5.0  # 5mm spacing between rows
                row_height = 0.0

            comp.position = (x + w/2, y + h/2)
            row_height = max(row_height, h)
            x += w + 3.0  # 3mm spacing between components

    def generate_kicad_pcb(self) -> str:
        """Generate KiCad PCB file content."""
        lines = [
            '(kicad_pcb (version 20231014) (generator "nexus_ee_design")',
            '',
            '  (general',
            '    (thickness 1.6)',
            '    (legacy_teardrops no)',
            '  )',
            '',
            '  (paper "A4")',
            '',
            self._generate_layers(),
            '',
            self._generate_setup(),
            '',
            self._generate_nets(),
            '',
            self._generate_footprints(),
            '',
            self._generate_board_outline(),
            '',
            ')'
        ]
        return '\n'.join(lines)

    def _generate_layers(self) -> str:
        """Generate layer definitions."""
        return '''  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
    (50 "User.1" user)
    (51 "User.2" user)
  )'''

    def _generate_setup(self) -> str:
        """Generate design rules setup."""
        return '''  (setup
    (pad_to_mask_clearance 0)
    (allow_soldermask_bridges_in_footprints no)
    (pcbplotparams
      (layerselection 0x00010fc_ffffffff)
      (plot_on_all_layers_selection 0x0000000_00000000)
      (disableapertmacros no)
      (usegerberextensions no)
      (usegerberattributes yes)
      (usegerberadvancedattributes yes)
      (creategerberjobfile yes)
      (dashed_line_dash_ratio 12.000000)
      (dashed_line_gap_ratio 3.000000)
      (svgprecision 4)
      (plotframeref no)
      (viasonmask no)
      (mode 1)
      (useauxorigin no)
      (hpglpennumber 1)
      (hpglpenspeed 20)
      (hpglpendiameter 15.000000)
      (pdf_front_fp_property_popups yes)
      (pdf_back_fp_property_popups yes)
      (dxfpolygonmode yes)
      (dxfimperialunits yes)
      (dxfusepcbnewfont yes)
      (psnegative no)
      (psa4output no)
      (plotreference yes)
      (plotvalue yes)
      (plotfptext yes)
      (plotinvisibletext no)
      (sketchpadsonfab no)
      (subtractmaskfromsilk no)
      (outputformat 1)
      (mirror no)
      (drillshape 1)
      (scaleselection 1)
      (outputdirectory "")
    )
  )'''

    def _generate_nets(self) -> str:
        """Generate net definitions."""
        lines = ['  (net 0 "")']
        # Add basic power nets
        lines.append('  (net 1 "GND")')
        lines.append('  (net 2 "VCC")')
        return '\n'.join(lines)

    def _generate_footprints(self) -> str:
        """Generate footprint placements."""
        if not self.components:
            return ''

        lines = []
        for i, comp in enumerate(self.components):
            fp = self._component_to_footprint(comp, i)
            lines.append(fp)
        return '\n\n'.join(lines)

    def _component_to_footprint(self, comp: Component, index: int) -> str:
        """Convert component to KiCad footprint S-expression."""
        x, y = comp.position
        # Simplified footprint - creates a placeholder
        return f'''  (footprint "{comp.footprint}"
    (layer "F.Cu")
    (uuid "{comp.uuid}")
    (at {x:.2f} {y:.2f} {comp.rotation})
    (property "Reference" "{comp.reference}"
      (at 0 -3 0)
      (layer "F.SilkS")
      (uuid "{uuid.uuid4()}")
      (effects (font (size 1 1) (thickness 0.15)))
    )
    (property "Value" "{comp.value}"
      (at 0 3 0)
      (layer "F.Fab")
      (uuid "{uuid.uuid4()}")
      (effects (font (size 1 1) (thickness 0.15)))
    )
    (property "Footprint" "{comp.footprint}"
      (at 0 0 0)
      (unlocked yes)
      (layer "F.Fab")
      (hide yes)
      (uuid "{uuid.uuid4()}")
      (effects (font (size 1.27 1.27) (thickness 0.15)))
    )
    (pad "1" smd rect
      (at -1 0)
      (size 1 1.5)
      (layers "F.Cu" "F.Paste" "F.Mask")
      (uuid "{uuid.uuid4()}")
    )
    (pad "2" smd rect
      (at 1 0)
      (size 1 1.5)
      (layers "F.Cu" "F.Paste" "F.Mask")
      (uuid "{uuid.uuid4()}")
    )
  )'''

    def _generate_board_outline(self) -> str:
        """Generate board edge cuts."""
        ox, oy = self.board_outline.origin
        w, h = self.board_outline.width, self.board_outline.height

        return f'''  (gr_rect
    (start {ox:.2f} {oy:.2f})
    (end {ox + w:.2f} {oy + h:.2f})
    (stroke (width 0.1) (type solid))
    (fill none)
    (layer "Edge.Cuts")
    (uuid "{uuid.uuid4()}")
  )'''

    def generate(self) -> Dict[str, Any]:
        """Run the full generation pipeline."""
        try:
            # Parse schematic
            comp_count = self.parse_schematic()

            # Place components
            self.place_components()

            # Generate PCB
            pcb_content = self.generate_kicad_pcb()

            # Write output
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(pcb_content)

            # Calculate initial quality score
            # Basic scoring: more components = lower initial score (more to route)
            base_score = 70
            penalty = min(30, comp_count * 2)  # 2 points per component, max 30
            score = max(40, base_score - penalty)

            return {
                'success': True,
                'score': score,
                'filePath': str(self.output_path),
                'drcViolations': comp_count,  # Initially, all unrouted = violations
                'components': comp_count,
                'boardWidth': self.board_outline.width,
                'boardHeight': self.board_outline.height,
                'message': f'Generated PCB with {comp_count} components'
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0,
                'filePath': '',
                'drcViolations': 0,
                'components': 0,
                'message': f'Error: {str(e)}'
            }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'message': 'Usage: generate_pcb_layout.py <input_json_path>'
        }))
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(json.dumps({
            'success': False,
            'message': f'Input file not found: {input_path}'
        }))
        sys.exit(1)

    # Parse input
    try:
        config = json.loads(input_path.read_text())
    except json.JSONDecodeError as e:
        print(json.dumps({
            'success': False,
            'message': f'Invalid JSON: {e}'
        }))
        sys.exit(1)

    schematic_path = Path(config.get('schematicPath', ''))
    output_path = Path(config.get('outputPath', '/tmp/output.kicad_pcb'))

    # Generate PCB
    generator = PCBLayoutGenerator(schematic_path, output_path)
    result = generator.generate()

    # Output result
    print(json.dumps(result))


if __name__ == '__main__':
    main()
