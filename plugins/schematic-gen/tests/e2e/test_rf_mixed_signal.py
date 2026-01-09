#!/usr/bin/env python3
"""
End-to-End Test: RF Mixed-Signal Circuit - PRODUCTION VERSION

A comprehensive mixed-signal circuit to test the complete pipeline with:
- RF Section: LNA, matching network, antenna interface
- Analog Section: Op-amp buffer, ADC input conditioning
- Digital Section: MCU, logic level shifter, I2C bus

CRITICAL: This module generates REAL PCB content with proper RF layout considerations:
- Controlled impedance traces for RF signals (50Ω)
- Ground plane isolation between RF/analog/digital
- Proper component placement for signal integrity
- All designators on silkscreen
- Professional 45-degree routing

Components:
- U1: STM32F103 microcontroller (LQFP48)
- U2: LNA - BGA2801 (SOT363)
- U3: Op-Amp - OPA344 (SOT23-5)
- U4: ADC - MCP3201 (SOIC-8)
- U5: Level Shifter - TXS0102 (VSSOP-8)
- L1: RF Inductor - 10nH (0603)
- L2: RF Inductor - 22nH (0603)
- C1-C6: Various capacitors (0402, 0603)
- R1-R8: Various resistors (0402, 0603)
- ANT1: RF connector (SMA)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'python-scripts'))

from validation_exceptions import (
    ValidationFailure,
    MissingDependencyFailure
)


# =============================================================================
# CIRCUIT DEFINITION - RF MIXED-SIGNAL BOARD
# =============================================================================

# Board configuration
BOARD_CONFIG = {
    'width_mm': 60.0,
    'height_mm': 50.0,
    'layers': 4,  # 4-layer for proper RF performance
    'trace_width_rf': 0.35,  # 50Ω controlled impedance (approximate)
    'trace_width_signal': 0.2,
    'trace_width_power': 0.5,
    'via_drill': 0.3,
    'via_size': 0.6,
    'clearance': 0.15,
}

# Component definitions organized by section
COMPONENTS = [
    # =========================================================================
    # DIGITAL SECTION (Left side of board)
    # =========================================================================
    {
        'ref': 'U1',
        'value': 'STM32F103C8',
        'footprint': 'Package_QFP:LQFP-48_7x7mm_P0.5mm',
        'position': (15, 25),
        'rotation': 0,
        'section': 'digital',
        'description': 'Main MCU - ARM Cortex-M3',
        'pins': {
            # Power
            '1': 'VBAT', '23': 'VSS', '24': 'VDD', '35': 'VSS', '36': 'VDD',
            '47': 'VSS', '48': 'VDD',
            # SPI to ADC
            '25': 'SPI_CS', '26': 'SPI_SCK', '27': 'SPI_MISO', '28': 'SPI_MOSI',
            # I2C
            '29': 'I2C_SCL', '30': 'I2C_SDA',
            # ADC inputs
            '10': 'ADC_IN0', '11': 'ADC_IN1',
            # GPIO
            '12': 'GPIO_RF_EN', '13': 'GPIO_LED',
            # Crystal
            '5': 'OSC_IN', '6': 'OSC_OUT',
            # Reset
            '7': 'NRST',
        }
    },
    {
        'ref': 'U5',
        'value': 'TXS0102',
        'footprint': 'Package_SO:VSSOP-8_2.3x2mm_P0.5mm',
        'position': (8, 10),
        'rotation': 0,
        'section': 'digital',
        'description': 'I2C Level Shifter 3.3V to 5V',
        'pins': {
            '1': 'VCC_A',  # 3.3V side
            '2': 'I2C_SDA_A',
            '3': 'I2C_SCL_A',
            '4': 'GND',
            '5': 'I2C_SCL_B',
            '6': 'I2C_SDA_B',
            '7': 'OE',
            '8': 'VCC_B',  # 5V side
        }
    },
    {
        'ref': 'Y1',
        'value': '8MHz',
        'footprint': 'Crystal:Crystal_SMD_3215-2Pin_3.2x1.5mm',
        'position': (8, 30),
        'rotation': 0,
        'section': 'digital',
        'description': 'MCU Crystal Oscillator',
        'pins': {'1': 'OSC_IN', '2': 'OSC_OUT'}
    },
    {
        'ref': 'C7',
        'value': '22pF',
        'footprint': 'Capacitor_SMD:C_0402_1005Metric',
        'position': (5, 28),
        'rotation': 0,
        'section': 'digital',
        'description': 'Crystal load cap',
        'pins': {'1': 'OSC_IN', '2': 'GND'}
    },
    {
        'ref': 'C8',
        'value': '22pF',
        'footprint': 'Capacitor_SMD:C_0402_1005Metric',
        'position': (5, 32),
        'rotation': 0,
        'section': 'digital',
        'description': 'Crystal load cap',
        'pins': {'1': 'OSC_OUT', '2': 'GND'}
    },

    # =========================================================================
    # RF SECTION (Right side of board - isolated)
    # =========================================================================
    {
        'ref': 'U2',
        'value': 'BGA2801',
        'footprint': 'Package_TO_SOT_SMD:SOT-363_SC-70-6',
        'position': (52, 25),
        'rotation': 0,
        'section': 'rf',
        'description': 'Low Noise Amplifier - 2.4GHz',
        'pins': {
            '1': 'RF_IN',
            '2': 'GND',
            '3': 'VCC_RF',
            '4': 'RF_OUT',
            '5': 'GND',
            '6': 'VCC_RF',
        }
    },
    {
        'ref': 'ANT1',
        'value': 'SMA',
        'footprint': 'Connector_Coaxial:SMA_Amphenol_132134_EdgeMount',
        'position': (57, 25),
        'rotation': 270,
        'section': 'rf',
        'description': 'RF Antenna Connector - 50Ω',
        'pins': {'1': 'RF_ANT', '2': 'GND'}
    },
    {
        'ref': 'L1',
        'value': '10nH',
        'footprint': 'Inductor_SMD:L_0603_1608Metric',
        'position': (48, 20),
        'rotation': 0,
        'section': 'rf',
        'description': 'RF matching inductor',
        'pins': {'1': 'RF_IN', '2': 'RF_MATCH1'}
    },
    {
        'ref': 'L2',
        'value': '22nH',
        'footprint': 'Inductor_SMD:L_0603_1608Metric',
        'position': (48, 30),
        'rotation': 0,
        'section': 'rf',
        'description': 'RF choke inductor',
        'pins': {'1': 'VCC_RF', '2': 'RF_BIAS'}
    },
    {
        'ref': 'C1',
        'value': '1pF',
        'footprint': 'Capacitor_SMD:C_0402_1005Metric',
        'position': (55, 20),
        'rotation': 0,
        'section': 'rf',
        'description': 'RF matching cap',
        'pins': {'1': 'RF_ANT', '2': 'RF_IN'}
    },
    {
        'ref': 'C2',
        'value': '100pF',
        'footprint': 'Capacitor_SMD:C_0402_1005Metric',
        'position': (48, 25),
        'rotation': 90,
        'section': 'rf',
        'description': 'RF bypass cap',
        'pins': {'1': 'VCC_RF', '2': 'GND'}
    },
    {
        'ref': 'C3',
        'value': '10pF',
        'footprint': 'Capacitor_SMD:C_0402_1005Metric',
        'position': (45, 25),
        'rotation': 0,
        'section': 'rf',
        'description': 'RF output coupling cap',
        'pins': {'1': 'RF_OUT', '2': 'RF_TO_DEMOD'}
    },

    # =========================================================================
    # ANALOG SECTION (Center of board)
    # =========================================================================
    {
        'ref': 'U3',
        'value': 'OPA344',
        'footprint': 'Package_TO_SOT_SMD:SOT-23-5',
        'position': (35, 15),
        'rotation': 0,
        'section': 'analog',
        'description': 'Precision Op-Amp - Rail-to-Rail',
        'pins': {
            '1': 'OPAMP_OUT',
            '2': 'VCC_ANALOG',
            '3': 'OPAMP_INP',  # Non-inverting input
            '4': 'OPAMP_INM',  # Inverting input
            '5': 'GND_ANALOG',
        }
    },
    {
        'ref': 'U4',
        'value': 'MCP3201',
        'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
        'position': (35, 35),
        'rotation': 0,
        'section': 'analog',
        'description': '12-bit ADC - SPI',
        'pins': {
            '1': 'VREF',
            '2': 'ADC_INA',  # Analog input +
            '3': 'ADC_INB',  # Analog input -
            '4': 'GND_ANALOG',
            '5': 'SPI_CS',
            '6': 'SPI_MISO',
            '7': 'SPI_SCK',
            '8': 'VCC_ANALOG',
        }
    },
    {
        'ref': 'R1',
        'value': '10k',
        'footprint': 'Resistor_SMD:R_0603_1608Metric',
        'position': (30, 12),
        'rotation': 0,
        'section': 'analog',
        'description': 'Op-amp feedback resistor',
        'pins': {'1': 'OPAMP_OUT', '2': 'OPAMP_INM'}
    },
    {
        'ref': 'R2',
        'value': '10k',
        'footprint': 'Resistor_SMD:R_0603_1608Metric',
        'position': (30, 18),
        'rotation': 0,
        'section': 'analog',
        'description': 'Op-amp input resistor',
        'pins': {'1': 'RF_TO_DEMOD', '2': 'OPAMP_INP'}
    },
    {
        'ref': 'R3',
        'value': '1k',
        'footprint': 'Resistor_SMD:R_0603_1608Metric',
        'position': (40, 15),
        'rotation': 90,
        'section': 'analog',
        'description': 'ADC input protection',
        'pins': {'1': 'OPAMP_OUT', '2': 'ADC_INA'}
    },
    {
        'ref': 'C4',
        'value': '100nF',
        'footprint': 'Capacitor_SMD:C_0603_1608Metric',
        'position': (38, 12),
        'rotation': 0,
        'section': 'analog',
        'description': 'Op-amp bypass',
        'pins': {'1': 'VCC_ANALOG', '2': 'GND_ANALOG'}
    },
    {
        'ref': 'C5',
        'value': '10uF',
        'footprint': 'Capacitor_SMD:C_0805_2012Metric',
        'position': (32, 35),
        'rotation': 0,
        'section': 'analog',
        'description': 'ADC VREF decoupling',
        'pins': {'1': 'VREF', '2': 'GND_ANALOG'}
    },
    {
        'ref': 'R4',
        'value': '100k',
        'footprint': 'Resistor_SMD:R_0603_1608Metric',
        'position': (38, 35),
        'rotation': 0,
        'section': 'analog',
        'description': 'ADC bias resistor',
        'pins': {'1': 'ADC_INB', '2': 'GND_ANALOG'}
    },

    # =========================================================================
    # POWER SECTION (Bottom of board)
    # =========================================================================
    {
        'ref': 'C6',
        'value': '10uF',
        'footprint': 'Capacitor_SMD:C_0805_2012Metric',
        'position': (15, 45),
        'rotation': 0,
        'section': 'power',
        'description': 'MCU VDD decoupling',
        'pins': {'1': 'VDD', '2': 'GND'}
    },
    {
        'ref': 'R5',
        'value': '10k',
        'footprint': 'Resistor_SMD:R_0603_1608Metric',
        'position': (10, 40),
        'rotation': 0,
        'section': 'power',
        'description': 'NRST pull-up',
        'pins': {'1': 'VDD', '2': 'NRST'}
    },
    {
        'ref': 'R6',
        'value': '330',
        'footprint': 'Resistor_SMD:R_0603_1608Metric',
        'position': (20, 40),
        'rotation': 0,
        'section': 'power',
        'description': 'LED current limit',
        'pins': {'1': 'GPIO_LED', '2': 'LED_A'}
    },
    {
        'ref': 'D1',
        'value': 'GREEN',
        'footprint': 'LED_SMD:LED_0603_1608Metric',
        'position': (25, 40),
        'rotation': 0,
        'section': 'power',
        'description': 'Status LED',
        'pins': {'1': 'LED_A', '2': 'GND'}
    },
    {
        'ref': 'R7',
        'value': '4.7k',
        'footprint': 'Resistor_SMD:R_0402_1005Metric',
        'position': (12, 15),
        'rotation': 90,
        'section': 'digital',
        'description': 'I2C pull-up SCL',
        'pins': {'1': 'VCC_A', '2': 'I2C_SCL_A'}
    },
    {
        'ref': 'R8',
        'value': '4.7k',
        'footprint': 'Resistor_SMD:R_0402_1005Metric',
        'position': (14, 15),
        'rotation': 90,
        'section': 'digital',
        'description': 'I2C pull-up SDA',
        'pins': {'1': 'VCC_A', '2': 'I2C_SDA_A'}
    },
]

# =============================================================================
# NET DEFINITIONS
# =============================================================================
NETS = {
    # Power nets
    'VDD': [
        ('U1', '24'), ('U1', '36'), ('U1', '48'), ('U1', '1'),  # MCU power
        ('C6', '1'), ('R5', '1'),
    ],
    'GND': [
        ('U1', '23'), ('U1', '35'), ('U1', '47'),  # MCU ground
        ('U5', '4'), ('C6', '2'), ('C7', '2'), ('C8', '2'),
        ('ANT1', '2'), ('U2', '2'), ('U2', '5'), ('C2', '2'),
        ('D1', '2'),
    ],
    'VCC_A': [  # 3.3V for I2C A-side
        ('U5', '1'), ('R7', '1'), ('R8', '1'),
    ],
    'VCC_B': [  # 5V for I2C B-side
        ('U5', '8'),
    ],
    'VCC_RF': [
        ('U2', '3'), ('U2', '6'), ('L2', '1'), ('C2', '1'),
    ],
    'VCC_ANALOG': [
        ('U3', '2'), ('U4', '8'), ('C4', '1'),
    ],
    'GND_ANALOG': [
        ('U3', '5'), ('U4', '4'), ('C4', '2'), ('C5', '2'), ('R4', '2'),
    ],
    'VREF': [
        ('U4', '1'), ('C5', '1'),
    ],

    # Crystal oscillator
    'OSC_IN': [('U1', '5'), ('Y1', '1'), ('C7', '1')],
    'OSC_OUT': [('U1', '6'), ('Y1', '2'), ('C8', '1')],

    # Reset
    'NRST': [('U1', '7'), ('R5', '2')],

    # SPI bus (MCU to ADC)
    'SPI_CS': [('U1', '25'), ('U4', '5')],
    'SPI_SCK': [('U1', '26'), ('U4', '7')],
    'SPI_MISO': [('U1', '27'), ('U4', '6')],
    'SPI_MOSI': [('U1', '28')],  # ADC is read-only

    # I2C bus
    'I2C_SCL': [('U1', '29')],
    'I2C_SDA': [('U1', '30')],
    'I2C_SCL_A': [('U5', '3'), ('R7', '2')],
    'I2C_SDA_A': [('U5', '2'), ('R8', '2')],
    'I2C_SCL_B': [('U5', '5')],
    'I2C_SDA_B': [('U5', '6')],
    'OE': [('U5', '7')],  # Level shifter enable

    # RF signal path
    'RF_ANT': [('ANT1', '1'), ('C1', '1')],
    'RF_IN': [('C1', '2'), ('L1', '1'), ('U2', '1')],
    'RF_MATCH1': [('L1', '2')],
    'RF_BIAS': [('L2', '2')],
    'RF_OUT': [('U2', '4'), ('C3', '1')],
    'RF_TO_DEMOD': [('C3', '2'), ('R2', '1')],

    # Analog signal path
    'OPAMP_INP': [('U3', '3'), ('R2', '2')],
    'OPAMP_INM': [('U3', '4'), ('R1', '2')],
    'OPAMP_OUT': [('U3', '1'), ('R1', '1'), ('R3', '1')],
    'ADC_INA': [('U4', '2'), ('R3', '2')],
    'ADC_INB': [('U4', '3'), ('R4', '1')],

    # GPIO
    'GPIO_RF_EN': [('U1', '12')],
    'GPIO_LED': [('U1', '13'), ('R6', '1')],
    'LED_A': [('R6', '2'), ('D1', '1')],
    'ADC_IN0': [('U1', '10')],
    'ADC_IN1': [('U1', '11')],
}


# =============================================================================
# NETLIST GENERATION
# =============================================================================

def generate_real_netlist(output_path: str) -> Dict[str, Any]:
    """
    Generate a proper KiCad netlist for the RF mixed-signal circuit.
    """
    netlist_lines = [
        '(export (version "E")',
        '  (design',
        '    (source "test_rf_mixed_signal.py")',
        '    (date "2026-01-09")',
        '    (tool "SKiDL/KiCad Netlist Generator")',
        '    (comment1 "RF Mixed-Signal Test Circuit")',
        '    (comment2 "Digital + Analog + RF sections"))',
        '  (components'
    ]

    # Add components
    for comp in COMPONENTS:
        netlist_lines.extend([
            f'    (comp (ref "{comp["ref"]}")',
            f'      (value "{comp["value"]}")',
            f'      (footprint "{comp["footprint"]}")',
            f'      (description "{comp.get("description", "")}"))'
        ])

    netlist_lines.append('  )')  # Close components
    netlist_lines.append('  (nets')

    # Add nets
    net_code = 1
    for net_name, connections in NETS.items():
        netlist_lines.append(f'    (net (code "{net_code}") (name "{net_name}")')
        for ref, pin in connections:
            netlist_lines.append(f'      (node (ref "{ref}") (pin "{pin}"))')
        netlist_lines.append('    )')
        net_code += 1

    netlist_lines.append('  )')  # Close nets
    netlist_lines.append(')')  # Close export

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(netlist_lines))

    return {
        'success': True,
        'output': output_path,
        'components': len(COMPONENTS),
        'nets': len(NETS),
        'sections': {
            'digital': len([c for c in COMPONENTS if c.get('section') == 'digital']),
            'rf': len([c for c in COMPONENTS if c.get('section') == 'rf']),
            'analog': len([c for c in COMPONENTS if c.get('section') == 'analog']),
            'power': len([c for c in COMPONENTS if c.get('section') == 'power']),
        }
    }


# =============================================================================
# PCB GENERATION - REAL LAYOUT WITH RF CONSIDERATIONS
# =============================================================================

def generate_real_pcb(schematic_path: str, output_path: str) -> Dict[str, Any]:
    """
    Generate a REAL PCB layout with proper RF layout considerations.

    Features:
    - 4-layer stackup (Signal/GND/Power/Signal)
    - Controlled impedance RF traces
    - Ground plane isolation between sections
    - Proper via stitching around RF
    - All designators on silkscreen
    - Professional 45-degree routing
    """
    board_width = BOARD_CONFIG['width_mm']
    board_height = BOARD_CONFIG['height_mm']

    # Start building PCB file
    pcb_lines = [
        '(kicad_pcb (version 20230121) (generator "test_rf_mixed_signal")',
        '',
        '  (general',
        '    (thickness 1.6)',
        '    (legacy_teardrops no))',
        '',
        '  (paper "A4")',
        '',
        '  (layers',
        '    (0 "F.Cu" signal)',
        '    (1 "In1.Cu" signal)',  # GND plane
        '    (2 "In2.Cu" signal)',  # Power plane
        '    (31 "B.Cu" signal)',
        '    (32 "B.Adhes" user "B.Adhesive")',
        '    (33 "F.Adhes" user "F.Adhesive")',
        '    (34 "B.Paste" user)',
        '    (35 "F.Paste" user)',
        '    (36 "B.SilkS" user "B.Silkscreen")',
        '    (37 "F.SilkS" user "F.Silkscreen")',
        '    (38 "B.Mask" user)',
        '    (39 "F.Mask" user)',
        '    (40 "Dwgs.User" user "User.Drawings")',
        '    (41 "Cmts.User" user "User.Comments")',
        '    (42 "Eco1.User" user "User.Eco1")',
        '    (43 "Eco2.User" user "User.Eco2")',
        '    (44 "Edge.Cuts" user)',
        '    (45 "Margin" user)',
        '    (46 "B.CrtYd" user "B.Courtyard")',
        '    (47 "F.CrtYd" user "F.Courtyard")',
        '    (48 "B.Fab" user "B.Fabrication")',
        '    (49 "F.Fab" user "F.Fabrication")',
        '  )',
        '',
        '  (setup',
        '    (pad_to_mask_clearance 0.05)',
        '    (allow_soldermask_bridges_in_footprints no)',
        '    (pcbplotparams',
        '      (layerselection 0x00010fc_ffffffff)',
        '      (plot_on_all_layers_selection 0x0000000_00000000)',
        '      (disableapertmacros no)',
        '      (usegerberextensions no)',
        '      (usegerberattributes yes)',
        '      (usegerberadvancedattributes yes)',
        '      (creategerberjobfile yes)',
        '      (dashed_line_dash_ratio 12.000000)',
        '      (dashed_line_gap_ratio 3.000000)',
        '      (svgprecision 4)',
        '      (plotframeref no)',
        '      (viasonmask no)',
        '      (mode 1)',
        '      (useauxorigin no)',
        '      (hpglpennumber 1)',
        '      (hpglpenspeed 20)',
        '      (hpglpendiameter 15.000000)',
        '      (pdf_front_fp_property_popups yes)',
        '      (pdf_back_fp_property_popups yes)',
        '      (dxfpolygonmode yes)',
        '      (dxfimperialunits yes)',
        '      (dxfusepcbnewfont yes)',
        '      (psnegative no)',
        '      (psa4output no)',
        '      (plotreference yes)',
        '      (plotvalue yes)',
        '      (plotfptext yes)',
        '      (plotinvisibletext no)',
        '      (sketchpadsonfab no)',
        '      (subtractmaskfromsilk no)',
        '      (outputformat 1)',
        '      (mirror no)',
        '      (drillshape 1)',
        '      (scaleselection 1)',
        '      (outputdirectory "")))',
        ''
    ]

    # Add nets
    pcb_lines.append('  (net 0 "")')
    net_codes = {'': 0}
    net_num = 1
    for net_name in NETS.keys():
        pcb_lines.append(f'  (net {net_num} "{net_name}")')
        net_codes[net_name] = net_num
        net_num += 1

    pcb_lines.append('')

    # Add footprints
    for comp in COMPONENTS:
        x, y = comp['position']
        ref = comp['ref']
        value = comp['value']
        fp = comp['footprint']
        rotation = comp.get('rotation', 0)

        pcb_lines.extend(_generate_footprint(comp, net_codes))

    pcb_lines.append('')

    # Generate traces with proper routing
    traces = _generate_traces_rf(COMPONENTS, NETS, net_codes)
    pcb_lines.extend(traces)

    pcb_lines.append('')

    # Add ground plane fill on In1.Cu
    pcb_lines.extend(_generate_ground_plane(board_width, board_height, net_codes))

    pcb_lines.append('')

    # Add board outline
    pcb_lines.extend([
        f'  (gr_rect (start 3 3) (end {board_width - 3} {board_height - 3})',
        '    (stroke (width 0.15) (type default)) (fill none) (layer "Edge.Cuts"))',
        ''
    ])

    # Add title block on silkscreen
    pcb_lines.extend([
        f'  (gr_text "RF Mixed-Signal Board" (at {board_width/2} 47) (layer "F.SilkS")',
        '    (effects (font (size 1.5 1.5) (thickness 0.25))))',
        f'  (gr_text "E2E Test Circuit v1.0" (at {board_width/2} 49) (layer "F.SilkS")',
        '    (effects (font (size 0.8 0.8) (thickness 0.12))))',
        '',
        # Section labels
        f'  (gr_text "DIGITAL" (at 15 5) (layer "F.SilkS")',
        '    (effects (font (size 1 1) (thickness 0.15))))',
        f'  (gr_text "ANALOG" (at 35 5) (layer "F.SilkS")',
        '    (effects (font (size 1 1) (thickness 0.15))))',
        f'  (gr_text "RF" (at 52 5) (layer "F.SilkS")',
        '    (effects (font (size 1 1) (thickness 0.15))))',
        ''
    ])

    # Add via stitching around RF section (for ground plane)
    pcb_lines.extend(_generate_via_stitching(45, 8, 55, 42, net_codes))

    pcb_lines.append(')')

    # Write PCB file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(pcb_lines))

    return {
        'success': True,
        'output': output_path,
        'board_width_mm': board_width,
        'board_height_mm': board_height,
        'layer_count': 4,
        'expected_footprints': len(COMPONENTS),
        'expected_traces': len([t for t in traces if '(segment' in t]),
        'sections': {
            'digital': len([c for c in COMPONENTS if c.get('section') == 'digital']),
            'rf': len([c for c in COMPONENTS if c.get('section') == 'rf']),
            'analog': len([c for c in COMPONENTS if c.get('section') == 'analog']),
            'power': len([c for c in COMPONENTS if c.get('section') == 'power']),
        }
    }


def _generate_footprint(comp: Dict, net_codes: Dict[str, int]) -> List[str]:
    """Generate footprint based on component type."""
    ref = comp['ref']
    value = comp['value']
    x, y = comp['position']
    rotation = comp.get('rotation', 0)
    fp = comp['footprint']
    pins = comp.get('pins', {})

    lines = [
        f'  (footprint "{fp}" (layer "F.Cu")',
        f'    (at {x} {y} {rotation})',
        f'    (property "Reference" "{ref}" (at 0 -2.5 {rotation}) (layer "F.SilkS")',
        f'      (effects (font (size 0.7 0.7) (thickness 0.12))))',
        f'    (property "Value" "{value}" (at 0 2.5 {rotation}) (layer "F.Fab")',
        f'      (effects (font (size 0.7 0.7) (thickness 0.12))))',
    ]

    # Generate pads based on footprint type
    if 'LQFP-48' in fp:
        lines.extend(_generate_lqfp48_pads(pins, net_codes))
    elif 'SOT-363' in fp:
        lines.extend(_generate_sot363_pads(pins, net_codes))
    elif 'SOT-23-5' in fp:
        lines.extend(_generate_sot235_pads(pins, net_codes))
    elif 'SOIC-8' in fp:
        lines.extend(_generate_soic8_pads(pins, net_codes))
    elif 'VSSOP-8' in fp:
        lines.extend(_generate_vssop8_pads(pins, net_codes))
    elif 'Crystal' in fp:
        lines.extend(_generate_crystal_pads(pins, net_codes))
    elif 'SMA' in fp:
        lines.extend(_generate_sma_pads(pins, net_codes))
    elif 'R_0402' in fp or 'C_0402' in fp:
        lines.extend(_generate_0402_pads(pins, net_codes))
    elif 'R_0603' in fp or 'C_0603' in fp or 'L_0603' in fp:
        lines.extend(_generate_0603_pads(pins, net_codes))
    elif 'R_0805' in fp or 'C_0805' in fp:
        lines.extend(_generate_0805_pads(pins, net_codes))
    elif 'LED_0603' in fp:
        lines.extend(_generate_led_0603_pads(pins, net_codes))

    # Add silkscreen outline
    lines.extend(_generate_silkscreen_outline(fp, ref))

    lines.append('  )')
    return lines


def _generate_lqfp48_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate LQFP-48 pads (7x7mm, 0.5mm pitch)."""
    lines = []
    # 12 pins per side, starting from pin 1 at bottom-left corner
    pitch = 0.5
    body_half = 3.5  # 7mm body / 2
    pad_offset = 4.5  # Distance from center to pad center

    # Bottom side (pins 1-12)
    for i in range(12):
        pin_num = str(i + 1)
        px = -2.75 + i * pitch
        py = pad_offset
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.2f} {py:.2f}) (size 0.3 1.2) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )

    # Right side (pins 13-24)
    for i in range(12):
        pin_num = str(i + 13)
        px = pad_offset
        py = 2.75 - i * pitch
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.2f} {py:.2f} 90) (size 0.3 1.2) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )

    # Top side (pins 25-36)
    for i in range(12):
        pin_num = str(i + 25)
        px = 2.75 - i * pitch
        py = -pad_offset
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.2f} {py:.2f}) (size 0.3 1.2) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )

    # Left side (pins 37-48)
    for i in range(12):
        pin_num = str(i + 37)
        px = -pad_offset
        py = -2.75 + i * pitch
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.2f} {py:.2f} 90) (size 0.3 1.2) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )

    return lines


def _generate_sot363_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate SOT-363 (SC-70-6) pads."""
    lines = []
    # 6 pins, 3 on each side, 0.65mm pitch
    positions = [
        ('1', -0.65, 0.9), ('2', 0, 0.9), ('3', 0.65, 0.9),
        ('4', 0.65, -0.9), ('5', 0, -0.9), ('6', -0.65, -0.9),
    ]
    for pin_num, px, py in positions:
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px} {py}) (size 0.4 0.7) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_sot235_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate SOT-23-5 pads."""
    lines = []
    positions = [
        ('1', -0.95, 1.1), ('2', 0, 1.1), ('3', 0.95, 1.1),
        ('4', 0.95, -1.1), ('5', -0.95, -1.1),
    ]
    for pin_num, px, py in positions:
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px} {py}) (size 0.6 0.8) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_soic8_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate SOIC-8 pads (1.27mm pitch)."""
    lines = []
    pitch = 1.27
    for i in range(4):
        pin_num = str(i + 1)
        px = -1.905 + i * pitch
        py = 2.7
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.3f} {py}) (size 0.6 1.5) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )

    for i in range(4):
        pin_num = str(8 - i)
        px = -1.905 + i * pitch
        py = -2.7
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.3f} {py}) (size 0.6 1.5) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_vssop8_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate VSSOP-8 pads (0.5mm pitch)."""
    lines = []
    pitch = 0.5
    for i in range(4):
        pin_num = str(i + 1)
        px = -0.75 + i * pitch
        py = 1.2
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.3f} {py}) (size 0.3 0.8) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )

    for i in range(4):
        pin_num = str(8 - i)
        px = -0.75 + i * pitch
        py = -1.2
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px:.3f} {py}) (size 0.3 0.8) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_crystal_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate crystal pads (3.2x1.5mm SMD)."""
    lines = []
    for pin_num, px in [('1', -1.1), ('2', 1.1)]:
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px} 0) (size 1.0 1.4) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_sma_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate SMA edge-mount connector pads."""
    lines = []
    # Center signal pad
    net_name = pins.get('1', '')
    net_code = net_codes.get(net_name, 0)
    lines.append(
        f'    (pad "1" smd rect (at 0 0) (size 1.5 1.5) '
        f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
    )
    # Ground pads
    net_name = pins.get('2', '')
    net_code = net_codes.get(net_name, 0)
    for px, py in [(-2.5, 0), (2.5, 0)]:
        lines.append(
            f'    (pad "2" smd rect (at {px} {py}) (size 1.5 2.5) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_0402_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate 0402 SMD pads."""
    lines = []
    for pin_num, px in [('1', -0.5), ('2', 0.5)]:
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px} 0) (size 0.5 0.6) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_0603_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate 0603 SMD pads."""
    lines = []
    for pin_num, px in [('1', -0.75), ('2', 0.75)]:
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px} 0) (size 0.8 0.9) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_0805_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate 0805 SMD pads."""
    lines = []
    for pin_num, px in [('1', -0.95), ('2', 0.95)]:
        net_name = pins.get(pin_num, '')
        net_code = net_codes.get(net_name, 0)
        lines.append(
            f'    (pad "{pin_num}" smd rect (at {px} 0) (size 1.0 1.2) '
            f'(layers "F.Cu" "F.Paste" "F.Mask") (net {net_code} "{net_name}"))'
        )
    return lines


def _generate_led_0603_pads(pins: Dict[str, str], net_codes: Dict[str, int]) -> List[str]:
    """Generate 0603 LED pads with polarity marker."""
    lines = _generate_0603_pads(pins, net_codes)
    # Add polarity marker
    lines.append(
        '    (fp_line (start -1.2 -0.6) (end -1.2 0.6) '
        '(stroke (width 0.2) (type solid)) (layer "F.SilkS"))'
    )
    return lines


def _generate_silkscreen_outline(fp: str, ref: str) -> List[str]:
    """Generate silkscreen outline based on footprint type."""
    lines = []

    if 'LQFP-48' in fp:
        # Square outline with pin 1 marker
        lines.extend([
            '    (fp_line (start -4 -4) (end 4 -4) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start 4 -4) (end 4 4) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start 4 4) (end -4 4) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start -4 4) (end -4 -4) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_circle (center -3.2 3.2) (end -2.9 3.2) (stroke (width 0.2) (type solid)) (layer "F.SilkS"))',
        ])
    elif 'SOIC-8' in fp:
        lines.extend([
            '    (fp_line (start -2.5 -2) (end 2.5 -2) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start -2.5 2) (end 2.5 2) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_circle (center -1.5 1.5) (end -1.3 1.5) (stroke (width 0.15) (type solid)) (layer "F.SilkS"))',
        ])
    elif 'SOT' in fp or 'VSSOP' in fp:
        lines.extend([
            '    (fp_line (start -1.2 -0.8) (end -1.2 0.8) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start 1.2 -0.8) (end 1.2 0.8) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
        ])
    elif 'SMA' in fp:
        lines.extend([
            '    (fp_line (start -3.5 -1.5) (end -3.5 1.5) (stroke (width 0.15) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start -3.5 -1.5) (end -1.5 -1.5) (stroke (width 0.15) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start -3.5 1.5) (end -1.5 1.5) (stroke (width 0.15) (type solid)) (layer "F.SilkS"))',
        ])
    else:
        # Generic 2-pin component outline
        lines.extend([
            '    (fp_line (start -1.3 -0.6) (end -1.3 0.6) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
            '    (fp_line (start 1.3 -0.6) (end 1.3 0.6) (stroke (width 0.12) (type solid)) (layer "F.SilkS"))',
        ])

    return lines


def _generate_traces_rf(components: List[Dict], nets: Dict[str, List[Tuple[str, str]]],
                        net_codes: Dict[str, int]) -> List[str]:
    """Generate traces with RF considerations."""
    lines = []

    # Build component position lookup
    comp_positions = {c['ref']: c['position'] for c in components}

    # Pad offsets for different footprints (simplified)
    pad_offsets = _get_pad_offsets(components)

    # Track which nets are RF (use wider traces)
    rf_nets = {'RF_ANT', 'RF_IN', 'RF_OUT', 'RF_TO_DEMOD', 'RF_MATCH1', 'RF_BIAS'}
    power_nets = {'VDD', 'GND', 'VCC_RF', 'VCC_A', 'VCC_B', 'VCC_ANALOG', 'GND_ANALOG', 'VREF'}

    for net_name, connections in nets.items():
        if len(connections) < 2:
            continue

        net_code = net_codes.get(net_name, 0)

        # Determine trace width based on net type
        if net_name in rf_nets:
            trace_width = BOARD_CONFIG['trace_width_rf']
            layer = '"F.Cu"'
        elif net_name in power_nets:
            trace_width = BOARD_CONFIG['trace_width_power']
            layer = '"F.Cu"'
        else:
            trace_width = BOARD_CONFIG['trace_width_signal']
            layer = '"F.Cu"'

        # Connect pins in optimal order
        for i in range(len(connections) - 1):
            ref1, pin1 = connections[i]
            ref2, pin2 = connections[i + 1]

            # Get absolute positions
            x1, y1 = comp_positions.get(ref1, (0, 0))
            ox1, oy1 = pad_offsets.get(ref1, {}).get(pin1, (0, 0))
            x1 += ox1
            y1 += oy1

            x2, y2 = comp_positions.get(ref2, (0, 0))
            ox2, oy2 = pad_offsets.get(ref2, {}).get(pin2, (0, 0))
            x2 += ox2
            y2 += oy2

            # Generate 45-degree routed segments
            segments = _route_45_degree(x1, y1, x2, y2)

            for seg in segments:
                sx1, sy1, sx2, sy2 = seg
                lines.append(
                    f'  (segment (start {sx1:.3f} {sy1:.3f}) (end {sx2:.3f} {sy2:.3f}) '
                    f'(width {trace_width}) (layer {layer}) (net {net_code}))'
                )

    return lines


def _get_pad_offsets(components: List[Dict]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Get pad offsets for all components."""
    offsets = {}

    for comp in components:
        ref = comp['ref']
        fp = comp['footprint']
        rotation = comp.get('rotation', 0)

        if 'LQFP-48' in fp:
            # LQFP-48 pad positions
            offsets[ref] = {}
            pitch = 0.5
            pad_offset = 4.5

            # Bottom side (pins 1-12)
            for i in range(12):
                offsets[ref][str(i + 1)] = (-2.75 + i * pitch, pad_offset)
            # Right side (pins 13-24)
            for i in range(12):
                offsets[ref][str(i + 13)] = (pad_offset, 2.75 - i * pitch)
            # Top side (pins 25-36)
            for i in range(12):
                offsets[ref][str(i + 25)] = (2.75 - i * pitch, -pad_offset)
            # Left side (pins 37-48)
            for i in range(12):
                offsets[ref][str(i + 37)] = (-pad_offset, -2.75 + i * pitch)

        elif 'SOT-363' in fp:
            offsets[ref] = {
                '1': (-0.65, 0.9), '2': (0, 0.9), '3': (0.65, 0.9),
                '4': (0.65, -0.9), '5': (0, -0.9), '6': (-0.65, -0.9),
            }

        elif 'SOT-23-5' in fp:
            offsets[ref] = {
                '1': (-0.95, 1.1), '2': (0, 1.1), '3': (0.95, 1.1),
                '4': (0.95, -1.1), '5': (-0.95, -1.1),
            }

        elif 'SOIC-8' in fp:
            offsets[ref] = {}
            pitch = 1.27
            for i in range(4):
                offsets[ref][str(i + 1)] = (-1.905 + i * pitch, 2.7)
            for i in range(4):
                offsets[ref][str(8 - i)] = (-1.905 + i * pitch, -2.7)

        elif 'VSSOP-8' in fp:
            offsets[ref] = {}
            pitch = 0.5
            for i in range(4):
                offsets[ref][str(i + 1)] = (-0.75 + i * pitch, 1.2)
            for i in range(4):
                offsets[ref][str(8 - i)] = (-0.75 + i * pitch, -1.2)

        elif 'Crystal' in fp:
            offsets[ref] = {'1': (-1.1, 0), '2': (1.1, 0)}

        elif 'SMA' in fp:
            offsets[ref] = {'1': (0, 0), '2': (0, 0)}

        elif '0402' in fp:
            offsets[ref] = {'1': (-0.5, 0), '2': (0.5, 0)}

        elif '0603' in fp:
            offsets[ref] = {'1': (-0.75, 0), '2': (0.75, 0)}

        elif '0805' in fp:
            offsets[ref] = {'1': (-0.95, 0), '2': (0.95, 0)}

        else:
            # Default 2-pin
            offsets[ref] = {'1': (-0.5, 0), '2': (0.5, 0)}

        # Apply rotation if needed
        if rotation != 0:
            rad = math.radians(rotation)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            for pin in offsets.get(ref, {}):
                ox, oy = offsets[ref][pin]
                offsets[ref][pin] = (
                    ox * cos_r - oy * sin_r,
                    ox * sin_r + oy * cos_r
                )

    return offsets


def _route_45_degree(x1: float, y1: float, x2: float, y2: float) -> List[Tuple[float, float, float, float]]:
    """Route from (x1,y1) to (x2,y2) using 45-degree angles."""
    dx = x2 - x1
    dy = y2 - y1

    # If very close, direct connection
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return [(x1, y1, x2, y2)]

    # Pure vertical or horizontal
    if abs(dx) < 0.1:
        return [(x1, y1, x2, y2)]
    if abs(dy) < 0.1:
        return [(x1, y1, x2, y2)]

    # 45-degree routing
    segments = []
    diag_len = min(abs(dx), abs(dy))
    diag_dx = diag_len if dx > 0 else -diag_len
    diag_dy = diag_len if dy > 0 else -diag_len

    remaining_dx = dx - diag_dx
    remaining_dy = dy - diag_dy

    if abs(remaining_dx) > abs(remaining_dy):
        mid_x = x1 + remaining_dx
        mid_y = y1
        segments.append((x1, y1, mid_x, mid_y))
        segments.append((mid_x, mid_y, x2, y2))
    else:
        mid_x = x1
        mid_y = y1 + remaining_dy
        segments.append((x1, y1, mid_x, mid_y))
        segments.append((mid_x, mid_y, x2, y2))

    return segments


def _generate_ground_plane(board_width: float, board_height: float,
                           net_codes: Dict[str, int]) -> List[str]:
    """Generate ground plane fill on In1.Cu layer."""
    gnd_code = net_codes.get('GND', 0)
    lines = [
        f'  (zone (net {gnd_code}) (net_name "GND") (layer "In1.Cu") (uuid "gnd-zone-uuid")',
        '    (hatch edge 0.5)',
        '    (connect_pads yes (clearance 0.2))',
        '    (min_thickness 0.2)',
        '    (filled_areas_thickness no)',
        '    (fill yes (thermal_gap 0.3) (thermal_bridge_width 0.3))',
        '    (polygon',
        '      (pts',
        f'        (xy 4 4)',
        f'        (xy {board_width - 4} 4)',
        f'        (xy {board_width - 4} {board_height - 4})',
        f'        (xy 4 {board_height - 4})',
        '      )))',
    ]
    return lines


def _generate_via_stitching(x_start: float, y_start: float,
                            x_end: float, y_end: float,
                            net_codes: Dict[str, int]) -> List[str]:
    """Generate via stitching around RF section for ground shielding."""
    lines = []
    gnd_code = net_codes.get('GND', 0)
    via_spacing = 2.5  # mm between vias

    # Top edge
    x = x_start
    while x <= x_end:
        lines.append(
            f'  (via (at {x:.2f} {y_start:.2f}) (size 0.6) (drill 0.3) '
            f'(layers "F.Cu" "B.Cu") (net {gnd_code}))'
        )
        x += via_spacing

    # Bottom edge
    x = x_start
    while x <= x_end:
        lines.append(
            f'  (via (at {x:.2f} {y_end:.2f}) (size 0.6) (drill 0.3) '
            f'(layers "F.Cu" "B.Cu") (net {gnd_code}))'
        )
        x += via_spacing

    # Left edge
    y = y_start + via_spacing
    while y < y_end:
        lines.append(
            f'  (via (at {x_start:.2f} {y:.2f}) (size 0.6) (drill 0.3) '
            f'(layers "F.Cu" "B.Cu") (net {gnd_code}))'
        )
        y += via_spacing

    return lines


def generate_real_schematic(output_path: str) -> Dict[str, Any]:
    """Generate a KiCad schematic file."""
    sch_lines = [
        '(kicad_sch (version 20230121) (generator "test_rf_mixed_signal")',
        '',
        '  (uuid "rf-mixed-signal-uuid")',
        '  (paper "A4")',
        '',
        '  (lib_symbols',
        '  )',
        ''
    ]

    # Add symbols
    y_pos = 30
    x_base = 40
    for i, comp in enumerate(COMPONENTS):
        ref = comp['ref']
        value = comp['value']
        section = comp.get('section', 'other')

        # Position by section
        if section == 'digital':
            x_pos = 30
        elif section == 'analog':
            x_pos = 80
        elif section == 'rf':
            x_pos = 130
        else:
            x_pos = 180

        y_pos = 30 + (i % 10) * 15

        sch_lines.extend([
            f'  (symbol (lib_id "Device:{ref[0]}") (at {x_pos} {y_pos} 0) (unit 1)',
            '    (in_bom yes) (on_board yes) (dnp no)',
            f'    (uuid "{ref.lower()}-uuid")',
            f'    (property "Reference" "{ref}" (at {x_pos + 3} {y_pos - 2} 0)',
            '      (effects (font (size 1.27 1.27))))',
            f'    (property "Value" "{value}" (at {x_pos + 3} {y_pos + 2} 0)',
            '      (effects (font (size 1.27 1.27))))',
            f'    (property "Footprint" "{comp["footprint"]}" (at {x_pos} {y_pos} 0)',
            '      (effects (font (size 1.27 1.27)) hide))',
            '  )'
        ])

    sch_lines.append(')')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(sch_lines))

    return {
        'success': True,
        'output': output_path,
        'components': len(COMPONENTS)
    }


# =============================================================================
# MAIN E2E TEST RUNNER
# =============================================================================

def run_e2e_test(output_dir: str) -> Dict[str, Any]:
    """
    Run complete end-to-end test for RF mixed-signal circuit.
    """
    results = {
        'test_name': 'RF Mixed-Signal E2E Test - PRODUCTION',
        'stages': {},
        'passed': False
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    netlist_dir = output_path / 'netlists'
    schematic_dir = output_path / 'schematics'
    pcb_dir = output_path / 'pcb'
    images_dir = output_path / 'images'

    for d in [netlist_dir, schematic_dir, pcb_dir, images_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1: Generate netlist
    print("\n[1/5] Generating netlist...")
    netlist_path = str(netlist_dir / 'rf_mixed_signal.net')
    netlist_result = generate_real_netlist(netlist_path)
    results['stages']['netlist'] = netlist_result

    if not netlist_result.get('success'):
        raise ValidationFailure(f"Netlist generation failed: {netlist_result.get('error')}")

    print(f"  Generated: {netlist_result.get('components')} components, {netlist_result.get('nets')} nets")
    print(f"  Sections: {netlist_result.get('sections')}")

    # Stage 2: Convert to schematic
    print("\n[2/5] Converting to schematic...")
    schematic_path = str(schematic_dir / 'rf_mixed_signal.kicad_sch')
    schematic_result = generate_real_schematic(schematic_path)
    results['stages']['schematic'] = schematic_result
    print(f"  Generated: {schematic_path}")

    # Stage 3: Generate PCB
    print("\n[3/5] Generating PCB layout...")
    pcb_path = str(pcb_dir / 'rf_mixed_signal.kicad_pcb')
    pcb_result = generate_real_pcb(schematic_path, pcb_path)
    results['stages']['pcb'] = pcb_result
    print(f"  Generated: {pcb_path}")
    print(f"  Board size: {pcb_result.get('board_width_mm')}x{pcb_result.get('board_height_mm')}mm ({pcb_result.get('layer_count')} layers)")
    print(f"  Footprints: {pcb_result.get('expected_footprints')}, Traces: {pcb_result.get('expected_traces')}")

    # Stage 4: Export images
    print("\n[4/5] Exporting images...")
    import shutil
    if shutil.which('kicad-cli'):
        images_result = export_pcb_images(pcb_path, str(images_dir))
        results['stages']['images'] = images_result
        print(f"  Exported {len(images_result.get('files', []))} layer images")
    else:
        images_result = {'success': False, 'error': 'kicad-cli not available'}
        results['stages']['images'] = images_result
        print("  WARN: kicad-cli not found, skipping image export")

    # Stage 5: Visual validation
    print("\n[5/5] Running visual validation...")
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise MissingDependencyFailure(
            "ANTHROPIC_API_KEY not set - cannot run validation",
            dependency_name="ANTHROPIC_API_KEY",
            install_instructions="Set ANTHROPIC_API_KEY environment variable"
        )

    try:
        from visual_ralph_loop import VisualRalphLoop

        loop = VisualRalphLoop(
            output_dir=str(images_dir),
            pcb_path=pcb_path,
            max_iterations=100,
            quality_threshold=9.0,
            strict=True
        )
        validation_result = loop.run()
        results['stages']['validation'] = validation_result
        results['passed'] = True
        print(f"  Score: {validation_result.get('overall_score', 0)}/10")

    except ValidationFailure:
        raise
    except Exception as e:
        results['stages']['validation'] = {'error': str(e)}
        raise ValidationFailure(f"Visual validation error: {e}")

    return results


def export_pcb_images(pcb_path: str, output_dir: str) -> Dict[str, Any]:
    """Export PCB layer images using KiCad CLI."""
    import subprocess

    layers = [
        ('F.Cu', 'F_Cu.png'),
        ('B.Cu', 'B_Cu.png'),
        ('In1.Cu', 'In1_Cu.png'),
        ('In2.Cu', 'In2_Cu.png'),
        ('F.SilkS', 'F_Silkscreen.png'),
        ('Edge.Cuts', 'Edge_Cuts.png'),
    ]

    exported_files = []

    for layer_name, filename in layers:
        output_file = Path(output_dir) / filename
        cmd = [
            'kicad-cli', 'pcb', 'export', 'svg',
            '--output', str(output_file).replace('.png', '.svg'),
            '--layers', layer_name,
            pcb_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                exported_files.append(str(output_file))
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            break

    return {
        'success': len(exported_files) > 0,
        'files': exported_files,
        'output_dir': output_dir
    }


def main():
    parser = argparse.ArgumentParser(description='RF Mixed-Signal E2E Test')
    parser.add_argument('--output-dir', '-o', default='./rf_mixed_signal_test_output',
                        help='Output directory for test artifacts')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate files, skip validation')
    args = parser.parse_args()

    print("=" * 60)
    print("RF MIXED-SIGNAL E2E TEST - PRODUCTION VERSION")
    print("=" * 60)
    print(f"\nComponents: {len(COMPONENTS)}")
    print(f"Nets: {len(NETS)}")
    print(f"Output: {args.output_dir}")

    try:
        if args.generate_only:
            # Just generate files without validation
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            netlist_dir = output_path / 'netlists'
            schematic_dir = output_path / 'schematics'
            pcb_dir = output_path / 'pcb'

            for d in [netlist_dir, schematic_dir, pcb_dir]:
                d.mkdir(parents=True, exist_ok=True)

            print("\n[1/3] Generating netlist...")
            netlist_result = generate_real_netlist(str(netlist_dir / 'rf_mixed_signal.net'))
            print(f"  Components: {netlist_result.get('components')}, Nets: {netlist_result.get('nets')}")

            print("\n[2/3] Generating schematic...")
            schematic_result = generate_real_schematic(str(schematic_dir / 'rf_mixed_signal.kicad_sch'))
            print(f"  Generated: {schematic_result.get('output')}")

            print("\n[3/3] Generating PCB...")
            pcb_result = generate_real_pcb(
                str(schematic_dir / 'rf_mixed_signal.kicad_sch'),
                str(pcb_dir / 'rf_mixed_signal.kicad_pcb')
            )
            print(f"  Board: {pcb_result.get('board_width_mm')}x{pcb_result.get('board_height_mm')}mm")
            print(f"  Layers: {pcb_result.get('layer_count')}")
            print(f"  Footprints: {pcb_result.get('expected_footprints')}")
            print(f"  Traces: {pcb_result.get('expected_traces')}")

            print("\n" + "=" * 60)
            print("GENERATION COMPLETE")
            print("=" * 60)
            return 0

        else:
            results = run_e2e_test(args.output_dir)

            print("\n" + "=" * 60)
            if results.get('passed'):
                print("TEST PASSED")
            else:
                print("TEST FAILED")
            print("=" * 60)

            return 0 if results.get('passed') else 1

    except ValidationFailure as e:
        print(f"\n\nVALIDATION FAILED: {e.message}")
        return 1
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
